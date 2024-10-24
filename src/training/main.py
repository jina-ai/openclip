import logging
import os
import shutil
import sys
import warnings
from datetime import datetime

import deepspeed.comm
import numpy as np
import torch
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_clip import create_model_and_transforms, get_tokenizer, resize_eva_pos_embed
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.eval import evaluate
from training.factory import create_dataloaders, create_losses
from training.optimizer import create_optimizer
from training.params import parse_args
from training.scheduler import create_scheduler
from training.train import DatasetRecordHistory, train_one_epoch
from training.utils import (
    copy_codebase,
    get_latest_checkpoint,
    pytorch_load,
    random_seed,
    remote_sync,
    setup_logging,
    start_sync_process,
)

LATEST_CHECKPOINT_NAME = 'epoch-latest.pt'

warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.checkpoint')
warnings.filterwarnings('ignore', category=UserWarning, module='xformers.ops.fmha')


def main(args):
    args, dsinit = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use,
        # easier if we don't use / in name as a rule?
        _model_name_safe = args.model.replace('/', '-')
        datestr = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        if args.distributed:
            # sync date_str from master to all ranks
            datestr = broadcast_object(args, datestr)
        args.name = '-'.join(
            [
                datestr,
                f'model_{_model_name_safe}',
                f'lr_{args.lr}',
                f'b_{args.batch_size}',
                f'j_{args.workers}',
                f'p_{args.precision}',
            ]
        )

    _resume_latest = args.resume == 'latest'
    _resume_logs = args.resume_logs and (args.resume is not None)
    _log_base_path = os.path.join(args.logs, args.name)

    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(_log_base_path, exist_ok=True)
        _log_filename = f'out-{args.rank}.log' if args.log_local else 'out.log'
        args.log_path = os.path.join(_log_base_path, _log_filename)
        if os.path.exists(args.log_path) and not (_resume_latest or _resume_logs):
            logger.error(
                f'Experiment {args.name} already exists. Use --name to '
                'specify a new experiment.'
            )
            return -1

    # setup logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(
        args.log_path, args.log_level, args.rank, include_host=True, include_rank=True
    )

    # setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(_log_base_path, 'checkpoints')

    if is_master(args):
        args.tensorboard_path = (
            os.path.join(_log_base_path, 'tensorboard') if args.tensorboard else ''
        )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if _resume_latest:
        _resume_from = None
        _checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of
        # the local checkpoints folder.
        if args.remote_sync is not None:
            _checkpoint_path = os.path.join(args.remote_sync, args.name, 'checkpoints')
            if args.save_most_recent:
                logger.error(
                    'Error. Cannot use save-most-recent with remote_sync and '
                    'resume latest.'
                )
                return -1
            if args.remote_sync_protocol != 's3':
                logger.error(
                    'Error. Sync protocol not supported when using resume latest.'
                )
                return -1

        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system
            # is under stress, however it's very difficult to fully work around such
            # situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                _resume_from = os.path.join(_checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(_resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    _resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                _resume_from = get_latest_checkpoint(
                    _checkpoint_path, remote=args.remote_sync is not None
                )
            if _resume_from:
                logger.info(f'Found latest checkpoint: {_resume_from}.')
            else:
                logger.info(f'No latest checkpoint found in {_checkpoint_path}.')

        if args.distributed:
            # sync found checkpoint path to all ranks
            _resume_from = broadcast_object(args, _resume_from)

        args.resume = _resume_from

    if args.copy_codebase:
        copy_codebase(args.logs, args.name)

    # start the sync proces if remote-sync is not None
    _remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        logger.info('Checking remote sync ...')

        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logger.info('Remote sync successful')
        else:
            logger.error('Error: remote sync failed, exiting ...')
            return -1

        # if all looks good, start a process to do this every
        # args.remote_sync_frequency seconds
        _remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        _remote_sync_process.start()

    if args.precision == 'fp16':
        logger.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for '
            'training.'
        )

    if args.horovod:
        logger.info(
            f'Running in horovod mode with multiple processes / nodes. '
            f'Device: {args.device}. Process (global: {args.rank}, local '
            f'{args.local_rank}), total {args.world_size}.'
        )
    elif args.distributed:
        logger.info(
            f'Running in distributed mode with multiple processes. '
            f'Device: {args.device}. Process (global: {args.rank}, local '
            f'{args.local_rank}), total {args.world_size}.'
        )
    else:
        logger.info(f'Running with a single process, device {args.device}.')

    distill_model = None
    args.distill = (
        args.distill_model is not None and args.distill_pretrained is not None
    )
    if args.distill:
        assert (
            args.accum_freq == 1
        ), 'Model distillation does not work with gradient accumulation'
        assert (
            'coca' not in args.model.lower()
        ), 'Model distillation does not work with CoCa models'

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)

    _model_kwargs = {'freeze_logit_scale': False}
    if args.siglip:
        _model_kwargs.update(
            {
                'init_logit_scale': np.log(1 / 0.1),
                'init_logit_bias': -10,
            }
        )
    else:
        _model_kwargs.update(
            {
                'init_logit_scale': np.log(1 / 0.07),
                'init_logit_bias': None,
            }
        )

    if args.temperature:
        _model_kwargs['init_logit_scale'] = np.log(1 / args.temperature)

    if args.freeze_temperature:
        _model_kwargs['freeze_logit_scale'] = True

    model, preprocess_cfg, preprocess_train, preprocess_val = (
        create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            pretrained_hf=not args.hf_random_init,
            **_model_kwargs,
        )
    )
    if args.distill:
        distill_model, _, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        logger.warning(
            '=> using a layer from bitsandbytes.\n'
            '   this is an experimental feature which requires two extra pip installs\n'
            '   pip install bitsandbytes triton'
            '   please make sure to use triton 2.0.0'
        )
        import bitsandbytes as bnb

        from open_clip.utils import replace_linear

        logger.debug(f'=> replacing linear layers with {args.use_bnb_linear}')
        _linear_replacement_cls = getattr(
            bnb.nn.triton_based_modules, args.use_bnb_linear
        )
        replace_linear(model, _linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_groups,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    _params_file = ''
    if is_master(args):
        logger.info(f'Model: {str(model)}')
        logger.debug('Parameters:')
        _params_file = os.path.join(args.logs, args.name, 'params.txt')
        with open(_params_file, 'w') as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logger.debug(f'  {name}: {val}')
                f.write(f'{name}: {val}\n')

    if args.distributed and not (args.horovod or args.deepspeed):
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        _ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            _ddp_args['static_graph'] = True

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **_ddp_args
        )
        if args.distill:
            distill_model = torch.nn.parallel.DistributedDataParallel(
                distill_model, device_ids=[device], **_ddp_args
            )

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data or args.dataset_type == 'synthetic':
        model, optimizer, scaler = create_optimizer(
            args=args, model=model, dsinit=dsinit
        )

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.deepspeed:
            if os.path.exists(args.resume):
                import glob

                _all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch-*'))
                _latest_ckpt = -1
                for ckpt in _all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        _latest_ckpt = max(int(t), _latest_ckpt)

                if _latest_ckpt >= 0:
                    start_epoch = _latest_ckpt
                    _, client_states = model.load_checkpoint(
                        args.resume, tag=f'epoch-{_latest_ckpt}'
                    )
                    logger.info(
                        f"=> resuming from checkpoint '{args.resume}' "
                        f'(epoch {_latest_ckpt})'
                    )
                else:
                    logger.info(f"=> no checkpoint found at '{args.resume}'")
            else:
                logger.info(f"=> '{args.resume}' does not exist!")
        else:
            checkpoint = pytorch_load(
                os.path.join(args.resume, 'state.pt'), map_location='cpu'
            )
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint['epoch']
                sd = checkpoint['state_dict']
                if not args.distributed and next(iter(sd.items()))[0].startswith(
                    'module'
                ):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                sd = {key: value for key, value in sd.items() if 'rope' not in key}
                resize_eva_pos_embed(sd, model)
                model.load_state_dict(sd, strict=False)
                if optimizer is not None:
                    checkpoint['optimizer']['param_groups'] = optimizer.state_dict()[
                        'param_groups'
                    ]

                    # Retrieve the tensors
                    exp_avg = checkpoint['optimizer']['state'][800]['exp_avg']
                    exp_avg_sq = checkpoint['optimizer']['state'][800]['exp_avg_sq']

                    # Calculate padding needed on the sequence dimension
                    new_seq_len = sd.get(
                        'visual.pos_embed', sd.get('module.visual.pos_embed')
                    ).size(1)
                    old_seq_len = exp_avg.size(1)
                    pad_size = new_seq_len - old_seq_len
                    if pad_size > 0:
                        logger.info(
                            f'Paddind position embedding optimizer gradients: '
                            f'length {old_seq_len} -> {new_seq_len}'
                        )
                        pad = (0, 0, 0, pad_size)
                        checkpoint['optimizer']['state'][800]['exp_avg'] = (
                            torch.nn.functional.pad(exp_avg, pad)
                        )
                        checkpoint['optimizer']['state'][800]['exp_avg_sq'] = (
                            torch.nn.functional.pad(exp_avg_sq, pad)
                        )

                    optimizer.load_state_dict(checkpoint['optimizer'])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logger.info(
                    f"=> resuming from checkpoint '{args.resume}' "
                    f'(epoch {start_epoch})'
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logger.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )

    embed_dim = (
        model.module.embed_dim
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.embed_dim
    )
    loss, mtl_losses = create_losses(args, embed_dim=embed_dim)

    tokenizer = get_tokenizer(args.model, context_length=args.max_sequence_length)
    data = create_dataloaders(
        args,
        preprocess_train,
        preprocess_val,
        epoch=start_epoch,
        tokenizer=tokenizer,
        preprocess_cfg=preprocess_cfg,
        mtl_losses=mtl_losses,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    dataset_records = None
    if args.save_dataset_records and is_master(args):
        dataset_records = DatasetRecordHistory()
        if args.resume:
            records_ckpt = os.path.join(args.resume, 'dataset-records.bin')
            if os.path.exists(records_ckpt):
                dataset_records = DatasetRecordHistory.load(records_ckpt)

    # create scheduler if training
    scheduler = None
    if 'train' in data and optimizer is not None:
        _total_steps = (
            data['train'].dataloader.num_batches // args.accum_freq
        ) * args.epochs
        _cooldown_steps = None
        if args.epochs_cooldown is not None:
            _cooldown_steps = (
                data['train'].dataloader.num_batches // args.accum_freq
            ) * args.epochs_cooldown
        scheduler = create_scheduler(
            optimizer=optimizer,
            baselr=args.lr,
            warmup_steps=args.warmup,
            total_steps=_total_steps,
            cooldown_steps=_cooldown_steps,
            cooldown_power=args.lr_cooldown_power,
            cooldown_end_lr=args.lr_cooldown_end,
            scheduler_type=args.lr_scheduler,
        )

    args.save_logs = args.logs and args.logs.lower() != 'none'
    writer = None
    if args.save_logs and is_master(args) and args.tensorboard:
        assert tensorboard is not None, 'Please install tensorboard.'
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'

        logger.debug('Starting WandB ...')
        args.train_sz = data['train'].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data['val'].dataloader.num_samples

        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if (_resume_latest or _resume_logs) else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')

        wandb.save(_params_file)
        logger.debug('Finished setting up WandB')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logger.info('Compiling model ...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode

            convert_int8_model_to_inference_mode(model)
        evaluate(
            model,
            preprocess_val,
            tokenizer,
            data,
            start_epoch,
            args,
            tb_writer=writer,
        )
        return

    if args.evaluate_on_start:
        evaluate(
            model,
            preprocess_val,
            tokenizer,
            data,
            start_epoch,
            args,
            tb_writer=writer,
        )

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logger.info(f'Starting epoch {epoch}')

        if args.save_dataset_records and not is_master(args):
            dataset_records = DatasetRecordHistory()

        dataset_records = train_one_epoch(
            model,
            data,
            preprocess_val,
            tokenizer,
            loss,
            mtl_losses,
            epoch,
            optimizer,
            scaler,
            scheduler,
            distill_model,
            args,
            dataset_records=dataset_records,
            tb_writer=writer,
        )
        _completed_epoch = epoch + 1

        # saving checkpoints
        if args.save_logs:

            # --- CREATE CHECKPOINT DIRECTORY
            _ckpt_dir = os.path.join(args.checkpoint_path, f'epoch-{_completed_epoch}')
            if is_master(args):
                os.makedirs(_ckpt_dir, exist_ok=False)

            # wait for all ranks
            if args.deepspeed:
                deepspeed.comm.barrier()
            else:
                torch.distributed.barrier()

            # --- SAVE MODEL
            if args.deepspeed:
                _ds_checkpoint_path = os.path.join(args.logs, args.name, 'checkpoints')
                if _completed_epoch == args.epochs or (
                    args.save_frequency > 0
                    and (_completed_epoch % args.save_frequency) == 0
                ):
                    client_state = {'epoch': _completed_epoch}
                    model.save_checkpoint(
                        save_dir=_ds_checkpoint_path,
                        tag=f'epoch-{str(_completed_epoch)}',
                        client_state=client_state,
                    )
            elif is_master(args):
                _checkpoint_dict = {
                    'epoch': _completed_epoch,
                    'name': args.name,
                    'state_dict': original_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scaler is not None:
                    _checkpoint_dict['scaler'] = scaler.state_dict()

                if _completed_epoch == args.epochs or (
                    args.save_frequency > 0
                    and (_completed_epoch % args.save_frequency) == 0
                ):
                    _model_ckpt_path = os.path.join(_ckpt_dir, 'state.pt')
                    torch.save(_checkpoint_dict, _model_ckpt_path)

                if args.save_most_recent:
                    # try not to corrupt the latest checkpoint if save fails
                    _tmp_save_path = os.path.join(args.checkpoint_path, 'tmp.pt')
                    _latest_save_path = os.path.join(
                        args.checkpoint_path, LATEST_CHECKPOINT_NAME
                    )
                    torch.save(_checkpoint_dict, _tmp_save_path)
                    os.replace(_tmp_save_path, _latest_save_path)

            # --- SAVE DATASET RECORDS

            if not is_master(args):
                if args.save_dataset_records:
                    dataset_records.save(
                        os.path.join(
                            _ckpt_dir, f'worker{args.rank}-dataset-records.bin'
                        )
                    )

            # save text dataset checkpoints
            if data['train-text'] is not None:
                data['train-text'][0].write_to_json(os.path.join(
                    _ckpt_dir, f'worker{args.rank}-s3-dataset.json',
                ))

            # save MTL dataset checkpoints
            if data['train-mtl'] is not None:
                data['train-mtl'][0].write_to_json(os.path.join(
                    _ckpt_dir, f'worker{args.rank}-mtl-dataset.json',
                ))

            # wait for all ranks again
            if args.deepspeed:
                deepspeed.comm.barrier()
            else:
                torch.distributed.barrier()

            if is_master(args):
                if args.save_dataset_records:
                    for rank in range(args.world_size):
                        if rank != args.rank:
                            worker_ckpt = os.path.join(
                                _ckpt_dir, f'worker{rank}-dataset-records.bin'
                            )
                            rank_dataset_records = DatasetRecordHistory.load(
                                worker_ckpt
                            )
                            dataset_records.merge(rank_dataset_records)
                            os.remove(worker_ckpt)
                    dataset_records.save(
                        os.path.join(_ckpt_dir, f'dataset-records.bin')
                    )

            # --- HOUSEKEEPING
            if is_master(args):
                if args.delete_previous_checkpoint:
                    _previous_checkpoint = os.path.join(
                        args.checkpoint_path, f'epoch-{_completed_epoch - 1}'
                    )
                    if os.path.exists(_previous_checkpoint):
                        shutil.rmtree(_previous_checkpoint)

        if is_master(args):
            evaluate(
                model,
                preprocess_val,
                tokenizer,
                data,
                _completed_epoch,
                args,
                tb_writer=writer,
            )

    # stop wandb
    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync
    if _remote_sync_process is not None:
        _remote_sync_process.terminate()

        logger.info('Final remote sync ...')
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logger.info('Final remote sync successful')
        else:
            logger.info('Final remote sync failed')


if __name__ == '__main__':
    main(sys.argv[1:])
