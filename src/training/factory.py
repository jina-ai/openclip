import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import DataLoader
from training.data import (
    InputType,
    MultiDataset,
    dynamic_collate,
    get_csv_dataset,
    get_imagenet,
    get_synthetic_dataset,
    get_wds_dataset,
)
from training.loss import (
    CoCaLoss,
    DistillInfoNCELoss,
    InfoNCELoss,
    MatryoshkaInfoNCELoss,
    MatryoshkaSigLIPLoss,
    SigLIPLoss,
    ThreeTowersLoss,
)
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

DEFAULT_CONTEXT_LENGTH = 77


def _create_contrastive_loss(args):
    if args.distill:
        return DistillInfoNCELoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif 'coca' in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif '3towers' in args.model.lower():
        return ThreeTowersLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )

    if args.siglip:
        assert not args.horovod, 'Horovod not currently supported for SigLIP'
        if args.matryoshka:
            return MatryoshkaSigLIPLoss(rank=args.rank, world_size=args.world_size)
        return SigLIPLoss(rank=args.rank, world_size=args.world_size)

    if args.matryoshka:
        return MatryoshkaInfoNCELoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )

    return InfoNCELoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def _create_mtl_losses(args):
    import training.loss as loss_module

    try:
        lossinit = json.loads(args.mtl_loss)
    except json.JSONDecodeError:
        lossinit = [{'name': loss} for loss in args.mtl_loss.split(',')]

    lossinit = lossinit or [{'name': 'InfoNCELoss'}]
    d: Dict[str, Any]
    for d in lossinit:
        d['name'] = getattr(loss_module, d['name'])
        if 'tasks' not in d:
            d['tasks'] = '*'
        if 'options' not in d:
            d['options'] = {}

    losses = {}
    for d in lossinit:
        for task in d['tasks']:
            logging.debug(f'Setting up loss: {d["name"].__name__}')
            lossfn = d['name'](**d['options'])
            losses[task] = lossfn

    return losses


def _create_multimodal_dataloader(
    args, preprocess_train, preprocess_val, tokenizer, batch_size, epoch
):
    if args.dataset_type == 'auto':
        _train_ext = args.train_data.split('.')[-1]
        _val_ext = args.val_data.split('.')[-1]
        if _train_ext in ['csv', 'tsv']:
            _train_dataset_type = 'csv'
        elif _train_ext in ['tar']:
            _train_dataset_type = 'webdataset'
        else:
            raise ValueError(
                f'Tried to figure out dataset type, but failed for extension '
                f'{_train_ext}.'
            )
        if _val_ext in ['csv', 'tsv']:
            _val_dataset_type = 'csv'
        elif _val_ext in ['tar']:
            _val_dataset_type = 'webdataset'
        else:
            raise ValueError(
                f'Tried to figure out dataset type, but failed for extension '
                f'{_val_ext}.'
            )
    else:
        _train_dataset_type = args.dataset_type
        _val_dataset_type = args.dataset_type

    data = {}
    if _train_dataset_type == 'webdataset':
        data['train'] = get_wds_dataset(
            shards=args.train_data,
            preprocess_fn=preprocess_train,
            num_samples=args.train_num_samples,
            is_train=True,
            tokenizer=tokenizer,
            upsampling_factors=args.train_upsampling_factors,
            workers=args.workers,
            batch_size=batch_size,
            seed=args.seed,
            epoch=epoch,
            dataset_resampled=args.dataset_resampled,
            world_size=args.world_size,
        )
    elif _train_dataset_type == 'csv':
        data['train'] = get_csv_dataset(
            fname=args.train_data,
            preprocess_fn=preprocess_train,
            is_train=True,
            tokenizer=tokenizer,
            csv_image_key=args.csv_image_key,
            csv_caption_key=args.csv_caption_key,
            csv_separator=args.csv_separator,
            distributed=args.distributed,
            workers=args.workers,
            batch_size=batch_size,
        )
    elif _train_dataset_type == 'synthetic':
        data['train'] = get_synthetic_dataset(
            num_samples=args.train_num_samples,
            preprocess_fn=preprocess_train,
            is_train=True,
            tokenizer=tokenizer,
            distributed=args.distributed,
            workers=args.workers,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f'Unsupported train dataset type: {_train_dataset_type}')

    if _val_dataset_type == 'webdataset':
        data['val'] = get_wds_dataset(
            shards=args.val_data,
            preprocess_fn=preprocess_val,
            num_samples=args.val_num_samples,
            is_train=False,
            tokenizer=tokenizer,
            workers=args.workers,
            batch_size=batch_size,
            seed=args.seed,
            epoch=epoch,
            world_size=args.world_size,
        )
    elif _val_dataset_type == 'csv':
        data['val'] = get_csv_dataset(
            fname=args.val_data,
            preprocess_fn=preprocess_val,
            is_train=False,
            tokenizer=tokenizer,
            csv_image_key=args.csv_image_key,
            csv_caption_key=args.csv_caption_key,
            csv_separator=args.csv_separator,
            distributed=args.distributed,
            workers=args.workers,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f'Unsupported val dataset type: {_val_dataset_type}')

    if args.imagenet_val is not None:
        data['imagenet-val'] = get_imagenet(
            args, (preprocess_train, preprocess_val), 'val'
        )
    if args.imagenet_v2 is not None:
        data['imagenet-v2'] = get_imagenet(
            args, (preprocess_train, preprocess_val), 'v2'
        )

    return data


def _create_s3_dataloader(
    datasets: List[str],
    bucket: str,
    input_type_dict: Dict[str, Any],
    tokenizer: Union[str, Any],
    sampling_rates: Optional[List[float]] = None,
    resume: Optional[str] = None,
    batch_size: int = 32,
    max_sequence_length: int = DEFAULT_CONTEXT_LENGTH,
    prefix: str = 's3',
    max_shards: Optional[int] = None,
    max_batches: Optional[int] = None,
    num_batches: int = 0,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
):
    dataset = None
    if resume:
        checkpoint = os.path.join(resume, f'worker{rank}-{prefix}-dataset.json')
        if os.path.isfile(checkpoint):
            logging.debug(f'Loading from checkpoint {checkpoint} ...')
            dataset = MultiDataset.load_from_json(
                checkpoint,
                world_size=world_size,
                global_rank=rank,
            )

    if dataset is None:
        logging.debug(f'Bucket: {bucket}, Datasets: {datasets}')
        sampling_rates = (
            {dataset: sr for dataset, sr in zip(datasets, sampling_rates)}
            if sampling_rates
            else None
        )
        dataset = MultiDataset(
            bucket=bucket,
            batch_size=batch_size,
            input_type_dict=input_type_dict,
            datasets=datasets,
            max_shards=max_shards,
            world_size=world_size,
            global_rank=rank,
            sampling_rates=sampling_rates,
            num_batches=num_batches,
            max_batches=max_batches,
            seed=seed,
            synchronous=True,
        )

    logging.debug('Setting up the S3 dataloader')

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, force_download=True)

    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=partial(
            dynamic_collate,
            tokenizer=tokenizer,
            tokenizer_options={
                'padding': 'max_length',
                'truncation': True,
                'max_length': max_sequence_length,
                'return_tensors': 'pt',
            },
            input_type_dict=input_type_dict,
        ),
        batch_size=batch_size,
        pin_memory=True,
    )
    return dataset, dataloader


class _DummyDataloader:
    def __iter__(self):
        return self

    def __next__(self):
        return None, (None, None)


def create_losses(args):
    loss = _create_contrastive_loss(args)
    mtllosses = None
    if args.train_data_mtl:
        mtllosses = _create_mtl_losses(args)
    return loss, mtllosses


def create_dataloaders(
    args,
    preprocess_train,
    preprocess_val,
    epoch: int,
    tokenizer: Any,
    mtl_losses: Optional[Dict[str, Any]] = None,
):
    logging.info('Creating the multimodal dataloaders ...')
    batch_size = args.batch_size
    s3_batch_size = 0

    if args.train_data_s3:
        s3_batch_size = batch_size // 2
        batch_size = batch_size - s3_batch_size

    logging.debug(f'Batch size: {batch_size}')
    data = _create_multimodal_dataloader(
        args=args,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
        epoch=epoch,
        batch_size=batch_size,
    )

    if args.train_data_s3:
        logging.info('Creating the S3 dataloader ...')
        logging.debug(f'Batch size: {s3_batch_size}')
        assert isinstance(tokenizer.tokenizer, PreTrainedTokenizer) or isinstance(
            tokenizer.tokenizer, PreTrainedTokenizerFast
        )
        upsampling_factors = None
        if args.train_data_s3_upsampling_factors:
            upsampling_factors = [
                float(v) for v in args.train_data_s3_upsampling_factors.split('::')
            ]
        data['train-s3'] = _create_s3_dataloader(
            datasets=args.train_data_s3.split('::'),
            bucket=args.train_data_s3_bucket,
            input_type_dict={'*': InputType.PAIR},
            tokenizer=tokenizer.tokenizer,
            sampling_rates=upsampling_factors,
            resume=args.resume,
            batch_size=s3_batch_size,
            max_sequence_length=args.max_sequence_length,
            prefix='s3',
            max_shards=args.s3_max_shards,
            max_batches=args.s3_max_batches,
            num_batches=args.s3_num_batches,
            seed=args.seed,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        data['train-s3'] = None

    if args.train_data_mtl:
        logging.info('Creating the MTL dataloader ...')
        logging.debug(f'Batch size: {args.mtl_batch_size}')
        assert isinstance(tokenizer.tokenizer, PreTrainedTokenizer) or isinstance(
            tokenizer.tokenizer, PreTrainedTokenizerFast
        )
        upsampling_factors = None
        if args.train_data_mtl_upsampling_factors:
            upsampling_factors = [
                float(v) for v in args.train_data_mtl_upsampling_factors.split('::')
            ]
        data['train-mtl'] = _create_s3_dataloader(
            datasets=args.train_data_mtl.split('::'),
            bucket=args.train_data_mtl_s3_bucket,
            input_type_dict={
                task: lossfn.input_type for task, lossfn in mtl_losses.items()
            },
            tokenizer=tokenizer.tokenizer,
            sampling_rates=upsampling_factors,
            resume=args.resume,
            batch_size=args.mtl_batch_size,
            max_sequence_length=args.mtl_max_sequence_length,
            prefix='mtl',
            max_shards=args.s3_max_shards,
            max_batches=args.s3_max_batches,
            num_batches=args.s3_num_batches,
            seed=args.seed,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        data['train-mtl'] = None

    return data
