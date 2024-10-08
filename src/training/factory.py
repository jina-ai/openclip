import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from open_clip.transform import PreprocessCfg
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from training.data import (
    InputType,
    MultiDataset,
    dynamic_collate,
    get_csv_dataset,
    get_synthetic_dataset,
    get_wds_dataset,
)
from training.loss import (
    CoCaLoss,
    DistillInfoNCELoss,
    InfoNCELoss,
    MatryoshkaOperator,
    SigLIPLoss,
    ThreeTowersLoss,
)

DEFAULT_CONTEXT_LENGTH = 77


def _create_contrastive_loss(args, embed_dim: int):
    logger.info('Creating the contrastive loss ...')
    if args.distill:
        logger.debug(f'Loss class: {DistillInfoNCELoss.__name__}')
        loss = DistillInfoNCELoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif 'coca' in args.model.lower():
        logger.debug(f'Loss class: {CoCaLoss.__name__}')
        loss = CoCaLoss(
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
        logger.debug(f'Loss class: {ThreeTowersLoss.__name__}')
        loss = ThreeTowersLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, 'Horovod not currently supported for SigLIP'

        logger.debug(f'Loss class: {SigLIPLoss.__name__}')
        loss = SigLIPLoss(
            rank=args.rank,
            world_size=args.world_size,
            bidirectional=args.siglip_bidirectional,
            chunked=args.siglip_chunked,
        )
    else:
        logger.debug(f'Loss class: {InfoNCELoss.__name__}')
        loss = InfoNCELoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )

    if args.matryoshka:
        dims = [int(dim) for dim in args.matryoshka_dims.split(',')]
        weights = (
            [float(weight) for weight in args.matryoshka_weights.split(',')]
            if args.matryoshka_weights
            else None
        )
        logger.info(f'Using Matryoshka with dims: {dims} and weights: {weights}')
        return MatryoshkaOperator(
            loss=loss, embed_dim=embed_dim, dims=dims, weights=weights
        )

    return loss


def _create_mtl_losses(args, embed_dim: int):
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

    dims, weights = [], []
    if args.matryoshka:
        dims = [int(dim) for dim in args.matryoshka_dims.split(',')]
        weights = (
            [float(weight) for weight in args.matryoshka_weights.split(',')]
            if args.matryoshka_weights
            else None
        )
        logger.info(f'Using Matryoshka with dims: {dims} and weights: {weights}')

    losses = {}
    for d in lossinit:
        for task in d['tasks']:
            logger.debug(f'Setting up loss: {d["name"].__name__}')
            lossfn = d['name'](**d['options'])
            if args.matryoshka:
                lossfn = MatryoshkaOperator(
                    loss=lossfn, embed_dim=embed_dim, dims=dims, weights=weights
                )
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
            upsampling_factors=args.train_data_upsampling_factors,
            group_ids=args.train_data_assigned_groupids,
            use_long_captions=False,
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

    return data


def _create_images_dataloader(
    args, preprocess_cfg: PreprocessCfg, tokenizer, batch_size, epoch
):
    from timm.data import RandomResizedCropAndInterpolation
    from torchvision import transforms

    # augmentation follows exactly the one described in SimCLR paper
    # https://arxiv.org/pdf/2002.05709v3
    # See Section 3 and Appendix A
    s = 1.0  # strength parameter, stronger color jitter equals better results
    imgsize = preprocess_cfg.size
    imgsize = imgsize if isinstance(imgsize, int) else imgsize[0]
    pipeline = [
        transforms.RandomHorizontalFlip(p=0.5),
        RandomResizedCropAndInterpolation(
            preprocess_cfg.size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=preprocess_cfg.interpolation,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=int(imgsize * 0.1) + 1, sigma=(0.1, 2.0)
                ),
            ],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(preprocess_cfg.mean),
            std=torch.tensor(preprocess_cfg.std),
        ),
    ]
    augmentation_fn = transforms.Compose(pipeline)

    return get_wds_dataset(
        shards=args.train_imgdata,
        preprocess_fn=augmentation_fn,
        num_samples=args.train_num_samples,
        is_train=True,
        tokenizer=tokenizer,
        upsampling_factors=args.train_imgdata_upsampling_factors,
        group_ids=args.train_imgdata_assigned_groupids,
        images_pairs=True,
        workers=args.workers,
        batch_size=batch_size,
        seed=args.seed,
        epoch=epoch,
        dataset_resampled=args.dataset_resampled,
        world_size=args.world_size,
    )


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
            logger.debug(f'Loading from checkpoint {checkpoint} ...')
            dataset = MultiDataset.load_from_json(
                checkpoint,
                world_size=world_size,
                global_rank=rank,
            )

    if dataset is None:
        logger.debug(f'Bucket: {bucket}, Datasets: {datasets}')
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

    logger.debug('Setting up the S3 dataloader')

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


def create_losses(args, embed_dim: int):
    loss = _create_contrastive_loss(args, embed_dim)
    mtllosses = None
    if args.train_mtldata:
        mtllosses = _create_mtl_losses(args, embed_dim)
    return loss, mtllosses


def create_dataloaders(
    args,
    preprocess_train,
    preprocess_val,
    epoch: int,
    tokenizer: Any,
    mtl_losses: Optional[Dict[str, Any]] = None,
    preprocess_cfg: Optional[PreprocessCfg] = None,
):
    batch_size = args.batch_size
    txt_batch_size = 0
    img_batch_size = 0

    if args.train_txtdata and args.train_imgdata:
        txt_batch_size = batch_size // 4
        img_batch_size = batch_size // 4
    elif args.train_txtdata:
        txt_batch_size = batch_size // 2
    elif args.train_imgdata:
        img_batch_size = batch_size // 2

    batch_size -= txt_batch_size + img_batch_size

    logger.info('Creating the multimodal dataloader ...')
    logger.debug(f'Batch size: {batch_size}')
    data = _create_multimodal_dataloader(
        args=args,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
        epoch=epoch,
        batch_size=batch_size,
    )

    if args.train_txtdata:
        logger.info('Creating the text pair S3 dataloader ...')
        logger.debug(f'Batch size: {txt_batch_size}')
        assert isinstance(tokenizer.tokenizer, PreTrainedTokenizer) or isinstance(
            tokenizer.tokenizer, PreTrainedTokenizerFast
        )
        upsampling_factors = None
        if args.train_txtdata_upsampling_factors:
            upsampling_factors = [
                float(v) for v in args.train_txtdata_upsampling_factors.split('::')
            ]
        data['train-text'] = _create_s3_dataloader(
            datasets=args.train_txtdata.split('::'),
            bucket=args.train_txtdata_s3bucket,
            input_type_dict={'*': InputType.PAIR},
            tokenizer=tokenizer.tokenizer,
            sampling_rates=upsampling_factors,
            resume=args.resume,
            batch_size=txt_batch_size,
            max_sequence_length=args.max_sequence_length,
            prefix='text',
            max_shards=args.s3_max_shards,
            max_batches=args.s3_max_batches,
            num_batches=args.s3_num_batches,
            seed=args.seed,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        data['train-text'] = None

    if args.train_imgdata:
        assert preprocess_cfg is not None
        logger.info('Creating the image pairs dataloader ...')
        logger.debug(f'Batch size: {img_batch_size}')
        data['train-image'] = _create_images_dataloader(
            args=args,
            preprocess_cfg=preprocess_cfg,
            tokenizer=tokenizer,
            batch_size=img_batch_size,
            epoch=epoch,
        )
    else:
        data['train-image'] = None

    if args.train_mtldata:
        logger.info('Creating the MTL dataloader ...')
        logger.debug(f'Batch size: {args.mtl_batch_size}')
        assert isinstance(tokenizer.tokenizer, PreTrainedTokenizer) or isinstance(
            tokenizer.tokenizer, PreTrainedTokenizerFast
        )
        upsampling_factors = None
        if args.train_mtldata_upsampling_factors:
            upsampling_factors = [
                float(v) for v in args.train_mtldata_upsampling_factors.split('::')
            ]
        data['train-mtl'] = _create_s3_dataloader(
            datasets=args.train_mtldata.split('::'),
            bucket=args.train_mtldata_s3bucket,
            input_type_dict={
                task: lossfn._loss.input_type
                if isinstance(lossfn, MatryoshkaOperator)
                else lossfn.input_type
                for task, lossfn in mtl_losses.items()
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
