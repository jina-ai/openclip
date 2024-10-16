import math
import time
import warnings
from collections import Counter, defaultdict

import yaml
from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from typing import List, Optional, Tuple

import torch
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from training.distributed import is_master
from training.utils import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        'image_features': model_out[0],
        'text_features': model_out[1],
        'logit_scale': model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, model, scaler=None, deepspeed=False):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    elif deepspeed:
        model.backward(total_loss)
    else:
        total_loss.backward()


class _DummyS3Dataloader:
    def __iter__(self):
        return self

    def __next__(self):
        return None, (None, None)


class _DummyWDSDataloader:
    def __iter__(self):
        return self

    def __next__(self):
        return None, None, None, None


class DatasetType(IntEnum):
    MULTIMODAL = 0
    TEXT = 1
    IMAGE = 2
    MTL = 3


@dataclass
class DatasetRecord:
    path: str
    count: int
    type: DatasetType


class DatasetRecordHistory:

    def __init__(self) -> None:
        self._dataset_ids = {}
        self._dataset_shard_ids = {}
        self._records = {}

    @property
    def datasets2ids(self):
        return self._dataset_ids

    @property
    def ids2datasets(self):
        return {v: k for k, v in self._dataset_ids.items()}

    def shards2ids(self, dataset_id: int):
        return self._dataset_shard_ids[dataset_id]

    def ids2shards(self, dataset_id: int):
        return {v: k for k, v in self._dataset_shard_ids[dataset_id].items()}

    def merge(self, other: 'DatasetRecordHistory'):
        for step, records in other._records.items():
            _new_records = []
            for did, sid, count, _type in records:
                dataset = other.ids2datasets[did]
                shard = other.ids2shards(did)[sid]
                _new_did = self.dataset_addget(dataset)
                _new_sid = self.dataset_shard_addget(shard, _new_did)
                _new_records.append((_new_did, _new_sid, count, _type))
            if step not in self._records:
                self._records[step] = []
            self._records[step].extend(_new_records)

    def dataset_addget(self, name: str) -> int:
        if name in self._dataset_ids:
            return self._dataset_ids[name]
        _new_id = len(self._dataset_ids)
        self._dataset_ids[name] = _new_id
        return _new_id

    def dataset_shard_addget(self, name: str, dataset_id: int) -> int:
        if dataset_id not in self._dataset_shard_ids:
            self._dataset_shard_ids[dataset_id] = {}

        _shard_ids = self._dataset_shard_ids[dataset_id]
        if name in _shard_ids:
            return _shard_ids[name]
        _new_id = len(_shard_ids)
        self._dataset_shard_ids[dataset_id][name] = _new_id
        return _new_id

    def add_records(self, records: List[DatasetRecord], step: int) -> None:
        if step not in self._records:
            self._records[step] = []

        for record in records:
            dataset, shard = self.get_dataset_and_shard_from_path(record.path)
            _dataset_id = self.dataset_addget(dataset)
            _shard_id = self.dataset_shard_addget(shard, _dataset_id)
            self._records[step].append(
                (_dataset_id, _shard_id, record.count, record.type.value)
            )

    @staticmethod
    def get_dataset_and_shard_from_path(path: str) -> Tuple[str, str]:
        if path.startswith('pipe:aws s3 cp s3://') and path.endswith('.tar -'):
            path = path.replace('pipe:aws s3 cp s3://', '')
            path = path.replace(' -', '')
        shard = path.split('/')[-1]
        if shard.startswith('shard') or shard.endswith('.tar'):
            dataset = '/'.join(path.split('/')[:-1])
            return dataset, shard

        return path, 'unknown'

    @property
    def state_dict(self):
        return {
            'dataset_ids': self._dataset_ids,
            'dataset_shard_ids': self._dataset_shard_ids,
            'records': self._records,
        }

    def report(self, start: int = 0, end: int = -1, report_shards: bool = False):
        _last_step = max(list(self._records.keys()))
        end = end if end > 0 else _last_step
        steps = [step for step in self._records.keys() if start <= step <= end]
        stats = {
            e.name: {
                'datasets': {
                    dataset: {
                        'shards': {
                            shard: {
                                'count': 0,
                                'percentage': 0.0,
                            }
                            for shard in self._dataset_shard_ids[dataset_id].keys()
                        },
                        'count': 0,
                        'percentage': 0.0,
                    }
                    for dataset, dataset_id in self._dataset_ids.items()
                },
                'count': 0,
                'percentage': 0.0
            }
            for e in DatasetType
        }
        total = 0
        for step in steps:
            for _dataset_id, _shard_id, count, _type_value in self._records[step]:
                _type = DatasetType(_type_value).name
                _dataset = self.ids2datasets[_dataset_id]
                _shard = self.ids2shards(_dataset_id)[_shard_id]
                total += count
                stats[_type]['count'] += count
                stats[_type]['datasets'][_dataset]['count'] += count
                stats[_type]['datasets'][_dataset]['shards'][_shard]['count'] += count

        for e in DatasetType:
            total_per_type = stats[e.name]['count']
            stats[e.name]['percentage'] = total_per_type / total if total else 0.0
            for dataset in stats[e.name]['datasets']:
                total_per_dataset = stats[e.name]['datasets'][dataset]['count']
                stats[e.name]['datasets'][dataset]['percentage'] = (
                    total_per_dataset / total_per_type
                ) if total_per_type else 0.0
                for shard in stats[e.name]['datasets'][dataset]['shards']:
                    total_per_shard = (
                        stats[e.name]['datasets'][dataset]['shards'][shard]['count']
                    )
                    (
                        stats[e.name]['datasets'][dataset]['shards'][shard][
                            'percentage'
                        ]
                    ) = (
                        total_per_shard / total_per_dataset
                        if total_per_dataset else 0.0
                    )
                if not report_shards:
                    _ = stats[e.name]['datasets'][dataset].pop('shards')
                else:
                    stats[e.name]['datasets'][dataset]['shards'] = {
                        k: v for k, v in sorted(
                            stats[e.name]['datasets'][dataset]['shards'].items(),
                            key=lambda x: x[1]['count'],
                            reverse=True
                        ) if v['count'] > 0
                    }
            stats[e.name]['datasets'] = {
                k: v for k, v in sorted(
                    stats[e.name]['datasets'].items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                ) if v['count'] > 0
            }
        stats = {
            k: v for k, v in sorted(
                stats.items(), key=lambda x: x[1]['count'], reverse=True
            ) if v['count'] > 0
        }
        return stats

    def yaml_report(
        self, fname: str, start: int = 0, end: int = -1, report_shards: bool = False
    ):
        with open(fname, 'w') as f:
            yaml.safe_dump(
                self.report(start, end, report_shards), f, sort_keys=False,
            )

    def save(self, f):
        torch.save(self.state_dict, f)

    @classmethod
    def load(cls, f):
        sd = torch.load(f)
        obj = cls()
        obj._dataset_ids = sd['dataset_ids']
        obj._dataset_shard_ids = sd['dataset_shard_ids']
        obj._records = sd['records']
        return obj


def train_one_epoch(
    model,
    data,
    loss,
    mtl_losses,
    epoch,
    optimizer,
    scaler,
    scheduler,
    distill_model,
    args,
    dataset_records: Optional[DatasetRecordHistory] = None,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        distill_model.eval()

    # set epoch in process safe manner via sampler or shared_epoch
    data['train'].set_epoch(epoch)
    train_dataloader = data['train'].dataloader

    train_text_dataloader = data['train-text']
    if train_text_dataloader is None:
        train_text_dataloader = _DummyS3Dataloader()
    else:
        _, train_text_dataloader = train_text_dataloader

    train_image_dataloader = data['train-image']
    if train_image_dataloader is None:
        train_image_dataloader = _DummyWDSDataloader()
    else:
        train_image_dataloader = train_image_dataloader.dataloader

    train_mtl_dataloader = data['train-mtl']
    if train_mtl_dataloader is None:
        train_mtl_dataloader = _DummyS3Dataloader()
    else:
        _, train_mtl_dataloader = train_mtl_dataloader

    _num_batches_per_epoch = train_dataloader.num_batches // args.accum_freq
    _sample_digits = math.ceil(math.log(train_dataloader.num_samples + 1, 10))

    accum_images, accum_texts, accum_features = [], [], {}
    (
        accum_mtl_datasets,
        accum_mtl_labels,
        accum_mtl_texts,
        accum_mtl_features,
    ) = [], [], [], []
    losses_m = {}
    _batch_time_m = AverageMeter()
    _data_time_m = AverageMeter()

    start = time.time()

    mtl_logit_scale = None
    if args.mtl_temperature:
        mtl_logit_scale = torch.tensor(
            [1 / args.mtl_temperature]
        ).to(device=device, dtype=input_dtype, non_blocking=True)

    _dataset_records = []

    # training loop
    for i, (
        (_, urls, images, texts),
        (textdataset, (text_pairs, _)),
        (_, image_urls, images_left, images_right),
        (mtldataset, (mtlbatch, mtllabels)),
    ) in enumerate(
        zip(
            train_dataloader,
            islice(train_text_dataloader, 1, None),
            train_image_dataloader,
            islice(train_mtl_dataloader, 1, None),
        )
    ):

        i_accum = i // args.accum_freq
        step = _num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        allimages = [images]
        alltexts = [texts]

        batch_size = texts.shape[0]
        texts_batch_size = 0
        images_batch_size = 0
        mtl_batch_size = 0
        if text_pairs:
            texts_batch_size = text_pairs[0]['input_ids'].shape[0] // 2
            for tp in text_pairs:
                tp.to(device=device)
                alltexts.append(tp['input_ids'])

        if (images_left is not None) and (images_right is not None):
            images_batch_size = images_left.shape[0]
            images_left = images_left.to(device=device)
            images_right = images_right.to(device=device)
            allimages.extend([images_left, images_right])

        run_mtl_training = False
        run_mtl_pair_training = False
        run_mtl_triplet_training = False
        if mtlbatch:
            run_mtl_training = True
            if len(mtllabels) == 0:
                run_mtl_pair_training = True
                mtl_batch_size = mtlbatch[0]['input_ids'].shape[0] // 2
                for mb in mtlbatch:
                    mb.to(device=device)
                    alltexts.append(mb['input_ids'])
            else:
                run_mtl_triplet_training = True
                mtllabels[0] = [label.to(device=device) for label in mtllabels[0]]
                mtl_batch_size = mtlbatch[0]['input_ids'].shape[0]
                for mb in mtlbatch:
                    mb.to(device=device)
                    alltexts.append(mb['input_ids'])

        allimages = torch.cat(allimages, dim=0)
        alltexts = torch.cat(alltexts, dim=0)

        if dataset_records is not None:
            for url, freq in Counter(urls).most_common():
                _dataset_records.append(
                    DatasetRecord(path=url, count=freq, type=DatasetType.MULTIMODAL)
                )
            if textdataset is not None:
                _dataset_records.append(
                    DatasetRecord(
                        path=textdataset, count=texts_batch_size, type=DatasetType.TEXT
                    )
                )
            if image_urls is not None:
                for url, freq in Counter(image_urls).most_common():
                    _dataset_records.append(
                        DatasetRecord(path=url, count=freq, type=DatasetType.IMAGE)
                    )
            if mtldataset is not None:
                _dataset_records.append(
                    DatasetRecord(
                        path=mtldataset, count=mtl_batch_size, type=DatasetType.MTL
                    )
                )

        _data_time_m.update(time.time() - start)

        if args.deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        if args.accum_freq == 1:
            # WITHOUT Gradient Accumulation

            with autocast():
                modelout = model(allimages, alltexts)
                if args.distill:
                    with torch.no_grad():
                        distill_model_out = distill_model(allimages, alltexts)
                    modelout.update(
                        {
                            'distill_left_features': distill_model_out[
                                'distill_image_features'
                            ],
                            'distill_right_features': distill_model_out[
                                'distill_text_features'
                            ],
                        }
                    )

                modelout['output_dict'] = True
                logit_scale = modelout['logit_scale']
                image_features = modelout.pop('image_features')
                text_features = modelout.pop('text_features')
                if run_mtl_training:
                    if run_mtl_pair_training:
                        mtl_features = [
                            text_features[-2 * mtl_batch_size:-mtl_batch_size],
                            text_features[-mtl_batch_size:],
                        ]
                        text_features = text_features[:-2*mtl_batch_size]
                    elif run_mtl_triplet_training:
                        mtl_features = [text_features[-mtl_batch_size:]]
                        text_features = text_features[:-mtl_batch_size]

                left_features = torch.cat(
                    [
                        image_features[:batch_size],
                        text_features[batch_size: batch_size + texts_batch_size,],
                        image_features[batch_size: batch_size + images_batch_size,],
                    ],
                    dim=0,
                )
                right_features = torch.cat(
                    [
                        text_features[:batch_size],
                        text_features[batch_size + texts_batch_size:,],
                        image_features[batch_size + images_batch_size:],
                    ],
                    dim=0,
                )

                losses = defaultdict(lambda: torch.zeros(1, device=device))

                _losses = loss(left_features, right_features, **modelout)
                contrastive_loss = sum(_losses.values())
                losses['contrastive_loss'] += contrastive_loss

                if run_mtl_training:
                    mtllossfn = (
                        mtl_losses[mtldataset]
                        if mtldataset in mtl_losses
                        else mtl_losses['*']
                    )
                    if mtl_logit_scale is not None:
                        _ = modelout.pop('logit_scale')
                        modelout['logit_scale'] = mtl_logit_scale
                    logit_scale_mtl = modelout['logit_scale']

                    mtlloss = mtllossfn(*mtl_features, *mtllabels, **modelout)
                    losses['mtl_loss'] = args.mtl_loss_weight * mtlloss

            total_loss = sum(losses.values())
            losses['loss'] += total_loss
            backward(total_loss, model, scaler=scaler, deepspeed=args.deepspeed)

        else:
            # WITH Gradient Accumulation

            # First, cache the features without any gradient tracking.
            with torch.no_grad():

                mtltexts = None
                if run_mtl_triplet_training:
                    mtltexts = alltexts[-mtl_batch_size:]
                    alltexts = alltexts[:-mtl_batch_size]

                accum_images.append(allimages)
                accum_texts.append(alltexts)
                accum_mtl_datasets.append(mtldataset)
                accum_mtl_texts.append(mtltexts)
                accum_mtl_labels.append(mtllabels)

                with autocast():
                    modelout = model(allimages, alltexts)
                    image_features = modelout.pop('image_features')
                    text_features = modelout.pop('text_features')
                    if run_mtl_training and run_mtl_pair_training:
                        mtl_features = [
                            text_features[-2 * mtl_batch_size:-mtl_batch_size],
                            text_features[-mtl_batch_size:],
                        ]
                        text_features = text_features[:-2 * mtl_batch_size]
                        accum_mtl_features.append(mtl_features)
                    accumulated = {
                        'left_features': torch.cat(
                            [
                                image_features[:batch_size],
                                text_features[batch_size: batch_size+texts_batch_size,],
                                image_features[
                                    batch_size: batch_size + images_batch_size,
                                ],
                            ],
                            dim=0,
                        ),
                        'right_features': torch.cat(
                            [
                                text_features[:batch_size],
                                text_features[batch_size + texts_batch_size:,],
                                image_features[batch_size + images_batch_size:],
                            ],
                            dim=0,
                        ),
                    }
                    for key, val in accumulated.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            if len(accum_mtl_labels) not in {0, args.accum_freq}:
                warnings.warn(
                    f'Out of {args.accum_freq} gradient accumulation steps, '
                    f'{len(accum_mtl_labels)} contain labels and '
                    f'{len(accum_mtl_features)} are aligned pairs. '
                    f'Gradient accumulation cannot work with inconsistent data'
                )
                accum_images, accum_texts, accum_features = [], [], {}
                (
                    accum_mtl_datasets,
                    accum_mtl_labels,
                    accum_mtl_texts,
                    accum_mtl_features,
                ) = [], [], [], []
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features
            # from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            if args.deepspeed:
                model.zero_grad()
                model.micro_steps = 0
            else:
                optimizer.zero_grad()

            losses = defaultdict(lambda: torch.zeros(1, device=device))

            for k in range(args.accum_freq):
                with autocast():
                    allimages = accum_images[k]
                    alltexts = accum_texts[k]
                    mtltexts = accum_mtl_texts[k]

                    if run_mtl_triplet_training:
                        alltexts = torch.cat([alltexts, mtltexts], dim=0)

                    modelout = model(allimages, alltexts)
                    modelout['output_dict'] = True
                    image_features = modelout.pop('image_features')
                    text_features = modelout.pop('text_features')

                    if run_mtl_training:
                        if run_mtl_pair_training:
                            mtl_features = [
                                text_features[-2 * mtl_batch_size:-mtl_batch_size],
                                text_features[-mtl_batch_size:],
                            ]
                            text_features = text_features[:-2 * mtl_batch_size]
                        elif run_mtl_triplet_training:
                            mtl_features = [text_features[-mtl_batch_size:]]
                            text_features = text_features[:-mtl_batch_size]

                    modelout['left_features'] = torch.cat(
                        [
                            image_features[:batch_size],
                            text_features[batch_size: batch_size + texts_batch_size,],
                            image_features[
                                batch_size: batch_size + images_batch_size,
                            ],
                        ],
                        dim=0,
                    )
                    modelout['right_features'] = torch.cat(
                        [
                            text_features[:batch_size],
                            text_features[batch_size + texts_batch_size:,],
                            image_features[batch_size + images_batch_size:],
                        ],
                        dim=0,
                    )

                    inputs_no_accum = {}
                    inputs_no_accum['logit_scale'] = logit_scale = modelout.pop(
                        'logit_scale'
                    )
                    if 'logit_bias' in modelout:
                        inputs_no_accum['logit_bias'] = modelout.pop('logit_bias')

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:k] + [modelout[key]] + accumulated[k + 1:]
                        )

                    _losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    contrastive_loss = sum(_losses.values())
                    losses['contrastive_loss'] += contrastive_loss
                    total_loss = contrastive_loss

                    if run_mtl_training:
                        mtldataset = accum_mtl_datasets[k]
                        mtllossfn = (
                            mtl_losses[mtldataset]
                            if mtldataset in mtl_losses
                            else mtl_losses['*']
                        )
                        if mtl_logit_scale is not None:
                            inputs_no_accum['logit_scale'] = mtl_logit_scale

                        logit_scale_mtl = inputs_no_accum['logit_scale']

                        if run_mtl_pair_training:
                            inputs = []
                            _cached_features = list(zip(*accum_mtl_features))
                            for idx, _cached_feature in enumerate(_cached_features):
                                inputs.append(
                                    torch.cat(
                                        _cached_feature[:k]
                                        + (mtl_features[idx],)
                                        + _cached_feature[k + 1:]
                                    )
                                )
                            mtlloss = mtllossfn(
                                *inputs, **inputs_no_accum, output_dict=False
                            )
                            del inputs
                        elif run_mtl_triplet_training:
                            mtllabels = accum_mtl_labels[k]
                            mtlloss = mtllossfn(
                                *mtl_features,
                                *mtllabels,
                                **inputs_no_accum,
                                output_dict=False,
                            )

                        mtlloss = args.mtl_loss_weight * mtlloss
                        losses['mtl_loss'] += mtlloss
                        total_loss += mtlloss

                losses['loss'] += total_loss
                backward(total_loss, model, scaler=scaler, deepspeed=args.deepspeed)

            losses['contrastive_loss'] = losses['contrastive_loss'] / args.accum_freq
            losses['mtl_loss'] = losses['mtl_loss'] / args.accum_freq
            losses['loss'] = losses['loss'] / args.accum_freq

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        elif args.deepspeed:
            model.step()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}
            (
                accum_mtl_datasets,
                accum_mtl_labels,
                accum_mtl_texts,
                accum_mtl_features,
            ) = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        if dataset_records is not None:
            dataset_records.add_records(records=_dataset_records, step=step)
            _dataset_records = []

        _batch_time_m.update(time.time() - start)
        start = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == _num_batches_per_epoch
        ):
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            num_text_samples = (
                batch_count * texts_batch_size * args.accum_freq * args.world_size
            )
            num_image_samples = (
                batch_count * images_batch_size * args.accum_freq * args.world_size
            )
            num_mtl_samples = (
                batch_count * mtl_batch_size * args.accum_freq * args.world_size
            )
            num_contrastive_samples = num_samples + num_text_samples + num_image_samples
            num_total_samples = num_contrastive_samples + num_mtl_samples

            samples_per_epoch = train_dataloader.num_samples
            percent_complete = 100.0 * batch_count / _num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            temperature_scalar = 1 / logit_scale_scalar

            logit_scale_mtl_scalar = 0.0
            temperature_mtl_scalar = 0.0
            if mtl_losses is not None:
                logit_scale_mtl_scalar = logit_scale_mtl.item()
                temperature_mtl_scalar = 1 / logit_scale_mtl_scalar

            loss_log = ' - '.join(
                [
                    f'{loss_name.replace("_", "-")}: '
                    f'{loss_m.val:#.5g} (avg {loss_m.avg:#.5g})'
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq
                * (batch_size + texts_batch_size + images_batch_size + mtl_batch_size)
                * args.world_size
                / _batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq
                * (batch_size + texts_batch_size + images_batch_size + mtl_batch_size)
                / _batch_time_m.val
            )
            last_layer_lr = optimizer.param_groups[-1]['lr']
            first_layer_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f'--------- STEP {step} \n'
                f'Epoch: {epoch} [{num_samples:>{_sample_digits}}/'
                f'{samples_per_epoch} ({percent_complete:.0f}%)] - '
                f'Total samples seen: {num_total_samples} ({num_contrastive_samples} '
                f'contrastive & {num_mtl_samples} MTL)\n'
                f'\tTimings: data {_data_time_m.avg:.3f}s '
                f'batch {_batch_time_m.avg:.3f}s - '
                f'Samples/s: {samples_per_second:#g}/s, '
                f'{samples_per_second_per_gpu:#g}/s/gpu\n'
                f'\tLR: first layer -> {first_layer_lr:5f} last layer -> '
                f'{last_layer_lr:5f} - '
                f'Logit scale: {logit_scale_scalar:.3f} '
                f'(temperature {temperature_scalar:.3f}) '
                f'MTL Logit scale: {logit_scale_mtl_scalar:.3f} '
                f'(temperature {temperature_mtl_scalar:.3f})\n'
                f'\tLosses: {loss_log}'
            )

            # Save train loss / etc. Using non avg meter values as loggers have
            # their own smoothing
            logdata = {
                'data-time': _data_time_m.val,
                'batch-time': _batch_time_m.val,
                'samples-per-second': samples_per_second,
                'samples-per-second-per-gpu': samples_per_second_per_gpu,
                'logit-scale': logit_scale_scalar,
                'temperature': temperature_scalar,
                'logit-scale-mtl': logit_scale_mtl_scalar,
                'temperature-mtl': temperature_mtl_scalar,
                'first-layer-lr': first_layer_lr,
                'last-layer-lr': last_layer_lr,
            }
            logdata.update(
                {name.replace('_', '-'): val.val for name, val in losses_m.items()}
            )
            logdata = {'train/' + name: val for name, val in logdata.items()}

            if tb_writer is not None:
                for name, val in logdata.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb'
                wandb.log(logdata, step=step)

            # resetting batch / data time meters per log window
            _batch_time_m.reset()
            _data_time_m.reset()

    return dataset_records
