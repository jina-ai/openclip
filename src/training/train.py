import logging
import math
import time
import warnings
from collections import defaultdict
from itertools import islice

import torch

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


class _DummyDataloader:
    def __iter__(self):
        return self

    def __next__(self):
        return None, (None, None)


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

    train_s3_dataloader = data['train-s3'] or _DummyDataloader()
    train_mtl_dataloader = data['train-mtl'] or _DummyDataloader()

    _num_batches_per_epoch = train_dataloader.num_batches // args.accum_freq
    _sample_digits = math.ceil(math.log(train_dataloader.num_samples + 1, 10))

    accum_images, accum_texts, accum_features = [], [], {}
    accum_mtl_datasets, accum_mtl_batches, accum_mtl_labels, accum_mtl_features = (
        [],
        [],
        [],
        [],
    )
    losses_m = {}
    _batch_time_m = AverageMeter()
    _data_time_m = AverageMeter()

    start = time.time()

    # training loop
    for i, (batch, (_, (s3batch, _), (mtldataset, (mtlbatch, mtllabels)))) in enumerate(
        zip(
            train_dataloader,
            islice(train_s3_dataloader, 1, None),
            islice(train_mtl_dataloader, 1, None),
        )
    ):
        i_accum = i // args.accum_freq
        step = _num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        alltexts = [texts]
        batch_size = texts.shape[0]
        s3_batch_size = 0
        if s3batch:
            for b in s3batch:
                b.to(device=device)
                alltexts.append(b['input_ids'])
            s3_batch_size = s3batch[0].shape[0]
        if mtlbatch:
            for b in mtlbatch:
                b.to(device=device)
        if mtllabels:
            mtllabels[0] = [label.to(device=device) for label in mtllabels[0]]

        alltexts = torch.cat(alltexts, dim=0)

        _data_time_m.update(time.time() - start)

        if args.deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        if args.accum_freq == 1:
            # WITHOUT Gradient Accumulation

            with autocast():
                modelout = model(images, alltexts)
                if args.distill:
                    with torch.no_grad():
                        distill_model_out = distill_model(images, alltexts)
                    modelout.update(
                        {f'distill_{k}': v for k, v in distill_model_out.items()}
                    )
                modelout['output_dict'] = True
                image_features = modelout.pop('image_features')
                text_features = modelout.pop('text_features')
                left_features = torch.cat(
                    [
                        image_features,
                        text_features[batch_size : batch_size + s3_batch_size,],
                    ],
                    dim=0,
                )
                right_features = torch.cat(
                    [
                        text_features[:batch_size],
                        text_features[batch_size + s3_batch_size :,],
                    ],
                    dim=0,
                )
                losses = loss(left_features, right_features, **modelout)

                if mtl_losses is not None:
                    mtllossfn = (
                        mtl_losses[mtldataset]
                        if mtldataset in mtl_losses
                        else mtl_losses['*']
                    )
                    modelouts = [model(None, b['input_ids']) for b in mtlbatch]
                    mtlfeats = []
                    for out in modelouts:
                        mtlfeats.append(out.pop('text_features'))
                        _ = out.pop('image_features')
                    losskwargs = modelouts[0]
                    losskwargs['output_dict'] = True
                    mtlloss = mtllossfn(*mtlfeats, *mtllabels, **losskwargs)
                    losses['mtl_loss'] = args.mtl_loss_weight * mtlloss

            total_loss = sum(losses.values())
            losses['loss'] = total_loss
            backward(total_loss, model, scaler=scaler, deepspeed=args.deepspeed)

        else:
            # WITH Gradient Accumulation

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    modelout = model(images, alltexts)
                    image_features = modelout.pop('image_features')
                    text_features = modelout.pop('text_features')
                    modelout['left_features'] = torch.cat(
                        [
                            image_features,
                            text_features[batch_size : batch_size + s3_batch_size,],
                        ],
                        dim=0,
                    )
                    modelout['right_features'] = torch.cat(
                        [
                            text_features[:batch_size],
                            text_features[batch_size + s3_batch_size :,],
                        ],
                        dim=0,
                    )
                    for f in ('logit_scale', 'logit_bias'):
                        modelout.pop(f, None)

                    for key, val in modelout.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                    if mtl_losses is not None:
                        # if we have no labels == pair training
                        if len(mtllabels) == 0:
                            modelouts = [model(None, b['input_ids']) for b in mtlbatch]
                            mtlfeats = []
                            for out in modelouts:
                                mtlfeats.append(out.pop('text_features'))
                            accum_mtl_features.append(mtlfeats)
                        # else == triplet training
                        else:
                            accum_mtl_labels.append(mtllabels)

                accum_images.append(images)
                accum_texts.append(alltexts)
                if args.mtl:
                    accum_mtl_datasets.append(mtldataset)
                    accum_mtl_batches.append(mtlbatch)

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
                    accum_mtl_batches,
                    accum_mtl_labels,
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
                    images = accum_images[k]
                    alltexts = accum_texts[k]

                    modelout = model(images, alltexts)
                    modelout['output_dict'] = True
                    image_features = modelout.pop('image_features')
                    text_features = modelout.pop('text_features')
                    modelout['left_features'] = torch.cat(
                        [
                            image_features,
                            text_features[batch_size : batch_size + s3_batch_size,],
                        ],
                        dim=0,
                    )
                    modelout['right_features'] = torch.cat(
                        [
                            text_features[:batch_size],
                            text_features[batch_size + s3_batch_size :,],
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
                            accumulated[:k] + [modelout[key]] + accumulated[k + 1 :]
                        )

                    _losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    contrastive_loss = sum(_losses.values())
                    losses['contrastive_loss'] += contrastive_loss
                    total_loss = contrastive_loss

                    if mtl_losses is not None:
                        mtldataset = accum_mtl_datasets[k]
                        mtlbatch = accum_mtl_batches[k]
                        mtllossfn = (
                            mtl_losses[mtldataset]
                            if mtldataset in mtl_losses
                            else mtl_losses['*']
                        )
                        modelouts = [model(None, b['input_ids']) for b in mtlbatch]
                        inputs_no_accum = {
                            'logit_scale': modelouts[0].pop('logit_scale')
                        }
                        if 'logit_bias' in modelouts[0]:
                            inputs_no_accum['logit_bias'] = modelout.pop('logit_bias')
                        mtlfeats = []
                        for out in modelouts:
                            mtlfeats.append(out.pop('text_features'))

                        if len(accum_mtl_labels) == 0:
                            inputs = []
                            _cached_features = list(zip(*accum_mtl_features))
                            for idx, _cached_feature in enumerate(_cached_features):
                                inputs.append(
                                    torch.cat(
                                        _cached_feature[:k]
                                        + (mtlfeats[idx],)
                                        + _cached_feature[k + 1 :]
                                    )
                                )
                            mtlloss = mtllossfn(
                                *mtlfeats, **inputs_no_accum, output_dict=True
                            )
                            del inputs
                        else:
                            mtllabels = accum_mtl_labels[k]
                            mtlloss = mtllossfn(
                                *mtlfeats,
                                *mtllabels,
                                **inputs_no_accum,
                                output_dict=True,
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
                accum_mtl_batches,
                accum_mtl_labels,
                accum_mtl_features,
            ) = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # with torch.no_grad():
        #     unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        _batch_time_m.update(time.time() - start)
        start = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == _num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = train_dataloader.num_samples
            percent_complete = 100.0 * batch_count / _num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = ' - '.join(
                [
                    f'{loss_name.replace("_", " ")}: '
                    f'{loss_m.val:#.5g} (avg {loss_m.avg:#.5g})'
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / _batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / _batch_time_m.val
            )
            logging.info(
                f'Epoch: {epoch} [{num_samples:>{_sample_digits}}/'
                f'{samples_per_epoch} ({percent_complete:.0f}%)] - '
                f'Data Time: {_data_time_m.avg:.3f}s - '
                f'Batch Time: {_batch_time_m.avg:.3f}s - '
                f'Samples per Second: {samples_per_second:#g}/s, '
                f'{samples_per_second_per_gpu:#g}/s/gpu - '
                f'Last Layer LR: {optimizer.param_groups[-1]["lr"]:5f} - '
                f'Logit Scale: {logit_scale_scalar:.3f} - '
                f'LOSS | {loss_log}'
            )

            # Save train loss / etc. Using non avg meter values as loggers have
            # their own smoothing
            logdata = {
                'data_time': _data_time_m.val,
                'batch_time': _batch_time_m.val,
                'samples_per_second': samples_per_second,
                'samples_per_second_per_gpu': samples_per_second_per_gpu,
                'scale': logit_scale_scalar,
            }
            logdata.update(
                {
                    f'lr/{pgroup["###logging_descriptor"]}': pgroup['lr']
                    for pgroup in optimizer.param_groups
                }
            )
            logdata.update({name: val.val for name, val in losses_m.items()})

            logdata = {'train/' + name: val for name, val in logdata.items()}

            if tb_writer is not None:
                for name, val in logdata.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                logdata['step'] = step  # for backwards compatibility
                wandb.log(logdata, step=step)

            # resetting batch / data time meters per log window
            _batch_time_m.reset()
            _data_time_m.reset()
