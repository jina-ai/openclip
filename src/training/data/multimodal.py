import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Callable, Dict, Optional, Union

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SubsetRandomSampler,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split('::')
        weights = weights.split('::')
        assert len(weights) == len(urllist), (
            f'Expected the number of data components ({len(urllist)}) and weights '
            f'({len(weights)}) to match.'
        )
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ['train', 'val', 'v2']
    is_train = split == 'train'
    preprocess_train, preprocess_val = preprocess_fns

    if split == 'v2':
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ['WDS_EPOCH'] = '0'
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample: Dict[str, Any]):
    _has_caption = 'txt' in sample
    _has_image = any([ext in sample for ext in {'png', 'jpeg', 'jpg', 'webp'}])
    return _has_caption and _has_image


def log_and_continue(exn):
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, _=None):
    """
    Return function over iterator that groups key, value pairs into samples.
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample['fname'], filesample['data']
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a
        #  potential for this happening in the current LAION400m dataset if a tar ends
        #  with same prefix as the next begins, rare, but can happen since prefix
        #  aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample['__key__']
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample['__url__'])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader
        # workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in
            # different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class _SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: Union[DataLoader, wds.WebLoader]
    sampler: Optional[DistributedSampler] = None
    shared_epoch: Optional[_SharedEpoch] = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class _DETShuffle(wds.PipelineStage):
    def __init__(
        self,
        bufsize: int = 1000,
        initial: int = 100,
        seed: int = 0,
        epoch: Union[int, _SharedEpoch] = -1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, _SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess
            # (dataloader workers or train) situation as different workers may wrap
            # at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different
            # across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in
            # each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class _ResampledShards(IterableDataset):
    def __init__(
        self,
        urls: str,
        weights: Optional[str] = None,
        nshards: int = sys.maxsize,
        worker_seed: Optional[Callable] = None,
        deterministic: bool = False,
        epoch: Union[int, _SharedEpoch] = -1,
    ):
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), (
                f'Number of urls {len(self.urls)} and weights {len(self.weights)} '
                f'should match.'
            )
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, _SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess
            # (dataloader workers or train) situation as different workers may wrap
            # at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by
                # arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )


class _CSVDataset(Dataset):
    def __init__(
        self, input_filename, transforms, img_key, caption_key, sep='\t', tokenizer=None
    ):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class _SyntheticDataset(Dataset):
    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption='Dummy caption',
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size
        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = None
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(
    num_samples: int,
    preprocess_fn: Any,
    is_train: bool = False,
    tokenizer: Any = None,
    distributed: bool = False,
    workers: int = 1,
    batch_size: int = 32,
):
    image_size = preprocess_fn.transforms[0].size
    dataset = _SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=num_samples,
        tokenizer=tokenizer,
    )
    numsamples = len(dataset)
    sampler = DistributedSampler(dataset) if distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = numsamples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_csv_dataset(
    fname: str,
    preprocess_fn: Any,
    is_train: bool = False,
    tokenizer: Any = None,
    csv_image_key: str = 'image',
    csv_caption_key: str = 'caption',
    csv_separator: str = ',',
    distributed: bool = False,
    workers: int = 1,
    batch_size: int = 32,
):
    assert fname
    dataset = _CSVDataset(
        fname,
        preprocess_fn,
        img_key=csv_image_key,
        caption_key=csv_caption_key,
        sep=csv_separator,
        tokenizer=tokenizer,
    )
    numsamples = len(dataset)
    sampler = DistributedSampler(dataset) if distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = numsamples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_wds_dataset(
    shards: str,
    preprocess_fn: Any,
    num_samples: Optional[int] = None,
    is_train: bool = False,
    tokenizer: Any = None,
    upsampling_factors: Optional[str] = None,
    workers: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    epoch: int = 0,
    floor: bool = False,
    dataset_resampled: bool = False,
    world_size: int = 1,
):
    assert shards is not None
    resampled = dataset_resampled and is_train

    num_shards = None
    if is_train:
        if num_samples is None:
            num_samples, num_shards = get_dataset_size(shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for '
                    'the training dataset. Please specify it via `--train-num-samples` '
                    'if no dataset length info is present.'
                )
    else:
        num_samples = num_samples or 0

    shared_epoch = _SharedEpoch(epoch=epoch)

    if is_train and upsampling_factors is not None:
        assert resampled, (
            'Upsampling factors are only supported when sampling with '
            'replacement (with --dataset-resampled).'
        )

    if resampled:
        pipeline = [
            _ResampledShards(
                shards,
                weights=upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    _DETShuffle(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to
                # each worker at each node
                tarfile_to_samples_nothrow,
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to
                # each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode('pilrgb', handler=log_and_continue),
            wds.rename(image='jpg;png;jpeg;webp', text='txt'),
            wds.map_dict(image=preprocess_fn, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple('image', 'text'),
            wds.batched(batch_size, partial=not is_train),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(shards)[0])
            assert (
                num_shards >= workers * world_size
            ), 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on
        # each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)
