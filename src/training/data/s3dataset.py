import csv
import gzip
import json
import os
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from itertools import islice
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from torch.utils.data import IterableDataset
from training.data.utils import (
    SimLMCrossEncoder,
    add_instruction,
    download_shard,
    get_dataset_info,
    get_directories,
    get_shard_size,
    get_shards,
    log_on_rank,
)

csv.field_size_limit(sys.maxsize)


class InputType(IntEnum):
    PAIR = 2
    TRIPLET = 3
    SCORED_TRIPLET = 4
    MULTIPLE_NEGATIVES = 5
    MULTIPLE_NEGATIVES_WITHOUT_SCORES = 6
    PAIR_WITH_SCORES = 7
    TEXT_WITH_LABEL = 8


def get_tuple_length(input_type: InputType):
    if input_type in (InputType.PAIR, InputType.PAIR_WITH_SCORES):
        return 2
    elif input_type in (InputType.TRIPLET, InputType.SCORED_TRIPLET):
        return 3
    elif input_type in (InputType.TEXT_WITH_LABEL,):
        return 1
    elif input_type in (
        InputType.MULTIPLE_NEGATIVES,
        InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES,
    ):
        return 9


class S3Dataset(IterableDataset):
    def __init__(
        self,
        bucket: str,
        dataset: str,
        world_size: int,
        global_rank: int,
        input_type_dict: Dict[str, str],
        task_type: Optional[str] = None,
        directory: Optional[str] = None,
        max_shards: Optional[int] = None,
        dialect: Literal['csv', 'tsv'] = 'tsv',
        shard_num: int = 0,
        index: Optional[int] = None,
        ce_model_name='intfloat/simlm-msmarco-reranker',
        interleaved=False,
        task_implementation: Literal['none', 'instruction-based'] = 'none',
    ):
        """A dataset that iterates through shards of a dataset stored
            in an S3 bucket.

        :param bucket: The name of the bucket where the data is stored.
        :param dataset: The name of the dataset to iterate over.
        :param input_type_dict: A dictionary mapping datasets to input types.
        :param max_shards: The maximum number of shards to iterate over before
            returning. None by default.
        :param dialect: The type of file that the data is stored in, either csv or tsv.
        :param shard_num: The index of the shard that is currently being processed (this
            is set when continuing from checkpoint).
        :param index: The row index that is processed next (set when continuing from
            checkpoint).
        :param interleaved: Whether to partition shards across workers or iterate over
            them with a stride and offset.
        """
        super().__init__()
        self._bucket = bucket
        self._directory = directory
        self._max_shards = max_shards
        self._dialect = dialect
        self._current_shard_num = shard_num
        self._current_index = (
            index if index is not None else (global_rank if interleaved else 0)
        )
        self._task_type = task_type
        self._task_implementation = task_implementation

        self._dataset = dataset
        shards = get_shards(
            bucket_name=self._bucket, dataset=self._dataset, directory=directory
        )

        assert dataset in input_type_dict or '*' in input_type_dict
        input_type = (
            input_type_dict[dataset]
            if dataset in input_type_dict
            else input_type_dict['*']
        )
        self._tuple_len = get_tuple_length(input_type)

        if input_type in (
            InputType.SCORED_TRIPLET,
            InputType.MULTIPLE_NEGATIVES,
        ):
            self._add_ce_scores = 'true'
        elif input_type == InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES:
            self._add_ce_scores = 'zeros'
        else:
            self._add_ce_scores = 'false'

        if input_type in (InputType.PAIR_WITH_SCORES, InputType.TEXT_WITH_LABEL):
            self._input_has_score = True
        else:
            self._input_has_score = False

        num_workers = world_size
        rank = global_rank
        self._rank = rank
        self._interleaved = interleaved
        if interleaved:
            start_index = 0
            stop_index = len(shards)
            self._stride = num_workers
        else:
            start_index = int(len(shards) * rank / num_workers)
            stop_index = int(len(shards) * (rank + 1) / num_workers)
            self._stride = 1
        self._shards = shards[start_index:stop_index]

        self._start_index = start_index
        self._stop_index = stop_index

        if max_shards is not None:
            self._shards = self._shards[:max_shards]

        # calculating number of pairs in dataset
        if len(self._shards) > 0:
            shard_len = get_shard_size(self._bucket, self._shards[0], self._dialect)

            if 1 < len(shards) <= stop_index and (
                max_shards is None or max_shards >= len(shards)
            ):
                end_shard_len = get_shard_size(
                    self._bucket, self._shards[-1], self._dialect
                )
            else:
                end_shard_len = shard_len
            self._num_pairs = ((len(self._shards) - 1) * shard_len) + end_shard_len
        else:
            self._num_pairs = 0

        log_on_rank(
            (
                f'worker {rank}/{num_workers} taking shards {start_index} - '
                f'{stop_index} for dataset {dataset}, total number of pairs: '
                f'{self._num_pairs}'
            )
        )

        if self._add_ce_scores == 'true':
            self._cross_encoder_model = self._load_ce_model(ce_model_name)

        self._tmpdir = tempfile.TemporaryDirectory()
        self._thread_pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _load_ce_model(ce_model_name):
        if ce_model_name.startswith('cross-encoder/'):
            from sentence_transformers import CrossEncoder

            return CrossEncoder(ce_model_name, max_length=512)
        else:
            return SimLMCrossEncoder(model_name=ce_model_name)

    def _get_ce_scores(self, query: str, pos: str, *negs: str):
        scores = self._cross_encoder_model.predict(
            [(query, pos), *[(query, neg) for neg in negs]]
        )
        return scores

    def _async_download_shard(self, shard_num: int):
        return self._thread_pool.submit(
            download_shard,
            self._bucket,
            self._shards[shard_num],
            self._tmpdir.name,
        )

    def __len__(self):
        return self._num_pairs // self._stride

    def __iter__(self):
        if self._current_shard_num >= len(self._shards):
            return

        # Start download of first shard
        future_shard_pth = self._async_download_shard(self._current_shard_num)

        for shard_num, shard in enumerate(
            self._shards[self._current_shard_num :], self._current_shard_num
        ):
            self._current_shard = shard
            self._current_shard_num = shard_num
            shard_pth = future_shard_pth.result()

            if shard_num + 1 < len(self._shards):
                # Start download of next shard in a different thread
                future_shard_pth = self._async_download_shard(shard_num + 1)
            else:
                # There is no next shard
                future_shard_pth = None

            if shard_pth.endswith('.gz'):
                file = gzip.open(shard_pth, 'rt')
            elif shard_pth.endswith(f'.{self._dialect}'):
                file = open(shard_pth, 'r')
            else:
                raise ValueError(f'Shard {shard_pth} has unknown file extension.')

            reader = csv.reader(
                file,
                dialect='excel-tab' if self._dialect == 'tsv' else 'excel',
            )

            for row in islice(reader, self._current_index, None, self._stride):
                self._current_index += self._stride
                assert len(row) >= self._tuple_len, (
                    f'Dataset {self._dataset}, shard {shard}, '
                    f'row {self._current_index}, row length '
                    f'{len(row)}, tuple length {self._tuple_len}'
                )
                out = [row[x] for x in range(self._tuple_len)]
                if self._add_ce_scores == 'true':
                    scores = self._get_ce_scores(*out)
                elif self._add_ce_scores == 'false':
                    scores = None
                elif self._add_ce_scores == 'zeros':
                    scores = [0.0] * (len(out) - 1)
                else:
                    raise ValueError(
                        f'add_ce_scores must be one of true, false, or zeros, '
                        f'got {self._add_ce_scores}'
                    )

                if self._input_has_score:
                    scores = [float(row[-1])]

                if self._task_implementation == 'instruction-based':
                    out = add_instruction(out, task_type=self._task_type)

                yield (
                    self._dataset,
                    (
                        out,
                        scores,
                    ),
                )
            file.close()

            if not os.path.exists(self._bucket):  # local bucket
                os.remove(shard_pth)
            self._current_index = self._rank if self._interleaved else 0

    def cleanup(self):
        if self._tmpdir is not None:
            log_on_rank(f'Cleaning up dataset {self._dataset}')
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._thread_pool.shutdown()
            self._thread_pool = None


def _list_to_tuple(data):
    if isinstance(data, list):
        return tuple(_list_to_tuple(item) for item in data)
    return data


def _path_to_dir(pth: str) -> str:
    return '/'.join(pth.split('/')[:-1])


def _path_to_name(pth: str) -> str:
    return pth.split('/')[-1]


class MultiDataset(IterableDataset):
    def __init__(
        self,
        bucket: str,
        world_size: int,
        global_rank: int,
        batch_size: int,
        input_type_dict: Dict[str, str],
        datasets: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
        max_shards: Optional[int] = None,
        sampling_rates: Optional[Dict[str, float]] = None,
        task_types: Optional[Dict[str, str]] = None,
        task_implementation: Literal['none', 'instruction-based'] = 'none',
        max_batches: Optional[int] = None,
        num_batches: int = 0,
        dialect: Literal['csv', 'tsv'] = 'tsv',
        rng_state: Optional[Union[Tuple, List]] = None,
        seed: int = 0,
        absolute_sampling_rates: bool = False,
        synchronous: bool = False,
        **kwargs,
    ):
        """A dataset that creates multiple S3 datasets and iterates over all of them at
        random.

        :param bucket: The name of the bucket where the data is stored.
        :param fabric: A fabric instance to get information about the current process'
            rank.
        :param batch_size: The number of pairs in a single batch, used to determine
            the number of pairs to yield before changing datasets.
        :param datasets: A list of datasets to create 'torch.utils.data.Dataset'
            instances for. Each dataset should specify the complete path within
            the S3 bucket.
        :param max_shards: The maximum number of shards to iterate over for each
            dataset.
        :param sampling_rates: A dictionary containing the sampling rates of each
            dataset.
            Datasets with a higher sampling rate will be sampled from more often.
        :param task_types: A dictionary containing the task type for each dataset.
        :param task_implementation: The implementation of the task, either 'none',
            or 'instruction-based'.
        :param max_batches: The maximum number of batches after which to stop.
        :param absolute_sampling_rates: Whether the provided sampling rates should be
            understood as absolute or as up/down-sampling rates.
        :param synchronous: Whether workers take different shards or iterate over the
            same shards with an offset and stride (default is different shards).
        """
        super().__init__()
        self._bucket = bucket
        self._batch_size = batch_size
        self._world_size = world_size
        self._global_rank = global_rank
        self._input_type_dict = input_type_dict
        self._synchronous = synchronous
        self._task_types = task_types or dict()
        self._task_implementation = task_implementation

        datasets: Optional[Union[List[str], List[Dict[str, Any]]]]
        if datasets is None:
            if not os.path.exists(self._bucket):
                datasets = [
                    os.path.join(bucket, folder)
                    for folder in get_dataset_info(bucket, bucket).keys()
                ]
            else:
                bucket_name = os.path.basename(self._bucket)
                dset_parent_dir = os.path.join(self._bucket, bucket_name)
                datasets = [
                    os.path.join(bucket_name, folder)
                    for folder in get_directories(dset_parent_dir)
                ]

        if isinstance(datasets, list):
            self._datasets = {
                dspath: S3Dataset(
                    bucket,
                    _path_to_name(dspath),
                    world_size=world_size,
                    global_rank=global_rank,
                    input_type_dict=input_type_dict,
                    task_type=self._task_types.get(dspath),
                    task_implementation=self._task_implementation,
                    directory=_path_to_dir(dspath),
                    max_shards=max_shards,
                    dialect=dialect,
                    interleaved=synchronous,
                )
                for dspath in datasets
            }
        else:
            datasets: Dict[str, Dict[str, Any]]
            self._datasets = {
                dspath: S3Dataset(
                    bucket=bucket,
                    dataset=_path_to_name(dspath),
                    directory=_path_to_dir(dspath),
                    world_size=world_size,
                    global_rank=global_rank,
                    interleaved=synchronous,
                    input_type_dict=input_type_dict,
                    task_type=self._task_types.get(dspath),
                    task_implementation=self._task_implementation,
                    max_shards=dataset['max_shards'],
                    dialect=dataset['dialect'],
                    shard_num=dataset['current_shard_num'],
                    index=dataset['current_index'],
                    **(
                        {'ce_model_name': kwargs['ce_model_name']}
                        if 'ce_model_name' in kwargs
                        else {}
                    ),
                )
                for dspath, dataset in datasets.items()
            }

        self._sampling_rates = {
            ds_path: len(dataset) for ds_path, dataset in self._datasets.items()
        }

        if sampling_rates is not None:
            if absolute_sampling_rates:
                required_rates = set(self._sampling_rates.keys())
                provided_rates = set(sampling_rates.keys())
                if provided_rates != required_rates:
                    raise ValueError(
                        f'Trying to use absolute sampling rates, but provided'
                        f'rates do not match datasets. Got sampling rates for '
                        f'{provided_rates} but need {required_rates}.'
                    )
                self._sampling_rates = sampling_rates
            else:
                # Use up/down-sampling rates
                for key, value in sampling_rates.items():
                    if key in self._sampling_rates:
                        self._sampling_rates[key] = self._sampling_rates[key] * value
                    else:
                        raise ValueError(
                            f'Sampling rate given for {key} has no '
                            f'corresponding dataset.'
                        )

        for ds_path, dataset in list(self._datasets.items()):
            if len(dataset._shards) == 0:
                del self._sampling_rates[ds_path]
                self._datasets[ds_path].cleanup()
                del self._datasets[ds_path]
        self._max_batches = max_batches
        self._num_batches = num_batches

        self.current_dataset = None

        if rng_state is None:
            seed_offset = 0 if synchronous else global_rank
            # multiply base seed by 64 to avoid overlaps in multi-gpu training
            self._rng = random.Random(64 * seed + seed_offset)
        else:
            self._rng = random.Random()
            self._rng.setstate(_list_to_tuple(rng_state))

    def __iter__(self):
        sources = {name: iter(ds) for name, ds in self._datasets.items()}
        while not self._max_batches or self._max_batches > self._num_batches:
            dataset = self._rng.choices(
                list(self._sampling_rates.keys()),
                weights=list(self._sampling_rates.values()),
            )[0]
            self.current_dataset = dataset

            self._num_batches += 1
            for _ in range(self._batch_size):
                try:
                    yield next(sources[dataset])
                except StopIteration:
                    log_on_rank(f'reached the end of dataset {dataset}, rebuilding')
                    self._datasets[dataset].cleanup()
                    self._datasets[dataset] = self.rebuild_dataset(
                        dataset, self._world_size, self._global_rank
                    )
                    sources[dataset] = iter(self._datasets[dataset])
                    yield next(sources[dataset])

        self._num_batches = 0

    def __len__(self):
        return self._max_batches * self._batch_size

    def rebuild_dataset(self, dataset: str, world_size: int, global_rank: int):
        return S3Dataset(
            bucket=self._bucket,
            dataset=_path_to_name(dataset),
            world_size=world_size,
            global_rank=global_rank,
            directory=_path_to_dir(dataset),
            max_shards=self._datasets[dataset]._max_shards,
            dialect=self._datasets[dataset]._dialect,
            input_type_dict=self._input_type_dict,
            task_type=self._task_types.get(dataset),
            task_implementation=self._task_implementation,
            interleaved=self._synchronous,
        )

    def state_dict(self):
        return {
            'bucket': self._bucket,
            'batch_size': self._batch_size,
            'datasets': {
                ds_path: {
                    'current_shard_num': dataset._current_shard_num,
                    'current_index': dataset._current_index,
                    'dialect': dataset._dialect,
                    'max_shards': dataset._max_shards,
                }
                for ds_path, dataset in self._datasets.items()
            },
            'sampling_rates': self._sampling_rates,
            'task_types': self._task_types,
            'task_implementation': self._task_implementation,
            'max_batches': self._max_batches,
            'num_batches': self._num_batches,
            'input_type_dict': self._input_type_dict,
            'rng': self._rng.getstate(),
            'synchronous': self._synchronous,
        }

    def cleanup(self):
        for ds in self._datasets.values():
            ds.cleanup()

    @classmethod
    def load_state_dict(cls, state_dict, world_size: int, global_rank: int):
        return cls(
            bucket=state_dict['bucket'],
            world_size=world_size,
            global_rank=global_rank,
            input_type_dict=state_dict['input_type_dict'],
            datasets=state_dict['datasets'],
            sampling_rates=state_dict['sampling_rates'],
            task_types=state_dict['task_types'],
            task_implementation=state_dict['task_implementation'],
            batch_size=state_dict['batch_size'],
            max_batches=state_dict['max_batches'],
            num_batches=state_dict['num_batches'],
            rng_state=state_dict['rng'],
            synchronous=state_dict['synchronous'],
        )

    def write_to_json(self, fname: str):
        directory_path = os.path.dirname(fname)
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            with open(fname, 'w') as json_file:
                json.dump(self.state_dict(), json_file)
        except Exception as e:
            _ = str(e)
            log_on_rank('File already exist, skipping.')  # avoid race condition

    @classmethod
    def load_from_json(cls, fname: str, world_size: int, global_rank: int):
        with open(fname, 'r') as json_file:
            return cls.load_state_dict(json.load(json_file), world_size, global_rank)

    @property
    def num_batches(self):
        return self._num_batches
