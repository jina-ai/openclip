import csv
import gzip
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

import boto3
import torch
from aiohttp import ClientError
from loguru import logger
from torch.distributed import get_rank as torch_get_rank
from training.distributed import is_using_distributed
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
)

s3_client = boto3.client('s3', region_name='eu-central-1')


INSTRUCTION_CONFIG = MappingProxyType(
    {
        '': ('', ''),
        'retrieval': ('Query: ', 'Document for retrieval: '),
        'sts': ('Statement for clustering: ', 'Statement for clustering: '),
        'reranking': ('Query: ', 'Document for reranking: '),
        'clustering': ('Statement for clustering: ', 'Statement for clustering: '),
        'classification': (
            'Statement for classification: ',
            'Statement for classification: ',
        ),
    }
)


def get_rank(group=None):
    if is_using_distributed():
        return torch_get_rank(group)
    return 0


def log_on_rank(message):
    try:
        rank = get_rank()
        logger.debug(f'[rank={rank}]{message}')
    except RuntimeError:
        logger.debug(message)


class SimLMCrossEncoder:
    def __init__(
        self,
        model_name: str = 'intfloat/simlm-msmarco-reranker',
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        if device:
            self._device = device
        else:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.eval()
        self._model.to(self._device)

    def predict(self, sentences: List[List[str]]) -> BatchEncoding:
        query, target = zip(*sentences)
        target = [f'-: {x}' for x in target]
        features = self._tokenizer(
            query,
            text_pair=target,
            max_length=192,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(self._device)
        with torch.no_grad():
            return self._model(**features, return_dict=True).logits[:, 0].cpu().detach()


def get_shards(dataset: str, bucket_name: str, directory: Optional[str] = None):
    if directory is None:
        directory = bucket_name
    if os.path.exists(bucket_name):  # local path
        shards = []
        for file in os.listdir(os.path.join(bucket_name, directory, dataset)):
            fname = os.path.join(bucket_name, directory, dataset, file)
            if os.path.isfile(fname):
                shards.append(fname)
    else:  # remote s3 bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=bucket_name, Prefix=f'{directory}/{dataset}/', Delimiter='/'
        )
        try:
            shards = [
                shard['Key']
                for page in pages
                for shard in page['Contents']
                if shard['Key'] != f'{directory}/{dataset}/'
            ]
        except KeyError as e:
            log_on_rank(f'KEY ERROR: {dataset}')
            raise e

    return shards


def get_tags(bucket_name: str, directory: Optional[str] = None):
    if directory is None:
        tags = s3_client.get_bucket_tagging(Bucket=bucket_name)
        tags = {tag['Key']: int(tag['Value']) for tag in tags['TagSet']}
    else:
        try:
            s3_client.download_file(
                Bucket=bucket_name, Key=f'{directory}/sizes.json', Filename='sizes.json'
            )
            with open('sizes.json', 'r') as file:
                tags = json.load(file)
        except ClientError:
            raise ValueError(f'No sizes file found in directory {directory}')
    return tags


def get_dataset_info(bucket_name, directory: Optional[str] = None):
    if directory is None:
        directory = bucket_name
    if os.path.exists(bucket_name):  # local path
        datasets = os.listdir(f'{bucket_name}/{directory}')
    else:  # remote s3 bucket
        result = s3_client.list_objects(
            Bucket=bucket_name, Prefix=f'{directory}/', Delimiter='/'
        )
        datasets = []
        for out in result.get('CommonPrefixes'):
            datasets.append(out.get('Prefix')[len(directory) + 1 : -1])
    # try to get size info
    try:
        tags = get_tags(bucket_name, directory)
    except Exception as _:
        log_on_rank(f'Could not retrieve size values for {bucket_name}/{directory}')
        tags = {}
    dataset_dict = {}
    for dataset in datasets:
        if dataset in tags:
            dataset_dict[dataset] = tags[dataset]
        else:
            dataset_dict[dataset] = None
    return dataset_dict


def download_shard(
    source_bucket: str,
    shard: str,
    target_dir: str,
):
    fname = shard.split('/')[-1]
    target_path = os.path.join(target_dir, fname)
    if os.path.exists(target_path):
        return target_path
    elif os.path.exists(shard):
        return shard
    log_on_rank(f'Downloading shard {shard} from {source_bucket}')
    s3_client.download_file(
        Bucket=source_bucket,
        Key=shard,
        Filename=target_path,
    )
    return target_path


def get_shard_size(source_bucket: str, shard: str, dialect: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = download_shard(
            source_bucket=source_bucket, shard=shard, target_dir=tmpdir
        )
        if shard_path.endswith('.gz'):
            with gzip.open(shard_path, 'rt') as gz_file:
                reader = csv.reader(
                    gz_file, dialect='excel-tab' if dialect == 'tsv' else 'excel'
                )
                entries = list(reader)
        elif shard_path.endswith(f'.{dialect}'):
            with open(shard_path, 'r') as file:
                reader = csv.reader(
                    file, dialect='excel-tab' if dialect == 'tsv' else 'excel'
                )
                entries = list(reader)
        if not os.path.exists(source_bucket):
            os.remove(shard_path)

    shard_len = len(entries)
    return shard_len


def get_directories(path: str):
    directories = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                directories.append(entry.name)
    return directories


def lookahead(f):
    lookahead.future = None
    thread_pool = ThreadPoolExecutor(max_workers=1)

    def g(*args, **kwargs):
        future = thread_pool.submit(f, *args, **kwargs)
        result = lookahead.future.result() if lookahead.future is not None else None
        lookahead.future = future
        return result

    return g


def add_instruction(
    texts,
    task_type: Optional[str],
    instruction_config: Dict[str, Tuple[str]] = INSTRUCTION_CONFIG,
):
    if task_type is None:
        task_type = ''
    first_prefix = instruction_config[task_type][0]
    remaining_prefixes = instruction_config[task_type][1]
    if len(texts) < 1:
        raise ValueError('Texts must contain at least one element')
    return [f'{first_prefix}{texts[0]}'] + [
        f'{remaining_prefixes}{text}' for text in texts[1:]
    ]
