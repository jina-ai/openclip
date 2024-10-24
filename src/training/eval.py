import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Union, cast

import numpy as np
import torch
import torch.nn.functional as f
from datasets import Dataset, load_dataset
from deepspeed import DeepSpeedEngine
from loguru import logger
from PIL import Image
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype

from training.distributed import is_master
from training.utils import get_autocast


def _get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {'image2text': logits_per_image, 'text2image': logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f'{name}-mean-rank'] = preds.mean() + 1
        metrics[f'{name}-median-rank'] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f'{name}-R@{k}'] = np.mean(preds < k)

    return metrics


def _maybe_compute_generative_loss(model_out):
    if 'logits' in model_out and 'labels' in model_out:
        token_logits = model_out['logits']
        token_labels = model_out['labels']
        return f.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def _filter_metrics(_metrics: Dict[str, float], select_metrics: Set[str]):
    _filtered_metrics = {}
    for key, value in _metrics.items():
        if len(select_metrics) > 0 and key not in select_metrics:
            continue
        if isinstance(value, float):
            _filtered_metrics[key] = value

    return _filtered_metrics


def _run_validation(model, data, epoch, args):
    if (
        'val' not in data
        or args.val_frequency == 0
        or ((epoch % args.val_frequency) != 0 and epoch != args.epochs)
    ):
        return {}

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting evaluation on the validation set ...')

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    device = torch.device(args.device)

    dataloader = data['val'].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute
    # very quickly
    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_image_features, all_text_features = [], []

    metrics = {}

    logger.info('Infering text and image features ...')

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            _, __, images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out['image_features']
                text_features = model_out['text_features']
                logit_scale = model_out['logit_scale']
                # features are accumulated in CPU tensors, otherwise GPU memory is
                # exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes
                # problematic
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                total_loss = (
                    f.cross_entropy(logits_per_image, labels)
                    + f.cross_entropy(logits_per_text, labels)
                ) / 2
                gen_loss = _maybe_compute_generative_loss(model_out)

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logger.info(
                    f'Eval epoch: {epoch} [{num_samples} / {samples_per_val}]\t'
                    f'Clip loss: {cumulative_loss / num_samples:.6f}\t'
                )

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logger.info(
                        f'Generative loss: '
                        f'{cumulative_gen_loss / num_samples:.6f}\t'
                    )

        logger.info('Calculating CLIP metrics, mean/median rank and recall ...')

        val_metrics = _get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update({**val_metrics, 'clip-loss': loss.item()})
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({'generative-loss': gen_loss.item()})

    logger.info('Finished!')
    logger.info('--------------------------------------------------------------------')

    return metrics


def _run_clip_benchmark(model, tokenizer, transform, epoch, args):
    if args.clip_benchmark_frequency == 0 or (
        (epoch % args.clip_benchmark_frequency) != 0 and epoch != args.epochs
    ):
        return {}

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting the CLIP benchmark ...')

    from clip_benchmark.run import CLIPBenchmarkModel, run_benchmark

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    autocast = get_autocast(args.precision)
    with autocast():
        results = run_benchmark(
            datasets=[t for t in args.clip_benchmark_datasets.split(',')],
            languages=[lang for lang in args.clip_benchmark_languages.split(',')],
            models=[
                CLIPBenchmarkModel(
                    name=args.model,
                    pretrained=args.name + f'-epoch#{epoch}',
                    module=module,
                    tokenizer=tokenizer,
                    transform=transform,
                )
            ],
            batch_size=args.clip_benchmark_batch_size,
            precision=args.precision,
            task='auto',
            output=None,
            dataset_root=args.clip_benchmark_dataset_root,
            webdataset_root=args.clip_benchmark_webdataset_root,
            distributed=False,
            recall_ks=[int(k) for k in args.clip_benchmark_recall_ks.split(',')],
        )
    metrics = {}
    for result in results:
        dataset = result['dataset']
        language = result['language']
        for k, v in result['metrics'].items():
            metrics[f'{dataset}-{language}-{k}'] = v

    logger.info('Finished CLIP benchmark!')
    logger.info('--------------------------------------------------------------------')

    return metrics


def _run_mteb_benchmark(model, tokenizer, epoch, args):
    if args.mteb_frequency == 0 or (
        (epoch % args.mteb_frequency) != 0 and epoch != args.epochs
    ):
        return {}

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting the MTEB benchmark ...')

    import iso639
    from mteb import MTEB, get_tasks
    from open_clip.model import CLIP
    from transformers import AutoTokenizer

    class _MTEBEncoder(torch.nn.Module):
        def __init__(
            self,
            clip_model: torch.nn.Module,
            _tokenizer: Any = None,
            hf_tokenizer_name: str = '',
            batch_size: int = 4,
            max_seq_length: int = 8192,
            device: Union[str, torch.device] = 'cpu',
        ):
            super(_MTEBEncoder, self).__init__()

            self._tokenizer = None
            self._batch_size = batch_size
            self._max_seq_length = max_seq_length
            self._device = device

            if isinstance(clip_model, DeepSpeedEngine):
                _model = clip_model.module
            elif isinstance(clip_model, torch.nn.parallel.DistributedDataParallel):
                _model = clip_model.module
            else:
                _model = clip_model

            self._model = _model

            if isinstance(_model, CLIP):
                assert _tokenizer is not None
                self._tokenizer = _tokenizer
                self._embed = self._clip_embed

            else:
                assert hf_tokenizer_name
                self._tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=hf_tokenizer_name,
                    trust_remote_code=True,
                    force_download=True,
                )
                self._embed = self._hf_embed

        @staticmethod
        def _mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def _hf_embed(self, sentences: list[str]):
            encoded_input = self._tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self._max_seq_length,
            ).to(self._device)

            model_output = self._model.text.transformer(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input['attention_mask']
            )
            sentence_embeddings = f.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.to(torch.float32).cpu().numpy()

        def _clip_embed(self, sentences: list[str]):
            x = self._tokenizer(sentences).to(self._device)
            sentence_embeddings = self._model.encode_text(x)
            return sentence_embeddings.to(torch.float32).cpu().numpy()

        @torch.no_grad()
        def encode(self, sentences: list[str], batch_size: int = 1, **_):
            embeddings = []
            with torch.inference_mode():
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i : i + batch_size]
                    embeddings.append(self._embed(batch))

            return np.concatenate(embeddings, axis=0)

    _mteb_model = _MTEBEncoder(
        clip_model=model,
        _tokenizer=tokenizer,
        hf_tokenizer_name=args.mteb_tokenizer_name,
        max_seq_length=args.mteb_max_sequence_length,
        device=args.device,
    )

    metrics = {}
    tasks = args.mteb_tasks.split(',')
    langs = [
        iso639.Language.match(lang).part3 for lang in args.mteb_languages.split(',')
    ]
    select_metrics = set(args.mteb_metrics.split(','))
    autocast = get_autocast(args.precision)

    with autocast():
        for task in tasks:
            split = 'dev' if task == 'MSMARCO' else 'test'
            mteb_tasks = get_tasks(tasks=[task], languages=langs)
            evaluation = MTEB(tasks=mteb_tasks)
            results = evaluation.run(
                model=_mteb_model,
                verbosity=0,
                eval_splits=[split],
                encode_kwargs={'batch_size': args.mteb_batch_size},
                output_folder=None,
                ignore_identical_ids=False,
            )
            results = results[0].scores
            for split, _results in results.items():
                for scores in _results:
                    subset = scores['hf_subset'].replace('-', '_')
                    mteb_metrics = _filter_metrics(scores, select_metrics)
                    for k, v in mteb_metrics.items():
                        metrics[f'{task}-{subset}-{split}-{k}'] = v

    logger.info('Finished MTEB benchmark!')
    logger.info('--------------------------------------------------------------------')

    return metrics


def _run_vidore_benchmark(model, tokenizer, transform, epoch, args):
    if args.vidore_benchmark_frequency == 0 or (
        (epoch % args.vidore_benchmark_frequency) != 0 and epoch != args.epochs
    ):
        return {}

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting the Vidore benchmark ...')

    # sanity check
    if args.vidore_dataset_name is None and args.vidore_collection_name is None:
        raise ValueError('Please provide a dataset name or collection name')
    elif (
        args.vidore_dataset_name is not None and args.vidore_collection_name is not None
    ):
        raise ValueError('Please provide only one of dataset name or collection name')

    import huggingface_hub
    from vidore_benchmark.evaluation.evaluate import evaluate_dataset
    from vidore_benchmark.retrievers import VisionRetriever
    from vidore_benchmark.utils.iter_utils import batched
    from vidore_benchmark.utils.torch_utils import get_torch_device

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    autocast = get_autocast(args.precision)

    class VidoreOpenCLIPRetriever(VisionRetriever):
        def __init__(self):
            super().__init__()
            self.device = get_torch_device(args.device)
            self.model = module
            self.tokenizer = tokenizer
            self.transform = transform

        @property
        def use_visual_embedding(self) -> bool:
            return True

        def forward_queries(
            self, queries, batch_size: int, **kwargs
        ) -> List[torch.Tensor]:
            list_emb_queries: List[torch.Tensor] = []
            with torch.no_grad(), autocast():
                for query_batch in tqdm(
                    batched(queries, batch_size),
                    desc='Query batch',
                    total=math.ceil(len(queries) / batch_size),
                ):
                    query_batch = cast(List[str], query_batch)
                    inputs_queries = self.tokenizer(query_batch).to(self.device)
                    qs = self.model.encode_text(inputs_queries)
                    qs /= qs.norm(dim=-1, keepdim=True)
                    list_emb_queries.append(qs)

            return list_emb_queries

        def forward_documents(
            self, documents, batch_size: int, **kwargs
        ) -> List[torch.Tensor]:
            list_emb_documents: List[torch.Tensor] = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for doc_batch in tqdm(
                    batched(documents, batch_size),
                    desc='Document batch',
                    total=math.ceil(len(documents) / batch_size),
                ):
                    doc_batch = cast(List[Image.Image], doc_batch)
                    list_doc = [
                        document.convert('RGB')
                        for document in doc_batch
                        if isinstance(document, Image.Image)
                    ]
                    input_image_processed = torch.cat(
                        [
                            self.transform(d).unsqueeze(0).to(self.device)
                            for d in list_doc
                        ],
                        dim=0,
                    )
                    ps = self.model.encode_image(input_image_processed)
                    ps /= ps.norm(dim=-1, keepdim=True)
                    list_emb_documents.append(ps)

            return list_emb_documents

        def get_scores(
            self,
            list_emb_queries: List[torch.Tensor],
            list_emb_documents: List[torch.Tensor],
            batch_size: Optional[int] = None,
        ) -> torch.Tensor:
            emb_queries = torch.cat(list_emb_queries, dim=0)
            emb_documents = torch.cat(list_emb_documents, dim=0)
            scores = torch.einsum('bd,cd->bc', emb_queries, emb_documents)
            return scores

    _select_metrics = set(args.vidore_metrics.split(','))

    retriever = VidoreOpenCLIPRetriever()
    metrics = {}

    _batchsize = args.vidore_batch_size
    _dataset_name = args.vidore_dataset_name
    _collection_name = args.vidore_collection_name
    _dataset_split = args.vidore_dataset_split

    if _dataset_name is not None:
        dataset_id = _dataset_name
        logger.info(f'Evaluating {dataset_id} ...')
        dataset = cast(Dataset, load_dataset(dataset_id, split=_dataset_split))
        with autocast():
            metrics = evaluate_dataset(
                retriever,
                dataset,
                batch_query=_batchsize,
                batch_doc=_batchsize,
            )
        metrics = _filter_metrics(metrics, _select_metrics)
        metrics = {f'{dataset_id}-{k}': v for k, v in metrics.items()}

    elif _collection_name is not None:
        collection = huggingface_hub.get_collection(_collection_name)
        datasets = collection.items
        metrics = {}
        averages = defaultdict(list)
        with autocast():
            for dataset in datasets:
                dataset_id = dataset.item_id
                logger.info(f'Evaluating {dataset_id} ...')
                dataset = cast(Dataset, load_dataset(dataset_id, split=_dataset_split))
                dataset_metrics = evaluate_dataset(
                    retriever, dataset, batch_query=_batchsize, batch_doc=_batchsize
                )
                dataset_metrics = _filter_metrics(dataset_metrics, _select_metrics)
                for k, v in dataset_metrics.items():
                    averages[k].append(v)
                    metrics[f'{dataset_id.replace("vidore/", "")}-{k}'] = v

            for k, v in averages.items():
                metrics[f'avg-{k}'] = sum(v) / len(v)

    logger.info('Finished Vidore benchmark!')
    logger.info('--------------------------------------------------------------------')

    return metrics


def _run_cbir_benchmark(model, transform, epoch, args):
    if args.cbir_benchmark_frequency == 0 or (
        (epoch % args.cbir_benchmark_frequency) != 0 and epoch != args.epochs
    ):
        return {}

    logger.info('--------------------------------------------------------------------')
    logger.info('Starting the CBIR benchmark ...')

    autocast = get_autocast(args.precision)
    metrics = {}

    def extract_features(model, images, labels, batch_size, device):
        all_features = []
        all_labels = []
        num_samples = len(images)
        num_batches = num_samples // batch_size + int(num_samples % batch_size != 0)

        for i in tqdm(range(num_batches)):
            # Manually slice the dataset to get the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            transformed_images = [transform(image).to(device) for image in batch_images]

            batch_tensor = torch.stack(transformed_images)

            with torch.no_grad():
                with autocast():
                    features = model(batch_tensor)['image_features']
            all_features.append(features)
            all_labels.extend(batch_labels)

        # Concatenate all features and labels
        return torch.cat(all_features), torch.tensor(all_labels, device=device)

    def compute_precision_at_k(features, labels, k=10):
        total_queries = len(labels)
        precision_at_k = 0

        for i in range(total_queries):
            # Get similarity scores for the current query
            similarity_scores = f.cosine_similarity(features[i].unsqueeze(0), features)

            # Sort the indices by similarity score (highest to lowest)
            sorted_indices = torch.argsort(similarity_scores, descending=True)

            # Get the top K similar images
            top_k_indices = sorted_indices[:k]

            # Calculate Precision@K
            relevant_items = sum(
                [1 for idx in top_k_indices if labels[idx] == labels[i]]
            )
            precision_at_k += relevant_items / k

        precision_at_k /= total_queries
        return precision_at_k

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        _model = model.module
    else:
        _model = model

    task_list = [
        'tanganke/stanford_cars',
        'uoft-cs/cifar100',
        'nelorth/oxford-flowers',
        'dpdl-benchmark/sun397',
    ]
    for dataset in task_list:
        dataset_id = dataset.split('/')[1]
        logger.info(f'Evaluating {dataset_id} ...')
        split = 'validation' if dataset_id == 'sun397' else 'test'
        ds = load_dataset(dataset)[split]

        if dataset_id == 'cifar100':
            images = ds['img']
            labels = ds['fine_label']
        else:
            images = ds['image']
            labels = ds['label']

        image_features, image_labels = extract_features(
            _model, images, labels, args.cbir_batch_size, args.device
        )
        precision_at_10 = compute_precision_at_k(image_features, image_labels, k=10)

        metrics[f'{dataset_id}-precision_at_10'] = precision_at_10

    logger.info('Finished CBIR benchmark!')
    logger.info('--------------------------------------------------------------------')
    return metrics


def _draw_similarity_graph(model, transform, tokenizer, epoch, args, step):
    if args.simgraph_frequency == 0 or (
        (epoch % args.simgraph_frequency) != 0 and epoch != args.epochs
    ):
        return None

    logger.info('--------------------------------------------------------------------')
    logger.info('Drawing the similarity graphs ...')

    def _create_similarities(_images, _queries, _docs):
        img2txt_pos_sims = []
        txt2txt_pos_sims = []
        img2txt_neg_sims = []
        txt2txt_neg_sims = []
        img2img_neg_sims = []
        for img, query, doc in zip(_images, _queries, _docs):
            img2txt_pos_sims.append(img @ doc.T)
            txt2txt_pos_sims.append(query @ doc.T)
            img2txt_neg_sims.append(img @ random.choice(_docs).T)
            txt2txt_neg_sims.append(query @ random.choice(_docs).T)
            img2img_neg_sims.append(img @ random.choice(_images).T)

        return (
            img2txt_pos_sims,
            txt2txt_pos_sims,
            img2txt_neg_sims,
            txt2txt_neg_sims,
            img2img_neg_sims,
        )

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        _model = model.module
    else:
        _model = model

    dataset = load_dataset('jxie/flickr8k')['test']

    text_queries = []
    final_images = []
    text_docs = []

    autocast = get_autocast(args.precision)
    for item in tqdm(dataset):
        text_inputs = tokenizer([item['caption_0'], item['caption_1']])
        image_inputs = transform(item['image'])

        images = image_inputs.unsqueeze(0).to(args.device)
        texts = text_inputs.to(args.device)

        with torch.no_grad():
            with autocast():
                output = _model(images, texts)
                image_features = output['image_features'].to('cpu').numpy()
                text_features = output['text_features'].to('cpu').numpy()

        final_images.append(image_features[0])
        text_queries.append(text_features[0])
        text_docs.append(text_features[1])

    img_txt_pos, txt_txt_pos, img_txt_neg, txt_txt_neg, img_img_neg = (
        _create_similarities(final_images, text_queries, text_docs)
    )

    import matplotlib.pyplot as plt

    _hist_kwargs = dict(bins=30, alpha=0.5, density=True)
    plt.figure(figsize=(10, 6))
    plt.hist(img_txt_pos, label='POSimg2txt', color='red', **_hist_kwargs)
    plt.hist(txt_txt_pos, label='POStxt2txt', color='blue', **_hist_kwargs)
    plt.hist(img_txt_neg, label='NEGimg2txt', color='orange', **_hist_kwargs)
    plt.hist(txt_txt_neg, label='NEGtxt2txt', color='lightblue', **_hist_kwargs)
    plt.hist(img_img_neg, label='NEGimg2img', color='lightgreen', **_hist_kwargs)

    plt.title(f'Cosine Similarity Distribution - {args.name} epoch #{epoch}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend(loc='upper left')

    plt.grid(True)

    if args.save_logs:
        plt.savefig(os.path.join(args.checkpoint_path, f'simgraph@{epoch}.png'))

    if args.wandb:
        wandb.log({'cossim-graph': wandb.Image(plt)}, step=step or 0)

    plt.close()

    logger.info('Finished!')
    logger.info('--------------------------------------------------------------------')


def evaluate(
    model: torch.nn.Module,
    transform: Any,
    tokenizer: Any,
    data,
    epoch: int,
    args,
    step_now: int = 0,
    tb_writer: Any = None,
):
    metrics = {}
    if not is_master(args):
        return metrics

    step = step_now

    model.eval()

    logger.info('--------------------------- EVALUATION -----------------------------')

    val_metrics = _run_validation(model, data, epoch, args)
    metrics.update({f'valset-{k}': v for k, v in val_metrics.items()})

    clip_benchmark_metrics = _run_clip_benchmark(
        model, tokenizer, transform, epoch, args
    )
    metrics.update({f'clipb-{k}': v for k, v in clip_benchmark_metrics.items()})

    mteb_metrics = _run_mteb_benchmark(model, tokenizer, epoch, args)
    metrics.update({f'mteb-{k}': v for k, v in mteb_metrics.items()})

    vidore_benchmark_metrics = _run_vidore_benchmark(
        model, tokenizer, transform, epoch, args
    )
    metrics.update({f'vidore-{k}': v for k, v in vidore_benchmark_metrics.items()})

    cbir_benchmark_metrics = _run_cbir_benchmark(model, transform, epoch, args)
    metrics.update({f'cbir-{k}': v for k, v in cbir_benchmark_metrics.items()})

    _draw_similarity_graph(model, transform, tokenizer, epoch, args, step)

    if not metrics:
        return {}

    logger.info(
        f'Eval epoch: {epoch} '
        + '\t'.join([f'{k}: {round(v, 4):.4f}' for k, v in metrics.items()])
    )
    logdata = {'val/' + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in logdata.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, 'results.jsonl'), 'a+') as fd:
            fd.write(json.dumps(metrics))
            fd.write('\n')

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        logdata['epoch'] = epoch
        wandb.log(logdata, step=step)

    logger.info('------------------------------ DONE --------------------------------')

    return metrics
