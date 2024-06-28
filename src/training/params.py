import argparse
import ast
import json
import os

import yaml


def _get_default_params(modelname):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    modelname = modelname.lower()
    if 'vit' in modelname:
        return {'lr': 5.0e-4, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1.0e-6}
    else:
        return {'lr': 5.0e-4, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(
                    value
                )  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            'The path to a YAML or JSON file with configuration parameters. '
            'The parameters in the file override the command line arguments.'
        ),
    )
    parser.add_argument(
        '--local_rank', type=int, default=0, help='Needed for DeepSpeed, ignore.'
    )

    # ---------------------------------------------------------------------------- DATA
    parser.add_argument(
        '--train-data',
        type=str,
        default=None,
        help=(
            'Path to file(s) with training data. When using type `webdataset` multiple '
            'datasources can be combined using the `::` separator.'
        ),
    )
    parser.add_argument(
        '--train-data-upsampling-factors',
        type=str,
        default=None,
        help=(
            'When using multiple data sources with webdataset and sampling with '
            'replacement, this can be used to upsample specific data sources. '
            'Similar to --train-data, this should be a string with as many numbers '
            'as there are data sources, separated by `::` (e.g. 1::2::0.5). '
            'By default, datapoints are sampled uniformly regardless of the '
            'dataset sizes.'
        ),
    )
    parser.add_argument(
        '--train-data-s3',
        type=str,
        default=None,
        help='Similar to --train-data, but reserved for datasets located in S3.',
    )
    parser.add_argument(
        '--train-data-s3-upsampling-factors',
        type=str,
        default=None,
        help='Similar to --train-data-upsampling-factors, but for --train-data-s3.',
    )
    parser.add_argument(
        '--train-data-s3-bucket',
        type=str,
        default=None,
        help=(
            'In case S3 datasets are provided in --train-data-s3, this argument '
            'specifies the S3 bucket where datasets are located.'
        ),
    )
    parser.add_argument(
        '--train-data-mtl',
        type=str,
        default=None,
        help=(
            'Similar to --train-data-s3 but the datasets specified here are used in '
            'parallel to the main training loop. Useful for joint Multi Task '
            'Training.'
        ),
    )
    parser.add_argument(
        '--train-data-mtl-upsampling-factors',
        type=str,
        default=None,
        help='Similar to --train-data-s3-upsampling-factors but for --train-data-mtl.',
    )
    parser.add_argument(
        '--train-data-mtl-s3-bucket',
        type=str,
        default=None,
        help='Similar to --train-data-s3-bucket but for --train-data-mtl.',
    )
    parser.add_argument(
        '--train-num-samples',
        type=int,
        default=None,
        help=(
            'Number of samples in the training dataset. Required for webdatasets '
            'if the dataset size are not available in the info files.'
        ),
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default=None,
        help='Path to file(s) with validation data',
    )
    parser.add_argument(
        '--val-num-samples',
        type=int,
        default=None,
        help=(
            'Number of samples in the validation dataset. Useful for webdatasets if '
            'the dataset sizes are not available in the info files.'
        ),
    )
    parser.add_argument(
        '--dataset-type',
        choices=['webdataset', 'csv', 'synthetic', 's3', 'auto'],
        default='auto',
        help='Indicates the type of dataset to process.',
    )
    parser.add_argument(
        '--dataset-resampled',
        default=False,
        action='store_true',
        help='Whether to use sampling with replacement for webdataset shard selection.',
    )
    parser.add_argument(
        '--s3-max-shards',
        type=int,
        default=None,
        help='The max shards of the S3 dataloader.',
    )
    parser.add_argument(
        '--s3-max-batches',
        type=int,
        default=None,
        help='The max batches of the S3 dataloader.',
    )
    parser.add_argument(
        '--s3-num-batches',
        type=int,
        default=0,
        help='The num batches of the S3 dataloader.',
    )
    parser.add_argument(
        '--csv-separator',
        type=str,
        default='\t',
        help='For CSV-like datasets, select which separator to use.',
    )
    parser.add_argument(
        '--csv-image-key',
        type=str,
        default='image',
        help='For CSV-like datasets, the name of the column with the image paths.',
    )
    parser.add_argument(
        '--csv-caption-key',
        type=str,
        default='title',
        help='For CSV-like datasets, the name of the column containing the captions.',
    )
    # --------------------------------------------------------------------------- MODEL
    parser.add_argument(
        '--model',
        type=str,
        default='RN50',
        help='Specify the CLIP model architecture to train.',
    )
    parser.add_argument(
        '--pretrained',
        default='',
        type=str,
        help=(
            'Initialize the model with pretrained weights by specifying a pretrained '
            'model tag or checkpoint file path.'
        ),
    )
    parser.add_argument(
        '--pretrained-image',
        default=False,
        action='store_true',
        help='Load imagenet pretrained weights for image tower backbone if available.',
    )
    parser.add_argument(
        '--hf-random-init',
        default=False,
        action='store_true',
        help='Randomly initialize the weights for a HuggingFace text tower if needed.',
    )
    parser.add_argument(
        '--lock-image',
        default=False,
        action='store_true',
        help='Freeze the image tower during training.',
    )
    parser.add_argument(
        '--lock-image-unlocked-groups',
        type=int,
        default=0,
        help='Leave the last n layer groups in the vision tower unlocked.',
    )
    parser.add_argument(
        '--lock-image-freeze-bn-stats',
        default=False,
        action='store_true',
        help=(
            'Freeze BatchNorm running stats in the vision tower for any locked layers.'
        ),
    )
    parser.add_argument(
        '--lock-text',
        default=False,
        action='store_true',
        help='Freeze the text tower during training.',
    )
    parser.add_argument(
        '--lock-text-unlocked-groups',
        type=int,
        default=0,
        help='Leave the last n layer groups in the text tower unlocked.',
    )
    parser.add_argument(
        '--lock-text-freeze-layer-norm',
        default=False,
        action='store_true',
        help='Freeze LayerNorm running stats in the text tower for any locked layers.',
    )
    parser.add_argument(
        '--force-quick-gelu',
        default=False,
        action='store_true',
        help='Force use of QuickGELU activation for non-OpenAI transformer models.',
    )
    parser.add_argument(
        '--force-patch-dropout',
        default=None,
        type=float,
        help=(
            'Override the patch dropout during training, usefule when fine-tuning with '
            'no dropout near the end.'
        ),
    )
    parser.add_argument(
        '--force-custom-text',
        default=False,
        action='store_true',
        help='Force use of CustomTextCLIP model (separate text-tower).',
    )
    parser.add_argument(
        '--use-bnb-linear',
        default=False,
        action='store_true',
        help=(
            'Replaces the network linear layers from the bitsandbytes library. '
            'Allows for int8 training/inference, etc.'
        ),
    )
    parser.add_argument(
        '--use-bn-sync',
        default=False,
        action='store_true',
        help='Whether to use batch norm sync.',
    )
    parser.add_argument(
        '--distill-model',
        default=None,
        help='Which model architecture to distill from, if any.',
    )
    parser.add_argument(
        '--distill-pretrained',
        default=None,
        help='Which pretrained weights to distill from, if any.',
    )
    parser.add_argument(
        '--torchscript',
        default=False,
        action='store_true',
        help=(
            'Run torch.jit.script on the model, uses jit version of OpenAI models '
            'if pretrained==openai.'
        ),
    )
    parser.add_argument(
        '--torchcompile',
        default=False,
        action='store_true',
        help='Run torch.compile() on the model, requires pytorch>=2.0.',
    )
    # ------------------------------------------------------------------------ TRAINING
    parser.add_argument(
        '--epochs',
        type=int,
        default=32,
        help=(
            'Number of epochs to train for. An epoch is defined as a full run over '
            '--train-num-samples'
        ),
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help='Batch size per GPU.'
    )
    parser.add_argument(
        '--mtl-batch-size',
        type=int,
        default=128,
        help='The batch size for the MTL training loop.',
    )
    parser.add_argument(
        '--workers', type=int, default=4, help='Number of dataloader workers per GPU.'
    )
    parser.add_argument(
        '--precision',
        choices=[
            'amp',
            'amp_bf16',
            'amp_bfloat16',
            'bf16',
            'fp16',
            'pure_bf16',
            'pure_fp16',
            'fp32',
        ],
        default='amp',
        help='Floating point precision.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument(
        '--grad-clip-norm',
        type=float,
        default=None,
        help='Gradient clipping threshold. If None, gradient clipping is disabled.',
    )
    parser.add_argument(
        '--accum-freq',
        type=int,
        default=1,
        help='Gradient accumulation frequency in steps.',
    )
    parser.add_argument(
        '--grad-checkpointing',
        default=False,
        action='store_true',
        help='Enable gradient checkpointing.',
    )
    # ----------------------------------------------------------------------- OPTIMIZER
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adamw', 'lamb'],
        default='adamw',
        help='Optimizer type.',
    )
    parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
    parser.add_argument(
        '--text-lr',
        type=float,
        default=None,
        help='Specify a different learning rate for the text tower.',
    )
    parser.add_argument(
        '--beta1', type=float, default=None, help='Adam beta 1 parameter.'
    )
    parser.add_argument(
        '--beta2', type=float, default=None, help='Adam beta 2 parameter.'
    )
    parser.add_argument(
        '--eps', type=float, default=None, help='Adam epsilon parameter.'
    )
    parser.add_argument('--wd', type=float, default=0.2, help='Weight decay parameter.')
    # ----------------------------------------------------------------------- SCHEDULER
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        choices=['cosine', 'const', 'const-cooldown'],
        default='cosine',
        help='LR scheduler type.',
    )
    parser.add_argument(
        '--skip-scheduler',
        action='store_true',
        default=False,
        help='Use this flag to skip LR scheduling.',
    )
    parser.add_argument(
        '--warmup', type=int, default=10000, help='Number of scheduler warmup steps.'
    )
    parser.add_argument(
        '--lr-cooldown-end',
        type=float,
        default=0.0,
        help='End learning rate for cooldown schedule. Default is 0.0.',
    )
    parser.add_argument(
        '--lr-cooldown-power',
        type=float,
        default=1.0,
        help='Power for polynomial cooldown schedule. Default: 1.0 (linear decay)',
    )
    parser.add_argument(
        '--epochs-cooldown',
        type=int,
        default=None,
        help=(
            'When a scheduler with cooldown is used, perform cooldown from '
            'total_epochs - epochs_cooldown onwards.'
        ),
    )
    parser.add_argument(
        '--llr-decay',
        type=float,
        default=1.0,
        help=(
            'Layerwise Learning Rate Decay (LLRD) factor to use. '
            'Value of 1.0 means no LLRD'
        ),
    )
    parser.add_argument(
        '--text-llr-decay',
        type=float,
        default=None,
        help=(
            'Specify a separate Layerwise Learning Rate Decay (LLRD) factor to use '
            'for the text tower. Value of 1.0 means no LLRD'
        ),
    )
    # ---------------------------------------------------------------------------- LOSS
    parser.add_argument(
        '--siglip',
        default=False,
        action='store_true',
        help='Use SigLIP (sigmoid) loss.',
    )
    parser.add_argument(
        '--matryoshka',
        default=False,
        action='store_true',
        help='Use Matryoshka loss.',
    )
    parser.add_argument(
        '--temperature',
        default=None,
        type=float,
        help='InfoNCE temperature parameter.',
    )
    parser.add_argument(
        '--freeze-temperature',
        default=False,
        action='store_true',
        help='Keep the temperature parameter constant during training.',
    )
    parser.add_argument(
        '--local-loss',
        default=False,
        action='store_true',
        help=(
            'Calculate loss with local features @ global (instead of realizing '
            'full global @ global matrix)'
        ),
    )
    parser.add_argument(
        '--gather-with-grad',
        default=False,
        action='store_true',
        help='Enable full distributed gradients for features gather.',
    )
    parser.add_argument(
        '--mtl-loss',
        type=str,
        default=None,
        help=(
            'Comma separated or JSON list of loss functions to use for MTL ' 'training.'
        ),
    )
    parser.add_argument(
        '--mtl-loss-weight',
        type=float,
        default=1.0,
        help='Comma separated weighing factors for the MTL losses.',
    )
    parser.add_argument(
        '--coca-caption-loss-weight',
        type=float,
        default=2.0,
        help='Weight assigned to caption loss in CoCa.',
    )
    parser.add_argument(
        '--coca-contrastive-loss-weight',
        type=float,
        default=1.0,
        help='Weight assigned to contrastive loss in CoCa.',
    )
    # ------------------------------------------------------------------------- LOGGING
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help=(
            'Optional experiment identifier when storing logs. If None the current '
            'timestamp is used.'
        ),
    )
    parser.add_argument(
        '--logs',
        type=str,
        default='./logs/',
        help='Where to store tensorboard/wandb logs. Use None to avoid storing logs.',
    )
    parser.add_argument(
        '--log-local',
        action='store_true',
        default=False,
        help='Log files on local master, otherwise global master only.',
    )
    parser.add_argument(
        '--log-every-n-steps',
        type=int,
        default=100,
        help='Log every n steps to console/tensorboard/wandb.',
    )
    parser.add_argument(
        '--report-to',
        default='',
        choices=['', 'wandb', 'tensorboard', 'wandb,tensorboard'],
        type=str,
        help='Choose a remote logging backend.',
    )
    parser.add_argument(
        '--wandb-notes', default='', type=str, help='Notes if logging with wandb.'
    )
    parser.add_argument(
        '--wandb-project-name',
        type=str,
        default='openclip',
        help='Name of the project if logging with wandb.',
    )
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='If true, more information are logged.',
    )
    parser.add_argument(
        '--copy-codebase',
        default=False,
        action='store_true',
        help=(
            'If true, we copy the entire base to the logs directory, and execute '
            'from there.'
        ),
    )
    # --------------------------------------------------------------------- CHECKPOINTS
    parser.add_argument(
        '--resume',
        default=None,
        type=str,
        help='Resume for a checkpoint path.',
    )
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=1,
        help='How often (in epochs) to save checkpoints.',
    )
    parser.add_argument(
        '--save-most-recent',
        action='store_true',
        default=False,
        help='Always save the most recent model trained to epoch-latest.pt.',
    )
    parser.add_argument(
        '--delete-previous-checkpoint',
        default=False,
        action='store_true',
        help='If true, delete previous checkpoint after storing a new one.',
    )
    parser.add_argument(
        '--remote-sync',
        type=str,
        default=None,
        help='Optionally sync with a remote path specified by this argument.',
    )
    parser.add_argument(
        '--remote-sync-frequency',
        type=int,
        default=300,
        help=(
            'How frequently to sync to a remote directly if --remote-sync is not None.'
        ),
    )
    parser.add_argument(
        '--remote-sync-protocol',
        choices=['s3', 'fsspec'],
        default='s3',
        help=(
            'What protocol to use for remote sync backups if --remote-sync is not None.'
        ),
    )
    # --------------------------------------------------------------------- DISTRIBUTED
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='The URL used to set up distributed training.',
    )
    parser.add_argument(
        '--dist-backend', default='nccl', type=str, help='The distributed backend.'
    )
    parser.add_argument(
        '--horovod',
        default=False,
        action='store_true',
        help='Use horovod for distributed training.',
    )
    parser.add_argument(
        '--deepspeed',
        action='store_true',
        default=False,
        help='Use deepspeed for distributed training.',
    )
    parser.add_argument(
        '--zero-stage',
        type=int,
        default=1,
        help='Stage of ZeRO algorithm, applicable if deepspeed is enabled.',
    )
    parser.add_argument(
        '--zero-bucket-size',
        type=int,
        default=1e6,
        help='ZeRO algorith allgather and reduce bucket size.',
    )
    parser.add_argument(
        '--ddp-static-graph',
        default=False,
        action='store_true',
        help='Enable static graph optimization for DDP in PyTorch >= 1.11.',
    )
    parser.add_argument(
        '--no-set-device-rank',
        default=False,
        action='store_true',
        help=(
            "Don't set device index from local rank (when CUDA_VISIBLE_DEVICES is "
            'restricted to one per proc).'
        ),
    )
    # ------------------------------------------------------------------- PREPROCESSING
    parser.add_argument(
        '--max-sequence-length',
        default=77,
        type=int,
        help='CLIP training max sequence length.',
    )
    parser.add_argument(
        '--mtl-max-sequence-length',
        type=int,
        default=77,
        help='The max sequence length of the MTL dataloader.',
    )
    parser.add_argument(
        '--force-image-size',
        type=int,
        nargs='+',
        default=None,
        help='Override default image size.',
    )
    parser.add_argument(
        '--image-mean',
        type=float,
        nargs='+',
        default=None,
        metavar='MEAN',
        help='Override default image mean value of dataset.',
    )
    parser.add_argument(
        '--image-std',
        type=float,
        nargs='+',
        default=None,
        metavar='STD',
        help='Override default image std value of dataset.',
    )
    parser.add_argument(
        '--image-interpolation',
        default=None,
        type=str,
        choices=['bicubic', 'bilinear', 'random'],
        help='Override default image resize interpolation.',
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None,
        type=str,
        choices=['shortest', 'longest', 'squash'],
        help='Override default image resize (& crop) mode during inference.',
    )
    parser.add_argument(
        '--aug-cfg',
        nargs='*',
        default={},
        action=ParseKwargs,
        help='Data augmentation config.',
    )
    # ---------------------------------------------------------------------- EVALUATION
    parser.add_argument(
        '--evaluate-on-start',
        action='store_true',
        default=False,
        help='Run evaluation before the first training epoch.',
    )
    parser.add_argument(
        '--val-frequency',
        type=int,
        default=1,
        help='How often (in epochs) to run evaluation with val data.',
    )
    parser.add_argument(
        '--zeroshot-frequency',
        type=int,
        default=1,
        help='How often (in epochs) to run zero shot evaluation.',
    )
    parser.add_argument(
        '--clip-benchmark-frequency',
        type=int,
        default=1,
        help='How often (in epochs) to run evaluation using the CLIP benchmark.',
    )
    parser.add_argument(
        '--mteb-frequency',
        type=int,
        default=1,
        help='How often (in epochs) to run evaluation on MTEB.',
    )
    parser.add_argument(
        '--imagenet-val',
        type=str,
        default=None,
        help='Path to ImageNet val set for conducting zero shot evaluation.',
    )
    parser.add_argument(
        '--imagenet-v2',
        type=str,
        default=None,
        help='Path to ImageNet v2 for conducting zero shot evaluation.',
    )
    parser.add_argument(
        '--clip-benchmark-datasets',
        type=str,
        default=(
            'wds/mscoco_captions,'
            'wds/multilingual_mscoco_captions,'
            'wds/flickr8k,'
            'wds/flickr30k,'
            'wds/flickr30k-200,'
            'wds/crossmodal3600,'
            'wds/xtd200,'
            'wds/imagenet1k'
        ),
        help='Specify datasets to use in CLIP benchmark.',
    )
    parser.add_argument(
        '--clip-benchmark-dataset-root',
        type=str,
        default=(
            'https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}'
            '/tree/main'
        ),
        help='Specify CLIP Benchmark datasets root path.',
    )
    parser.add_argument(
        '--clip-benchmark-languages',
        type=str,
        default='en',
        help='Specify CLIP Benchmark languages.',
    )
    parser.add_argument(
        '--clip-benchmark-recall-ks',
        type=str,
        default='1,5',
        help=(
            'Define a comma separated list of k values, at which metrics will be '
            'calculated in the CLIP Benchmark.'
        ),
    )
    parser.add_argument(
        '--mteb-tasks',
        type=str,
        default='STS12,STS15,STS17',
        help='Define a comma separated list of MTEB tasks to evaluate on.',
    )
    parser.add_argument(
        '--mteb-languages',
        type=str,
        default='en',
        help='Specify MTEB languages.',
    )
    parser.add_argument(
        '--mteb-max-sequence-length',
        type=int,
        default=8192,
        help='The max sequence length that will be used during MTEB evaluation.',
    )

    args = parser.parse_args(args)

    if args.config:
        if os.path.isfile(args.config):
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(args.config, 'r') as f:
                    config = json.load(f)
        else:
            print(f'The config file {args.config} does not exist!')
            exit(1)

        for k, v in config.items():
            key = k.replace('-', '_')
            if hasattr(args, key):
                setattr(args, key, v)

    # If some params are not passed, we use the default values based on model name.
    defaultparams = _get_default_params(args.model)
    for name, val in defaultparams.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.deepspeed:
        try:
            import deepspeed

            os.environ['ENV_TYPE'] = 'deepspeed'
            dsinit = deepspeed.initialize
        except ImportError or ModuleNotFoundError:
            print("DeepSpeed is not available, please run 'pip install deepspeed'")
            exit(1)
    else:
        os.environ['ENV_TYPE'] = 'pytorch'
        dsinit = None

    return args, dsinit
