from torch import nn, optim


def create_optimizer(args, model: nn.Module, dsinit=None):
    is_gain_or_bias = (
        lambda n, p: p.ndim < 2
        or 'bn' in n
        or 'ln' in n
        or 'bias' in n
        or 'logit_scale' in n
    )

    def is_text_module(n: str):
        return (
            n.startswith('text')
            or n.startswith('transformer')
            or n.startswith('module.text')  # for torch.DistributedDataParallel
            or n.startswith('module.transformer')  # for torch.DistributedDataParallel
        )

    def is_vision_module(n: str):
        return (
            n.startswith('visual')
            or n.startswith('module.visual')  # for torch.DistributedDataParallel
        )

    params = []
    _text_lr = args.text_lr if args.text_lr is not None else args.lr
    _text_counter = 0
    _vision_lr = args.lr
    _vision_counter = 0
    _text_llr_decay = (
        args.text_llr_decay if args.text_llr_decay is not None else args.llr_decay
    )
    _vision_llr_decay = args.llr_decay

    for name, param in reversed(list(model.named_parameters())):
        if param.requires_grad:
            _weight_decay = 0.0 if is_gain_or_bias(name, param) else args.wd

            lr = args.lr
            descriptor = ''
            if is_text_module(name):
                lr = _text_lr
                descriptor = f'type=text|depth={_text_counter}|name={name}|'
                _text_lr *= _text_llr_decay
                _text_counter += 1
            elif is_vision_module(name):
                lr = _vision_lr
                descriptor = f'type=vision|depth={_vision_counter}|name={name}|'
                _vision_lr *= _vision_llr_decay
                _vision_counter += 1

            params.append(
                {
                    'params': param,
                    'lr': lr,
                    'weight_decay': _weight_decay,
                    '###logging_descriptor': descriptor,
                }
            )

    if args.deepspeed:
        assert dsinit is not None
        scaler = None
        model, optimizer, _, _ = dsinit(
            args=args,
            model=model,
            model_parameters=params,
            dist_init_required=True,
        )
    else:
        if args.optimizer == 'lamb':
            try:
                from deepspeed.ops.lamb import FusedLamb
            except ModuleNotFoundError or ImportError:
                raise ModuleNotFoundError(
                    'DeepSpeed is required in order to use the LAMB optimizer, use '
                    "'pip install deepspeed' to install"
                )
            optimizer = FusedLamb(
                params=params, betas=(args.beta1, args.beta2), eps=args.eps
            )
        else:
            optimizer = optim.AdamW(
                params, betas=(args.beta1, args.beta2), eps=args.eps
            )

        if args.horovod:
            try:
                import horovod.torch as hvd
            except ImportError:
                raise ImportError('Horovod is not installed')

            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if args.precision == 'amp':
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
        else:
            scaler = None

    return model, optimizer, scaler
