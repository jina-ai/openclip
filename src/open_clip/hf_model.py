import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

"""
HF architecture mapping
"""

_HF_ARCH_DICT = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    'roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    'xlm-roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    'mt5': {
        'config_names': {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            'context_length': '',
            'vocab_size': 'vocab_size',
            'width': 'd_model',
            'heads': 'num_heads',
            'layers': 'num_layers',
            'layer_attr': 'block',
            'token_embeddings_attr': 'embed_tokens',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    'bert': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
        },
        'pooler': 'cls_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/m2m_100
    'm2m_100': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'd_model',
            'heads': 'encoder_attention_heads',
            'layers': 'encoder_layers',
        },
        'pooler': 'cls_pooler',
    },
}


"""
Pooling functions
"""

_POOLERS = {}


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """
    Max pooling
    """

    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """
    CLS token pooling
    """

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, _: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


"""
HF text model
"""


class HFTextEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj_type: str = None,
        proj_bias: bool = False,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        model_config_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = pooler_type == 'cls_pooler'
        model_config_kwargs = model_config_kwargs or {}

        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                code_revision=revision,
            )
            self.config.update(model_config_kwargs)
            create_func, model_args = (
                (AutoModel.from_pretrained, model_name_or_path)
                if pretrained
                else (AutoModel.from_config, self.config)
            )
            # TODO: do all model configs have this attribute?
            #  PretrainedConfig does so yes??
            if (
                hasattr(self.config, 'is_encoder_decoder')
                and self.config.is_encoder_decoder
            ):
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(
                    model_args,
                    trust_remote_code=trust_remote_code,
                    add_pooling_layer=uses_transformer_pooler,
                    code_revision=revision,
                )
        else:
            self.config = config
            self.config.update(model_config_kwargs)
            self.transformer = AutoModel.from_config(self.config)

        if pooler_type is None:  # get default arch pooler
            pooler_type = _HF_ARCH_DICT[self.config.model_type]['pooler']

        # FIXME downstream users of OpenCLIP models use these attr,
        #  need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(
            self.config, _HF_ARCH_DICT[self.config.model_type]['config_names']['width']
        )
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )

    def forward(self, x: torch.Tensor):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seq_len) != self.pooler.cls_token_position, :
            ]
            if isinstance(self.pooler, ClsPooler)
            else out.last_hidden_state
        )

        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )
            return

        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, _=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass


"""
HF vision model
"""


class HFVisionEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        image_size: int,
        output_dim: int,
        config: PretrainedConfig = None,
        pool_type: str = 'tok',
        proj_type: Optional[str] = None,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        hidden_drop: float = 0.0,
        drop_path: Optional[float] = None,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.image_size = (image_size, image_size)

        if config is None:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                hidden_dropout_prob=hidden_drop,
                attention_probs_dropout_prob=attn_drop,
                drop_path_rate=drop_path,
            )
            create_func, model_args = (
                (AutoModel.from_pretrained, model_name_or_path)
                if pretrained
                else (AutoModel.from_config, self.config)
            )
            self.transformer = create_func(
                model_args,
                trust_remote_code=trust_remote_code,
                hidden_dropout_prob=hidden_drop,
                attention_probs_dropout_prob=attn_drop,
            )
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)

        if 'dinov2' in model_name_or_path:
            self.transformer.embeddings.mask_token.requires_grad = False

        assert pool_type in ('tok', 'avg', 'none')
        self.pool_type = pool_type

        d_model = self.config.hidden_size
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        # returns a tuple of (final hidden states, token pooled outputs)
        x = self.transformer(x)[0]
        pooled, tokens = self._global_pool(x)
        projected = self.proj(pooled)

        return projected

    def lock(self, unlocked_layers: int = 0, freeze_bn_stats: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_bn_stats) if 'LayerNorm' in n.split('.') else False
                )
            return

        # TODO: make it work if unlocked_layers !=0
        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_bn_stats) if 'LayerNorm' in n.split('.') else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, *_, **__):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
