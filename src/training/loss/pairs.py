from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as f

try:
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    dist = None
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from training.data import InputType


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
        bidirectional: bool = True,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ) -> None:
        super().__init__()

        self._local_loss = local_loss
        self._gather_with_grad = gather_with_grad
        self._rank = rank
        self._world_size = world_size
        self._use_horovod = use_horovod

        self._cache_labels = cache_labels
        self._previous_batchsize = 0
        self._labels = {}

        self._logit_scale = torch.tensor([1 / temperature])
        self._bidirectional = bidirectional

        assert has_distributed, (
            'Package torch.distributed did not import correctly, please use a PyTorch '
            'version with distributed support.'
        )
        if use_horovod:
            assert hvd is not None, 'Please install horovod'

    @property
    def input_type(self) -> InputType:
        return InputType.PAIR

    def get_distributed_features(self, features: torch.Tensor):
        if self._use_horovod:
            if self._gather_with_grad:
                _all_features = hvd.allgather(features)
            else:
                with torch.no_grad():
                    _all_features = hvd.allgather(features)
                if not self._local_loss:
                    # ensure grads for local rank when all_* features don't have
                    # a gradient
                    _gathered_features = list(
                        _all_features.chunk(self._world_size, dim=0)
                    )
                    _gathered_features[self._rank] = features
                    _all_features = torch.cat(_gathered_features, dim=0)
        else:
            # We gather tensors from all gpus
            if self._gather_with_grad:
                _all_features = torch.cat(
                    torch.distributed.nn.all_gather(features), dim=0
                )
            else:
                _gathered_features = [
                    torch.zeros_like(features) for _ in range(self._world_size)
                ]
                dist.all_gather(_gathered_features, features)

                if not self._local_loss:
                    # ensure grads for local rank when all_* features don't have
                    # a gradient
                    _gathered_features[self._rank] = features

                _all_features = torch.cat(_gathered_features, dim=0)

        return _all_features

    def get_labels(self, device: str, batchsize: int) -> torch.Tensor:
        if self._previous_batchsize != batchsize or device not in self._labels:
            labels = torch.arange(batchsize, device=device, dtype=torch.long)
            if self._world_size > 1 and self._local_loss:
                labels = labels + batchsize * self._rank
            if self._cache_labels:
                self._labels[device] = labels
                self._previous_batchsize = batchsize
            return labels

        return self._labels[device]

    def get_distributed_logits(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _right_over_left = None
        if self._world_size > 1:
            _all_left_features = self.get_distributed_features(left_features)
            _all_right_features = self.get_distributed_features(right_features)
            if self._local_loss:
                _left_over_right = left_features @ _all_right_features.T
                if self._bidirectional:
                    _right_over_left = right_features @ _all_left_features.T
            else:
                _left_over_right = _all_left_features @ _all_right_features.T
                if self._bidirectional:
                    _right_over_left = _left_over_right.T
        else:
            _left_over_right = left_features @ right_features.T
            if self._bidirectional:
                _right_over_left = right_features @ left_features.T

        _left_over_right *= logit_scale
        _right_over_left *= logit_scale

        return _left_over_right, _right_over_left

    def infonce(
        self,
        left_logits: torch.Tensor,
        right_logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss = f.cross_entropy(left_logits, labels)
        if self._bidirectional:
            loss += f.cross_entropy(right_logits, labels)
            loss = loss / 2
        return loss

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale
        left_logits, right_logits = self.get_distributed_logits(
            left_features,
            right_features,
            logit_scale,
        )
        labels = self.get_labels(left_features.device, left_features.shape[0])
        loss = self.infonce(left_logits, right_logits, labels)

        return loss if output_dict else {'contrastive_loss': loss}


class CoCaLoss(nn.Module):
    def __init__(
        self,
        caption_loss_weight: float = 1.0,
        clip_loss_weight: float = 1.0,
        pad_id: int = 0,
        **infonce_kwargs,
    ):
        super().__init__()
        self._caption_loss_weight = caption_loss_weight
        self._infonce_loss_weight = clip_loss_weight
        self._infonce_loss = InfoNCELoss(**infonce_kwargs)
        self._caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        _infonce_loss = torch.tensor(0)

        if self._clip_loss_weight:
            _infonce_loss = self._infonce_loss(
                left_features,
                right_features,
                logit_scale,
                output_dict=False,
            )
            _infonce_loss = self._infonce_loss_weight * _infonce_loss

        _caption_loss = self._caption_loss(logits.permute(0, 2, 1), labels)
        _caption_loss = _caption_loss * self._caption_loss_weight

        if output_dict:
            return {'contrastive_loss': _infonce_loss, 'caption_loss': _caption_loss}

        return _infonce_loss, _caption_loss


class DistillInfoNCELoss(nn.Module):
    def __init__(self, **infonce_kwargs):
        super().__init__()
        self._infonce_loss = InfoNCELoss(**infonce_kwargs)

    @staticmethod
    def distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor):
        return (
            -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1))
            .sum(dim=1)
            .mean(dim=0)
        )

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        distill_left_features: torch.Tensor,
        distill_right_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        distill_logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        left_logits, right_logits = self._infonce_loss.get_distributed_logits(
            left_features, right_features, logit_scale
        )
        (
            distill_left_logits,
            distill_right_logits,
        ) = self._infonce_loss.get_distributed_logits(
            distill_left_features, distill_right_features, distill_logit_scale
        )
        labels = self._infonce_loss.get_labels(
            left_features.device, left_features.shape[0]
        )
        infonce_loss = self._infonce_loss.infonce(left_logits, right_logits, labels)
        distillation_loss = (
            self.distillation_loss(distill_left_logits, left_logits)
            + self.distillation_loss(distill_right_logits, right_logits)
        ) / 2
        if output_dict:
            return {'contrastive_loss': infonce_loss, 'distill_loss': distillation_loss}
        return infonce_loss, distillation_loss


class ThreeTowersLoss(nn.Module):
    def __init__(self, **infonce_kwargs):
        super().__init__()
        self._infonce_loss = InfoNCELoss(**infonce_kwargs)

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        teacher_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        features = {
            'left': left_features,
            'right': right_features,
            'teacher': teacher_features,
        }
        loss = 0.0
        for k1, v1 in features.items():
            for k2, v2 in features.items():
                if k1 == k2:
                    continue
                loss += self._infonce_loss(v1, v2, logit_scale)
        loss /= 6
        if output_dict:
            return {'contrastive_loss': loss}
        return loss


class MatryoshkaInfoNCELoss(InfoNCELoss):
    def __init__(
        self,
        temperature: float = 0.05,
        bidirectional: bool = True,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
        dims: Sequence[int] = (16, 32, 64, 128, 256, 512),
        weights: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            temperature=temperature,
            bidirectional=bidirectional,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        if weights:
            assert len(weights) == len(dims)
        self._dims = dims
        self._weights = weights if weights else [1] * len(self._dims)

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        loss = 0.0
        for dim, weight in zip(self._dims, self._weights):
            _left_composites = left_features[..., :dim]
            _right_composites = right_features[..., :dim]
            _composite_loss = super().forward(
                left_features, right_features, logit_scale, output_dict=False
            )
            loss += _composite_loss * weight

        return {'contrastive_loss': loss} if output_dict else loss
