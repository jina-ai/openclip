from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as f


def _neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def _neighbour_exchange_bidirectional(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return _neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


class NeighbourExchangeBidirectional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return _neighbour_exchange_bidirectional(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidirectional.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidirectional.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) -
    https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={
          Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas
      },
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        temperature: float = 0.05,
        logit_bias: Optional[torch.Tensor] = None,
        bidirectional: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        self._rank = rank
        self._world_size = world_size
        self._logit_scale = torch.exp(torch.log(torch.tensor([1 / temperature])))
        self._logit_bias = logit_bias
        self._bidirectional = bidirectional

    @staticmethod
    def sigmoid_loss(
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        negative_only: bool = False,
    ):
        device = left_features.device
        dtype = left_features.dtype
        batchsize = left_features.shape[0]

        logits = logit_scale * left_features @ right_features.T
        if logit_bias is not None:
            logits += logit_bias

        labels = -torch.ones((batchsize, batchsize), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(batchsize, device=device, dtype=dtype) + labels

        return -f.logsigmoid(labels * logits).sum() / left_features.shape[0]

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale
        logit_bias = logit_bias or self._logit_bias

        loss = self.sigmoid_loss(left_features, right_features, logit_scale, logit_bias)

        if self._world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self._rank + 1) % self._world_size
            left_rank = (self._rank - 1 + self._world_size) % self._world_size
            if self._bidirectional:
                right_features_to_right = right_features_to_left = right_features
                num_bidir, remainder = divmod(self._world_size - 1, 2)
                for i in range(num_bidir):
                    _text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        right_features_to_left,
                        right_features_to_right,
                    )
                    for _f in _text_features_recv:
                        loss += self.sigmoid_loss(
                            left_features,
                            _f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    (
                        right_features_to_left,
                        right_features_to_right,
                    ) = _text_features_recv
                if remainder:
                    _text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, right_features_to_right
                    )
                    loss += self.sigmoid_loss(
                        left_features,
                        _text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                right_features_to_right = right_features
                for i in range(self._world_size - 1):
                    right_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, right_features_to_right
                    )
                    loss += self.sigmoid_loss(
                        left_features,
                        right_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    right_features_to_right = right_features_from_left

        return {'contrastive_loss': loss} if output_dict else loss


class MatryoshkaSigLIPLoss(SigLIPLoss):
    def __init__(
        self,
        temperature: float = 0.05,
        logit_bias: Optional[torch.Tensor] = None,
        bidirectional: bool = True,
        rank: int = 0,
        world_size: int = 1,
        dims: Sequence[int] = (16, 32, 64, 128, 256, 512),
        weights: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            temperature=temperature,
            bidirectional=bidirectional,
            logit_bias=logit_bias,
            rank=rank,
            world_size=world_size,
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
        logit_bias: Optional[torch.Tensor] = None,
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
