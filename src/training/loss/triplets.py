import itertools
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from training.data import InputType


class CoSentSTSLoss(nn.Module):
    """
    CoSent Loss for STS data.
    Adapted from:
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CoSENTLoss.py
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self._logit_scale = torch.tensor([1 / temperature])

    @property
    def input_type(self):
        return InputType.PAIR_WITH_SCORES

    def forward(
        self,
        embeddings_u: torch.Tensor,
        embeddings_v: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale
        scores = (embeddings_u * embeddings_v).sum(dim=-1)
        scores = scores * logit_scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)
        return {'sts_loss': loss} if output_dict else loss


class CoSentClusteringLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        """
        Computes a loss that tries to maximize the similarity of text values with the
        same label and to minimize the similarity values with different labels.
        """
        super().__init__()
        self._logit_scale = torch.tensor([1 / temperature])

    @property
    def input_type(self):
        return InputType.TEXT_WITH_LABEL

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale

        # group embeddings by labels
        distinct_labels = {}
        idx = 0
        for label in labels:
            if label.item() not in distinct_labels:
                distinct_labels[label.item()] = idx
                idx += 1
        all_vectors = [[] for _ in distinct_labels]
        for label, embedding in zip(labels, embeddings):
            all_vectors[distinct_labels[label.item()]].append(embedding)
        all_vectors = [torch.stack(group) for group in all_vectors]
        sum_value = torch.tensor(1.0, requires_grad=True)
        if (
            len(all_vectors) > 1
        ):  # without negatives you can not calculate the cosent loss
            for i in range(len(all_vectors)):
                # select vectors of one row
                group_vecs = all_vectors[i]
                if (
                    len(group_vecs) < 2
                ):  # if there is only one element there are not positive pairs
                    continue
                group_vecs = torch.nn.functional.normalize(group_vecs, p=2, dim=1)
                # select remaining vectors
                other_vecs = torch.cat(all_vectors[:i] + all_vectors[i + 1 :])
                other_vecs = torch.nn.functional.normalize(other_vecs, p=2, dim=1)
                pos_sim_values = (
                    group_vecs @ group_vecs.T
                )  # contains unwanted 1s in diagonal
                neg_sim_values = group_vecs @ other_vecs.T
                sum_exp_neg_sim_values = torch.sum(
                    torch.exp(neg_sim_values * logit_scale)
                )
                exp_pos_sim_values = torch.exp(-pos_sim_values * logit_scale)
                exp_pos_sim_values = exp_pos_sim_values * (
                    1 - torch.eye(exp_pos_sim_values.shape[0]).to(embeddings.device)
                )  # remove unwanted 1s
                sum_value = sum_value + torch.sum(
                    exp_pos_sim_values * sum_exp_neg_sim_values
                )
        loss = torch.log(sum_value)
        return {'clustering_loss': loss} if output_dict else loss


class InfoNCEHardNegativeLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, bidirectional: bool = False):
        super().__init__()
        self._logit_scale = torch.tensor([1 / temperature])
        self._bidirectional = bidirectional

    @property
    def input_type(self):
        return InputType.TRIPLET

    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        logit_scale = logit_scale or self._logit_scale
        batchsize = anchor_features.shape[0]
        targets = torch.cat([positive_features, negative_features])
        labels = torch.arange(start=0, end=batchsize, device=anchor_features.device)

        scores = logit_scale * anchor_features @ targets
        loss = f.cross_entropy(scores, labels)
        if self._bidirectional:
            scores = logit_scale * positive_features @ anchor_features
            loss += f.cross_entropy(scores, labels)
            loss = loss / 2

        return {'contrastive_loss': loss} if output_dict else loss


class MultiCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 1.0,
        temperature: float = 0.05,
        bidirectional: bool = False,
        single_info_nce: bool = False,
    ):
        super().__init__()

        if bidirectional and not single_info_nce:
            raise ValueError(
                'Bidirectional loss should only be used with single InfoNCE loss.'
            )

        self._kl_loss = nn.KLDivLoss(reduction='batchmean')
        self._infonce_loss = InfoNCEHardNegativeLoss(
            temperature=temperature, bidirectional=bidirectional
        )
        self._alpha = alpha
        self._beta = beta
        self._single_info_nce = single_info_nce

    @property
    def input_type(self):
        if self._beta != 0:
            return InputType.MULTIPLE_NEGATIVES
        else:
            return InputType.MULTIPLE_NEGATIVES_WITHOUT_SCORES

    @staticmethod
    def _iterate_batch(inputs, mask):
        """Iterate through a batch row by row."""
        lower_limit = 0
        for upper_limit in itertools.accumulate(mask, lambda x, y: x + y):
            yield inputs[lower_limit:upper_limit]
            lower_limit = upper_limit

    def forward(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        row_sizes: np.ndarray,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        scores_masks = row_sizes - 1
        loss = None
        batch_count = 0
        anchor_features = []
        positive_features = []
        negative_features = []

        for embrow, scorerow in zip(
            self._iterate_batch(features, row_sizes),
            self._iterate_batch(scores, scores_masks),
        ):
            _pos_score = torch.functional.F.cosine_similarity(
                embrow[0].unsqueeze(0), embrow[1].unsqueeze(0)
            )
            _neg_scores = torch.functional.F.cosine_similarity(
                embrow[0].unsqueeze(0), embrow[2:]
            )
            _emb_scores = torch.nn.functional.log_softmax(
                torch.cat([_pos_score, _neg_scores]), dim=0
            )
            if self._beta > 0:
                ce_scores = torch.nn.functional.softmax(torch.stack(scorerow))
                _kl_loss = self._kl_loss(_emb_scores, ce_scores)
            else:
                _kl_loss = 0.0

            if loss is None:
                loss = self._beta * _kl_loss
            else:
                loss += self._beta * _kl_loss

            if not self._single_info_nce:
                loss += self._alpha * self._info_nce_loss(
                    embrow[0].unsqueeze(0),
                    embrow[1].unsqueeze(0),
                    embrow[2:],
                    logit_scale=logit_scale,
                )
            else:
                anchor_features.append(embrow[0].unsqueeze(0))
                positive_features.append(embrow[1].unsqueeze(0))
                negative_features.append(embrow[2:])
            batch_count += 1

        loss /= batch_count
        if self._single_info_nce:
            loss += self._alpha * self._info_nce_loss(
                torch.cat(anchor_features),
                torch.cat(positive_features),
                torch.cat(negative_features),
                logit_scale=logit_scale,
            )

        return {'contrastive_loss': loss} if output_dict else loss


class MatryoshkaMultiCELoss(MultiCELoss):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 1.0,
        temperature: float = 0.05,
        bidirectional: bool = False,
        single_info_nce: bool = False,
        dims: Sequence[int] = (16, 32, 64, 128, 256, 512),
        weights: Optional[Sequence[int]] = None,
    ):
        super().__init__(
            alpha,
            beta,
            temperature,
            bidirectional,
            single_info_nce,
        )
        if weights:
            assert len(weights) == len(dims)
        self._dims = dims
        self._weights = weights if weights else [1] * len(self._dims)

    def forward(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        row_sizes: np.ndarray,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        scores_masks = row_sizes - 1
        weighted_loss = 0.0
        for dim, weight in zip(self._dims, self._weights):
            loss = None
            batch_count = 0
            anchor_features = []
            positive_features = []
            negative_features = []
            for embrow, scorerow in zip(
                self._iterate_batch(features, row_sizes),
                self._iterate_batch(scores, scores_masks),
            ):
                anc = embrow[0].unsqueeze(0)[:, :dim]
                pos = embrow[1].unsqueeze(0)[:, :dim]
                neg = embrow[2:, :dim]

                pos_score = torch.functional.F.cosine_similarity(anc, pos)
                neg_scores = torch.functional.F.cosine_similarity(anc, neg)
                emb_scores = torch.nn.functional.log_softmax(
                    torch.cat([pos_score, neg_scores]), dim=0
                )
                if self._beta > 0:
                    ce_scores = torch.nn.functional.softmax(torch.stack(scorerow))
                    kl_loss = self._kl_loss(emb_scores, ce_scores)
                else:
                    kl_loss = 0.0

                if loss is None:
                    loss = self._beta * kl_loss
                else:
                    loss += self._beta * kl_loss

                if not self._single_info_nce:
                    loss += self._alpha * self._info_nce_loss(
                        anc,
                        pos,
                        neg,
                        logit_scale=logit_scale,
                    )
                else:
                    anchor_features.append(anc)
                    positive_features.append(pos)
                    negative_features.append(neg)
                batch_count += 1

            loss /= batch_count

            if self._single_info_nce:
                loss += self._alpha * self._info_nce_loss(
                    torch.cat(anchor_features),
                    torch.cat(positive_features),
                    torch.cat(negative_features),
                    logit_scale=logit_scale,
                )
            weighted_loss += loss * weight

        return weighted_loss
