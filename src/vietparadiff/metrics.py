"""Dependency-free metrics shared by training and paper evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def normalized_error(
    reference: Sequence[object],
    hypothesis: Sequence[object],
) -> float:
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def binary_auc_eer(
    labels: Sequence[bool],
    scores: Sequence[float],
) -> tuple[float, float]:
    if len(labels) != len(scores) or len(labels) < 2:
        raise ValueError("labels/scores phải cùng length >= 2.")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Verification scores phải hữu hạn.")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC/EER cần cả positive và negative pairs.")

    ordered = sorted(
        enumerate(scores),
        key=lambda item: item[1],
    )
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and ordered[end][1] == ordered[index][1]
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(
            bool(labels[item_index])
            for item_index, _ in ordered[index:end]
        )
        index = end
    auc = (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    thresholds = [math.inf, *sorted(set(scores), reverse=True), -math.inf]
    best_eer = 1.0
    best_gap = math.inf
    for threshold in thresholds:
        false_positive = sum(
            (not label) and score >= threshold
            for label, score in zip(labels, scores, strict=True)
        )
        false_negative = sum(
            label and score < threshold
            for label, score in zip(labels, scores, strict=True)
        )
        fpr = false_positive / negatives
        fnr = false_negative / positives
        gap = abs(fpr - fnr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (fpr + fnr) / 2.0
    return float(auc), float(best_eer)


def cosine_similarity(
    first: Tensor,
    second: Tensor,
) -> Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Cosine inputs phải có cùng shape [N,D].")
    return (F.normalize(first, dim=1) * F.normalize(second, dim=1)).sum(
        dim=1
    )


def mean_pairwise_cosine_distance(embeddings: Tensor) -> float:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Diversity cần embeddings [N>=2,D].")
    normalized = F.normalize(embeddings.float(), dim=1)
    similarity = normalized @ normalized.transpose(0, 1)
    indices = torch.triu_indices(
        embeddings.shape[0],
        embeddings.shape[0],
        offset=1,
        device=embeddings.device,
    )
    return float((1.0 - similarity[indices[0], indices[1]]).mean())


def _polynomial_kernel(first: Tensor, second: Tensor) -> Tensor:
    return (first @ second.transpose(0, 1) / first.shape[1] + 1.0).pow(3)


def _unbiased_mmd(first: Tensor, second: Tensor) -> Tensor:
    count_first = first.shape[0]
    count_second = second.shape[0]
    if count_first < 2 or count_second < 2:
        raise ValueError("MMD subset cần ít nhất hai sample mỗi distribution.")
    kernel_first = _polynomial_kernel(first, first)
    kernel_second = _polynomial_kernel(second, second)
    cross = _polynomial_kernel(first, second)
    same_first = (
        kernel_first.sum() - kernel_first.diagonal().sum()
    ) / (count_first * (count_first - 1))
    same_second = (
        kernel_second.sum() - kernel_second.diagonal().sum()
    ) / (count_second * (count_second - 1))
    return same_first + same_second - 2.0 * cross.mean()


def style_distribution_mmd(
    real_embeddings: Tensor,
    generated_embeddings: Tensor,
    *,
    subset_size: int = 100,
    subsets: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    if (
        real_embeddings.ndim != 2
        or generated_embeddings.ndim != 2
        or real_embeddings.shape[1] != generated_embeddings.shape[1]
    ):
        raise ValueError("MMD embeddings phải có shape [N,D] cùng D.")
    if not torch.isfinite(real_embeddings).all() or not torch.isfinite(
        generated_embeddings
    ).all():
        raise ValueError("MMD embeddings phải hữu hạn.")
    actual_subset = min(
        subset_size,
        real_embeddings.shape[0],
        generated_embeddings.shape[0],
    )
    if actual_subset < 2 or subsets <= 0 or seed < 0:
        raise ValueError("MMD subset/subsets/seed không hợp lệ.")
    real = real_embeddings.detach().float().cpu()
    generated = generated_embeddings.detach().float().cpu()
    generator = torch.Generator().manual_seed(seed)
    values: list[Tensor] = []
    for _ in range(subsets):
        real_indices = torch.randperm(
            real.shape[0],
            generator=generator,
        )[:actual_subset]
        generated_indices = torch.randperm(
            generated.shape[0],
            generator=generator,
        )[:actual_subset]
        values.append(
            _unbiased_mmd(
                real.index_select(0, real_indices),
                generated.index_select(0, generated_indices),
            )
        )
    stacked = torch.stack(values)
    return (
        float(stacked.mean()),
        float(stacked.std(unbiased=subsets > 1)),
    )


__all__ = [
    "binary_auc_eer",
    "cosine_similarity",
    "edit_distance",
    "style_distribution_mmd",
    "mean_pairwise_cosine_distance",
    "normalized_error",
]
