"""Known-answer tests for paper metrics."""

from __future__ import annotations

import pytest
import torch

from vietparadiff.metrics import (
    binary_auc_eer,
    edit_distance,
    mean_pairwise_cosine_distance,
    normalized_error,
    style_distribution_mmd,
)


def test_edit_metrics_known_answers() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert normalized_error("ab", "acb") == pytest.approx(0.5)


def test_auc_eer_perfect_verification() -> None:
    auc, eer = binary_auc_eer(
        [True, True, False, False],
        [0.9, 0.8, 0.2, 0.1],
    )
    assert auc == pytest.approx(1.0)
    assert eer == pytest.approx(0.0)


def test_diversity_and_style_mmd_known_answers() -> None:
    orthogonal = torch.eye(2)
    assert mean_pairwise_cosine_distance(orthogonal) == pytest.approx(1.0)
    mean, std = style_distribution_mmd(
        orthogonal,
        orthogonal,
        subset_size=2,
        subsets=1,
        seed=7,
    )
    assert mean == pytest.approx(-2.375)
    assert std == pytest.approx(0.0)


def test_metric_invalid_contracts_raise() -> None:
    with pytest.raises(ValueError):
        binary_auc_eer([True, True], [0.1, 0.2])
    with pytest.raises(ValueError):
        style_distribution_mmd(
            torch.ones(1, 2),
            torch.ones(1, 2),
        )
