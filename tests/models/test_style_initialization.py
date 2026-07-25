"""Initialization and two-stage style/layout contract tests."""

from __future__ import annotations

from dataclasses import fields

import torch

from src.models import (
    DualFrequencyStyleEncoder,
    StyleCondition,
    StyleEncoderConfig,
    VietParaDiffInput,
)


def test_fusion_gate_starts_near_raw_only() -> None:
    model = DualFrequencyStyleEncoder(
        StyleEncoderConfig(use_pretrained_backbone=False)
    )

    assert torch.count_nonzero(model.fusion_gate.weight) == 0
    assert torch.equal(
        model.fusion_gate.bias,
        torch.full_like(model.fusion_gate.bias, -4.0),
    )
    assert torch.allclose(
        torch.sigmoid(model.fusion_gate.bias),
        torch.full_like(model.fusion_gate.bias, 0.01798621),
    )


def test_layout_head_starts_at_neutral_scales() -> None:
    model = DualFrequencyStyleEncoder(
        StyleEncoderConfig(use_pretrained_backbone=False)
    ).eval()
    reference = torch.zeros(1, 1, 256, 32)

    with torch.no_grad():
        style = model(reference, torch.ones_like(reference, dtype=torch.bool))

    assert torch.equal(style.layout_scales, torch.ones(1, 3))
    assert torch.count_nonzero(model.layout_head[-1].weight) == 0
    assert torch.count_nonzero(model.layout_head[-1].bias) == 0


def test_top_level_input_requires_precomputed_style_condition() -> None:
    names = {field.name for field in fields(VietParaDiffInput)}

    assert "style_condition" in names
    assert "reference_images" not in names
    assert "reference_valid_mask" not in names
    assert VietParaDiffInput.__annotations__["style_condition"] == "StyleCondition"
    assert StyleCondition is not None

