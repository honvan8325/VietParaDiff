"""Tests for line-aware harmonization without boxes or hard masks."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    GraphemeCondition,
    ParagraphUNetConfig,
    TextGuidedInterLineHarmonizer,
)


def make_condition(line_ids: list[int]) -> GraphemeCondition:
    length = len(line_ids)
    return GraphemeCondition(
        base_context=torch.randn(1, length, 768),
        shape_context=torch.randn(1, length, 768),
        tone_context=torch.randn(1, length, 768),
        attention_mask=torch.ones(1, length, dtype=torch.bool),
        line_ids=torch.tensor([line_ids], dtype=torch.long),
    )


def test_one_line_paragraph_bypasses_spatial_harmonization() -> None:
    config = ParagraphUNetConfig()
    harmonizer = TextGuidedInterLineHarmonizer(768, config)
    features = torch.randn(1, 768, 2, 4)
    global_style = torch.randn(1, 768)

    updated, tokens, norm = harmonizer(
        features,
        make_condition([0, 0, 0]),
        global_style,
    )

    assert torch.equal(updated, features)
    assert norm.item() == 0.0
    assert tokens.shape == (1, 8, config.harmonizer_dim)
    assert torch.count_nonzero(tokens[:, 1:]) == 0


def test_multi_line_harmonizer_has_text_attention_gradients() -> None:
    config = ParagraphUNetConfig()
    harmonizer = TextGuidedInterLineHarmonizer(768, config)
    features = torch.randn(1, 768, 2, 4, requires_grad=True)
    global_style = torch.randn(1, 768, requires_grad=True)

    updated, tokens, norm = harmonizer(
        features,
        make_condition([0, 0, 1, 1]),
        global_style,
    )
    loss = updated.square().mean() + tokens.square().mean()
    loss.backward()

    assert updated.shape == features.shape
    assert tokens.shape == (1, 8, config.harmonizer_dim)
    assert torch.equal(updated, features)
    assert norm.item() == 0.0
    assert harmonizer.text_projection.weight.grad is not None
    assert torch.isfinite(
        harmonizer.text_projection.weight.grad
    ).all()
    assert harmonizer.line_to_spatial.in_proj_weight.grad is not None
    assert torch.isfinite(
        harmonizer.line_to_spatial.in_proj_weight.grad
    ).all()
    assert harmonizer.output_projection.weight.grad is not None


def test_shape_and_tone_cannot_leak_into_deep_harmonizer() -> None:
    torch.manual_seed(17)
    config = ParagraphUNetConfig(dropout=0.0)
    harmonizer = TextGuidedInterLineHarmonizer(
        768,
        config,
    ).eval()
    base = torch.randn(1, 4, 768)
    shape = torch.randn(1, 4, 768)
    tone = torch.randn(1, 4, 768)
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    line_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    first = GraphemeCondition(
        base_context=base,
        shape_context=shape,
        tone_context=tone,
        attention_mask=attention_mask,
        line_ids=line_ids,
    )
    second = GraphemeCondition(
        base_context=base.clone(),
        shape_context=shape + 10.0,
        tone_context=tone - 10.0,
        attention_mask=attention_mask,
        line_ids=line_ids,
    )
    features = torch.randn(1, 768, 2, 4)
    global_style = torch.randn(1, 768)

    with torch.no_grad():
        _, first_tokens, _ = harmonizer(
            features,
            first,
            global_style,
        )
        _, second_tokens, _ = harmonizer(
            features,
            second,
            global_style,
        )

    assert torch.equal(first_tokens, second_tokens)


def test_harmonizer_rejects_active_line_id_outside_contract() -> None:
    harmonizer = TextGuidedInterLineHarmonizer(
        768,
        ParagraphUNetConfig(),
    )

    with pytest.raises(ValueError, match="Active line IDs"):
        harmonizer(
            torch.randn(1, 768, 2, 4),
            make_condition([0, 8]),
            torch.randn(1, 768),
        )
