"""Regression tests for topology-stable model behavior ablations."""

from __future__ import annotations

import torch

from vietparadiff.models.config import (
    ParagraphUNetConfig,
    StyleEncoderConfig,
)
from vietparadiff.models.generator import ConditionedLevel
from vietparadiff.models.grapheme import GraphemeCondition
from vietparadiff.models.style import (
    DualFrequencyStyleEncoder,
    StyleCondition,
)


def _condition() -> GraphemeCondition:
    return GraphemeCondition(
        base_context=torch.randn(1, 3, 768),
        shape_context=torch.randn(1, 3, 768),
        tone_context=torch.randn(1, 3, 768),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        line_ids=torch.zeros(1, 3, dtype=torch.long),
    )


def _style() -> StyleCondition:
    return StyleCondition(
        local_tokens=torch.randn(1, 2, 768),
        global_style=torch.randn(1, 768),
        layout_scales=torch.ones(1, 3),
        valid_feature_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )


def test_condition_flags_preserve_state_dict_topology() -> None:
    enabled = ConditionedLevel(
        32,
        ParagraphUNetConfig(),
        attention_mode="row",
        use_shape=True,
        use_tone=True,
    )
    disabled = ConditionedLevel(
        32,
        ParagraphUNetConfig(
            use_shape_condition=False,
            use_tone_condition=False,
            use_local_style_tokens=False,
            use_harmonizer=False,
        ),
        attention_mode="row",
        use_shape=True,
        use_tone=True,
    )
    assert {
        key: tuple(value.shape)
        for key, value in enabled.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in disabled.state_dict().items()
    }
    disabled.load_state_dict(enabled.state_dict(), strict=True)


def test_disabled_text_and_local_branches_cannot_change_level_output() -> None:
    torch.manual_seed(19)
    level = ConditionedLevel(
        32,
        ParagraphUNetConfig(
            dropout=0.0,
            use_shape_condition=False,
            use_tone_condition=False,
            use_local_style_tokens=False,
        ),
        attention_mode="row",
        use_shape=True,
        use_tone=True,
    ).eval()
    with torch.no_grad():
        assert level.shape_adapter is not None
        assert level.tone_adapter is not None
        level.shape_adapter.output_projection.weight.normal_()
        level.tone_adapter.output_projection.weight.normal_()
    features = torch.randn(1, 32, 2, 4)
    timestep = torch.randn(1, 1024)
    first_condition = _condition()
    first_style = _style()
    second_condition = GraphemeCondition(
        base_context=first_condition.base_context,
        shape_context=first_condition.shape_context + 100.0,
        tone_context=first_condition.tone_context - 100.0,
        attention_mask=first_condition.attention_mask,
        line_ids=first_condition.line_ids,
    )
    second_style = StyleCondition(
        local_tokens=first_style.local_tokens + 100.0,
        global_style=first_style.global_style,
        layout_scales=first_style.layout_scales,
        valid_feature_mask=first_style.valid_feature_mask,
    )

    with torch.no_grad():
        first, first_shape, first_tone, _ = level(
            features,
            timestep,
            first_condition,
            first_style,
        )
        second, second_shape, second_tone, _ = level(
            features,
            timestep,
            second_condition,
            second_style,
        )

    assert torch.equal(first, second)
    assert first_shape == second_shape == []
    assert first_tone == second_tone == []


def test_disabled_high_frequency_branch_cannot_affect_style_output() -> None:
    torch.manual_seed(23)
    encoder = DualFrequencyStyleEncoder(
        StyleEncoderConfig(
            use_pretrained_backbone=False,
            use_high_frequency_style=False,
        )
    ).eval()
    image = torch.randn(1, 1, 256, 32).clamp(-1.0, 1.0)
    mask = torch.ones_like(image, dtype=torch.bool)
    with torch.no_grad():
        first = encoder(image, mask)
        for parameter in encoder.hf_stem.parameters():
            parameter.add_(torch.randn_like(parameter) * 100.0)
        encoder.fusion_gate.weight.normal_(mean=0.0, std=100.0)
        encoder.fusion_gate.bias.normal_(mean=0.0, std=100.0)
        second = encoder(image, mask)

    assert torch.equal(first.local_tokens, second.local_tokens)
    assert torch.equal(first.global_style, second.global_style)
