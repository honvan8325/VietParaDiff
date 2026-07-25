"""Regression tests for true scale-separated grapheme contexts."""

from __future__ import annotations

import torch

from src.models import (
    FactorizedGraphemeEncoder,
    GraphemeBatch,
    GraphemeVocabulary,
    ParagraphFormatter,
    TextEncoderConfig,
)


def make_config() -> tuple[GraphemeVocabulary, TextEncoderConfig]:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    return vocabulary, TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
    )


def make_two_sample_batch(
    *,
    shape_ids: tuple[int, int],
    tone_ids: tuple[int, int],
) -> GraphemeBatch:
    return GraphemeBatch(
        base_ids=torch.tensor([[2], [2]]),
        shape_ids=torch.tensor(shape_ids)[:, None],
        tone_ids=torch.tensor(tone_ids)[:, None],
        case_ids=torch.tensor([[3], [3]]),
        class_ids=torch.tensor([[2], [2]]),
        line_ids=torch.zeros(2, 1, dtype=torch.long),
        position_in_line_ids=torch.zeros(2, 1, dtype=torch.long),
        height_bucket_ids=torch.zeros(2, dtype=torch.long),
        attention_mask=torch.ones(2, 1, dtype=torch.bool),
    )


def test_shared_transformer_excludes_shape_and_tone_embeddings() -> None:
    _, config = make_config()
    model = FactorizedGraphemeEncoder(config)

    assert model.input_projection.in_features == 3 * config.component_embedding_dim


def test_tone_change_cannot_leak_into_base_or_shape_context() -> None:
    torch.manual_seed(11)
    _, config = make_config()
    model = FactorizedGraphemeEncoder(config).eval()
    batch = make_two_sample_batch(shape_ids=(2, 2), tone_ids=(2, 3))

    with torch.no_grad():
        condition = model(batch)

    assert torch.equal(condition.base_context[0], condition.base_context[1])
    assert torch.equal(condition.shape_context[0], condition.shape_context[1])
    assert not torch.equal(condition.tone_context[0], condition.tone_context[1])


def test_shape_change_cannot_leak_into_base_or_tone_context() -> None:
    torch.manual_seed(13)
    _, config = make_config()
    model = FactorizedGraphemeEncoder(config).eval()
    batch = make_two_sample_batch(shape_ids=(2, 4), tone_ids=(2, 2))

    with torch.no_grad():
        condition = model(batch)

    assert torch.equal(condition.base_context[0], condition.base_context[1])
    assert not torch.equal(condition.shape_context[0], condition.shape_context[1])
    assert torch.equal(condition.tone_context[0], condition.tone_context[1])


def test_hard_empty_line_keeps_spacing_but_has_no_active_mask() -> None:
    _, config = make_config()
    paragraph = ParagraphFormatter(config).format("a\n\nb")
    active = paragraph.line_slot_mask.sum(dim=(-2, -1)) > 0

    assert paragraph.lines == ("a", "", "b")
    assert active.tolist() == [True, False, True, False, False, False, False, False]

