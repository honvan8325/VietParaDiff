"""Tests for the line-level HTR teacher contract."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.models import HTRConfig, VietnameseHTR


def make_htr() -> VietnameseHTR:
    return VietnameseHTR(
        HTRConfig(
            raw_vocab_size=20,
            base_vocab_size=20,
            shape_vocab_size=8,
            tone_vocab_size=8,
        )
    )


def test_rejects_paragraph_height() -> None:
    with pytest.raises(ValueError, match="ảnh một dòng cao 64px"):
        make_htr()(torch.zeros(1, 1, 128, 64))


def test_depthwise_branch_uses_group_norm_not_batch_norm() -> None:
    model = make_htr()

    assert isinstance(model.blocks[0].conv_group_norm, nn.GroupNorm)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())


def test_padding_is_zero_before_first_conformer_block() -> None:
    model = make_htr().eval()
    captured: list[tuple[Tensor, Tensor]] = []

    def capture(
        _module: nn.Module,
        inputs: tuple[Tensor, Tensor],
    ) -> None:
        captured.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    handle = model.blocks[0].register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(
                torch.randn(2, 1, 64, 64),
                torch.tensor([64, 32], dtype=torch.long),
            )
    finally:
        handle.remove()

    features, padding = captured[0]
    assert torch.count_nonzero(features[padding]) == 0
    assert torch.count_nonzero(features[~padding]) > 0


def test_valid_widths_must_share_image_device() -> None:
    images = torch.zeros(1, 1, 64, 64)
    valid_widths = torch.empty(1, dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="phải cùng device"):
        make_htr()(images, valid_widths)
