"""Writer verifier architecture and training primitive tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from vietparadiff.models import (
    ArcFaceHead,
    WriterEncoderConfig,
    WriterStyleEncoder,
)
from vietparadiff.training.writer import split_writer_ids


def test_arcface_target_margin_matches_formula() -> None:
    head = ArcFaceHead(256, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1.0
        head.weight[1, 1] = 1.0
    embeddings = torch.zeros(2, 256)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    logits = head(embeddings, torch.tensor([0, 1]))
    clamped = torch.tensor(1.0 - 1e-7)
    expected = 30.0 * (
        clamped * torch.cos(torch.tensor(0.5))
        - torch.sqrt(1.0 - clamped.square())
        * torch.sin(torch.tensor(0.5))
    )
    assert float(logits[0, 0].detach()) == pytest.approx(
        float(expected), rel=1e-5
    )
    assert float(logits[1, 1].detach()) == pytest.approx(
        float(expected), rel=1e-5
    )
    assert abs(float(logits[0, 1].detach())) < 1e-4


def test_writer_split_is_deterministic_and_writer_disjoint() -> None:
    records = [
        {
            "id": f"{writer}-{sample}",
            "canonical_writer_id": f"writer-{writer:02d}",
        }
        for writer in range(20)
        for sample in range(2)
    ]
    first = split_writer_ids(
        records,
        seed=42,
        validation_fraction=0.1,
    )
    second = split_writer_ids(
        list(reversed(records)),
        seed=42,
        validation_fraction=0.1,
    )
    assert first == second
    train, validation = first
    assert len(train) == 18
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)


def test_writer_encoder_has_finite_gradient_and_strict_roundtrip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    config = WriterEncoderConfig()
    model = WriterStyleEncoder(config).train()
    images = torch.randn(2, 1, 128, 64).clamp(-1.0, 1.0)
    embeddings = model(images)
    assert embeddings.shape == (2, 256)
    assert torch.allclose(
        embeddings.norm(dim=1),
        torch.ones(2),
        atol=1e-5,
    )
    loss = F.mse_loss(
        embeddings,
        torch.roll(embeddings.detach(), shifts=1, dims=0),
    )
    loss.backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()

    checkpoint = tmp_path / "writer.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = WriterStyleEncoder(config)
    restored.load_checkpoint(checkpoint)
    assert {
        key: tuple(value.shape)
        for key, value in restored.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }
