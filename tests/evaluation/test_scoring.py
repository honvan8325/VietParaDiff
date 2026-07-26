from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from vietparadiff.evaluation.scoring import (
    _generated_writer_embedding,
    _load_real_writer_embeddings,
    validate_independent_htr_checkpoints,
)
from vietparadiff.models.writer import WriterStyleEncoder
from vietparadiff.training.writer import WriterImageProcessor


def test_scoring_rejects_guidance_htr_as_evaluator(
    tmp_path: Path,
) -> None:
    guidance = tmp_path / "guidance.pt"
    evaluation = tmp_path / "evaluation.pt"
    guidance.write_bytes(b"same checkpoint")
    evaluation.write_bytes(b"same checkpoint")
    with pytest.raises(ValueError, match="phải độc lập"):
        validate_independent_htr_checkpoints(guidance, evaluation)

    evaluation.write_bytes(b"independent checkpoint")
    guidance_hash, evaluation_hash = (
        validate_independent_htr_checkpoints(guidance, evaluation)
    )
    assert guidance_hash != evaluation_hash


def test_blank_sample_skips_writer_preprocessing(tmp_path: Path) -> None:
    class _MustNotRun:
        def __call__(self, path: Path) -> torch.Tensor:
            raise AssertionError(f"writer preprocessing ran for {path}")

    embedding = _generated_writer_embedding(
        tmp_path / "blank.png",
        {"blank_output": True},
        cast(WriterImageProcessor, _MustNotRun()),
        cast(WriterStyleEncoder, _MustNotRun()),
        torch.device("cpu"),
    )
    assert embedding is None


def test_real_gallery_is_loaded_even_when_generated_output_is_blank(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    class _Processor:
        def __call__(self, path: Path) -> torch.Tensor:
            calls.append(path)
            value = 1.0 if path.name == "reference.png" else 2.0
            return torch.full((1, 2, 2), value)

    class _Writer:
        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            values = images.mean(dim=(1, 2, 3))
            return torch.stack((values, values + 1.0), dim=1)

    results = [
        {
            "reference_id": "reference",
            "reference_image": "reference.png",
            "target_id": "target",
            "target_image": "target.png",
            "canonical_writer_id": "writer",
        }
    ]
    references, targets, writers = _load_real_writer_embeddings(
        results,
        image_root=tmp_path,
        processor=cast(WriterImageProcessor, _Processor()),
        model=cast(WriterStyleEncoder, _Writer()),
        device=torch.device("cpu"),
    )
    assert set(references) == {"reference"}
    assert set(targets) == {"target"}
    assert writers == {"reference": "writer"}
    assert calls == [
        tmp_path / "reference.png",
        tmp_path / "target.png",
    ]
