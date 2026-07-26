"""Deterministic external-baseline adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vietparadiff.baselines import (
    ExternalBaselineConfig,
    normalize_paragraph_output,
    stitch_word_images,
)


def _word(path: Path, width: int) -> None:
    image = Image.new("L", (width, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 6, width - 3, 25), fill=0)
    image.save(path)
    image.close()


def test_unicode_word_stitch_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "tiếng.png"
    second = tmp_path / "Việt.png"
    _word(first, 40)
    _word(second, 60)
    one = stitch_word_images(
        [[first, second]],
        line_ranges=[(20, 100)],
        output_height=384,
    )
    two = stitch_word_images(
        [[first, second]],
        line_ranges=[(20, 100)],
        output_height=384,
    )
    assert one.tobytes() == two.tobytes()
    assert one.size == (1024, 384)
    one.close()
    two.close()


def test_paragraph_normalization_preserves_aspect_ratio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paragraph.png"
    image = Image.new("L", (200, 100), 255)
    ImageDraw.Draw(image).rectangle((10, 10, 189, 89), fill=0)
    image.save(source)
    image.close()
    normalized = normalize_paragraph_output(
        source,
        output_height=384,
    )
    assert normalized.size == (1024, 384)
    assert normalized.getbbox() == (0, 0, 1024, 384)
    normalized.close()


def test_baseline_config_rejects_unpinned_commit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pin commit"):
        ExternalBaselineConfig(
            name="one_dm",
            checkout=tmp_path,
            expected_commit="0" * 40,
            checkpoint=tmp_path / "checkpoint.pt",
            checkpoint_sha256="0" * 64,
            command=(
                "python",
                "worker.py",
                "--requests={requests}",
                "--output={output_dir}",
            ),
            test_pairs=tmp_path / "pairs.jsonl",
            image_root=tmp_path,
            generator_model_config=tmp_path / "model.json",
            output_dir=tmp_path / "output",
            base_seed=42,
            samples_per_pair=3,
        )
