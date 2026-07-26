"""Known-failure tests for the full manifest auditor."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vietparadiff.data.audit import (
    DatasetAuditor,
    compute_dataset_snapshot,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (4, 4), 255)
    image.putpixel((1, 1), 0)
    image.save(path)
    image.close()


def test_audit_detects_core_invariant_failures(tmp_path: Path) -> None:
    split = tmp_path / "splits"
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    _image(first)
    duplicate.write_bytes(first.read_bytes())
    train_record = {
        "id": "train-line",
        "image": str(first),
        "text": "aaaaaaaaaaaaaaaa",
        "canonical_writer_id": "writer-shared",
        "level": "line",
        "width": 4,
        "height": 4,
    }
    _write_jsonl(
        split / "htr" / "train_lines.jsonl",
        [
            train_record,
            {
                **train_record,
                "id": "duplicate-line",
                "image": str(duplicate),
            },
        ],
    )
    _write_jsonl(
        split / "htr" / "train_words.jsonl",
        [
            {
                **train_record,
                "id": "train-word",
                "image": str(duplicate),
                "level": "word",
            }
        ],
    )
    _write_jsonl(
        split / "htr" / "test_lines.jsonl",
        [
            {
                **train_record,
                "id": "test-line",
                "image": str(tmp_path / "missing.png"),
            }
        ],
    )
    _write_jsonl(
        split / "vietparadiff" / "pretrain_targets.jsonl",
        [
            {
                "id": "bad-target",
                "image": str(first),
                "text": "\n".join(str(index) for index in range(9)),
                "canonical_writer_id": "writer-target",
                "formatter_mode": "physical_lines",
            }
        ],
    )
    _write_jsonl(
        split / "vietparadiff" / "pretrain_references.jsonl",
        [
            {
                "id": "other-reference",
                "image": str(first),
                "text": "reference",
                "canonical_writer_id": "different-writer",
            }
        ],
    )

    report = DatasetAuditor(
        split,
        image_root=tmp_path,
        workers=1,
    ).run()
    codes = set(report["error_counts"])
    assert report["schema_version"] == 3
    assert report["dataset_snapshot_sha256"]
    assert report["image_inventory_sha256"]
    assert report["manifest_sha256"]
    assert {
        "writer_leakage",
        "missing_image",
        "duplicate_cross_writer",
        "formatter_rejection",
        "reference_ineligible",
    }.issubset(codes)
    assert report["hard_error_count"] > 0
