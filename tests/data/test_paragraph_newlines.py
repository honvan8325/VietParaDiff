"""Dataset-builder tests for paragraph transcripts with physical line breaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vietparadiff.data import uithwdb
from vietparadiff.data.paragraph_labels import (
    align_sequential_paragraph_lines,
    join_paragraph_lines,
)


def save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (64, 32), color=255).save(path)


def read_manifest(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_line_label_helpers_preserve_content_and_order() -> None:
    matches, unmatched = align_sequential_paragraph_lines(
        (("p1.png", "Dòng một Dòng hai"),),
        (("1.png", "Dòng một"), ("2.png", "Dòng hai")),
    )

    assert unmatched == ()
    assert matches == {"p1.png": ("Dòng một", "Dòng hai")}
    assert (
        join_paragraph_lines(
            "Dòng một Dòng hai",
            matches["p1.png"],
        )
        == "Dòng một\nDòng hai"
    )


def test_join_paragraph_lines_rejects_changed_content() -> None:
    with pytest.raises(ValueError, match="không ghép khớp"):
        join_paragraph_lines("Dòng một Dòng hai", ("Dòng một", "sai"))


def test_uithwdb_builder_uses_ordered_line_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level_dirs = {
        level: tmp_path / f"raw_{level}"
        for level in ("word", "line", "paragraph")
    }
    for level, root in level_dirs.items():
        writer = root / "train_data" / "1"
        writer.mkdir(parents=True)
        labels: dict[str, str] = {}
        if level == "line":
            labels = {"1.png": "Dòng một", "2.png": "Dòng hai"}
        elif level == "paragraph":
            labels = {"1.png": "Dòng một Dòng hai"}
        (writer / "label.json").write_text(
            json.dumps(labels, ensure_ascii=False),
            encoding="utf-8",
        )
        for image_name in labels:
            save_image(writer / image_name)

    output = tmp_path / "uithwdb"
    monkeypatch.setattr(uithwdb, "LEVEL_DIRS", level_dirs)
    monkeypatch.setattr(uithwdb, "OUT", output)
    monkeypatch.setattr(uithwdb, "IMAGES", output / "images")
    monkeypatch.setattr(uithwdb, "MANIFEST", output / "manifest.jsonl")
    uithwdb.build_uithwdb_dataset()

    records = read_manifest(output / "manifest.jsonl")
    paragraph = next(
        record for record in records if record["level"] == "paragraph"
    )
    assert paragraph["text"] == "Dòng một\nDòng hai"


def test_uithwdb_builder_rejects_missing_native_line_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level_dirs = {
        level: tmp_path / f"uit_{level}"
        for level in ("word", "line", "paragraph")
    }
    for level, root in level_dirs.items():
        writer = root / "train_data" / "7"
        writer.mkdir(parents=True)
        labels: dict[str, str] = {}
        if level == "line":
            labels = {"1.png": "Đoạn đầu"}
        elif level == "paragraph":
            labels = {
                "1.png": "Đoạn đầu còn lại",
                "2.png": "Đoạn sau kết thúc",
            }
        (writer / "label.json").write_text(
            json.dumps(labels, ensure_ascii=False),
            encoding="utf-8",
        )
        for image_name in labels:
            save_image(writer / image_name)

    output = tmp_path / "uithwdb_fallback"
    monkeypatch.setattr(uithwdb, "LEVEL_DIRS", level_dirs)
    monkeypatch.setattr(uithwdb, "OUT", output)
    monkeypatch.setattr(uithwdb, "IMAGES", output / "images")
    monkeypatch.setattr(uithwdb, "MANIFEST", output / "manifest.jsonl")
    uithwdb.build_uithwdb_dataset()

    paragraphs = [
        record
        for record in read_manifest(output / "manifest.jsonl")
        if record["level"] == "paragraph"
    ]
    assert paragraphs == []
    report = json.loads(
        (output / "build_report.json").read_text(encoding="utf-8")
    )
    assert report["expected_rejection_count"] == 2
    assert {
        item["reason"] for item in report["expected_rejections"]
    } == {"missing_native_line_alignment"}
