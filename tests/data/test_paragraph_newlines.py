"""Dataset-builder tests for paragraph transcripts with physical line breaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data import uithwdb, vnondb
from src.data.paragraph_labels import (
    align_sequential_paragraph_lines,
    join_paragraph_lines,
    split_indexed_line_stem,
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
    assert split_indexed_line_stem("writer_document_3_12") == (
        "writer_document_3",
        12,
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
    monkeypatch.setattr(
        uithwdb,
        "VNON_LINE_DIR",
        tmp_path / "missing_vnondb_lines",
    )
    monkeypatch.setattr(
        uithwdb,
        "VNON_PARAGRAPH_DIR",
        tmp_path / "missing_vnondb_paragraphs",
    )

    uithwdb.build_uithwdb_dataset()

    records = read_manifest(output / "manifest.jsonl")
    paragraph = next(
        record for record in records if record["level"] == "paragraph"
    )
    assert paragraph["text"] == "Dòng một\nDòng hai"


def test_uithwdb_builder_uses_unique_vnondb_fallback(
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

    vnon_lines = tmp_path / "vnon_lines"
    vnon_paragraphs = tmp_path / "vnon_paragraphs"
    vnon_lines.mkdir()
    vnon_paragraphs.mkdir()
    paragraph_labels = (
        ("20240101_0001_doc_0", ("Đoạn đầu", "còn lại")),
        ("20240101_0001_doc_1", ("Đoạn sau", "kết thúc")),
    )
    for paragraph_stem, lines in paragraph_labels:
        (vnon_paragraphs / f"{paragraph_stem}.txt").write_text(
            " ".join(lines),
            encoding="utf-8",
        )
        for index, text in enumerate(lines):
            (vnon_lines / f"{paragraph_stem}_{index}.txt").write_text(
                text,
                encoding="utf-8",
            )

    output = tmp_path / "uithwdb_fallback"
    monkeypatch.setattr(uithwdb, "LEVEL_DIRS", level_dirs)
    monkeypatch.setattr(uithwdb, "OUT", output)
    monkeypatch.setattr(uithwdb, "IMAGES", output / "images")
    monkeypatch.setattr(uithwdb, "MANIFEST", output / "manifest.jsonl")
    monkeypatch.setattr(uithwdb, "VNON_LINE_DIR", vnon_lines)
    monkeypatch.setattr(uithwdb, "VNON_PARAGRAPH_DIR", vnon_paragraphs)

    uithwdb.build_uithwdb_dataset()

    paragraphs = [
        record
        for record in read_manifest(output / "manifest.jsonl")
        if record["level"] == "paragraph"
    ]
    assert [record["text"] for record in paragraphs] == [
        "Đoạn đầu\ncòn lại",
        "Đoạn sau\nkết thúc",
    ]


def test_vnondb_builder_uses_parent_line_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level_dirs = {
        level: tmp_path / f"raw_{level}"
        for level in ("word", "line", "paragraph")
    }
    for root in level_dirs.values():
        root.mkdir(parents=True)

    paragraph_stem = "20240101_0001_document_0"
    save_image(level_dirs["paragraph"] / f"{paragraph_stem}.png")
    (level_dirs["paragraph"] / f"{paragraph_stem}.txt").write_text(
        "Dòng một Dòng hai",
        encoding="utf-8",
    )
    for index, text in ((0, "Dòng một"), (1, "Dòng hai")):
        stem = f"{paragraph_stem}_{index}"
        save_image(level_dirs["line"] / f"{stem}.png")
        (level_dirs["line"] / f"{stem}.txt").write_text(
            text,
            encoding="utf-8",
        )

    output = tmp_path / "vnondb"
    monkeypatch.setattr(vnondb, "LEVEL_DIRS", level_dirs)
    monkeypatch.setattr(vnondb, "OUT", output)
    monkeypatch.setattr(vnondb, "IMAGES", output / "images")
    monkeypatch.setattr(vnondb, "MANIFEST", output / "manifest.jsonl")

    vnondb.build_vnondb_dataset()

    records = read_manifest(output / "manifest.jsonl")
    paragraph = next(
        record for record in records if record["level"] == "paragraph"
    )
    assert paragraph["text"] == "Dòng một\nDòng hai"
