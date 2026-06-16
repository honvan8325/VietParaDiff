"""Dataset manifest schema and JSONL I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class LineAnnotation:
    """One line in a paragraph image."""

    text: str
    box: list[float]


@dataclass
class TokenAnnotation:
    """One grapheme/token annotation."""

    surface: str
    box: list[float]
    line_id: int


@dataclass
class ManifestRow:
    """One paragraph sample in the JSONL manifest."""

    id: str
    writer_id: str
    document_id: str
    image: str
    transcript: str
    lines: list[LineAnnotation] = field(default_factory=list)
    tokens: list[TokenAnnotation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ManifestRow":
        return ManifestRow(
            id=str(data["id"]),
            writer_id=str(data["writer_id"]),
            document_id=str(data.get("document_id", data["writer_id"])),
            image=str(data["image"]),
            transcript=str(data["transcript"]),
            lines=[LineAnnotation(str(x["text"]), list(map(float, x["box"]))) for x in data.get("lines", [])],
            tokens=[TokenAnnotation(str(x["surface"]), list(map(float, x["box"])), int(x["line_id"])) for x in data.get("tokens", [])],
            meta=dict(data.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "writer_id": self.writer_id,
            "document_id": self.document_id,
            "image": self.image,
            "transcript": self.transcript,
            "lines": [{"text": line.text, "box": line.box} for line in self.lines],
            "tokens": [{"surface": tok.surface, "box": tok.box, "line_id": tok.line_id} for tok in self.tokens],
            "meta": self.meta,
        }


def read_jsonl(path: str | Path) -> list[ManifestRow]:
    """Read a manifest JSONL file."""

    rows: list[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(ManifestRow.from_dict(json.loads(line)))
    return rows


def write_jsonl(path: str | Path, rows: list[ManifestRow]) -> None:
    """Write rows as UTF-8 JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
