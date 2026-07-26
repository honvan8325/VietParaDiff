"""Utilities for reconstructing paragraph transcripts from line annotations."""

from __future__ import annotations

from collections.abc import Sequence


def flatten_whitespace(text: str) -> str:
    """Collapse all whitespace so flat and newline transcripts can be compared."""
    return " ".join(text.split())


def join_paragraph_lines(flat_text: str, lines: Sequence[str]) -> str:
    """Join ordered non-empty line labels and verify they preserve the text."""
    cleaned = tuple(line.strip() for line in lines)
    if not cleaned or any(not line for line in cleaned):
        raise ValueError("Paragraph phải có ít nhất một line label không rỗng.")
    paragraph_text = "\n".join(cleaned)
    if flatten_whitespace(paragraph_text) != flatten_whitespace(flat_text):
        raise ValueError("Line labels không ghép khớp paragraph transcript.")
    return paragraph_text


def align_sequential_paragraph_lines(
    paragraphs: Sequence[tuple[str, str]],
    lines: Sequence[tuple[str, str]],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Match ordered UIT-HWDB lines to ordered paragraphs by exact text content."""
    matches: dict[str, tuple[str, ...]] = {}
    unmatched: list[str] = []
    cursor = 0

    for paragraph_id, paragraph_text in paragraphs:
        target = flatten_whitespace(paragraph_text)
        candidate_lines: list[str] = []
        candidate_cursor = cursor

        while candidate_cursor < len(lines):
            candidate_lines.append(lines[candidate_cursor][1].strip())
            candidate_cursor += 1
            candidate = flatten_whitespace(" ".join(candidate_lines))

            if candidate == target:
                matches[paragraph_id] = tuple(candidate_lines)
                cursor = candidate_cursor
                break
            if len(candidate) >= len(target):
                unmatched.append(paragraph_id)
                break
        else:
            unmatched.append(paragraph_id)

    return matches, tuple(unmatched)


def split_indexed_line_stem(line_stem: str) -> tuple[str, int]:
    """Return the paragraph stem and numeric line index from a VNOnDB line stem."""
    paragraph_stem, separator, index_text = line_stem.rpartition("_")
    if not separator or not paragraph_stem or not index_text.isdigit():
        raise ValueError(
            "VNOnDB line stem phải kết thúc bằng chỉ số dòng số nguyên, "
            f"nhận {line_stem!r}."
        )
    return paragraph_stem, int(index_text)
