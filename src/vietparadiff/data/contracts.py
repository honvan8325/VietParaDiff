"""Shared scientific contracts for split construction and runtime sampling."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

__all__ = [
    "eligible_reference",
    "excluded_source_line_ids",
    "normalize_content",
]


def normalize_content(text: object) -> str:
    """Return the canonical text form used by every pairing contract."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Transcript phải là string không rỗng.")
    return " ".join(unicodedata.normalize("NFC", text).split())


def excluded_source_line_ids(
    target: Mapping[str, object],
) -> frozenset[str]:
    """Read exact stitch sources without accepting a legacy top-level field."""
    augmentation = target.get("augmentation")
    if augmentation is None:
        return frozenset()
    if not isinstance(augmentation, Mapping):
        raise ValueError(
            f"Target {target.get('id', '<unknown>')} có augmentation lỗi."
        )
    raw = augmentation.get("source_line_ids")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or not raw
        or not all(isinstance(value, str) and value for value in raw)
    ):
        raise ValueError(
            f"Target {target.get('id', '<unknown>')} có "
            "augmentation.source_line_ids lỗi."
        )
    return frozenset(raw)


def eligible_reference(
    target: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    excluded_reference_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    """Apply the exact same reference policy in build, runtime, and audit."""
    excluded = (
        excluded_source_line_ids(target)
        if excluded_reference_ids is None
        else frozenset(excluded_reference_ids)
    )
    target_id = str(target.get("id", target.get("target_id", "")))
    reference_id = str(
        reference.get("id", reference.get("reference_id", ""))
    )
    if not target_id or not reference_id or target_id == reference_id:
        return False
    if reference_id in excluded:
        return False
    if reference.get("level") != "line":
        return False
    if bool(reference.get("synthetic", False)):
        return False
    target_writer = target.get("canonical_writer_id")
    reference_writer = reference.get("canonical_writer_id")
    if (
        not isinstance(target_writer, str)
        or not target_writer
        or target_writer != reference_writer
    ):
        return False
    target_text = target.get("text", target.get("target_text"))
    reference_text = reference.get("text", reference.get("reference_text"))
    try:
        return normalize_content(target_text) != normalize_content(
            reference_text
        )
    except ValueError:
        return False
