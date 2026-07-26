"""Reviewed UIT-HWDB/VNOnDB writer crosswalk artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vietparadiff.artifacts import sha256_file

__all__ = [
    "ApprovedWriterPair",
    "WriterCoverage",
    "load_writer_crosswalk",
]


@dataclass(frozen=True, slots=True)
class ApprovedWriterPair:
    uithwdb_writer_id: str
    vnondb_writer_id: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class WriterCoverage:
    approved: tuple[ApprovedWriterPair, ...]
    proven_independent: frozenset[str]
    unresolved: frozenset[str]
    excluded: frozenset[str]
    artifact_sha256: str
    candidate_report_sha256: str

    @property
    def approved_writer_ids(self) -> frozenset[str]:
        return frozenset(
            writer
            for pair in self.approved
            for writer in (
                pair.uithwdb_writer_id,
                pair.vnondb_writer_id,
            )
        )


def _strings(value: object, field: str) -> frozenset[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"crosswalk.{field} phải là list string.")
    values = tuple(value)
    if len(values) != len(set(values)):
        raise ValueError(f"crosswalk.{field} chứa ID trùng.")
    return frozenset(values)


def _proven_independent(value: object) -> frozenset[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise ValueError("crosswalk.proven_independent phải là list.")
    writer_ids: list[str] = []
    for index, item in enumerate(value):
        expected = {
            "writer_id",
            "reason",
            "evidence_path",
            "evidence_sha256",
        }
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError(
                f"crosswalk.proven_independent[{index}] sai schema."
            )
        writer_id = item["writer_id"]
        reason = item["reason"]
        evidence_path = item["evidence_path"]
        if (
            not isinstance(writer_id, str)
            or not writer_id
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(evidence_path, str)
            or not evidence_path
        ):
            raise ValueError(
                "Proven-independent writer/reason/evidence không hợp lệ."
            )
        expected_hash = _sha256(
            item["evidence_sha256"],
            "independence evidence",
        )
        evidence = Path(evidence_path)
        if not evidence.is_file():
            raise FileNotFoundError(
                f"Thiếu proven-independent evidence: {evidence}"
            )
        if sha256_file(evidence) != expected_hash:
            raise ValueError(
                f"Proven-independent evidence hash mismatch: {evidence}"
            )
        writer_ids.append(writer_id)
    if len(writer_ids) != len(set(writer_ids)):
        raise ValueError("proven_independent chứa writer ID trùng.")
    return frozenset(writer_ids)


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} phải là lowercase SHA-256.")
    return value


def load_writer_crosswalk(path: Path) -> WriterCoverage:
    """Load a human-approved, complete cross-dataset identity decision."""
    if not path.is_file():
        raise FileNotFoundError(
            "Thiếu approved Vietnamese writer crosswalk: "
            f"{path}. Hãy review candidate report trước khi split."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "status",
        "approved",
        "candidate_report",
        "candidate_report_sha256",
        "proven_independent",
        "unresolved",
        "excluded",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Writer crosswalk sai schema.")
    if raw["schema_version"] != 1 or raw["status"] != "approved":
        raise ValueError(
            "Writer crosswalk phải có schema_version=1 và status=approved."
        )
    candidate_path = Path(str(raw["candidate_report"]))
    candidate_hash = _sha256(
        raw["candidate_report_sha256"],
        "candidate_report_sha256",
    )
    if not candidate_path.is_file() or sha256_file(candidate_path) != candidate_hash:
        raise ValueError("Crosswalk candidate report hash mismatch.")
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        not isinstance(candidate_payload, Mapping)
        or candidate_payload.get("schema_version") != 1
        or candidate_payload.get("status") != "candidate_only"
        or not isinstance(candidate_payload.get("candidates"), Sequence)
    ):
        raise ValueError("Crosswalk candidate report sai schema.")
    evidence_by_vnondb = {
        str(item.get("vnondb_writer_id")): str(
            item.get("evidence_sha256")
        )
        for item in candidate_payload["candidates"]
        if isinstance(item, Mapping)
    }
    approved_raw = raw["approved"]
    if (
        not isinstance(approved_raw, Sequence)
        or isinstance(approved_raw, (str, bytes))
    ):
        raise ValueError("crosswalk.approved phải là list.")
    pairs: list[ApprovedWriterPair] = []
    for index, item in enumerate(approved_raw):
        keys = {
            "uithwdb_writer_id",
            "vnondb_writer_id",
            "evidence_sha256",
        }
        if not isinstance(item, Mapping) or set(item) != keys:
            raise ValueError(f"crosswalk.approved[{index}] sai schema.")
        uit = item["uithwdb_writer_id"]
        vnon = item["vnondb_writer_id"]
        if (
            not isinstance(uit, str)
            or not uit
            or not isinstance(vnon, str)
            or not vnon
        ):
            raise ValueError("Approved writer IDs phải là string không rỗng.")
        pairs.append(
            ApprovedWriterPair(
                uit,
                vnon,
                _sha256(item["evidence_sha256"], "evidence_sha256"),
            )
        )
        if evidence_by_vnondb.get(vnon) != pairs[-1].evidence_sha256:
            raise ValueError(
                "Approved pair không bind candidate evidence hiện tại."
            )
    uit_ids = [pair.uithwdb_writer_id for pair in pairs]
    vnon_ids = [pair.vnondb_writer_id for pair in pairs]
    if len(uit_ids) != len(set(uit_ids)) or len(vnon_ids) != len(
        set(vnon_ids)
    ):
        raise ValueError("Approved crosswalk phải one-to-one.")
    proven = _proven_independent(raw["proven_independent"])
    unresolved = _strings(raw["unresolved"], "unresolved")
    excluded = _strings(raw["excluded"], "excluded")
    approved_ids = set(uit_ids) | set(vnon_ids)
    categories = (
        approved_ids,
        set(proven),
        set(unresolved),
        set(excluded),
    )
    for index, left in enumerate(categories):
        for right in categories[index + 1 :]:
            if left & right:
                raise ValueError(
                    "Writer không được xuất hiện ở nhiều crosswalk states."
                )
    return WriterCoverage(
        tuple(pairs),
        proven,
        unresolved,
        excluded,
        sha256_file(path),
        candidate_hash,
    )


def evidence_digest(payload: Mapping[str, object]) -> str:
    """Hash canonical candidate evidence for manual review."""
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
