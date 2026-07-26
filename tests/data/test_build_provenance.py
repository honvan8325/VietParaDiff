"""Regression tests for truthful, hash-bound dataset build provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vietparadiff.data.audit import compute_dataset_snapshot
from vietparadiff.data.build_provenance import (
    BUILDER_CONFIGS,
    BuildIssues,
    raw_inventory_sha256,
    write_build_report,
)
from vietparadiff.data.crosswalk import load_writer_crosswalk


def test_build_report_records_issues_and_binds_raw_manifest(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample.bin").write_bytes(b"raw")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"accepted"}\n', encoding="utf-8")
    issues = BuildIssues()
    issues.reject("missing", "missing_native_line_alignment", "label.json")
    issues.warn("accepted", "metadata_count_mismatch", "forms.txt")
    report = tmp_path / "build_report.json"
    write_build_report(
        dataset="uithwdb",
        raw_root=raw,
        manifest=manifest,
        output=report,
        builder_config=BUILDER_CONFIGS["uithwdb"],
        accepted_count=1,
        expected_rejections=issues.expected_rejections,
        hard_errors=issues.hard_errors,
        warnings=issues.warnings,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["raw_inventory_sha256"] == raw_inventory_sha256(raw)
    assert payload["accepted_count"] == 1
    assert payload["expected_rejection_count"] == 1
    assert payload["warning_count"] == 1
    assert payload["hard_error_count"] == 0
    assert set(payload["expected_rejections"][0]) >= {
        "record_id",
        "reason",
        "source",
    }


def test_snapshot_changes_when_writer_or_build_provenance_changes(
    tmp_path: Path,
) -> None:
    split = tmp_path / "data" / "splits"
    manifest = split / "htr" / "train_lines.jsonl"
    manifest.parent.mkdir(parents=True)
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    manifest.write_text(
        json.dumps({"id": "line", "image": str(image)}) + "\n",
        encoding="utf-8",
    )
    writers = split / "writers"
    writers.mkdir()
    writer_artifact = writers / "train.json"
    writer_artifact.write_text('{"writers":[]}\n', encoding="utf-8")
    first = compute_dataset_snapshot(split, image_root=tmp_path)
    writer_artifact.write_text('{"writers":["changed"]}\n', encoding="utf-8")
    second = compute_dataset_snapshot(split, image_root=tmp_path)
    assert first.provenance_inventory_sha256 != (
        second.provenance_inventory_sha256
    )
    assert first.dataset_snapshot_sha256 != second.dataset_snapshot_sha256


def test_proven_independent_requires_real_hash_bound_evidence(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "candidate_only",
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "candidate_report": str(candidate),
                "candidate_report_sha256": hashlib.sha256(
                    candidate.read_bytes()
                ).hexdigest(),
                "approved": [],
                "proven_independent": [
                    {
                        "writer_id": "uithwdb_1",
                        "reason": "authoritative roster",
                        "evidence_path": str(tmp_path / "missing.json"),
                        "evidence_sha256": "0" * 64,
                    }
                ],
                "unresolved": [],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="evidence"):
        load_writer_crosswalk(crosswalk)
