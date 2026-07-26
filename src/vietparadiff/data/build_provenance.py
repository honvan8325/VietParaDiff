"""Strict build provenance shared by normalized dataset builders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vietparadiff.artifacts import sha256_file

__all__ = [
    "BuildIssues",
    "BUILDER_CONFIGS",
    "builder_config_sha256",
    "git_provenance",
    "raw_inventory_sha256",
    "write_build_report",
]

BUILDER_CONFIGS: dict[str, dict[str, object]] = {
    "cvl": {"levels": ["paragraph", "line", "word"]},
    "iam": {
        "levels": ["paragraph", "line", "word"],
        "paragraph_padding": 20,
        "line_padding": 10,
        "word_padding": 4,
    },
    "uithwdb": {
        "native_line_alignment_required": True,
        "external_annotation_fallback": False,
    },
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class BuildIssues:
    """Collect every raw record excluded or diagnosed by a builder."""

    expected_rejections: list[dict[str, object]] = field(default_factory=list)
    hard_errors: list[dict[str, object]] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)

    @staticmethod
    def _record(
        record_id: str,
        reason: str,
        source: Path | str,
        detail: str | None,
    ) -> dict[str, object]:
        if not record_id or not reason:
            raise ValueError("Build issue cần record_id và reason không rỗng.")
        record: dict[str, object] = {
            "record_id": record_id,
            "reason": reason,
            "source": str(source),
        }
        if detail:
            record["detail"] = detail
        return record

    def reject(
        self,
        record_id: str,
        reason: str,
        source: Path | str,
        detail: str | None = None,
    ) -> None:
        self.expected_rejections.append(
            self._record(record_id, reason, source, detail)
        )

    def hard(
        self,
        record_id: str,
        reason: str,
        source: Path | str,
        detail: str | None = None,
    ) -> None:
        self.hard_errors.append(
            self._record(record_id, reason, source, detail)
        )

    def warn(
        self,
        record_id: str,
        reason: str,
        source: Path | str,
        detail: str | None = None,
    ) -> None:
        self.warnings.append(
            self._record(record_id, reason, source, detail)
        )


def raw_inventory_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset root không tồn tại: {root}")
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_provenance() -> dict[str, object]:
    source_scope = ("src", "scripts", "pyproject.toml", "uv.lock")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *source_scope],
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_entries: list[str] = []
    for value in untracked_output.splitlines():
        path = Path(value)
        if path.is_file() and (
            value in {"pyproject.toml", "uv.lock"}
            or value.startswith(("src/", "scripts/"))
        ):
            untracked_entries.append(f"{value}:{sha256_file(path)}")
    untracked_hash = _canonical_sha256(sorted(untracked_entries))
    return {
        "commit": commit,
        "source_scope": list(source_scope),
        "dirty": bool(patch or untracked_entries),
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_inventory_sha256": untracked_hash,
    }


def builder_config_sha256(config: Mapping[str, object]) -> str:
    return _canonical_sha256(dict(config))


def write_build_report(
    *,
    dataset: str,
    raw_root: Path,
    manifest: Path,
    output: Path,
    builder_config: Mapping[str, object],
    accepted_count: int,
    expected_rejections: Sequence[Mapping[str, object]],
    hard_errors: Sequence[Mapping[str, object]],
    warnings: Sequence[Mapping[str, object]] = (),
) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(f"Build manifest chưa tồn tại: {manifest}")
    payload = {
        "schema_version": 2,
        "dataset": dataset,
        "git": git_provenance(),
        "raw_root": str(raw_root.resolve()),
        "raw_inventory_sha256": raw_inventory_sha256(raw_root),
        "builder_config": dict(builder_config),
        "builder_config_sha256": builder_config_sha256(builder_config),
        "accepted_count": accepted_count,
        "expected_rejection_count": len(expected_rejections),
        "expected_rejections": [dict(item) for item in expected_rejections],
        "hard_error_count": len(hard_errors),
        "hard_errors": [dict(item) for item in hard_errors],
        "warning_count": len(warnings),
        "warnings": [dict(item) for item in warnings],
        "output_manifest": str(manifest),
        "output_manifest_sha256": sha256_file(manifest),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
