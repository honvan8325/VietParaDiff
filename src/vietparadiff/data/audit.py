"""Full manifest, image, split, formatter, and CTC audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from vietparadiff.data.contracts import (
    eligible_reference,
    excluded_source_line_ids,
    normalize_content,
)
from vietparadiff.data.build_provenance import (
    BUILDER_CONFIGS,
    builder_config_sha256,
    git_provenance,
    raw_inventory_sha256,
)
from vietparadiff.data.pipeline import (
    HTRImageProcessor,
    HTRVocabulary,
)
from vietparadiff.models.config import TextEncoderConfig
from vietparadiff.models.grapheme import (
    GraphemeVocabulary,
    ParagraphFormatter,
)

NORMALIZED_REAL_DATASETS = ("cvl", "iam", "uithwdb")


@dataclass(frozen=True, slots=True)
class AuditIssue:
    severity: Literal["hard_error", "expected_rejection", "warning"]
    code: str
    manifest: str
    record_id: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "manifest": self.manifest,
            "record_id": self.record_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    manifest_sha256: dict[str, str]
    provenance_sha256: dict[str, str]
    image_inventory_sha256: str
    provenance_inventory_sha256: str
    dataset_snapshot_sha256: str
    image_count: int

    def report_fields(self) -> dict[str, object]:
        return {
            "manifest_sha256": dict(self.manifest_sha256),
            "provenance_sha256": dict(self.provenance_sha256),
            "image_inventory_sha256": self.image_inventory_sha256,
            "provenance_inventory_sha256": (
                self.provenance_inventory_sha256
            ),
            "dataset_snapshot_sha256": self.dataset_snapshot_sha256,
            "snapshot_image_count": self.image_count,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_key(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _digest_entries(entries: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(entry.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_snapshot(
    manifest_sha256: Mapping[str, str],
    image_sha256: Mapping[str, str],
    provenance_sha256: Mapping[str, str] | None = None,
) -> DatasetSnapshot:
    provenance_sha256 = provenance_sha256 or {}
    manifest_entries = [
        f"manifest:{path}:{digest}"
        for path, digest in manifest_sha256.items()
    ]
    image_entries = [
        f"image:{path}:{digest}"
        for path, digest in image_sha256.items()
    ]
    provenance_entries = [
        f"provenance:{path}:{digest}"
        for path, digest in provenance_sha256.items()
    ]
    return DatasetSnapshot(
        manifest_sha256=dict(sorted(manifest_sha256.items())),
        provenance_sha256=dict(sorted(provenance_sha256.items())),
        image_inventory_sha256=_digest_entries(image_entries),
        provenance_inventory_sha256=_digest_entries(
            provenance_entries
        ),
        dataset_snapshot_sha256=_digest_entries(
            [*manifest_entries, *image_entries, *provenance_entries]
        ),
        image_count=len(image_sha256),
    )


def _records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number} JSON không hợp lệ."
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} phải là JSON object.")
        records.append(dict(payload))
    if not records:
        raise ValueError(f"Manifest không được rỗng: {path}")
    return records


def _split_kind(path: Path) -> str:
    name = path.name.lower()
    return "test" if name.startswith("test") else "train"


def _resolve(image_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else image_root / path


def _image_info(path: Path) -> tuple[int, int, str, str]:
    digest = _file_sha256(path)
    with Image.open(path) as image:
        image.load()
        return image.width, image.height, image.mode, digest


def _provenance_hashes(split_root: Path) -> dict[str, str]:
    data_root = split_root.parent
    provenance_paths: set[Path] = set()
    for dataset in NORMALIZED_REAL_DATASETS:
        provenance_paths.add(data_root / dataset / "manifest.jsonl")
        provenance_paths.add(data_root / dataset / "build_report.json")
    provenance_paths.add(
        data_root / "uithwdb_augmented" / "manifest.jsonl"
    )
    provenance_paths.update((split_root / "writers").glob("*.json"))
    return {
        _relative_key(path, data_root): (
            _file_sha256(path) if path.is_file() else "<missing>"
        )
        for path in sorted(provenance_paths)
    }


def _ctc_required(ids: Sequence[int]) -> int:
    repeats = sum(
        first == second
        for first, second in zip(ids[:-1], ids[1:], strict=True)
    )
    return len(ids) + repeats


def compute_dataset_snapshot(
    split_root: Path,
    *,
    image_root: Path,
) -> DatasetSnapshot:
    """Hash every split manifest and every stage-participating image."""
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy split root: {split_root}"
        )
    manifests = sorted(split_root.rglob("*.jsonl"))
    if not manifests:
        raise ValueError("Không tìm thấy split manifests.")
    manifest_sha256 = {
        _relative_key(path, split_root): _file_sha256(path)
        for path in manifests
    }
    provenance_sha256 = _provenance_hashes(split_root)
    image_paths: set[Path] = set()
    for manifest in manifests:
        if manifest.name == "rejected_targets.jsonl":
            continue
        for record in _records(manifest):
            for field in ("image", "target_image", "reference_image"):
                value = record.get(field)
                if value is not None:
                    image_paths.add(
                        _resolve(image_root, value).resolve()
                    )
    image_sha256 = {
        _relative_key(path, image_root): (
            _file_sha256(path) if path.is_file() else "<missing>"
        )
        for path in sorted(image_paths)
    }
    return _build_snapshot(
        manifest_sha256,
        image_sha256,
        provenance_sha256,
    )


class DatasetAuditor:
    def __init__(
        self,
        split_root: Path,
        *,
        image_root: Path,
        workers: int = 8,
    ) -> None:
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy split root: {split_root}"
            )
        if workers <= 0:
            raise ValueError("Audit workers phải dương.")
        self.split_root = split_root
        self.image_root = image_root
        self.workers = workers
        self.issues: list[AuditIssue] = []

    def _issue(
        self,
        code: str,
        manifest: Path,
        record_id: object,
        detail: str,
        *,
        severity: Literal[
            "hard_error", "expected_rejection", "warning"
        ] = "hard_error",
    ) -> None:
        self.issues.append(
            AuditIssue(
                severity,
                code,
                str(manifest),
                str(record_id),
                detail,
            )
        )

    def run(self) -> dict[str, object]:
        manifests = sorted(self.split_root.rglob("*.jsonl"))
        if not manifests:
            raise ValueError("Không tìm thấy split manifests.")
        loaded: dict[Path, list[dict[str, object]]] = {}
        writer_sets: dict[str, set[str]] = {
            "train": set(),
            "test": set(),
        }
        image_uses: dict[Path, set[str]] = defaultdict(set)
        image_levels: dict[Path, set[str]] = defaultdict(set)
        image_records: dict[
            Path, list[tuple[str, str, str, str, str]]
        ] = defaultdict(list)
        image_expected: dict[Path, list[tuple[Path, str, int | None, int | None]]] = defaultdict(list)
        id_to_writer: dict[str, str] = {}
        record_counts: dict[str, int] = {}
        manifest_sha256 = {
            _relative_key(path, self.split_root): _file_sha256(path)
            for path in manifests
        }

        for manifest in manifests:
            try:
                records = _records(manifest)
            except (ValueError, OSError) as error:
                self._issue("manifest_read", manifest, "-", str(error))
                continue
            loaded[manifest] = records
            record_counts[str(manifest)] = len(records)
            seen_ids: set[str] = set()
            participates_in_stage = (
                manifest.name != "rejected_targets.jsonl"
            )
            kind = _split_kind(manifest)
            for index, record in enumerate(records):
                record_id = str(
                    record.get("id", record.get("pair_id", f"row_{index}"))
                )
                if manifest.name == "rejected_targets.jsonl":
                    reason = record.get("rejection_reason_code")
                    if not isinstance(reason, str) or not reason:
                        self._issue(
                            "rejection_missing_reason",
                            manifest,
                            record_id,
                            "Expected rejection thiếu reason code.",
                        )
                    else:
                        self._issue(
                            reason,
                            manifest,
                            record_id,
                            str(record.get("rejection_reason", reason)),
                            severity="expected_rejection",
                        )
                if record_id in seen_ids:
                    self._issue(
                        "duplicate_id",
                        manifest,
                        record_id,
                        "ID trùng trong cùng manifest.",
                    )
                seen_ids.add(record_id)
                writer = record.get("canonical_writer_id")
                if isinstance(writer, str) and writer:
                    if participates_in_stage:
                        writer_sets[kind].add(writer)
                    if "id" in record:
                        previous = id_to_writer.get(str(record["id"]))
                        if previous is not None and previous != writer:
                            self._issue(
                                "writer_identity",
                                manifest,
                                record_id,
                                f"ID map tới {previous} và {writer}.",
                            )
                        id_to_writer[str(record["id"])] = writer
                image_fields = (
                    ("image", record.get("image")),
                    ("target_image", record.get("target_image")),
                    ("reference_image", record.get("reference_image")),
                )
                for field, value in image_fields:
                    if value is None or not participates_in_stage:
                        continue
                    path = _resolve(self.image_root, value).resolve()
                    image_uses[path].add(kind)
                    image_levels[path].add(
                        str(
                            record.get(
                                "level",
                                "target"
                                if field == "target_image"
                                else "reference",
                            )
                        )
                    )
                    if field == "image":
                        image_records[path].append(
                            (
                                str(record.get("dataset", "")),
                                str(record.get("level", "")),
                                str(record.get("canonical_writer_id", "")),
                                str(record.get("text", "")),
                                str(record.get("id", "")),
                            )
                        )
                    expected_width = (
                        int(record["width"])
                        if field == "image"
                        and isinstance(record.get("width"), int)
                        else None
                    )
                    expected_height = (
                        int(record["height"])
                        if field == "image"
                        and isinstance(record.get("height"), int)
                        else None
                    )
                    image_expected[path].append(
                        (
                            manifest,
                            record_id,
                            expected_width,
                            expected_height,
                        )
                    )

        leakage = writer_sets["train"] & writer_sets["test"]
        for writer in sorted(leakage):
            self._issue(
                "writer_leakage",
                self.split_root,
                writer,
                "Writer xuất hiện ở cả train và test manifests.",
            )
        for path, uses in image_uses.items():
            if uses == {"train", "test"}:
                self._issue(
                    "image_path_leakage",
                    self.split_root,
                    path,
                    "Cùng image path xuất hiện ở train và test.",
                )

        image_results: dict[Path, tuple[int, int, str, str]] = {}
        existing = [path for path in image_expected if path.is_file()]
        for path in image_expected:
            if not path.is_file():
                for manifest, record_id, _, _ in image_expected[path]:
                    self._issue(
                        "missing_image",
                        manifest,
                        record_id,
                        str(path),
                    )
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                path: executor.submit(_image_info, path)
                for path in existing
            }
            for path, future in futures.items():
                try:
                    image_results[path] = future.result()
                except Exception as error:
                    for manifest, record_id, _, _ in image_expected[path]:
                        self._issue(
                            "image_decode",
                            manifest,
                            record_id,
                            f"{path}: {error}",
                        )
        hash_splits: dict[str, set[str]] = defaultdict(set)
        hash_paths: dict[str, set[str]] = defaultdict(set)
        for path, (width, height, mode, digest) in image_results.items():
            if mode not in {"1", "L", "RGB", "RGBA"}:
                for manifest, record_id, _, _ in image_expected[path]:
                    self._issue(
                        "image_mode",
                        manifest,
                        record_id,
                        f"{path} có mode={mode}.",
                    )
            hash_splits[digest].update(image_uses[path])
            hash_paths[digest].add(str(path))
            for manifest, record_id, expected_width, expected_height in image_expected[path]:
                if expected_width is not None and expected_width != width:
                    self._issue(
                        "width_mismatch",
                        manifest,
                        record_id,
                        f"expected={expected_width}, actual={width}.",
                    )
                if expected_height is not None and expected_height != height:
                    self._issue(
                        "height_mismatch",
                        manifest,
                        record_id,
                        f"expected={expected_height}, actual={height}.",
                    )
        for digest, splits in hash_splits.items():
            if splits == {"train", "test"}:
                self._issue(
                    "image_content_leakage",
                    self.split_root,
                    digest,
                    ", ".join(sorted(hash_paths[digest])),
                )
            paths = [Path(value) for value in hash_paths[digest]]
            records = sorted(
                {
                    item
                    for path in paths
                    for item in image_records[path]
                }
            )
            if len(records) > 1:
                writers = {item[2] for item in records}
                texts = {
                    normalize_content(item[3])
                    for item in records
                    if item[3].strip()
                }
                levels = {item[1] for item in records}
                if len(writers) > 1:
                    self._issue(
                        "duplicate_cross_writer",
                        self.split_root,
                        digest,
                        ", ".join(sorted(hash_paths[digest])),
                    )
                elif len(texts) > 1:
                    self._issue(
                        "duplicate_label_conflict",
                        self.split_root,
                        digest,
                        ", ".join(sorted(texts)),
                    )
                elif len(levels) > 1:
                    self._issue(
                        "duplicate_cross_level",
                        self.split_root,
                        digest,
                        ", ".join(sorted(levels)),
                        severity="warning",
                    )
                else:
                    self._issue(
                        "duplicate_same_identity",
                        self.split_root,
                        digest,
                        ", ".join(sorted(hash_paths[digest])),
                        severity="warning",
                    )

        self._audit_formatter(loaded)
        self._audit_ctc(loaded)
        self._audit_references(loaded, id_to_writer)
        self._audit_provenance(loaded)
        issue_counts: dict[str, int] = defaultdict(int)
        severity_counts: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            issue_counts[issue.code] += 1
            severity_counts[issue.severity] += 1
        image_sha256 = {
            _relative_key(path, self.image_root): (
                image_results[path][3]
                if path in image_results
                else (
                    _file_sha256(path)
                    if path.is_file()
                    else "<missing>"
                )
            )
            for path in image_expected
        }
        snapshot = _build_snapshot(
            manifest_sha256,
            image_sha256,
            _provenance_hashes(self.split_root),
        )
        return {
            "schema_version": 3,
            "split_root": str(self.split_root),
            "image_root": str(self.image_root),
            "manifest_count": len(manifests),
            "record_counts": record_counts,
            "unique_image_count": len(image_expected),
            "decoded_image_count": len(image_results),
            "train_writer_count": len(writer_sets["train"]),
            "test_writer_count": len(writer_sets["test"]),
            "hard_error_count": severity_counts["hard_error"],
            "expected_rejection_count": severity_counts[
                "expected_rejection"
            ],
            "warning_count": severity_counts["warning"],
            # Compatibility field for older report consumers. It now means
            # blocking errors only.
            "error_count": severity_counts["hard_error"],
            "error_counts": dict(sorted(issue_counts.items())),
            "issues": [issue.to_dict() for issue in self.issues],
            "errors": [
                issue.to_dict()
                for issue in self.issues
                if issue.severity == "hard_error"
            ],
            **snapshot.report_fields(),
        }

    def _audit_provenance(
        self,
        loaded: Mapping[Path, Sequence[Mapping[str, object]]],
    ) -> None:
        data_root = self.split_root.parent
        current_git = git_provenance()
        for dataset in NORMALIZED_REAL_DATASETS:
            report_path = data_root / dataset / "build_report.json"
            manifest = data_root / dataset / "manifest.jsonl"
            if not report_path.is_file():
                self._issue(
                    "missing_build_report",
                    report_path,
                    dataset,
                    "Normalized dataset thiếu build_report.json.",
                )
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                self._issue(
                    "build_report_read",
                    report_path,
                    dataset,
                    str(error),
                )
                continue
            required = {
                "schema_version",
                "dataset",
                "git",
                "raw_root",
                "raw_inventory_sha256",
                "builder_config",
                "builder_config_sha256",
                "accepted_count",
                "expected_rejection_count",
                "expected_rejections",
                "hard_error_count",
                "hard_errors",
                "warning_count",
                "warnings",
                "output_manifest_sha256",
            }
            if not isinstance(report, Mapping) or not required.issubset(
                report
            ):
                self._issue(
                    "build_report_schema",
                    report_path,
                    dataset,
                    "Build report thiếu provenance fields.",
                )
                continue
            if report["schema_version"] != 2:
                self._issue(
                    "build_report_schema",
                    report_path,
                    dataset,
                    f"expected=2, actual={report['schema_version']}.",
                )
            if report["dataset"] != dataset:
                self._issue(
                    "build_report_dataset",
                    report_path,
                    dataset,
                    f"actual={report['dataset']}.",
                )
            expected_config = BUILDER_CONFIGS[dataset]
            if (
                report["builder_config"] != expected_config
                or report["builder_config_sha256"]
                != builder_config_sha256(expected_config)
            ):
                self._issue(
                    "build_config_contract",
                    report_path,
                    dataset,
                    "Resolved builder config/hash không khớp code hiện tại.",
                )
            if report["git"] != current_git:
                self._issue(
                    "build_git_provenance",
                    report_path,
                    dataset,
                    "Build report không bind current Git commit/dirty patch.",
                )
            raw_root = Path(str(report["raw_root"]))
            try:
                current_raw_hash = raw_inventory_sha256(raw_root)
            except (OSError, ValueError) as error:
                self._issue(
                    "build_raw_inventory",
                    report_path,
                    dataset,
                    str(error),
                )
            else:
                if current_raw_hash != report["raw_inventory_sha256"]:
                    self._issue(
                        "build_raw_inventory",
                        report_path,
                        dataset,
                        "Raw inventory đã đổi từ lúc build.",
                    )
            for count_name, list_name in (
                ("expected_rejection_count", "expected_rejections"),
                ("hard_error_count", "hard_errors"),
                ("warning_count", "warnings"),
            ):
                values = report[list_name]
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes))
                    or report[count_name] != len(values)
                ):
                    self._issue(
                        "build_report_counts",
                        report_path,
                        dataset,
                        f"{count_name} không khớp {list_name}.",
                    )
                    continue
                for index, issue in enumerate(values):
                    if (
                        not isinstance(issue, Mapping)
                        or not all(
                            isinstance(issue.get(field), str)
                            and bool(str(issue[field]).strip())
                            for field in ("record_id", "reason", "source")
                        )
                    ):
                        self._issue(
                            "build_issue_provenance",
                            report_path,
                            f"{dataset}:{list_name}:{index}",
                            "Build issue thiếu record_id/reason/source.",
                        )
            if manifest.is_file():
                try:
                    accepted = len(_records(manifest))
                except (OSError, ValueError) as error:
                    self._issue(
                        "build_manifest_read",
                        report_path,
                        dataset,
                        str(error),
                    )
                else:
                    if report["accepted_count"] != accepted:
                        self._issue(
                            "build_accepted_count",
                            report_path,
                            dataset,
                            (
                                f"report={report['accepted_count']}, "
                                f"manifest={accepted}."
                            ),
                        )
            if int(report["hard_error_count"]) != 0:
                self._issue(
                    "build_hard_errors",
                    report_path,
                    dataset,
                    f"count={report['hard_error_count']}.",
                )
            if (
                not manifest.is_file()
                or report["output_manifest_sha256"]
                != _file_sha256(manifest)
            ):
                self._issue(
                    "build_manifest_hash",
                    report_path,
                    dataset,
                    "Build report không bind current normalized manifest.",
                )

        for manifest, records in loaded.items():
            if manifest.name == "rejected_targets.jsonl":
                continue
            for record in records:
                writer_id = record.get("writer_id")
                if writer_id is None:
                    continue
                if (
                    not isinstance(writer_id, str)
                    or not writer_id
                    or record.get("canonical_writer_id") != writer_id
                ):
                    self._issue(
                        "canonical_writer_identity",
                        manifest,
                        str(
                            record.get(
                                "id",
                                record.get("pair_id", "-"),
                            )
                        ),
                        (
                            "Pipeline UIT-HWDB-only yêu cầu "
                            "canonical_writer_id == writer_id."
                        ),
                    )

    def _formatter(self) -> ParagraphFormatter:
        vocabulary = GraphemeVocabulary.default_vietnamese()
        config = TextEncoderConfig(
            base_vocab_size=len(vocabulary.base_to_id),
            shape_vocab_size=len(vocabulary.shape_to_id),
            tone_vocab_size=len(vocabulary.tone_to_id),
            case_vocab_size=len(vocabulary.case_to_id),
            class_vocab_size=len(vocabulary.class_to_id),
        )
        return ParagraphFormatter(config)

    def _audit_formatter(
        self,
        loaded: Mapping[Path, Sequence[Mapping[str, object]]],
    ) -> None:
        formatter = self._formatter()
        for manifest, records in loaded.items():
            if manifest.name not in {
                "pretrain_targets.jsonl",
                "finetune_targets_real.jsonl",
                "finetune_targets_synthetic.jsonl",
                "test_pairs.jsonl",
            }:
                continue
            for index, record in enumerate(records):
                record_id = record.get(
                    "id",
                    record.get("pair_id", f"row_{index}"),
                )
                text = record.get("text", record.get("target_text"))
                if not isinstance(text, str) or not text.strip():
                    self._issue(
                        "formatter_text",
                        manifest,
                        record_id,
                        "Target text thiếu hoặc rỗng.",
                    )
                    continue
                preserve = (
                    record.get("formatter_mode") == "physical_lines"
                    or manifest.name == "test_pairs.jsonl"
                )
                try:
                    formatter.format(
                        text,
                        preserve_physical_lines=preserve,
                    )
                except Exception as error:
                    self._issue(
                        "formatter_rejection",
                        manifest,
                        record_id,
                        str(error),
                    )

    def _audit_ctc(
        self,
        loaded: Mapping[Path, Sequence[Mapping[str, object]]],
    ) -> None:
        train_lines = self.split_root / "htr" / "train_lines.jsonl"
        train_words = self.split_root / "htr" / "train_words.jsonl"
        if not train_lines.is_file() or not train_words.is_file():
            self._issue(
                "ctc_manifest",
                self.split_root,
                "-",
                "Thiếu HTR train manifests.",
            )
            return
        try:
            vocabulary = HTRVocabulary.build_from_manifests(
                (train_lines, train_words)
            )
        except Exception as error:
            self._issue(
                "ctc_manifest",
                self.split_root,
                "-",
                str(error),
            )
            return
        processor = HTRImageProcessor()
        for manifest, records in loaded.items():
            if manifest.parent.name != "htr":
                continue
            for index, record in enumerate(records):
                record_id = str(record.get("id", f"row_{index}"))
                text = record.get("text")
                image = record.get("image")
                if not isinstance(text, str) or image is None:
                    self._issue(
                        "ctc_schema",
                        manifest,
                        record_id,
                        "HTR record thiếu text/image.",
                    )
                    continue
                path = _resolve(self.image_root, image)
                if not path.is_file():
                    continue
                try:
                    processed_width = int(processor(path)["valid_width"])
                    valid_width = max(
                        processed_width,
                        vocabulary.minimum_input_width(text),
                    )
                    input_length = (valid_width + 3) // 4
                    heads = vocabulary.encode(text)
                    required = max(
                        _ctc_required(head.tolist()) for head in heads
                    )
                    if required > input_length:
                        self._issue(
                            "ctc_infeasible",
                            manifest,
                            record_id,
                            f"required={required}, input={input_length}.",
                        )
                except Exception as error:
                    self._issue(
                        "ctc_processing",
                        manifest,
                        record_id,
                        str(error),
                    )

    def _audit_references(
        self,
        loaded: Mapping[Path, Sequence[Mapping[str, object]]],
        id_to_writer: Mapping[str, str],
    ) -> None:
        reference_records: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        records_by_id: dict[str, Mapping[str, object]] = {}
        for records in loaded.values():
            for record in records:
                sample_id = record.get("id")
                if isinstance(sample_id, str) and sample_id:
                    records_by_id[sample_id] = record
        for manifest, records in loaded.items():
            if "references" not in manifest.name:
                continue
            for record in records:
                writer = record.get("canonical_writer_id")
                if isinstance(writer, str):
                    reference_records[writer].append(record)
        for manifest, records in loaded.items():
            if (
                "targets" not in manifest.name
                or manifest.name == "rejected_targets.jsonl"
            ):
                continue
            for index, record in enumerate(records):
                writer = record.get("canonical_writer_id")
                record_id = str(record.get("id", f"row_{index}"))
                if not isinstance(writer, str):
                    continue
                try:
                    excluded = set(excluded_source_line_ids(record))
                except ValueError as error:
                    self._issue(
                        "reference_source_contract",
                        manifest,
                        record_id,
                        str(error),
                    )
                    continue
                candidates = [
                    candidate
                    for candidate in reference_records.get(writer, [])
                    if eligible_reference(
                        record,
                        candidate,
                        excluded_reference_ids=excluded,
                    )
                ]
                if not candidates:
                    self._issue(
                        "reference_ineligible",
                        manifest,
                        record_id,
                        f"Writer {writer} không có eligible reference.",
                    )
        test_pairs = self.split_root / "vietparadiff" / "test_pairs.jsonl"
        if test_pairs in loaded:
            for record in loaded[test_pairs]:
                writer = str(record["canonical_writer_id"])
                resolved: dict[str, Mapping[str, object]] = {}
                for field in ("target_id", "reference_id"):
                    sample_id = str(record[field])
                    actual = id_to_writer.get(sample_id)
                    if actual is not None and actual != writer:
                        self._issue(
                            "test_pair_writer",
                            test_pairs,
                            record["pair_id"],
                            f"{field} writer={actual}, pair writer={writer}.",
                        )
                    source = records_by_id.get(sample_id)
                    if source is None:
                        self._issue(
                            "test_pair_missing_record",
                            test_pairs,
                            record["pair_id"],
                            f"Không resolve được {field}={sample_id}.",
                        )
                    else:
                        resolved[field] = source
                target = resolved.get("target_id")
                reference = resolved.get("reference_id")
                if target is not None and (
                    target.get("level") != "paragraph"
                    or bool(target.get("synthetic", False))
                ):
                    self._issue(
                        "test_pair_target_contract",
                        test_pairs,
                        record["pair_id"],
                        "Final target phải là real paragraph.",
                    )
                if (
                    target is not None
                    and reference is not None
                    and not eligible_reference(target, reference)
                ):
                    self._issue(
                        "test_pair_reference_contract",
                        test_pairs,
                        record["pair_id"],
                        "Final reference không đạt shared eligibility.",
                    )


__all__ = [
    "AuditIssue",
    "DatasetAuditor",
    "DatasetSnapshot",
    "compute_dataset_snapshot",
]
