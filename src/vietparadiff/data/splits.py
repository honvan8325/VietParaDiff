"""Create writer-disjoint manifests for every VietParaDiff training stage."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from vietparadiff.data.contracts import (
    eligible_reference,
    excluded_source_line_ids,
)
from vietparadiff.models.config import TextEncoderConfig
from vietparadiff.models.grapheme import (
    SHAPE_MARKS,
    TONE_MARKS,
    GraphemeVocabulary,
    ParagraphFormatter,
    VietnameseGraphemeFactorizer,
)

__all__ = [
    "SplitConfig",
    "create_data_splits",
]


REAL_DATASETS = ("cvl", "iam", "uithwdb")
ALL_DATASETS = REAL_DATASETS + ("uithwdb_augmented",)
VIETNAMESE_DATASETS = ("uithwdb",)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuration for deterministic writer-level train/test splitting."""

    data_root: Path = Path("data")
    output_root: Path = Path("data/splits")
    test_fraction: float = 0.2
    seed: int = 42
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.data_root, Path):
            raise TypeError("data_root phải là pathlib.Path.")
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root phải là pathlib.Path.")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction phải nằm trong (0, 1).")
        if not isinstance(self.seed, int):
            raise TypeError("seed phải là int.")


def _load_manifest(path: Path, dataset: str) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records: list[dict[str, object]] = []
    required = {"id", "image", "text", "writer_id", "level"}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"Missing keys {sorted(missing)} at {path}:{line_number}."
                )
            if record["level"] not in {"paragraph", "line", "word"}:
                raise ValueError(
                    f"Invalid level at {path}:{line_number}: "
                    f"{record['level']!r}."
                )
            if not all(
                isinstance(record[key], str) and record[key]
                for key in ("id", "image", "text", "writer_id")
            ):
                raise ValueError(
                    f"Invalid string field at {path}:{line_number}."
                )
            enriched = dict(record)
            enriched["dataset"] = dataset
            records.append(enriched)
    by_digest_level: dict[tuple[str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    without_file: list[dict[str, object]] = []
    for record in records:
        image = Path(str(record["image"]))
        if not image.is_file():
            without_file.append(record)
            continue
        digest = _sha256_path(image)
        by_digest_level[(digest, str(record["level"]))].append(record)
    canonical: list[dict[str, object]] = list(without_file)
    for (digest, level), group in sorted(by_digest_level.items()):
        identities = {
            (
                str(record["writer_id"]),
                _flat_text(str(record["text"])),
            )
            for record in group
        }
        if len(identities) > 1:
            writers = {identity[0] for identity in identities}
            code = (
                "cross-writer metadata conflict"
                if len(writers) > 1
                else "label conflict"
            )
            raise ValueError(
                f"Exact duplicate {code} trong {dataset}/{level}: "
                f"sha256={digest}, ids="
                f"{sorted(str(record['id']) for record in group)}."
            )
        ordered = sorted(group, key=lambda record: str(record["id"]))
        kept = dict(ordered[0])
        if len(ordered) > 1:
            kept["duplicate_provenance"] = {
                "image_sha256": digest,
                "canonical_id": str(kept["id"]),
                "duplicate_ids": [
                    str(record["id"]) for record in ordered[1:]
                ],
            }
        canonical.append(kept)
    return sorted(canonical, key=lambda record: str(record["id"]))


def _flat_text(text: str) -> str:
    return " ".join(text.split())


def _canonical_writer_map(
    manifests: Mapping[str, Sequence[dict[str, object]]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    mapping: dict[str, str] = {}
    families: dict[str, tuple[str, ...]] = {}

    for dataset in REAL_DATASETS:
        writer_ids = sorted(
            {str(record["writer_id"]) for record in manifests[dataset]}
        )
        for writer_id in writer_ids:
            if writer_id in mapping:
                raise ValueError(
                    "Writer ID collision giữa normalized datasets: "
                    f"{writer_id}."
                )
            mapping[writer_id] = writer_id
            families[writer_id] = (writer_id,)

    real_uit_writers = {
        str(record["writer_id"]) for record in manifests["uithwdb"]
    }
    augmented_writers = {
        str(record["writer_id"])
        for record in manifests["uithwdb_augmented"]
    }
    unknown_augmented = augmented_writers - real_uit_writers
    if unknown_augmented:
        raise ValueError(
            "Synthetic writer không tồn tại trong UIT-HWDB: "
            f"{sorted(unknown_augmented)}."
        )
    return mapping, families


def _writer_splits(
    families: Mapping[str, tuple[str, ...]],
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for canonical_id, members in families.items():
        if canonical_id.startswith("cvl_"):
            group = "cvl"
        elif canonical_id.startswith("iam_"):
            group = "iam"
        elif canonical_id.startswith("uithwdb_"):
            group = "uithwdb"
        else:
            raise RuntimeError(
                f"Unknown canonical writer family: {canonical_id} {members}"
            )
        groups[group].append(canonical_id)

    test_writers: set[str] = set()
    for group, writer_ids in groups.items():
        ranked = sorted(
            writer_ids,
            key=lambda writer_id: hashlib.sha256(
                f"{seed}:{group}:{writer_id}".encode()
            ).digest(),
        )
        test_count = max(
            1,
            min(
                len(ranked) - 1,
                int(len(ranked) * test_fraction + 0.5),
            ),
        )
        test_writers.update(ranked[:test_count])
    train_writers = set(families) - test_writers
    if train_writers & test_writers:
        raise RuntimeError("Canonical writer leakage giữa train và test.")
    return train_writers, test_writers


@lru_cache(maxsize=None)
def _supported_generator_text(text: str) -> bool:
    factorizer = VietnameseGraphemeFactorizer()
    vocabulary = GraphemeVocabulary.default_vietnamese()
    decomposed = unicodedata.normalize("NFD", text)
    if any(
        unicodedata.combining(character)
        and character not in SHAPE_MARKS
        and character not in TONE_MARKS
        for character in decomposed
    ):
        return False
    try:
        graphemes = factorizer.factorize(text)
    except (TypeError, ValueError):
        return False
    return all(
        grapheme.base in vocabulary.base_to_id
        and grapheme.shape in vocabulary.shape_to_id
        and grapheme.tone in vocabulary.tone_to_id
        and grapheme.case in vocabulary.case_to_id
        and grapheme.class_name in vocabulary.class_to_id
        for grapheme in graphemes
    )


def _with_canonical_writer(
    record: Mapping[str, object],
    writer_map: Mapping[str, str],
) -> dict[str, object] | None:
    writer_id = str(record["writer_id"])
    canonical_id = writer_map.get(writer_id)
    if canonical_id is None:
        return None
    output = dict(record)
    output["canonical_writer_id"] = canonical_id
    return output


def _filter_records(
    manifests: Mapping[str, Sequence[dict[str, object]]],
    datasets: Iterable[str],
    *,
    level: str,
    writers: set[str],
    writer_map: Mapping[str, str],
) -> list[dict[str, object]]:
    output = []
    for dataset in datasets:
        for record in manifests[dataset]:
            if record["level"] != level:
                continue
            enriched = _with_canonical_writer(record, writer_map)
            if enriched is None:
                continue
            if enriched["canonical_writer_id"] in writers:
                output.append(enriched)
    return sorted(output, key=lambda record: str(record["id"]))


def _writer_payload(
    writer_ids: set[str],
    families: Mapping[str, tuple[str, ...]],
    manifests: Mapping[str, Sequence[dict[str, object]]],
    config: SplitConfig,
) -> dict[str, object]:
    datasets_by_writer: dict[str, set[str]] = defaultdict(set)
    for dataset, records in manifests.items():
        for record in records:
            datasets_by_writer[str(record["writer_id"])].add(dataset)
    writers = []
    for canonical_id in sorted(writer_ids):
        members = families[canonical_id]
        writers.append(
            {
                "canonical_writer_id": canonical_id,
                "member_writer_ids": list(members),
                "datasets": sorted(
                    {
                        dataset
                        for member in members
                        for dataset in datasets_by_writer[member]
                    }
                ),
            }
        )
    return {
        "seed": config.seed,
        "test_fraction": config.test_fraction,
        "count": len(writers),
        "writers": writers,
    }


@lru_cache(maxsize=1)
def _training_formatter() -> ParagraphFormatter:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    return ParagraphFormatter(
        TextEncoderConfig(
            base_vocab_size=len(vocabulary.base_to_id),
            shape_vocab_size=len(vocabulary.shape_to_id),
            tone_vocab_size=len(vocabulary.tone_to_id),
            case_vocab_size=len(vocabulary.case_to_id),
            class_vocab_size=len(vocabulary.class_to_id),
        )
    )


def _eligible_references(
    target: Mapping[str, object],
    references: Sequence[dict[str, object]],
    *,
    excluded_reference_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    excluded = excluded_reference_ids or set()
    return [
        reference
        for reference in references
        if eligible_reference(
            target,
            reference,
            excluded_reference_ids=excluded,
        )
    ]


def _select_reference(
    target: Mapping[str, object],
    references: Sequence[dict[str, object]],
    seed: int,
    *,
    excluded_reference_ids: set[str] | None = None,
) -> dict[str, object] | None:
    eligible = _eligible_references(
        target,
        references,
        excluded_reference_ids=excluded_reference_ids,
    )
    if not eligible:
        return None
    eligible.sort(key=lambda record: str(record["id"]))
    digest = hashlib.sha256(
        f"{seed}:{target['id']}".encode()
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(eligible)
    return eligible[index]


def _rejected_target(
    target: Mapping[str, object],
    *,
    stage: str,
    reason_code: str,
    reason: str,
) -> dict[str, object]:
    rejected = dict(target)
    rejected["rejection_stage"] = stage
    rejected["rejection_reason_code"] = reason_code
    rejected["rejection_reason"] = reason
    return rejected


def _partition_generator_targets(
    targets: Sequence[dict[str, object]],
    references_by_writer: Mapping[
        str,
        Sequence[dict[str, object]],
    ],
    *,
    stage: str,
    synthetic: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    formatter = _training_formatter()
    for target in targets:
        if not _supported_generator_text(str(target["text"])):
            rejected.append(
                _rejected_target(
                    target,
                    stage=stage,
                    reason_code="unsupported_grapheme",
                    reason=(
                        "Transcript chứa grapheme ngoài Vietnamese "
                        "factorizer/vocabulary."
                    ),
                )
            )
            continue
        try:
            formatter.format(
                str(target["text"]),
                preserve_physical_lines=True,
            )
        except (TypeError, ValueError) as error:
            rejected.append(
                _rejected_target(
                    target,
                    stage=stage,
                    reason_code="formatter_contract",
                    reason=str(error),
                )
            )
            continue

        excluded_reference_ids = (
            set(excluded_source_line_ids(target)) if synthetic else set()
        )

        canonical_id = str(target["canonical_writer_id"])
        eligible = _eligible_references(
            target,
            references_by_writer.get(canonical_id, ()),
            excluded_reference_ids=excluded_reference_ids,
        )
        if not eligible:
            rejected.append(
                _rejected_target(
                    target,
                    stage=stage,
                    reason_code="no_valid_reference",
                    reason=(
                        "Không có real line reference cùng canonical writer, "
                        "khác nội dung và không thuộc source_line_ids."
                    ),
                )
            )
            continue

        formatted_target = dict(target)
        formatted_target["formatter_mode"] = "physical_lines"
        accepted.append(formatted_target)
    return accepted, rejected


def _references_by_writer(
    references: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for reference in references:
        grouped[str(reference["canonical_writer_id"])].append(reference)
    return dict(grouped)


def _build_stage_outputs(
    manifests: Mapping[str, Sequence[dict[str, object]]],
    writer_map: Mapping[str, str],
    families: Mapping[str, tuple[str, ...]],
    train_writers: set[str],
    test_writers: set[str],
    config: SplitConfig,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    writer_files = {
        "writers/train.json": _writer_payload(
            train_writers,
            families,
            manifests,
            config,
        ),
        "writers/test.json": _writer_payload(
            test_writers,
            families,
            manifests,
            config,
        ),
    }

    jsonl_files: dict[str, list[dict[str, object]]] = {}
    jsonl_files["autokl/train_paragraphs.jsonl"] = _filter_records(
        manifests,
        REAL_DATASETS,
        level="paragraph",
        writers=train_writers,
        writer_map=writer_map,
    )
    jsonl_files["autokl/test_paragraphs.jsonl"] = _filter_records(
        manifests,
        REAL_DATASETS,
        level="paragraph",
        writers=test_writers,
        writer_map=writer_map,
    )
    jsonl_files["htr/train_lines.jsonl"] = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="line",
        writers=train_writers,
        writer_map=writer_map,
    )
    jsonl_files["htr/train_words.jsonl"] = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="word",
        writers=train_writers,
        writer_map=writer_map,
    )
    jsonl_files["htr/test_lines.jsonl"] = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="line",
        writers=test_writers,
        writer_map=writer_map,
    )
    jsonl_files["htr/test_words.jsonl"] = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="word",
        writers=test_writers,
        writer_map=writer_map,
    )

    rejected_targets: list[dict[str, object]] = []

    iam_targets = _filter_records(
        manifests,
        ("iam",),
        level="paragraph",
        writers=train_writers,
        writer_map=writer_map,
    )
    iam_references = _filter_records(
        manifests,
        ("iam",),
        level="line",
        writers=train_writers,
        writer_map=writer_map,
    )
    cvl_targets = _filter_records(
        manifests,
        ("cvl",),
        level="paragraph",
        writers=train_writers,
        writer_map=writer_map,
    )
    cvl_references = [
        record
        for record in _filter_records(
            manifests,
            ("cvl",),
            level="line",
            writers=train_writers,
            writer_map=writer_map,
        )
        if _supported_generator_text(str(record["text"]))
    ]
    pretrain_reference_pool = sorted(
        iam_references + cvl_references,
        key=lambda record: str(record["id"]),
    )
    pretrain_targets, pretrain_rejected = _partition_generator_targets(
        iam_targets + cvl_targets,
        _references_by_writer(pretrain_reference_pool),
        stage="pretrain",
    )
    rejected_targets.extend(pretrain_rejected)
    pretrain_writers = {
        str(record["canonical_writer_id"])
        for record in pretrain_targets
    }
    jsonl_files["vietparadiff/pretrain_targets.jsonl"] = sorted(
        pretrain_targets,
        key=lambda record: str(record["id"]),
    )
    jsonl_files["vietparadiff/pretrain_references.jsonl"] = [
        reference
        for reference in pretrain_reference_pool
        if reference["canonical_writer_id"] in pretrain_writers
    ]

    real_targets = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="paragraph",
        writers=train_writers,
        writer_map=writer_map,
    )
    synthetic_targets = _filter_records(
        manifests,
        ("uithwdb_augmented",),
        level="paragraph",
        writers=train_writers,
        writer_map=writer_map,
    )
    finetune_reference_pool = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="line",
        writers=train_writers,
        writer_map=writer_map,
    )
    references_by_id = {
        str(reference["id"]): reference
        for reference in finetune_reference_pool
    }
    for record in synthetic_targets:
        augmentation = record.get("augmentation")
        if (
            record.get("synthetic") is not True
            or not isinstance(augmentation, dict)
            or augmentation.get("type") != "line_stitch"
        ):
            raise ValueError(
                "UIT-HWDB augmented records phải là line_stitch synthetic."
            )
        source_line_ids = augmentation.get("source_line_ids")
        if (
            not isinstance(source_line_ids, list)
            or not source_line_ids
            or not all(
                isinstance(source_id, str) and source_id
                for source_id in source_line_ids
            )
        ):
            raise ValueError(
                f"Synthetic target {record['id']} phải có source_line_ids."
            )
        for source_id in source_line_ids:
            source = references_by_id.get(source_id)
            if source is None:
                raise ValueError(
                    f"Synthetic target {record['id']} tham chiếu source line "
                    f"không tồn tại trong train references: {source_id}."
                )
            if (
                source["canonical_writer_id"]
                != record["canonical_writer_id"]
            ):
                raise ValueError(
                    f"Synthetic target {record['id']} và source line "
                    f"{source_id} khác canonical writer."
                )
    finetune_references_by_writer = _references_by_writer(
        finetune_reference_pool
    )
    real_targets, real_rejected = _partition_generator_targets(
        real_targets,
        finetune_references_by_writer,
        stage="finetune_real",
    )
    synthetic_targets, synthetic_rejected = _partition_generator_targets(
        synthetic_targets,
        finetune_references_by_writer,
        stage="finetune_synthetic",
        synthetic=True,
    )
    rejected_targets.extend(real_rejected)
    rejected_targets.extend(synthetic_rejected)
    finetune_writers = {
        str(record["canonical_writer_id"])
        for record in real_targets + synthetic_targets
    }
    jsonl_files[
        "vietparadiff/finetune_targets_real.jsonl"
    ] = real_targets
    jsonl_files[
        "vietparadiff/finetune_targets_synthetic.jsonl"
    ] = synthetic_targets
    jsonl_files[
        "vietparadiff/finetune_references.jsonl"
    ] = [
        reference
        for reference in finetune_reference_pool
        if reference["canonical_writer_id"] in finetune_writers
    ]

    test_targets = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="paragraph",
        writers=test_writers,
        writer_map=writer_map,
    )
    test_references = _filter_records(
        manifests,
        VIETNAMESE_DATASETS,
        level="line",
        writers=test_writers,
        writer_map=writer_map,
    )
    test_references_by_writer = _references_by_writer(test_references)
    test_targets, test_rejected = _partition_generator_targets(
        test_targets,
        test_references_by_writer,
        stage="test",
    )
    rejected_targets.extend(test_rejected)

    pairs = []
    for target in test_targets:
        canonical_id = str(target["canonical_writer_id"])
        reference = _select_reference(
            target,
            test_references_by_writer[canonical_id],
            config.seed,
        )
        if reference is None:
            raise RuntimeError(
                f"Target {target['id']} đã qua coverage validation nhưng "
                "không chọn được reference."
            )
        pair_number = len(pairs) + 1
        pairs.append(
            {
                "pair_id": f"test_{pair_number:06d}",
                "canonical_writer_id": canonical_id,
                "target_id": target["id"],
                "target_image": target["image"],
                "target_text": target["text"],
                "reference_id": reference["id"],
                "reference_image": reference["image"],
            }
        )
    jsonl_files["vietparadiff/test_pairs.jsonl"] = pairs
    jsonl_files["vietparadiff/rejected_targets.jsonl"] = sorted(
        rejected_targets,
        key=lambda record: (
            str(record["rejection_stage"]),
            str(record["id"]),
        ),
    )
    return writer_files, jsonl_files


def _write_outputs(
    output_root: Path,
    writer_files: Mapping[str, object],
    jsonl_files: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    overwrite: bool,
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Split output already exists: {output_root}; use overwrite=True."
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            dir=output_root.parent,
        )
    )
    try:
        for relative_path, payload in writer_files.items():
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for relative_path, records in jsonl_files.items():
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def create_data_splits(
    config: SplitConfig | None = None,
) -> dict[str, int]:
    """Create every stage manifest and return record counts by output file."""
    config = config or SplitConfig()
    manifests = {
        dataset: _load_manifest(
            config.data_root / dataset / "manifest.jsonl",
            dataset,
        )
        for dataset in ALL_DATASETS
    }
    writer_map, families = _canonical_writer_map(manifests)
    train_writers, test_writers = _writer_splits(
        families,
        config.test_fraction,
        config.seed,
    )
    writer_files, jsonl_files = _build_stage_outputs(
        manifests,
        writer_map,
        families,
        train_writers,
        test_writers,
        config,
    )
    _write_outputs(
        config.output_root,
        writer_files,
        jsonl_files,
        overwrite=config.overwrite,
    )
    counts = {
        relative_path: len(records)
        for relative_path, records in jsonl_files.items()
    }
    counts["writers/train.json"] = len(train_writers)
    counts["writers/test.json"] = len(test_writers)
    return counts
