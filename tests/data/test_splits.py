"""Tests for deterministic stage-specific writer-disjoint data splits."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from vietparadiff.data.splits import (
    SplitConfig,
    _eligible_references,
    _load_manifest,
    _supported_generator_text,
    create_data_splits,
)
from PIL import Image
from vietparadiff.models import (
    GraphemeVocabulary,
    ParagraphFormatter,
    TextEncoderConfig,
)


def record(
    sample_id: str,
    writer_id: str,
    level: str,
    text: str,
    dataset: str,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "image": f"data/{dataset}/images/{sample_id}.png",
        "text": text,
        "writer_id": writer_id,
        "level": level,
        "width": 128,
        "height": 64,
    }


def write_manifest(
    root: Path,
    dataset: str,
    records: list[dict[str, object]],
) -> None:
    directory = root / dataset
    directory.mkdir(parents=True)
    with (directory / "manifest.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def make_fixture(root: Path) -> None:
    cvl = []
    iam = []
    uit = []
    vnon = []
    augmented = []
    for index in (1, 2):
        cvl_writer = f"cvl_{index:04d}"
        cvl.extend(
            (
                record(
                    f"cvl_p{index}_ok",
                    cvl_writer,
                    "paragraph",
                    f"Supported paragraph {index}",
                    "cvl",
                ),
                record(
                    f"cvl_p{index}_umlaut",
                    cvl_writer,
                    "paragraph",
                    f"Mailüfterl {index}",
                    "cvl",
                ),
                record(
                    f"cvl_p{index}_too_many_lines",
                    cvl_writer,
                    "paragraph",
                    "\n".join(f"line {line}" for line in range(9)),
                    "cvl",
                ),
                record(
                    f"cvl_l{index}",
                    cvl_writer,
                    "line",
                    f"Supported reference {index}",
                    "cvl",
                ),
            )
        )

        iam_writer = f"iam_{index:03d}"
        iam.extend(
            (
                record(
                    f"iam_p{index}",
                    iam_writer,
                    "paragraph",
                    f"IAM target {index}\nsecond line {index}",
                    "iam",
                ),
                record(
                    f"iam_l{index}",
                    iam_writer,
                    "line",
                    f"IAM reference {index}",
                    "iam",
                ),
            )
        )

        uit_writer = f"uithwdb_{index}"
        vnon_writer = f"vnondb_writer_{index}"
        paragraph_text = f"Mục tiêu {index}\ndòng thứ hai {index}"
        for dataset, writer_id, destination in (
            ("uithwdb", uit_writer, uit),
            ("vnondb", vnon_writer, vnon),
        ):
            destination.extend(
                (
                    record(
                        f"{dataset}_p{index}",
                        writer_id,
                        "paragraph",
                        paragraph_text,
                        dataset,
                    ),
                    record(
                        f"{dataset}_p{index}_no_reference",
                        writer_id,
                        "paragraph",
                        (
                            f"Mục tiêu {index}\n"
                            f"Nội dung tham chiếu {index}"
                        ),
                        dataset,
                    ),
                    record(
                        f"{dataset}_source_l{index}",
                        writer_id,
                        "line",
                        f"Mục tiêu {index}",
                        dataset,
                    ),
                    record(
                        f"{dataset}_reference_l{index}",
                        writer_id,
                        "line",
                        f"Nội dung tham chiếu {index}",
                        dataset,
                    ),
                    record(
                        f"{dataset}_w{index}",
                        writer_id,
                        "word",
                        f"từ{index}",
                        dataset,
                    ),
                )
            )

        synthetic = record(
            f"aug_p{index}",
            uit_writer,
            "paragraph",
            f"Tổng hợp {index}\nDòng ghép {index}",
            "uithwdb_augmented",
        )
        synthetic["synthetic"] = True
        synthetic["augmentation"] = {
            "type": "line_stitch",
            "source_dataset": "uithwdb",
            "source_line_ids": [f"uithwdb_source_l{index}"],
        }
        augmented.append(synthetic)

    write_manifest(root, "cvl", cvl)
    write_manifest(root, "iam", iam)
    write_manifest(root, "uithwdb", uit)
    write_manifest(root, "vnondb", vnon)
    write_manifest(root, "uithwdb_augmented", augmented)
    metadata = root / "metadata"
    metadata.mkdir()
    evidence = [f"{index}" * 64 for index in (1, 2)]
    candidate_path = metadata / "vietnamese_writer_crosswalk_candidates.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "candidate_only",
                "uithwdb_writer_ids": ["uithwdb_1", "uithwdb_2"],
                "vnondb_writer_ids": [
                    "vnondb_writer_1",
                    "vnondb_writer_2",
                ],
                "candidates": [
                    {
                        "vnondb_writer_id": f"vnondb_writer_{index}",
                        "evidence_sha256": value,
                    }
                    for index, value in enumerate(evidence, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "vietnamese_writer_crosswalk.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "candidate_report": str(candidate_path),
                "candidate_report_sha256": hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
                "approved": [
                    {
                        "uithwdb_writer_id": f"uithwdb_{index}",
                        "vnondb_writer_id": f"vnondb_writer_{index}",
                        "evidence_sha256": evidence[index - 1],
                    }
                    for index in (1, 2)
                ],
                "proven_independent": [],
                "unresolved": [],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )


def test_generator_text_filter_rejects_unsupported_marks() -> None:
    assert _supported_generator_text("Tiếng Việt đầy đủ.")
    assert not _supported_generator_text("Mailüfterl")


def test_create_data_splits_is_writer_disjoint_and_stage_correct(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    output = tmp_path / "splits"

    counts = create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=output,
            writer_crosswalk=(
                data_root
                / "metadata/vietnamese_writer_crosswalk.json"
            ),
            test_fraction=0.5,
            seed=7,
        )
    )

    train_payload = json.loads(
        (output / "writers/train.json").read_text(encoding="utf-8")
    )
    test_payload = json.loads(
        (output / "writers/test.json").read_text(encoding="utf-8")
    )
    train_writers = {
        item["canonical_writer_id"] for item in train_payload["writers"]
    }
    test_writers = {
        item["canonical_writer_id"] for item in test_payload["writers"]
    }
    assert train_writers.isdisjoint(test_writers)
    assert len(train_writers) == len(test_writers) == 3

    vietnamese_families = [
        item
        for item in train_payload["writers"] + test_payload["writers"]
        if item["canonical_writer_id"].startswith("vn_writer_")
    ]
    assert len(vietnamese_families) == 2
    assert all(len(item["member_writer_ids"]) == 2 for item in vietnamese_families)

    autokl = read_jsonl(output / "autokl/train_paragraphs.jsonl")
    assert autokl
    assert {item["level"] for item in autokl} == {"paragraph"}
    assert {item["dataset"] for item in autokl} <= {
        "cvl",
        "iam",
        "uithwdb",
        "vnondb",
    }

    htr_lines = read_jsonl(output / "htr/train_lines.jsonl")
    htr_words = read_jsonl(output / "htr/train_words.jsonl")
    assert {item["dataset"] for item in htr_lines + htr_words} <= {
        "uithwdb",
        "vnondb",
    }
    assert {item["level"] for item in htr_lines} == {"line"}
    assert {item["level"] for item in htr_words} == {"word"}

    pretrain_targets = read_jsonl(
        output / "vietparadiff/pretrain_targets.jsonl"
    )
    assert {item["dataset"] for item in pretrain_targets} == {"cvl", "iam"}
    assert not any("ü" in item["text"] for item in pretrain_targets)
    assert {
        item["formatter_mode"] for item in pretrain_targets
    } == {"physical_lines"}

    real_targets = read_jsonl(
        output / "vietparadiff/finetune_targets_real.jsonl"
    )
    synthetic_targets = read_jsonl(
        output / "vietparadiff/finetune_targets_synthetic.jsonl"
    )
    references = read_jsonl(
        output / "vietparadiff/finetune_references.jsonl"
    )
    assert {item["dataset"] for item in real_targets} == {
        "uithwdb",
        "vnondb",
    }
    assert {item["dataset"] for item in synthetic_targets} == {
        "uithwdb_augmented"
    }
    assert all(item["synthetic"] is True for item in synthetic_targets)
    assert {item["level"] for item in references} == {"line"}
    reference_groups: dict[str, list[dict[str, object]]] = {}
    for reference in references:
        reference_groups.setdefault(
            str(reference["canonical_writer_id"]),
            [],
        ).append(reference)
    for target in real_targets + synthetic_targets:
        source_ids = set(
            target.get("augmentation", {}).get("source_line_ids", [])
        )
        eligible = _eligible_references(
            target,
            reference_groups[str(target["canonical_writer_id"])],
            excluded_reference_ids=source_ids,
        )
        assert eligible
        assert all(item["id"] not in source_ids for item in eligible)

    pairs = read_jsonl(output / "vietparadiff/test_pairs.jsonl")
    assert pairs
    assert len(pairs) == counts["vietparadiff/test_pairs.jsonl"]
    for pair in pairs:
        assert pair["canonical_writer_id"] in test_writers
        reference = next(
            item
            for item in read_jsonl(output / "htr/test_lines.jsonl")
            if item["id"] == pair["reference_id"]
        )
        assert (
            reference["canonical_writer_id"]
            == pair["canonical_writer_id"]
        )
        assert " ".join(reference["text"].split()) != " ".join(
            pair["target_text"].split()
        )

    vocabulary = GraphemeVocabulary.default_vietnamese()
    formatter = ParagraphFormatter(
        TextEncoderConfig(
            base_vocab_size=len(vocabulary.base_to_id),
            shape_vocab_size=len(vocabulary.shape_to_id),
            tone_vocab_size=len(vocabulary.tone_to_id),
            case_vocab_size=len(vocabulary.case_to_id),
            class_vocab_size=len(vocabulary.class_to_id),
        )
    )
    for target in pretrain_targets + real_targets + synthetic_targets:
        formatter.format(
            str(target["text"]),
            preserve_physical_lines=True,
        )
    for pair in pairs:
        formatter.format(
            str(pair["target_text"]),
            preserve_physical_lines=True,
        )

    rejected = read_jsonl(
        output / "vietparadiff/rejected_targets.jsonl"
    )
    rejected_ids = {str(item["id"]) for item in rejected}
    assert sum(item.endswith("_umlaut") for item in rejected_ids) == 1
    assert sum(
        item.endswith("_too_many_lines") for item in rejected_ids
    ) == 1
    assert {
        item["rejection_reason_code"] for item in rejected
    } >= {
        "unsupported_grapheme",
        "formatter_contract",
    }

    test_member_ids = {
        member
        for writer in test_payload["writers"]
        for member in writer["member_writer_ids"]
    }
    expected_test_targets = {
        str(item["id"])
        for dataset in ("uithwdb", "vnondb")
        for item in read_jsonl(data_root / dataset / "manifest.jsonl")
        if item["level"] == "paragraph"
        and item["writer_id"] in test_member_ids
    }
    paired_test_targets = {str(item["target_id"]) for item in pairs}
    rejected_test_targets = {
        str(item["id"])
        for item in rejected
        if item["rejection_stage"] == "test"
    }
    assert paired_test_targets.isdisjoint(rejected_test_targets)
    assert (
        paired_test_targets | rejected_test_targets
        == expected_test_targets
    )


def test_split_output_is_deterministic_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = dict(
        data_root=data_root,
        writer_crosswalk=(
            data_root / "metadata/vietnamese_writer_crosswalk.json"
        ),
        test_fraction=0.5,
        seed=13,
    )

    create_data_splits(SplitConfig(output_root=first, **common))
    create_data_splits(SplitConfig(output_root=second, **common))

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files

    with pytest.raises(FileExistsError, match="overwrite=True"):
        create_data_splits(SplitConfig(output_root=first, **common))

    create_data_splits(
        SplitConfig(output_root=first, overwrite=True, **common)
    )
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == first_files


def test_unresolved_cross_dataset_writers_are_quarantined(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    crosswalk_path = (
        data_root / "metadata/vietnamese_writer_crosswalk.json"
    )
    payload = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    removed = payload["approved"].pop()
    payload["unresolved"] = [
        removed["uithwdb_writer_id"],
        removed["vnondb_writer_id"],
    ]
    crosswalk_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "splits"
    create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=output,
            writer_crosswalk=crosswalk_path,
            test_fraction=0.5,
            seed=7,
        )
    )
    unresolved = set(payload["unresolved"])
    for manifest in output.rglob("*.jsonl"):
        if manifest.name == "rejected_targets.jsonl":
            continue
        for record in read_jsonl(manifest):
            assert record.get("writer_id") not in unresolved
    coverage = json.loads(
        (output / "writers/crosswalk_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(coverage["unresolved"]) == unresolved


def test_exact_duplicate_policy_preserves_semantic_levels(
    tmp_path: Path,
) -> None:
    image = tmp_path / "same.png"
    Image.new("L", (8, 8), 255).save(image)
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "id": "line-a",
            "image": str(image),
            "text": "cấp",
            "writer_id": "writer",
            "level": "line",
        },
        {
            "id": "line-b",
            "image": str(image),
            "text": "cấp",
            "writer_id": "writer",
            "level": "line",
        },
        {
            "id": "paragraph-a",
            "image": str(image),
            "text": "cấp",
            "writer_id": "writer",
            "level": "paragraph",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    loaded = _load_manifest(manifest, "fixture")
    assert {record["level"] for record in loaded} == {
        "line",
        "paragraph",
    }
    line = next(record for record in loaded if record["level"] == "line")
    assert line["duplicate_provenance"]["duplicate_ids"] == ["line-b"]

    records[1]["text"] = "khác"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label conflict"):
        _load_manifest(manifest, "fixture")
