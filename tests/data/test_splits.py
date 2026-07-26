"""Tests for deterministic UIT-HWDB-only Vietnamese data splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vietparadiff.data.splits import (
    SplitConfig,
    _eligible_references,
    _load_manifest,
    _supported_generator_text,
    create_data_splits,
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
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_fixture(root: Path) -> None:
    cvl: list[dict[str, object]] = []
    iam: list[dict[str, object]] = []
    uit: list[dict[str, object]] = []
    augmented: list[dict[str, object]] = []
    for index in (1, 2):
        cvl_writer = f"cvl_{index:04d}"
        cvl.extend(
            (
                record(
                    f"cvl_p{index}",
                    cvl_writer,
                    "paragraph",
                    f"Supported paragraph {index}",
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
        uit.extend(
            (
                record(
                    f"uithwdb_p{index}",
                    uit_writer,
                    "paragraph",
                    f"Mục tiêu {index}\ndòng thứ hai {index}",
                    "uithwdb",
                ),
                record(
                    f"uithwdb_source_l{index}",
                    uit_writer,
                    "line",
                    f"Dòng nguồn {index}",
                    "uithwdb",
                ),
                record(
                    f"uithwdb_reference_l{index}",
                    uit_writer,
                    "line",
                    f"Nội dung tham chiếu {index}",
                    "uithwdb",
                ),
                record(
                    f"uithwdb_w{index}",
                    uit_writer,
                    "word",
                    f"từ{index}",
                    "uithwdb",
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
    write_manifest(root, "uithwdb_augmented", augmented)


def split_fixture(
    tmp_path: Path,
    *,
    seed: int = 7,
) -> tuple[Path, Path, dict[str, int]]:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    output = tmp_path / "splits"
    counts = create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=output,
            test_fraction=0.5,
            seed=seed,
        )
    )
    return data_root, output, counts


def test_generator_text_filter_rejects_unsupported_marks() -> None:
    assert _supported_generator_text("Tiếng Việt đầy đủ.")
    assert not _supported_generator_text("Mailüfterl")


def test_split_is_writer_disjoint_and_uses_correct_stage_levels(
    tmp_path: Path,
) -> None:
    _, output, counts = split_fixture(tmp_path)
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
    for item in train_payload["writers"] + test_payload["writers"]:
        assert item["member_writer_ids"] == [
            item["canonical_writer_id"]
        ]

    autokl = read_jsonl(output / "autokl/train_paragraphs.jsonl")
    assert {item["dataset"] for item in autokl} <= {
        "cvl",
        "iam",
        "uithwdb",
    }
    assert {item["level"] for item in autokl} == {"paragraph"}

    htr = read_jsonl(output / "htr/train_lines.jsonl")
    htr += read_jsonl(output / "htr/train_words.jsonl")
    assert {item["dataset"] for item in htr} == {"uithwdb"}
    assert {item["level"] for item in htr} == {"line", "word"}

    real = read_jsonl(
        output / "vietparadiff/finetune_targets_real.jsonl"
    )
    synthetic = read_jsonl(
        output / "vietparadiff/finetune_targets_synthetic.jsonl"
    )
    references = read_jsonl(
        output / "vietparadiff/finetune_references.jsonl"
    )
    assert {item["dataset"] for item in real} == {"uithwdb"}
    assert {item["dataset"] for item in synthetic} == {
        "uithwdb_augmented"
    }
    assert {item["dataset"] for item in references} == {"uithwdb"}
    assert {item["level"] for item in references} == {"line"}

    reference_groups: dict[str, list[dict[str, object]]] = {}
    for reference in references:
        reference_groups.setdefault(
            str(reference["canonical_writer_id"]),
            [],
        ).append(reference)
    for target in synthetic:
        source_ids = set(
            target["augmentation"]["source_line_ids"]
        )
        eligible = _eligible_references(
            target,
            reference_groups[str(target["canonical_writer_id"])],
            excluded_reference_ids=source_ids,
        )
        assert eligible
        assert all(item["id"] not in source_ids for item in eligible)

    pairs = read_jsonl(output / "vietparadiff/test_pairs.jsonl")
    assert len(pairs) == counts["vietparadiff/test_pairs.jsonl"]
    assert pairs
    for pair in pairs:
        assert str(pair["target_id"]).startswith("uithwdb_p")
        assert str(pair["reference_id"]).startswith("uithwdb_")
        assert str(pair["canonical_writer_id"]).startswith("uithwdb_")
    assert all(
        item["dataset"] != "uithwdb_augmented"
        for item in read_jsonl(output / "autokl/test_paragraphs.jsonl")
    )


def test_uithwdb_writer_id_is_preserved_as_canonical_identity(
    tmp_path: Path,
) -> None:
    _, output, _ = split_fixture(tmp_path)
    for manifest in output.rglob("*.jsonl"):
        if manifest.name in {"test_pairs.jsonl", "rejected_targets.jsonl"}:
            continue
        for item in read_jsonl(manifest):
            if item["dataset"].startswith("uithwdb"):
                assert item["canonical_writer_id"] == item["writer_id"]


def test_split_output_is_stable_under_input_reordering(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=first,
            test_fraction=0.5,
            seed=13,
        )
    )
    for dataset in ("cvl", "iam", "uithwdb", "uithwdb_augmented"):
        path = data_root / dataset / "manifest.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=second,
            test_fraction=0.5,
            seed=13,
        )
    )
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


def test_split_requires_overwrite_for_existing_output(
    tmp_path: Path,
) -> None:
    data_root, output, _ = split_fixture(tmp_path)
    config = SplitConfig(
        data_root=data_root,
        output_root=output,
        test_fraction=0.5,
        seed=7,
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        create_data_splits(config)
    create_data_splits(
        SplitConfig(
            data_root=data_root,
            output_root=output,
            test_fraction=0.5,
            seed=7,
            overwrite=True,
        )
    )


def test_unknown_augmented_writer_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    path = data_root / "uithwdb_augmented" / "manifest.jsonl"
    records = read_jsonl(path)
    records[0]["writer_id"] = "uithwdb_unknown"
    write_manifest(data_root, "uithwdb_augmented", records)
    with pytest.raises(ValueError, match="Synthetic writer"):
        create_data_splits(
            SplitConfig(
                data_root=data_root,
                output_root=tmp_path / "splits",
                test_fraction=0.5,
            )
        )


def test_writer_id_collision_between_datasets_is_rejected(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    make_fixture(data_root)
    path = data_root / "iam" / "manifest.jsonl"
    records = read_jsonl(path)
    for item in records:
        if item["writer_id"] == "iam_001":
            item["writer_id"] = "cvl_0001"
    write_manifest(data_root, "iam", records)
    with pytest.raises(ValueError, match="Writer ID collision"):
        create_data_splits(
            SplitConfig(
                data_root=data_root,
                output_root=tmp_path / "splits",
                test_fraction=0.5,
            )
        )


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
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    loaded = _load_manifest(manifest, "fixture")
    assert {item["level"] for item in loaded} == {
        "line",
        "paragraph",
    }
    line = next(item for item in loaded if item["level"] == "line")
    assert line["duplicate_provenance"]["duplicate_ids"] == ["line-b"]

    records[1]["text"] = "khác"
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label conflict"):
        _load_manifest(manifest, "fixture")
