"""Tests for training-time processors, datasets, samplers, and collators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from vietparadiff.data.pipeline import (
    AutoKLDataset,
    HTRDataset,
    HTRImageProcessor,
    HTRVocabulary,
    HeightBucketBatchSampler,
    ParagraphImageProcessor,
    ReferenceImageProcessor,
    VietParaDiffCollator,
    VietParaDiffDataset,
    WidthBucketBatchSampler,
    collate_autokl,
    collate_htr,
)
from vietparadiff.models import (
    GraphemeVocabulary,
    ParagraphFormatter,
    TextEncoderConfig,
)


def make_handwriting_image(
    path: Path,
    *,
    width: int,
    height: int,
    lines: int = 1,
) -> None:
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    line_height = max(3, (height - 40) // lines)
    for line in range(lines):
        y = 20 + line * line_height
        draw.line(
            (20, y, width - 20, min(height - 2, y + line_height // 3)),
            fill=0,
            width=max(2, line_height // 12),
        )
        draw.ellipse(
            (
                30,
                max(0, y - 4),
                min(width - 1, 50),
                min(height - 1, y + 12),
            ),
            outline=32,
            width=2,
        )
    image.save(path)
    image.close()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_reference_ids(
    samples: list[dict[str, object]],
) -> list[str]:
    return [str(sample["reference_id"]) for sample in samples]


def make_text_components() -> tuple[
    GraphemeVocabulary,
    ParagraphFormatter,
]:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    config = TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
    )
    return vocabulary, ParagraphFormatter(config)


def test_three_image_processors_preserve_contracts(tmp_path: Path) -> None:
    paragraph_path = tmp_path / "paragraph.png"
    line_path = tmp_path / "line.png"
    long_line_path = tmp_path / "long_line.png"
    make_handwriting_image(
        paragraph_path,
        width=640,
        height=300,
        lines=3,
    )
    make_handwriting_image(line_path, width=420, height=90)
    make_handwriting_image(long_line_path, width=1400, height=80)

    paragraph = ParagraphImageProcessor()(paragraph_path)
    paragraph_image = paragraph["image"]
    assert isinstance(paragraph_image, torch.Tensor)
    assert paragraph_image.shape == (
        1,
        paragraph["height_bucket"],
        1024,
    )
    assert paragraph["height_bucket"] in {
        384,
        512,
        640,
        768,
        896,
        1024,
        1280,
    }
    assert torch.isfinite(paragraph_image).all()
    assert paragraph_image.min() >= -1.0
    assert paragraph_image.max() <= 1.0
    assert (paragraph_image == 1.0).any()

    htr = HTRImageProcessor()(line_path)
    htr_image = htr["image"]
    assert isinstance(htr_image, torch.Tensor)
    assert htr_image.shape == (1, 64, htr["valid_width"])
    assert torch.equal(htr_image[:, :4], torch.ones_like(htr_image[:, :4]))
    assert torch.equal(htr_image[:, -4:], torch.ones_like(htr_image[:, -4:]))

    reference = ReferenceImageProcessor()(long_line_path)
    reference_image = reference["image"]
    valid_mask = reference["valid_mask"]
    assert reference_image.shape[0:2] == (1, 256)
    assert reference_image.shape[-1] <= 1536
    assert reference_image.shape[-1] % 32 == 0
    assert valid_mask.dtype == torch.bool
    assert valid_mask.shape == reference_image.shape
    assert valid_mask.any()
    assert torch.equal(
        reference_image.masked_select(~valid_mask),
        torch.ones_like(reference_image.masked_select(~valid_mask)),
    )


def test_htr_vocabulary_is_train_only_and_collate_pads_width(
    tmp_path: Path,
) -> None:
    line_image = tmp_path / "line.png"
    word_image = tmp_path / "word.png"
    test_image = tmp_path / "test.png"
    make_handwriting_image(line_image, width=500, height=80)
    make_handwriting_image(word_image, width=180, height=70)
    make_handwriting_image(test_image, width=260, height=70)
    train_manifest = tmp_path / "train.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    write_jsonl(
        train_manifest,
        [
            {
                "id": "line_1",
                "image": str(line_image),
                "text": "Chữ á",
                "level": "line",
            },
            {
                "id": "word_1",
                "image": str(word_image),
                "text": "đẹp",
                "level": "word",
            },
        ],
    )
    write_jsonl(
        test_manifest,
        [
            {
                "id": "test_1",
                "image": str(test_image),
                "text": "ỹ",
                "level": "word",
            }
        ],
    )

    vocabulary = HTRVocabulary.build_from_manifests(train_manifest)
    assert "ỹ" not in vocabulary.raw_to_id
    train_dataset = HTRDataset(train_manifest, vocabulary)
    test_dataset = HTRDataset(test_manifest, vocabulary)
    assert test_dataset[0]["raw_targets"].tolist() == [1]

    batch = collate_htr([train_dataset[0], train_dataset[1]])
    images = batch["images"]
    valid_widths = batch["valid_widths"]
    assert isinstance(images, torch.Tensor)
    assert images.shape[:3] == (2, 1, 64)
    assert images.shape[-1] % 4 == 0
    assert isinstance(valid_widths, torch.Tensor)
    assert (valid_widths <= images.shape[-1]).all()
    assert batch["raw_targets"].shape[0] == 2
    assert batch["target_lengths"].tolist() == [
        len("Chữ á"),
        len("đẹp"),
    ]
    shorter = int(valid_widths.argmin())
    assert torch.equal(
        images[shorter, :, :, valid_widths[shorter] :],
        torch.ones_like(
            images[shorter, :, :, valid_widths[shorter] :]
        ),
    )

    sampler = WidthBucketBatchSampler(
        train_dataset,
        batch_size=2,
        bucket_width=4096,
        shuffle=False,
    )
    assert list(sampler) == [[0, 1]]

    vocabulary_path = tmp_path / "htr_vocabulary.json"
    vocabulary.save(vocabulary_path)
    assert HTRVocabulary.load(vocabulary_path) == vocabulary


def test_htr_ctc_padding_preserves_detached_vietnamese_marks(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "marked.png"
    image = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 14, 24, 27), fill=0)
    draw.rectangle((14, 5, 16, 7), fill=0)  # detached tone/dot fixture
    image.save(image_path)
    image.close()
    manifest = tmp_path / "htr.jsonl"
    text = "ậ" * 24
    write_jsonl(
        manifest,
        [
            {
                "id": "marked",
                "image": str(image_path),
                "text": text,
                "level": "line",
            }
        ],
    )
    vocabulary = HTRVocabulary.build_from_manifests(manifest)
    processed = HTRImageProcessor()(image_path)
    sample = HTRDataset(manifest, vocabulary)[0]
    tensor = sample["image"]
    assert isinstance(tensor, torch.Tensor)
    assert int(sample["valid_width"]) > int(processed["valid_width"])
    required = vocabulary.minimum_input_width(text)
    assert int(sample["valid_width"]) == required
    original = tensor[:, :, : int(processed["valid_width"])]
    foreground = original < 0.5
    assert foreground[:, :28].any()  # detached upper mark remains
    assert foreground[:, 28:].any()  # main body remains
    assert torch.equal(
        tensor[:, :, int(processed["valid_width"]) :],
        torch.ones_like(tensor[:, :, int(processed["valid_width"]) :]),
    )


@pytest.mark.parametrize(
    ("text", "mark_box"),
    [
        ("á", (15, 3, 17, 5)),
        ("à", (12, 3, 14, 5)),
        ("ả", (15, 2, 17, 5)),
        ("ã", (13, 2, 18, 4)),
        ("ạ", (15, 27, 17, 29)),
        ("â", (12, 3, 18, 6)),
        ("ă", (12, 3, 18, 6)),
        ("ơ", (25, 10, 28, 13)),
        ("ư", (25, 10, 28, 13)),
        ("i", (15, 3, 17, 5)),
        (".", (27, 26, 29, 28)),
        (",", (27, 26, 29, 30)),
        ("!", (27, 4, 29, 8)),
        ("?", (26, 3, 30, 8)),
    ],
)
def test_htr_preprocessing_never_discards_meaningful_detached_components(
    tmp_path: Path,
    text: str,
    mark_box: tuple[int, int, int, int],
) -> None:
    path = tmp_path / f"fixture_{ord(text)}.png"
    image = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 12, 23, 24), fill=0)
    draw.rectangle(mark_box, fill=0)
    image.save(path)
    image.close()
    processed = HTRImageProcessor()(path)["image"]
    assert isinstance(processed, torch.Tensor)
    foreground = processed[0] < 0.5

    # Count 4-connected foreground regions after preprocessing. The fixture
    # deliberately contains a main body and a semantic detached mark.
    remaining = foreground.clone()
    components = 0
    while remaining.any():
        components += 1
        start = torch.nonzero(remaining, as_tuple=False)[0]
        stack = [(int(start[0]), int(start[1]))]
        remaining[stack[0]] = False
        while stack:
            row, column = stack.pop()
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < remaining.shape[0]
                    and 0 <= next_column < remaining.shape[1]
                    and bool(remaining[next_row, next_column])
                ):
                    remaining[next_row, next_column] = False
                    stack.append((next_row, next_column))
    assert components >= 2, f"Detached component bị mất cho {text!r}."


def test_autokl_height_sampler_and_collate(tmp_path: Path) -> None:
    images = []
    records = []
    for index, height in enumerate((220, 220, 650)):
        path = tmp_path / f"paragraph_{index}.png"
        make_handwriting_image(
            path,
            width=700,
            height=height,
            lines=2,
        )
        images.append(path)
        records.append(
            {
                "id": f"paragraph_{index}",
                "image": str(path),
                "level": "paragraph",
                "canonical_writer_id": f"writer_{index}",
            }
        )
    manifest = tmp_path / "autokl.jsonl"
    write_jsonl(manifest, records)
    dataset = AutoKLDataset(manifest)
    sampler = HeightBucketBatchSampler(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    for indices in sampler:
        buckets = {dataset.height_bucket(index) for index in indices}
        assert len(buckets) == 1
        batch = collate_autokl([dataset[index] for index in indices])
        assert batch["images"].shape[2] == batch["height_bucket"]
        assert batch["images"].shape[-1] == 1024


def test_vietparadiff_train_and_test_data_contracts(
    tmp_path: Path,
) -> None:
    target_image = tmp_path / "target.png"
    reference_source = tmp_path / "reference_source.png"
    reference_valid = tmp_path / "reference_valid.png"
    reference_third = tmp_path / "reference_third.png"
    make_handwriting_image(
        target_image,
        width=720,
        height=320,
        lines=2,
    )
    make_handwriting_image(reference_source, width=360, height=80)
    make_handwriting_image(reference_valid, width=520, height=90)
    make_handwriting_image(reference_third, width=440, height=85)
    target_manifest = tmp_path / "targets.jsonl"
    reference_manifest = tmp_path / "references.jsonl"
    test_pairs = tmp_path / "test_pairs.jsonl"
    writer = "vn_writer_test"
    target_text = "Dòng nguồn\nDòng đích"
    write_jsonl(
        target_manifest,
        [
            {
                "id": "target_real",
                "image": str(target_image),
                "text": target_text,
                "canonical_writer_id": writer,
                "formatter_mode": "physical_lines",
            },
            {
                "id": "target_synthetic",
                "image": str(target_image),
                "text": target_text,
                "canonical_writer_id": writer,
                "formatter_mode": "physical_lines",
                "synthetic": True,
                "augmentation": {
                    "type": "line_stitch",
                    "source_line_ids": ["line_source"],
                },
            },
        ],
    )
    write_jsonl(
        reference_manifest,
        [
            {
                "id": "line_source",
                "image": str(reference_source),
                "text": "Nội dung ngoài target",
                "canonical_writer_id": writer,
                "level": "line",
            },
            {
                "id": "line_valid",
                "image": str(reference_valid),
                "text": "Tham chiếu độc lập",
                "canonical_writer_id": writer,
                "level": "line",
            },
            {
                "id": "line_third",
                "image": str(reference_third),
                "text": "Một tham chiếu thứ ba",
                "canonical_writer_id": writer,
                "level": "line",
            },
        ],
    )
    write_jsonl(
        test_pairs,
        [
            {
                "pair_id": "test_000001",
                "canonical_writer_id": writer,
                "target_id": "target_real",
                "target_image": str(target_image),
                "target_text": target_text,
                "reference_id": "line_source",
                "reference_image": str(reference_source),
            }
        ],
    )
    vocabulary, formatter = make_text_components()
    train_dataset = VietParaDiffDataset(
        target_manifest,
        mode="train",
        reference_manifest=reference_manifest,
        formatter=formatter,
    )

    train_dataset.set_epoch(3)
    synthetic = train_dataset[1]
    assert synthetic["reference_id"] not in synthetic["source_line_ids"]
    assert synthetic["canonical_writer_id"] == writer
    first = train_dataset[0]["reference_id"]
    second = train_dataset[0]["reference_id"]
    assert first == second
    selected = set()
    for epoch in range(10):
        train_dataset.set_epoch(epoch)
        selected.add(train_dataset[0]["reference_id"])
    assert len(selected) > 1
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
        collate_fn=collect_reference_ids,
    )
    try:
        train_dataset.set_epoch(6)
        direct = train_dataset[0]["reference_id"]
        from_worker = next(iter(loader))[0]
        assert from_worker == direct
        train_dataset.set_epoch(7)
        assert next(iter(loader))[0] == train_dataset[0]["reference_id"]
    finally:
        iterator = loader._iterator
        if iterator is not None:
            iterator._shutdown_workers()

    samples = [train_dataset[0], synthetic]
    assert {
        train_dataset.height_bucket(index) for index in range(2)
    } == {samples[0]["height_bucket"]}
    sampler = HeightBucketBatchSampler(
        train_dataset,
        batch_size=2,
        shuffle=False,
    )
    assert list(sampler) == [[0, 1]]
    batch = VietParaDiffCollator(formatter, vocabulary)(samples)
    assert batch["target_images"].shape[0:2] == (2, 1)
    assert batch["target_images"].shape[-1] == 1024
    assert batch["reference_images"].shape[0:3] == (2, 1, 256)
    assert batch["reference_images"].shape[-1] % 32 == 0
    assert (
        batch["reference_valid_mask"].shape
        == batch["reference_images"].shape
    )
    assert batch["reference_valid_mask"].dtype == torch.bool
    assert batch["canonical_line_slots"].shape == (
        2,
        8,
        batch["output_height"] // 8,
        128,
    )
    assert batch["graphemes"].base_ids.shape[0] == 2

    test_dataset = VietParaDiffDataset(
        test_pairs,
        mode="test",
        formatter=formatter,
    )
    test_sample = test_dataset[0]
    assert test_sample["target_id"] == "target_real"
    assert test_sample["reference_id"] == "line_source"


def test_generator_processor_uses_exact_formatter_height(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "tall_paragraph.png"
    make_handwriting_image(
        image_path,
        width=1024,
        height=1100,
        lines=2,
    )
    processor = ParagraphImageProcessor()

    processed = processor(
        image_path,
        output_height=384,
    )

    assert processed["height_bucket"] == 384
    assert processed["image"].shape == (1, 384, 1024)


def test_training_dataset_rejects_target_without_reference(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    make_handwriting_image(image, width=300, height=80)
    targets = tmp_path / "targets.jsonl"
    references = tmp_path / "references.jsonl"
    write_jsonl(
        targets,
        [
            {
                "id": "target",
                "image": str(image),
                "text": "Toàn bộ nội dung",
                "canonical_writer_id": "writer",
                "formatter_mode": "physical_lines",
            }
        ],
    )
    write_jsonl(
        references,
        [
            {
                "id": "line",
                "image": str(image),
                "text": "Toàn bộ nội dung",
                "canonical_writer_id": "writer",
                "level": "line",
            }
        ],
    )
    _, formatter = make_text_components()
    with pytest.raises(ValueError, match="không có reference hợp lệ"):
        VietParaDiffDataset(
            targets,
            mode="train",
            reference_manifest=references,
            formatter=formatter,
        )
