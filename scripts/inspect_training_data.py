"""Render representative processed batches for visual data inspection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

from vietparadiff.data.pipeline import (
    AutoKLDataset,
    HTRDataset,
    HTRVocabulary,
    HeightBucketBatchSampler,
    VietParaDiffCollator,
    VietParaDiffDataset,
    collate_autokl,
    collate_htr,
)
from vietparadiff.models import (
    GraphemeVocabulary,
    ParagraphFormatter,
    TextEncoderConfig,
)


def _to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(
            f"Expected grayscale tensor [1,H,W], got {tuple(tensor.shape)}."
        )
    pixels = (
        tensor.detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .contiguous()
    )
    height, width = pixels.shape
    return Image.frombytes(
        "L",
        (width, height),
        bytes(pixels.untyped_storage()),
    )


def _save_grid(
    items: Sequence[tuple[torch.Tensor, str]],
    path: Path,
    *,
    columns: int = 2,
    panel_size: tuple[int, int] = (520, 340),
) -> None:
    if not items:
        raise ValueError(f"Không có item để render: {path.name}")
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new(
        "L",
        (columns * panel_size[0], rows * panel_size[1]),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    try:
        for index, (tensor, label) in enumerate(items):
            source = _to_image(tensor)
            try:
                thumbnail = ImageOps.contain(
                    source,
                    (panel_size[0] - 20, panel_size[1] - 45),
                    Image.Resampling.LANCZOS,
                )
            finally:
                source.close()
            x0 = (index % columns) * panel_size[0]
            y0 = (index // columns) * panel_size[1]
            canvas.paste(
                thumbnail,
                (
                    x0 + (panel_size[0] - thumbnail.width) // 2,
                    y0 + 30,
                ),
            )
            thumbnail.close()
            draw.text((x0 + 8, y0 + 8), label, fill=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    finally:
        canvas.close()


def _text_components() -> tuple[
    GraphemeVocabulary,
    ParagraphFormatter,
]:
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
    return vocabulary, formatter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process real split records and render six visual-check grids."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root containing splits/ and normalized images.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve image paths stored in manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_check"),
        help="Destination for six PNG inspection grids.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples shown per data family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic train reference selection.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size phải dương.")
    split_root = args.data_root / "splits"
    torch.manual_seed(args.seed)

    autokl = AutoKLDataset(
        split_root / "autokl/train_paragraphs.jsonl",
        image_root=args.image_root,
    )
    preferred = (
        ("uithwdb", 384),
        ("vnondb", 512),
        ("cvl", 896),
        ("iam", 1280),
    )
    auto_indices: list[int] = []
    for dataset_name, height_bucket in preferred:
        match = next(
            (
                index
                for index, record in enumerate(autokl.records)
                if record.get("dataset") == dataset_name
                and autokl.height_bucket(index) == height_bucket
            ),
            None,
        )
        if match is not None:
            auto_indices.append(match)
    if len(auto_indices) < args.batch_size:
        auto_indices.extend(
            index
            for index in range(len(autokl))
            if index not in auto_indices
        )
    auto_samples = [
        autokl[index]
        for index in auto_indices[: args.batch_size]
    ]
    for sample in auto_samples:
        collate_autokl([sample])
    _save_grid(
        [
            (
                sample["image"],
                f"{sample['sample_id']} · H={sample['height_bucket']}",
            )
            for sample in auto_samples
        ],
        args.output_dir / "autokl_batch.png",
    )

    htr_vocabulary = HTRVocabulary.build_from_manifests(
        (
            split_root / "htr/train_lines.jsonl",
            split_root / "htr/train_words.jsonl",
        )
    )
    htr_lines = HTRDataset(
        split_root / "htr/train_lines.jsonl",
        htr_vocabulary,
        image_root=args.image_root,
    )
    htr_words = HTRDataset(
        split_root / "htr/train_words.jsonl",
        htr_vocabulary,
        image_root=args.image_root,
    )
    line_samples = [
        htr_lines[index]
        for index in range(min(args.batch_size, len(htr_lines)))
    ]
    word_samples = [
        htr_words[index]
        for index in range(min(args.batch_size, len(htr_words)))
    ]
    line_batch = collate_htr(line_samples)
    word_batch = collate_htr(word_samples)
    _save_grid(
        [
            (image, sample_id)
            for image, sample_id in zip(
                line_batch["images"],
                line_batch["sample_ids"],
                strict=True,
            )
        ],
        args.output_dir / "htr_lines.png",
    )
    _save_grid(
        [
            (image, sample_id)
            for image, sample_id in zip(
                word_batch["images"],
                word_batch["sample_ids"],
                strict=True,
            )
        ],
        args.output_dir / "htr_words.png",
    )

    grapheme_vocabulary, formatter = _text_components()
    vietparadiff = VietParaDiffDataset(
        split_root / "vietparadiff/finetune_targets_real.jsonl",
        mode="train",
        reference_manifest=(
            split_root / "vietparadiff/finetune_references.jsonl"
        ),
        image_root=args.image_root,
        formatter=formatter,
        seed=args.seed,
    )
    viet_sampler = HeightBucketBatchSampler(
        vietparadiff,
        args.batch_size,
        shuffle=False,
    )
    pair_indices = next(iter(viet_sampler))
    pair_samples = [vietparadiff[index] for index in pair_indices]
    pair_batch = VietParaDiffCollator(
        formatter,
        grapheme_vocabulary,
    )(pair_samples)
    _save_grid(
        [
            (
                sample["reference_image"],
                f"ref:{sample['reference_id']}",
            )
            for sample in pair_samples
        ],
        args.output_dir / "style_references.png",
    )
    pair_items: list[tuple[torch.Tensor, str]] = []
    for sample in pair_samples:
        pair_items.extend(
            (
                (
                    sample["target_image"],
                    f"target:{sample['target_id']}",
                ),
                (
                    sample["reference_image"],
                    f"ref:{sample['reference_id']}",
                ),
            )
        )
    _save_grid(
        pair_items,
        args.output_dir / "vietparadiff_pairs.png",
    )
    canonical_items = []
    canonical_slots = pair_batch["canonical_line_slots"]
    for index, slots in enumerate(canonical_slots):
        weights = torch.arange(
            1,
            slots.shape[0] + 1,
            dtype=slots.dtype,
        )[:, None, None]
        visualization = 1.0 - 2.0 * (
            slots * weights / slots.shape[0]
        ).amax(dim=0, keepdim=True)
        canonical_items.append(
            (
                visualization,
                f"canonical:{pair_samples[index]['target_id']}",
            )
        )
    _save_grid(
        canonical_items,
        args.output_dir / "vietparadiff_canonical_slots.png",
    )

    for name in (
        "autokl_batch.png",
        "htr_lines.png",
        "htr_words.png",
        "style_references.png",
        "vietparadiff_pairs.png",
        "vietparadiff_canonical_slots.png",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
