"""Create synthetic paragraphs by stitching line samples from one dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TypedDict

from PIL import Image
from tqdm import tqdm

from vietparadiff.data.image_utils import save_normalized_image
from vietparadiff.cli_logging import get_logger

logger = get_logger(__name__)

CANVAS_WIDTH = 1024
MIN_SIDE_MARGIN = 20
MAX_SIDE_MARGIN = 80
MIN_VERTICAL_MARGIN = 20
MAX_VERTICAL_MARGIN = 80
MIN_LINE_GAP = 8
MAX_LINE_GAP = 30

MIN_LINES = 2
MAX_LINES = 8

MIN_WORDS_PER_LINE = 3
MIN_WRITER_WORD_RATIO = 0.60
MIN_PARAGRAPH_WORD_RATIO = 0.65
MIN_HEIGHT_RATIO = 0.60
MAX_HEIGHT_RATIO = 1.50
MAX_SELECTION_ATTEMPTS = 100
BACKGROUND_PERCENTILE = 0.90
BACKGROUND_TOLERANCE = 12

# Line counts follow the requested 40% / 40% / 20% distribution.
LINE_COUNT_BUCKETS = (
    ((2, 3, 4), 0.40),
    ((5, 6), 0.40),
    ((7, 8), 0.20),
)


@dataclass(frozen=True)
class LineSample:
    """One normalized line eligible for paragraph stitching."""

    id: str
    image: Path
    text: str
    writer_id: str
    word_count: int
    width: int
    height: int


class AugmentationMetadata(TypedDict):
    """Provenance stored with every synthetic paragraph."""

    type: str
    source_dataset: str
    source_line_ids: list[str]


class SyntheticParagraphRecord(TypedDict):
    """Manifest schema for one synthetic paragraph."""

    id: str
    image: str
    text: str
    writer_id: str
    level: str
    width: int
    height: int
    synthetic: bool
    augmentation: AugmentationMetadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the source dataset and augmentation options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create synthetic paragraphs from lines belonging to the same "
            "dataset writer."
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset directory name under --data-root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing <dataset>/manifest.jsonl (default: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Synthetic dataset directory. Defaults to "
            "<data-root>/augmented_<dataset>."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Number of synthetic paragraphs to create (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and layout (default: 42).",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=MIN_LINES,
        help="Minimum lines per paragraph (default: 2).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=MAX_LINES,
        help="Maximum lines per paragraph, capped at 8 (default: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset after generation succeeds.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, output: Path) -> None:
    """Validate ranges and protect source directories from deletion."""
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    if args.min_lines < MIN_LINES:
        raise ValueError("--min-lines must be at least 2")

    if args.max_lines > MAX_LINES:
        raise ValueError("--max-lines must not exceed 8")

    if args.max_lines < args.min_lines:
        raise ValueError("--max-lines must be greater than or equal to --min-lines")

    if Path(args.dataset).name != args.dataset or args.dataset in {".", ".."}:
        raise ValueError(f"Dataset must be a directory name: {args.dataset}")

    output = output.resolve()
    data_root = args.data_root.resolve()
    source_root = data_root / args.dataset

    if output in {
        Path.cwd().resolve(),
        Path("/").resolve(),
        data_root,
        source_root,
    }:
        raise ValueError(f"Refusing to use protected output directory: {output}")

    if source_root in output.parents:
        raise ValueError(f"Output must not be inside source data: {output}")


def load_lines_by_writer(
    data_root: Path,
    dataset: str,
) -> dict[str, list[LineSample]]:
    """Load every valid line directly from one dataset manifest."""
    manifest_path = data_root / dataset / "manifest.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    lines_by_writer: dict[str, list[LineSample]] = defaultdict(list)

    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {manifest_path}:{line_number}: "
                    f"{error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object at {manifest_path}:{line_number}"
                )

            if record.get("level") != "line":
                continue

            sample_id = record.get("id")
            image_value = record.get("image")
            text = record.get("text")
            writer_id = record.get("writer_id")

            if not all(
                isinstance(value, str) and value
                for value in (sample_id, image_value, text, writer_id)
            ):
                logger.warning(
                    f"Invalid line sample at {manifest_path}:{line_number}"
                )
                continue

            assert isinstance(sample_id, str)
            assert isinstance(image_value, str)
            assert isinstance(text, str)
            assert isinstance(writer_id, str)

            text = text.strip()

            if not text:
                logger.warning(
                    f"Empty line transcript at {manifest_path}:{line_number}"
                )
                continue

            image_path = Path(image_value)

            if not image_path.is_file():
                logger.warning(f"Line image not found: {image_path}")
                continue

            width = record.get("width")
            height = record.get("height")

            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or width <= 0
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height <= 0
            ):
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except OSError as error:
                    logger.warning(
                        f"Cannot read image size {image_path}: {error}"
                    )
                    continue

            lines_by_writer[writer_id].append(
                LineSample(
                    id=sample_id,
                    image=image_path,
                    text=text,
                    writer_id=writer_id,
                    word_count=len(text.split()),
                    width=width,
                    height=height,
                )
            )

    return dict(lines_by_writer)


def sample_line_count(
    rng: random.Random,
    min_lines: int,
    max_lines: int,
) -> int:
    """Sample a line count using the 40% / 40% / 20% bucket weights."""
    buckets = []

    for counts, weight in LINE_COUNT_BUCKETS:
        valid_counts = tuple(
            count for count in counts if min_lines <= count <= max_lines
        )

        if valid_counts:
            buckets.append((valid_counts, weight))

    selected_counts = rng.choices(
        population=[counts for counts, _ in buckets],
        weights=[weight for _, weight in buckets],
        k=1,
    )[0]
    return rng.choice(selected_counts)


def select_lines(
    lines_by_writer: dict[str, list[LineSample]],
    line_count: int,
    rng: random.Random,
) -> list[LineSample]:
    """Select non-repeating lines with compatible text length and height."""
    candidate_lines_by_writer: dict[str, list[LineSample]] = {}

    for writer_id, lines in lines_by_writer.items():
        median_word_count = median(line.word_count for line in lines)
        minimum_writer_words = max(
            MIN_WORDS_PER_LINE,
            round(median_word_count * MIN_WRITER_WORD_RATIO),
        )
        candidates = [
            line
            for line in lines
            if line.word_count >= minimum_writer_words
        ]

        if len(candidates) >= line_count:
            candidate_lines_by_writer[writer_id] = candidates

    eligible_writers = list(candidate_lines_by_writer)

    if not eligible_writers:
        raise RuntimeError(
            f"No writer has {line_count} lines with compatible text lengths"
        )

    for _ in range(MAX_SELECTION_ATTEMPTS):
        writer_id = rng.choice(eligible_writers)
        selected = rng.sample(
            candidate_lines_by_writer[writer_id],
            k=line_count,
        )
        average_words = sum(
            line.word_count for line in selected
        ) / len(selected)
        minimum_paragraph_words = max(
            MIN_WORDS_PER_LINE,
            round(average_words * MIN_PARAGRAPH_WORD_RATIO),
        )

        if any(
            line.word_count < minimum_paragraph_words
            for line in selected
        ):
            continue

        median_height = median(line.height for line in selected)

        if any(
            line.height < median_height * MIN_HEIGHT_RATIO
            or line.height > median_height * MAX_HEIGHT_RATIO
            for line in selected
        ):
            continue

        # The manifest does not guarantee source-page relationships, so the
        # compatible lines become a newly ordered paragraph.
        rng.shuffle(selected)
        return selected

    raise RuntimeError(
        "Cannot find a compatible group of lines "
        f"for a {line_count}-line paragraph"
    )


def estimate_background_level(grayscale: Image.Image) -> int:
    """Estimate a line's paper tone from its 90th-percentile pixel value."""
    target = grayscale.width * grayscale.height * BACKGROUND_PERCENTILE
    accumulated = 0

    for value, count in enumerate(grayscale.histogram()):
        accumulated += count

        if accumulated >= target:
            return value

    return 255


def whiten_line_background(grayscale: Image.Image) -> Image.Image:
    """Map each source paper tone to white while retaining antialiased ink."""
    background_level = estimate_background_level(grayscale)
    white_threshold = max(1, background_level - BACKGROUND_TOLERANCE)
    lookup_table = [
        255
        if value >= white_threshold
        else round(value * 255 / white_threshold)
        for value in range(256)
    ]
    return grayscale.point(lookup_table)


def normalize_line_for_canvas(
    image: Image.Image,
    max_width: int,
    target_height: int,
) -> Image.Image:
    """Whiten and resize a line toward one shared paragraph height."""
    grayscale = image.convert("L")

    try:
        normalized = whiten_line_background(grayscale)
    finally:
        grayscale.close()

    height_scale = target_height / normalized.height
    width_scale = max_width / normalized.width
    scale = min(height_scale, width_scale)
    resized_width = max(1, round(normalized.width * scale))
    resized_height = max(1, round(normalized.height * scale))

    if (resized_width, resized_height) == normalized.size:
        return normalized

    resized = normalized.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    normalized.close()
    return resized


def create_paragraph(
    line_samples: Sequence[LineSample],
    rng: random.Random,
) -> tuple[Image.Image, str]:
    """Compose prepared lines on a white 1024-pixel-wide paragraph canvas."""
    left_margin = rng.randint(MIN_SIDE_MARGIN, MAX_SIDE_MARGIN)
    right_margin = rng.randint(MIN_SIDE_MARGIN, MAX_SIDE_MARGIN)
    top_margin = rng.randint(MIN_VERTICAL_MARGIN, MAX_VERTICAL_MARGIN)
    bottom_margin = rng.randint(MIN_VERTICAL_MARGIN, MAX_VERTICAL_MARGIN)
    content_width = CANVAS_WIDTH - left_margin - right_margin

    if content_width <= 0:
        raise ValueError(f"Invalid content width: {content_width}")

    fitted_heights = (
        sample.height * min(1.0, content_width / sample.width)
        for sample in line_samples
    )
    target_line_height = max(1, round(median(fitted_heights)))
    prepared_lines = []
    line_gaps = []

    try:
        for index, sample in enumerate(line_samples):
            with Image.open(sample.image) as image:
                prepared_lines.append(
                    normalize_line_for_canvas(
                        image,
                        max_width=content_width,
                        target_height=target_line_height,
                    )
                )

            if index < len(line_samples) - 1:
                line_gaps.append(rng.randint(MIN_LINE_GAP, MAX_LINE_GAP))

        canvas_height = (
            top_margin
            + bottom_margin
            + sum(image.height for image in prepared_lines)
            + sum(line_gaps)
        )
        canvas = Image.new(
            mode="L",
            size=(CANVAS_WIDTH, canvas_height),
            color=255,
        )
        y = top_margin

        for index, line_image in enumerate(prepared_lines):
            canvas.paste(line_image, (left_margin, y))
            y += line_image.height

            if index < len(line_gaps):
                y += line_gaps[index]

        paragraph_text = "\n".join(sample.text for sample in line_samples)
        return canvas, paragraph_text
    finally:
        for image in prepared_lines:
            image.close()


def prepare_staging_directory(
    output: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Create an isolated staging directory beside the final output."""
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )

    staging = output.with_name(f".{output.name}.building-{os.getpid()}")

    if staging.exists():
        if not overwrite:
            raise FileExistsError(f"Staging directory already exists: {staging}")

        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink()

    images = staging / "images"
    images.mkdir(parents=True)
    return staging, images


def create_synthetic_record(
    sample_index: int,
    dataset: str,
    selected_lines: Sequence[LineSample],
    output: Path,
    staging_images: Path,
    rng: random.Random,
) -> SyntheticParagraphRecord:
    """Create and save one synthetic paragraph with source-line provenance."""
    sample_id = f"aug_{dataset}_{sample_index + 1:08d}"
    filename = f"{sample_id}.png"
    staging_image = staging_images / filename
    final_image = output / "images" / filename
    paragraph, paragraph_text = create_paragraph(selected_lines, rng)

    try:
        width, height = save_normalized_image(
            paragraph,
            staging_image,
            level="paragraph",
        )
    finally:
        paragraph.close()

    return {
        "id": sample_id,
        "image": final_image.as_posix(),
        "text": paragraph_text,
        "writer_id": selected_lines[0].writer_id,
        "level": "paragraph",
        "width": width,
        "height": height,
        "synthetic": True,
        "augmentation": {
            "type": "line_stitch",
            "source_dataset": dataset,
            "source_line_ids": [line.id for line in selected_lines],
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Generate synthetic paragraphs from the selected dataset manifest."""
    args = parse_args(argv)
    output = (
        args.output
        if args.output is not None
        else args.data_root / f"augmented_{args.dataset}"
    )
    validate_args(args, output)
    lines_by_writer = load_lines_by_writer(
        data_root=args.data_root,
        dataset=args.dataset,
    )
    lines_by_writer = {
        writer_id: lines
        for writer_id, lines in lines_by_writer.items()
        if len(lines) >= args.min_lines
    }

    if not lines_by_writer:
        raise RuntimeError(
            f"No writer has enough valid lines in dataset: {args.dataset}"
        )

    available_max_lines = min(
        args.max_lines,
        max(len(lines) for lines in lines_by_writer.values()),
    )
    staging, staging_images = prepare_staging_directory(
        output=output,
        overwrite=args.overwrite,
    )
    staging_manifest = staging / "manifest.jsonl"
    rng = random.Random(args.seed)
    created = 0
    attempts = 0
    max_attempts = args.samples * 5

    logger.info(f"Create synthetic paragraph dataset from: {args.dataset}")
    logger.info(f"Eligible writers: {len(lines_by_writer)}")
    logger.info(f"Requested paragraphs: {args.samples}")

    try:
        with staging_manifest.open("w", encoding="utf-8") as manifest:
            with tqdm(
                total=args.samples,
                desc=f"Stitch {args.dataset} paragraphs",
                unit="paragraph",
                dynamic_ncols=True,
            ) as progress:
                while created < args.samples and attempts < max_attempts:
                    attempts += 1
                    line_count = sample_line_count(
                        rng=rng,
                        min_lines=args.min_lines,
                        max_lines=available_max_lines,
                    )
                    selected_lines = select_lines(
                        lines_by_writer=lines_by_writer,
                        line_count=line_count,
                        rng=rng,
                    )

                    try:
                        record = create_synthetic_record(
                            sample_index=created,
                            dataset=args.dataset,
                            selected_lines=selected_lines,
                            output=output,
                            staging_images=staging_images,
                            rng=rng,
                        )
                    except (OSError, ValueError) as error:
                        logger.warning(
                            f"Cannot create synthetic paragraph on attempt "
                            f"{attempts}: {error}"
                        )
                        continue

                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    created += 1
                    progress.update()

        if created != args.samples:
            raise RuntimeError(
                f"Created {created}/{args.samples} paragraphs after "
                f"{attempts} attempts"
            )

        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()

        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(f"Synthetic paragraphs: {created}")
    logger.info(f"Writers: {len(lines_by_writer)}")
    logger.info(f"Manifest: {output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
