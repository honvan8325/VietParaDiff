"""Build a normalized dataset from UIT-HWDB's pre-segmented images."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from vietparadiff.data.image_utils import save_normalized_image
from vietparadiff.data.paragraph_labels import (
    align_sequential_paragraph_lines,
    flatten_whitespace,
    join_paragraph_lines,
    split_indexed_line_stem,
)
from vietparadiff.cli_logging import get_logger

__all__ = ["build_uithwdb_dataset"]

logger = get_logger(__name__)

RAW = Path("data/raw/UIT_HWDB")

OUT = Path("data/uithwdb")
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.jsonl"

LEVEL_DIRS = {
    "word": RAW / "UIT_HWDB_word",
    "line": RAW / "UIT_HWDB_line",
    "paragraph": RAW / "UIT_HWDB_paragraph",
}

VNON_PROCESSED = Path("data/raw/VNOnDB/Data_processed")
VNON_LINE_DIR = VNON_PROCESSED / "InkData_line_processed"
VNON_PARAGRAPH_DIR = VNON_PROCESSED / "InkData_paragraph_processed"


def _load_vnondb_line_fallbacks() -> dict[
    tuple[str, ...],
    list[dict[str, tuple[str, ...]]],
]:
    """Index VNOnDB line labels for UIT writers with incomplete line exports."""
    if not VNON_LINE_DIR.exists() or not VNON_PARAGRAPH_DIR.exists():
        return {}

    indexed_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for text_path in sorted(VNON_LINE_DIR.glob("*.txt")):
        try:
            text = text_path.read_text(encoding="utf-8-sig").strip()
            paragraph_stem, line_index = split_indexed_line_stem(
                text_path.stem
            )
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if text:
            indexed_lines[paragraph_stem].append((line_index, text))

    paragraphs_by_writer: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text_path in sorted(VNON_PARAGRAPH_DIR.glob("*.txt")):
        try:
            text = text_path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            continue
        parts = text_path.stem.split("_")
        if text and len(parts) >= 2:
            paragraphs_by_writer["_".join(parts[:2])].append(
                (text_path.stem, text)
            )

    fallbacks: dict[
        tuple[str, ...],
        list[dict[str, tuple[str, ...]]],
    ] = defaultdict(list)
    for paragraphs in paragraphs_by_writer.values():
        signature = tuple(
            sorted(flatten_whitespace(text) for _, text in paragraphs)
        )
        segments: dict[str, tuple[str, ...]] = {}
        for paragraph_stem, paragraph_text in paragraphs:
            line_texts = tuple(
                line_text
                for _, line_text in sorted(
                    indexed_lines.get(paragraph_stem, ()),
                    key=lambda item: item[0],
                )
            )
            try:
                join_paragraph_lines(paragraph_text, line_texts)
            except ValueError:
                continue
            segments[flatten_whitespace(paragraph_text)] = line_texts
        if segments:
            fallbacks[signature].append(segments)
    return dict(fallbacks)


def build_uithwdb_dataset() -> None:
    """Rebuild UIT-HWDB paragraph, line, and word samples.

    UIT-HWDB already provides separate level and writer directories. This
    builder reads each writer's ``label.json``, converts referenced images to
    width-limited grayscale PNG, assigns globally namespaced IDs, and writes
    the normalized JSONL manifest to ``data/uithwdb/manifest.jsonl``.

    Warning:
        ``data/uithwdb`` is deleted before the new dataset is written.
    """
    logger.info("Build UIT-HWDB dataset")

    if OUT.exists():
        logger.warning(f"Removing existing output folder: {OUT}")
        shutil.rmtree(OUT)

    IMAGES.mkdir(parents=True)

    manifest = []

    # Collect work at writer-directory granularity. Sorting numeric directory
    # names makes output ordering deterministic across filesystems.

    writer_jobs = []

    for level, level_root in LEVEL_DIRS.items():
        for split_folder in ["train_data", "test_data"]:
            split_root = level_root / split_folder

            if not split_root.exists():
                logger.warning(f"Data folder not found: {split_root}")
                continue

            writer_dirs = sorted(
                (path for path in split_root.glob("[0-9]*") if path.is_dir()),
                key=lambda path: int(path.name),
            )

            for writer_dir in writer_dirs:
                writer_jobs.append(
                    (
                        level,
                        split_folder,
                        writer_dir,
                    )
                )

    logger.info(f"Writer folders: {len(writer_jobs)}")

    labels_by_job: dict[tuple[str, str, str], dict[str, object]] = {}
    for level, split_folder, writer_dir in writer_jobs:
        key = (level, split_folder, writer_dir.name)
        label_path = writer_dir / "label.json"
        if not label_path.exists():
            logger.warning(f"Label not found: {label_path}")
            continue
        try:
            with label_path.open("r", encoding="utf-8") as file:
                labels = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Cannot read label file {label_path}: {error}")
            continue
        if not isinstance(labels, dict):
            logger.warning(f"Invalid label format: {label_path}")
            continue
        labels_by_job[key] = labels

    paragraph_lines: dict[tuple[str, str, str], tuple[str, ...]] = {}
    fallback_index = _load_vnondb_line_fallbacks()
    writer_keys = sorted(
        {
            (split_folder, writer_dir.name)
            for _, split_folder, writer_dir in writer_jobs
        },
        key=lambda item: (item[0], int(item[1])),
    )
    for split_folder, writer_raw in writer_keys:
        line_labels = labels_by_job.get(("line", split_folder, writer_raw))
        paragraph_labels = labels_by_job.get(
            ("paragraph", split_folder, writer_raw)
        )
        if paragraph_labels is None:
            continue
        if line_labels is None:
            line_labels = {}

        ordered_lines = sorted(
            (
                (name, text.strip())
                for name, text in line_labels.items()
                if isinstance(text, str) and text.strip()
            ),
            key=lambda item: int(Path(item[0]).stem),
        )
        ordered_paragraphs = sorted(
            (
                (name, text.strip())
                for name, text in paragraph_labels.items()
                if isinstance(text, str) and text.strip()
            ),
            key=lambda item: int(Path(item[0]).stem),
        )
        matches, unmatched = align_sequential_paragraph_lines(
            ordered_paragraphs,
            ordered_lines,
        )

        for image_name, line_texts in matches.items():
            paragraph_lines[
                (split_folder, writer_raw, image_name)
            ] = line_texts

        if unmatched:
            signature = tuple(
                sorted(
                    flatten_whitespace(text)
                    for _, text in ordered_paragraphs
                )
            )
            fallback_candidates = fallback_index.get(signature, ())
            if len(fallback_candidates) == 1:
                fallback = fallback_candidates[0]
                paragraph_text_by_name = dict(ordered_paragraphs)
                for image_name in unmatched:
                    line_texts = fallback.get(
                        flatten_whitespace(
                            paragraph_text_by_name[image_name]
                        )
                    )
                    if line_texts is not None:
                        paragraph_lines[
                            (split_folder, writer_raw, image_name)
                        ] = line_texts

    # Process every granularity through the same normalization path because
    # UIT-HWDB uses the same label-file format at all three levels.

    for level, split_folder, writer_dir in tqdm(
        writer_jobs,
        desc="Process UIT-HWDB",
        unit="writer",
        dynamic_ncols=True,
    ):
        writer_raw = writer_dir.name
        writer_id = f"uithwdb_{writer_raw}"

        label_path = writer_dir / "label.json"
        labels = labels_by_job.get((level, split_folder, writer_raw))
        if labels is None:
            continue

        image_names = sorted(
            labels.keys(),
            key=lambda name: int(Path(name).stem),
        )

        # label.json maps an image filename to its ground-truth transcript.
        # Iterating in numeric stem order produces a reproducible manifest.
        for image_name in image_names:
            text = labels[image_name]

            if not isinstance(text, str):
                logger.warning(f"Invalid transcript: {label_path} | {image_name}")
                continue

            text = text.strip()

            if not text:
                logger.warning(f"Empty transcript: {label_path} | {image_name}")
                continue
            if level == "paragraph":
                line_texts = paragraph_lines.get(
                    (split_folder, writer_raw, image_name)
                )
                if line_texts is None:
                    logger.warning(
                        "Line labels not found for paragraph: "
                        f"{label_path} | {image_name}"
                    )
                    continue
                try:
                    text = join_paragraph_lines(text, line_texts)
                except ValueError as error:
                    logger.warning(
                        f"{error} Paragraph: {label_path} | {image_name}"
                    )
                    continue

            image_source = writer_dir / image_name

            if not image_source.exists():
                logger.warning(f"Image not found: {image_source}")
                continue

            sample_raw = Path(image_name).stem

            sample_id = f"uithwdb_{level}_{writer_raw}_{sample_raw}"

            output_image = IMAGES / f"{sample_id}.png"

            try:
                with Image.open(image_source) as image:
                    image_width, image_height = save_normalized_image(
                        image,
                        output_image,
                        level=level,
                    )
            except OSError as error:
                logger.warning(f"Cannot process image {image_source}: {error}")
                continue

            manifest.append(
                {
                    "id": sample_id,
                    "image": output_image.as_posix(),
                    "text": text,
                    "writer_id": writer_id,
                    "level": level,
                    "width": image_width,
                    "height": image_height,
                }
            )

    # Keep Unicode text unescaped so Vietnamese transcripts remain readable
    # when the JSONL manifest is inspected directly.

    with MANIFEST.open("w", encoding="utf-8") as file:
        for sample in manifest:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    paragraph_count = sum(sample["level"] == "paragraph" for sample in manifest)

    line_count = sum(sample["level"] == "line" for sample in manifest)

    word_count = sum(sample["level"] == "word" for sample in manifest)

    writer_count = len({sample["writer_id"] for sample in manifest})

    logger.info(f"Writers: {writer_count}")
    logger.info(f"Paragraphs: {paragraph_count}")
    logger.info(f"Lines: {line_count}")
    logger.info(f"Words: {word_count}")
    logger.info(f"Total: {len(manifest)}")
    logger.info(f"Manifest: {MANIFEST}")
