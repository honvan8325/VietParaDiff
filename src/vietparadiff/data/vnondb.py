"""Build a normalized offline-image dataset from processed VNOnDB exports."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from vietparadiff.data.build_provenance import (
    BUILDER_CONFIGS,
    BuildIssues,
    write_build_report,
)
from vietparadiff.data.image_utils import save_normalized_image
from vietparadiff.data.paragraph_labels import (
    join_paragraph_lines,
    split_indexed_line_stem,
)
from vietparadiff.cli_logging import get_logger

__all__ = ["build_vnondb_dataset"]

logger = get_logger(__name__)

RAW = Path("data/raw/VNOnDB")
PROCESSED = RAW / "Data_processed"

OUT = Path("data/vnondb")
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.jsonl"

LEVEL_DIRS = {
    "word": PROCESSED / "InkData_word_processed",
    "line": PROCESSED / "InkData_line_processed",
    "paragraph": PROCESSED / "InkData_paragraph_processed",
}


def build_vnondb_dataset() -> None:
    """Rebuild VNOnDB paragraph, line, and word samples.

    Every source PNG is paired with a same-stem UTF-8 transcript file. Valid
    pairs are converted to width-limited grayscale PNG images and recorded in
    ``data/vnondb/manifest.jsonl``.

    Raises:
        FileNotFoundError: If any required level directory is missing.

    Warning:
        ``data/vnondb`` is deleted before the new dataset is written.
    """
    logger.info("Build VNOnDB dataset")
    issues = BuildIssues()

    for level, source_dir in LEVEL_DIRS.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Required VNOnDB folder not found: {source_dir}")

    paragraph_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for text_source in sorted(LEVEL_DIRS["line"].glob("*.txt")):
        try:
            text = text_source.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as error:
            logger.warning(f"Cannot read line transcript {text_source}: {error}")
            issues.hard(
                text_source.stem,
                "line_transcript_decode_failure",
                text_source,
                str(error),
            )
            continue
        if not text:
            logger.warning(f"Empty line transcript: {text_source}")
            issues.hard(
                text_source.stem,
                "empty_line_transcript",
                text_source,
            )
            continue
        try:
            paragraph_stem, line_index = split_indexed_line_stem(
                text_source.stem
            )
        except ValueError as error:
            logger.warning(f"{error} File: {text_source}")
            issues.hard(
                text_source.stem,
                "invalid_line_identifier",
                text_source,
                str(error),
            )
            continue
        paragraph_lines[paragraph_stem].append((line_index, text))

    if OUT.exists():
        logger.warning(f"Removing existing output folder: {OUT}")
        shutil.rmtree(OUT)

    IMAGES.mkdir(parents=True)

    manifest = []
    image_jobs = []

    # Build one deterministic job list across all sample granularities.

    for level, source_dir in LEVEL_DIRS.items():
        image_paths = sorted(source_dir.glob("*.png"))

        logger.info(f"{level.capitalize()} images: {len(image_paths)}")

        for image_path in image_paths:
            image_jobs.append(
                (
                    level,
                    image_path,
                )
            )

    logger.info(f"Total source images: {len(image_jobs)}")

    # Normalize image/transcript pairs into the shared manifest schema.

    for level, image_source in tqdm(
        image_jobs,
        desc="Process VNOnDB",
        unit="image",
        dynamic_ncols=True,
    ):
        text_source = image_source.with_suffix(".txt")

        if not text_source.exists():
            logger.warning(f"Transcript not found: {text_source}")
            issues.hard(
                image_source.stem,
                "missing_transcript",
                image_source,
            )
            continue

        try:
            text = text_source.read_text(
                encoding="utf-8-sig",
            ).strip()
        except UnicodeDecodeError as error:
            logger.warning(f"Cannot decode transcript {text_source}: {error}")
            issues.hard(
                image_source.stem,
                "transcript_decode_failure",
                text_source,
                str(error),
            )
            continue

        if not text:
            logger.warning(f"Empty transcript: {text_source}")
            issues.hard(
                image_source.stem,
                "empty_transcript",
                text_source,
            )
            continue

        raw_id = image_source.stem
        if level == "paragraph":
            indexed_lines = paragraph_lines.get(raw_id)
            if not indexed_lines:
                logger.warning(
                    f"Line transcripts not found for paragraph: {image_source}"
                )
                issues.reject(
                    image_source.stem,
                    "missing_native_line_alignment",
                    image_source,
                )
                continue
            line_texts = [
                line_text
                for _, line_text in sorted(
                    indexed_lines,
                    key=lambda item: item[0],
                )
            ]
            try:
                text = join_paragraph_lines(text, line_texts)
            except ValueError as error:
                logger.warning(f"{error} Paragraph: {text_source}")
                issues.hard(
                    image_source.stem,
                    "paragraph_line_label_conflict",
                    text_source,
                    str(error),
                )
                continue

        # VNOnDB embeds the writer key in the first two underscore-separated
        # fields. For example, ``20140603_0003_BCCTC_tg_0`` belongs to writer
        # ``20140603_0003``.
        id_parts = raw_id.split("_")

        if len(id_parts) < 2:
            logger.warning(f"Cannot determine writer ID: {image_source.name}")
            issues.hard(
                raw_id,
                "invalid_writer_identifier",
                image_source,
            )
            continue

        writer_raw = "_".join(id_parts[:2])
        writer_id = f"vnondb_{writer_raw}"

        sample_id = f"vnondb_{level}_{raw_id}"

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
            issues.hard(
                sample_id,
                "image_decode_failure",
                image_source,
                str(error),
            )
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

    # Write records only after all image/transcript validation is complete.

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
    write_build_report(
        dataset="vnondb",
        raw_root=next(iter(LEVEL_DIRS.values())).parent,
        manifest=MANIFEST,
        output=OUT / "build_report.json",
        builder_config=BUILDER_CONFIGS["vnondb"],
        accepted_count=len(manifest),
        expected_rejections=issues.expected_rejections,
        hard_errors=issues.hard_errors,
        warnings=issues.warnings,
    )
