"""Build a normalized dataset from UIT-HWDB's pre-segmented images."""

from __future__ import annotations

import json
import shutil
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
    align_sequential_paragraph_lines,
    join_paragraph_lines,
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
    issues = BuildIssues()

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
                issues.hard(
                    f"{level}:{split_folder}",
                    "missing_data_folder",
                    split_root,
                )
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
            issues.hard(
                f"{level}:{split_folder}:{writer_dir.name}",
                "missing_label_file",
                label_path,
            )
            continue
        try:
            with label_path.open("r", encoding="utf-8") as file:
                labels = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Cannot read label file {label_path}: {error}")
            issues.hard(
                f"{level}:{split_folder}:{writer_dir.name}",
                "invalid_label_file",
                label_path,
                str(error),
            )
            continue
        if not isinstance(labels, dict):
            logger.warning(f"Invalid label format: {label_path}")
            issues.hard(
                f"{level}:{split_folder}:{writer_dir.name}",
                "invalid_label_schema",
                label_path,
            )
            continue
        labels_by_job[key] = labels

    paragraph_lines: dict[tuple[str, str, str], tuple[str, ...]] = {}
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

        for image_name in unmatched:
            issues.reject(
                (
                    f"uithwdb_paragraph_{writer_raw}_"
                    f"{Path(image_name).stem}"
                ),
                "missing_native_line_alignment",
                (
                    LEVEL_DIRS["paragraph"]
                    / split_folder
                    / writer_raw
                    / image_name
                ),
            )

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
                issues.hard(
                    f"uithwdb_{level}_{writer_raw}_{Path(image_name).stem}",
                    "invalid_transcript_type",
                    label_path,
                )
                continue

            text = text.strip()

            if not text:
                logger.warning(f"Empty transcript: {label_path} | {image_name}")
                issues.hard(
                    f"uithwdb_{level}_{writer_raw}_{Path(image_name).stem}",
                    "empty_transcript",
                    label_path,
                )
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
                    # This path is already classified during native alignment.
                    continue
                try:
                    text = join_paragraph_lines(text, line_texts)
                except ValueError as error:
                    logger.warning(
                        f"{error} Paragraph: {label_path} | {image_name}"
                    )
                    issues.hard(
                        (
                            f"uithwdb_paragraph_{writer_raw}_"
                            f"{Path(image_name).stem}"
                        ),
                        "paragraph_line_label_conflict",
                        label_path,
                        str(error),
                    )
                    continue

            image_source = writer_dir / image_name

            if not image_source.exists():
                logger.warning(f"Image not found: {image_source}")
                issues.hard(
                    f"uithwdb_{level}_{writer_raw}_{Path(image_name).stem}",
                    "missing_image",
                    image_source,
                )
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
    write_build_report(
        dataset="uithwdb",
        raw_root=next(iter(LEVEL_DIRS.values())).parent,
        manifest=MANIFEST,
        output=OUT / "build_report.json",
        builder_config=BUILDER_CONFIGS["uithwdb"],
        accepted_count=len(manifest),
        expected_rejections=issues.expected_rejections,
        hard_errors=issues.hard_errors,
        warnings=issues.warnings,
    )
