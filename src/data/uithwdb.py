"""Build a normalized dataset from UIT-HWDB's pre-segmented images."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.data.image_utils import save_normalized_image
from src.logger import get_logger

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
