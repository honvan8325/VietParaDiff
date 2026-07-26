"""Build a normalized, multi-granularity dataset from the raw CVL corpus."""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from vietparadiff.data.image_utils import save_normalized_image
from vietparadiff.cli_logging import get_logger

__all__ = ["build_cvl_dataset"]

logger = get_logger(__name__)

RAW = Path("data/raw/CVL")
FULL = RAW / "cvl-database-1-1"
CROPPED = RAW / "cvl-database-cropped-1-1"

OUT = Path("data/cvl")
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.jsonl"

NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}


def build_cvl_dataset() -> None:
    """Rebuild CVL paragraph, line, and word samples.

    The builder indexes the original TIFF crops and PAGE XML annotations,
    converts every accepted image to a width-limited 8-bit grayscale PNG, and
    writes one JSON object per sample to ``data/cvl/manifest.jsonl``.

    Warning:
        ``data/cvl`` is deleted before the new dataset is written. Keep any
        manually added files outside that directory.
    """
    logger.info("Build CVL dataset")

    if OUT.exists():
        logger.warning(f"Removing existing output folder: {OUT}")
        shutil.rmtree(OUT)

    IMAGES.mkdir(parents=True)

    line_images = {}
    word_images = {}
    paragraph_images = {}
    xml_files = []
    manifest = []

    # Index the source crops once so XML identifiers can be resolved with
    # constant-time dictionary lookups during the main parsing loop.

    for split_folder in ["trainset", "testset"]:
        split_root = FULL / split_folder

        # A line image is named like ``0001-1-0.tif``. Its stem is the line
        # identifier used by the corresponding PAGE XML region.
        for image_path in (split_root / "lines").rglob(
            "[0-9][0-9][0-9][0-9]-[0-9]*-[0-9]*.tif"
        ):
            line_images[image_path.stem] = image_path

        # A word image is named like ``0001-1-0-0-Imagine.tif``. The trailing
        # transcription is not part of the XML word identifier, so only the
        # first four hyphen-separated components are retained.
        for image_path in (split_root / "words").rglob(
            "[0-9][0-9][0-9][0-9]-[0-9]*-[0-9]*-[0-9]*-*.tif"
        ):
            word_id = "-".join(image_path.stem.split("-", 4)[:4])

            word_images[word_id] = image_path

        # Attribute files such as ``0001-1_attributes.xml`` describe the
        # hierarchy and transcript for a complete handwritten page.
        xml_files.extend(
            sorted(
                (split_root / "xml").glob("[0-9][0-9][0-9][0-9]-[0-9]*_attributes.xml")
            )
        )

    # Cropped page images such as ``0001-1-cropped.tif`` provide the
    # paragraph-level sample associated with each XML page.
    for image_path in CROPPED.glob("[0-9][0-9][0-9][0-9]-[0-9]*-cropped.tif"):
        page_id = image_path.stem.removesuffix("-cropped")
        paragraph_images[page_id] = image_path

    logger.info(f"Line images: {len(line_images)}")
    logger.info(f"Word images: {len(word_images)}")
    logger.info(f"Paragraph images: {len(paragraph_images)}")
    logger.info(f"XML files: {len(xml_files)}")

    # Parse each PAGE XML document and materialize all three sample levels.

    for xml_path in tqdm(xml_files, desc="Parse XML", unit="file", dynamic_ncols=True):
        payload = xml_path.read_bytes()

        if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
            xml_text = payload.decode("utf-16")
        else:
            try:
                xml_text = payload.decode("utf-8-sig")
            except UnicodeDecodeError:
                xml_text = payload.decode("iso-8859-1")

        # Some CVL files declare UTF-16 even when their bytes are Latin-1.
        # After decoding the payload explicitly, normalize the declaration so
        # ElementTree does not try to reinterpret the already-decoded string.
        xml_text = re.sub(
            r'encoding=["\'][^"\']+["\']',
            'encoding="UTF-8"',
            xml_text,
            count=1,
        )

        root = ET.fromstring(xml_text)

        page = root.find("p:Page", NS)

        if page is None:
            logger.warning(f"Page not found: {xml_path.name}")
            continue

        page_id = Path(page.attrib.get("imageFilename", "")).stem

        if not page_id:
            page_id = xml_path.stem.removesuffix("_attributes")

        writer_raw = page_id.split("-")[0]
        writer_id = f"cvl_{writer_raw}"

        normalized_page_id = f"cvl_{page_id.replace('-', '_')}"

        # CVL marks the handwritten paragraph as an attribute region with
        # ``attrType=3`` (paragraph) and ``fontType=2`` (handwriting).

        handwritten_block = None

        for region in page.findall(
            ".//p:AttrRegion",
            NS,
        ):
            if (
                region.attrib.get("attrType") == "3"
                and region.attrib.get("fontType") == "2"
            ):
                handwritten_block = region
                break

        if handwritten_block is None:
            logger.warning(f"Handwritten block not found: {xml_path.name}")
            continue

        line_regions = [
            region
            for region in handwritten_block.findall(
                "./p:AttrRegion",
                NS,
            )
            if region.attrib.get("attrType") == "2"
        ]

        paragraph_lines = []
        line_records = []
        word_records = []

        # Build word records first so their transcripts can be joined in XML
        # order to produce the canonical line and paragraph transcripts.

        for line_region in line_regions:
            line_id = line_region.attrib["id"]

            normalized_line_id = f"cvl_{line_id.replace('-', '_')}"

            words = []

            word_regions = [
                region
                for region in line_region.findall(
                    "./p:AttrRegion",
                    NS,
                )
                if region.attrib.get("attrType") == "1"
            ]

            for word_region in word_regions:
                word_id = word_region.attrib.get("id")
                text = word_region.attrib.get("text")

                if word_id is None or text is None:
                    continue

                text = text.strip()

                if not text:
                    continue

                words.append(text)

                word_source = word_images.get(word_id)

                if word_source is None:
                    logger.warning(f"Word image not found: {word_id}")
                    continue

                normalized_word_id = f"cvl_{word_id.replace('-', '_')}"

                output_word = IMAGES / f"{normalized_word_id}.png"

                with Image.open(word_source) as image:
                    word_width, word_height = save_normalized_image(
                        image,
                        output_word,
                        level="word",
                    )

                word_records.append(
                    {
                        "id": normalized_word_id,
                        "image": output_word.as_posix(),
                        "text": text,
                        "writer_id": writer_id,
                        "level": "word",
                        "width": word_width,
                        "height": word_height,
                    }
                )

            line_text = " ".join(words)

            if not line_text:
                continue

            paragraph_lines.append(line_text)

            line_source = line_images.get(line_id)

            if line_source is None:
                logger.warning(f"Line image not found: {line_id}")
                continue

            output_line = IMAGES / f"{normalized_line_id}.png"

            with Image.open(line_source) as image:
                line_width, line_height = save_normalized_image(
                    image,
                    output_line,
                    level="line",
                )

            line_records.append(
                {
                    "id": normalized_line_id,
                    "image": output_line.as_posix(),
                    "text": line_text,
                    "writer_id": writer_id,
                    "level": "line",
                    "width": line_width,
                    "height": line_height,
                }
            )

        # A paragraph is accepted only when both its crop and the transcript
        # assembled from accepted child lines are available.

        paragraph_source = paragraph_images.get(page_id)
        paragraph_text = "\n".join(paragraph_lines)

        if paragraph_source is None:
            logger.warning(f"Paragraph image not found: {page_id}")
            continue

        if not paragraph_text:
            logger.warning(f"Paragraph text is empty: {page_id}")
            continue

        output_paragraph = IMAGES / f"{normalized_page_id}.png"

        with Image.open(paragraph_source) as image:
            paragraph_width, paragraph_height = save_normalized_image(
                image,
                output_paragraph,
                level="paragraph",
            )

        manifest.append(
            {
                "id": normalized_page_id,
                "image": output_paragraph.as_posix(),
                "text": paragraph_text,
                "writer_id": writer_id,
                "level": "paragraph",
                "width": paragraph_width,
                "height": paragraph_height,
            }
        )

        manifest.extend(line_records)
        manifest.extend(word_records)

    # Write UTF-8 JSON Lines so transcripts retain their original characters
    # and downstream readers can stream the manifest one sample at a time.

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

    logger.info(f"Paragraphs: {paragraph_count}")
    logger.info(f"Lines: {line_count}")
    logger.info(f"Words: {word_count}")
    logger.info(f"Total: {len(manifest)}")
    logger.info(f"Manifest: {MANIFEST}")
