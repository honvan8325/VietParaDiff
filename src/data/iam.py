"""Build normalized paragraph, line, and word crops from the IAM corpus."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.data.image_utils import save_normalized_image
from src.logger import get_logger

__all__ = ["build_iam_dataset"]

logger = get_logger(__name__)

RAW = Path("data/raw/IAM")
ARCHIVE = RAW / "archive"

FORMS = ARCHIVE / "forms" / "forms"
XML = ARCHIVE / "xml"
FORMS_TXT = ARCHIVE / "ascii" / "ascii" / "forms.txt"

OUT = Path("data/iam")
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.jsonl"

PARAGRAPH_PADDING = 20
LINE_PADDING = 10
WORD_PADDING = 4


def _parse_forms_metadata(path: Path) -> dict[str, dict]:
    """Parse IAM's form-level metadata index.

    Each non-comment row has the following fields::

        form_id writer_id sentence_count segmentation line_count
        segmented_line_count word_count segmented_word_count

    Malformed rows are logged and skipped so one damaged metadata entry does
    not prevent the remaining forms from being built.

    Args:
        path: Path to the original IAM ``forms.txt`` file.

    Returns:
        A mapping from form ID to writer and segmentation metadata.
    """

    metadata = {}

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 8:
            logger.warning(f"Invalid forms.txt row: {line}")
            continue

        (
            form_id,
            writer_id,
            sentence_count,
            segmentation,
            line_count,
            segmented_line_count,
            word_count,
            segmented_word_count,
        ) = parts[:8]

        try:
            metadata[form_id] = {
                "writer_id": writer_id,
                "sentence_count": int(sentence_count),
                "segmentation": segmentation,
                "line_count": int(line_count),
                "segmented_line_count": int(segmented_line_count),
                "word_count": int(word_count),
                "segmented_word_count": int(segmented_word_count),
            }
        except ValueError:
            logger.warning(f"Invalid numeric metadata: {line}")

    return metadata


def _get_bbox(element: ET.Element) -> tuple[int, int, int, int] | None:
    """Compute the union of valid ``cmp`` boxes below an XML element.

    Args:
        element: A paragraph, line, or word element from an IAM XML file.

    Returns:
        ``(left, top, right, bottom)`` in source-image coordinates, or
        ``None`` when the element contains no valid positive-size components.
    """

    boxes = []

    for component in element.findall(".//cmp"):
        try:
            x = int(component.attrib["x"])
            y = int(component.attrib["y"])
            width = int(component.attrib["width"])
            height = int(component.attrib["height"])
        except (KeyError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        boxes.append(
            (
                x,
                y,
                x + width,
                y + height,
            )
        )

    if not boxes:
        return None

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)

    return left, top, right, bottom


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Add padding to a box while clamping it to the image boundaries.

    Args:
        bbox: Unpadded ``(left, top, right, bottom)`` coordinates.
        padding: Number of pixels to add on all four sides.
        image_width: Width of the source form in pixels.
        image_height: Height of the source form in pixels.

    Returns:
        A padded box that remains entirely inside the source image.
    """
    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image_width, right + padding)
    bottom = min(image_height, bottom + padding)

    return left, top, right, bottom


def _save_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    padding: int,
    level: str,
) -> tuple[int, int]:
    """Crop, width-limit, and save one grayscale IAM sample as a PNG.

    Args:
        image: Grayscale source form.
        bbox: Tight component-based bounding box.
        output_path: Destination path for the PNG crop.
        padding: Context padding to add around the tight box.
        level: Sample level used to select the maximum output width.

    Returns:
        The final ``(width, height)`` written to ``output_path``.

    Raises:
        ValueError: If clamping produces an empty crop.
    """
    crop_box = _expand_bbox(
        bbox=bbox,
        padding=padding,
        image_width=image.width,
        image_height=image.height,
    )

    left, top, right, bottom = crop_box

    if left >= right or top >= bottom:
        raise ValueError(f"Invalid crop box: {crop_box}")

    crop = image.crop(crop_box)

    try:
        return save_normalized_image(crop, output_path, level=level)
    finally:
        crop.close()


def build_iam_dataset() -> None:
    """Rebuild IAM samples and their normalized JSONL manifest.

    A form image is opened once, then reused to crop its paragraph, line, and
    word samples. Word crops are emitted only for lines IAM marks as reliably
    segmented. All accepted images are stored as grayscale PNG files under
    ``data/iam/images`` with the shared level-specific width limits.

    Raises:
        FileNotFoundError: If a required IAM image, XML, or metadata directory
            is missing.

    Warning:
        ``data/iam`` is deleted before the new dataset is written.
    """
    logger.info("Build IAM dataset")

    required_paths = [
        FORMS,
        XML,
        FORMS_TXT,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required IAM path not found: {required_path}")

    if OUT.exists():
        logger.warning(f"Removing existing output folder: {OUT}")
        shutil.rmtree(OUT)

    IMAGES.mkdir(parents=True)

    # Index forms and annotations before processing. Dictionary lookup avoids
    # repeatedly scanning the large image directory for every XML document.

    form_images = {image_path.stem: image_path for image_path in FORMS.glob("*.png")}

    xml_files = sorted(XML.glob("*.xml"))

    forms_metadata = _parse_forms_metadata(FORMS_TXT)

    manifest = []

    segmentation_error_lines = 0

    logger.info(f"Form images: {len(form_images)}")
    logger.info(f"XML files: {len(xml_files)}")
    logger.info(f"Forms metadata: {len(forms_metadata)}")

    # Parse one XML annotation file for each source form.

    for xml_path in tqdm(
        xml_files,
        desc="Parse IAM XML",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            logger.warning(f"Cannot parse {xml_path.name}: {error}")
            continue

        form_id = root.attrib.get("id", xml_path.stem)

        form_source = form_images.get(form_id)

        if form_source is None:
            logger.warning(f"Form image not found: {form_id}")
            continue

        metadata = forms_metadata.get(form_id)

        writer_raw = root.attrib.get("writer-id")

        if metadata is not None:
            metadata_writer = metadata["writer_id"]

            if writer_raw is None:
                writer_raw = metadata_writer
            elif writer_raw != metadata_writer:
                logger.warning(
                    f"Writer mismatch for {form_id}: "
                    f"XML={writer_raw}, forms.txt={metadata_writer}"
                )

        if writer_raw is None:
            logger.warning(f"Writer ID not found: {form_id}")
            continue

        writer_id = f"iam_{writer_raw}"

        normalized_form_id = f"iam_{form_id.replace('-', '_')}"

        handwritten_part = root.find("./handwritten-part")

        if handwritten_part is None:
            logger.warning(f"Handwritten part not found: {form_id}")
            continue

        line_elements = list(handwritten_part.findall("./line"))

        if not line_elements:
            logger.warning(f"Handwritten lines not found: {form_id}")
            continue

        # Cross-check XML counts against forms.txt. Mismatches are diagnostic:
        # the XML remains the authoritative source used to create samples.

        if metadata is not None:
            actual_line_count = len(line_elements)

            actual_segmented_line_count = sum(
                line.attrib.get("segmentation") == "ok" for line in line_elements
            )

            actual_word_count = sum(
                len(line.findall("./word")) for line in line_elements
            )

            actual_segmented_word_count = sum(
                len(line.findall("./word"))
                for line in line_elements
                if line.attrib.get("segmentation") == "ok"
            )

            checks = [
                (
                    "line count",
                    actual_line_count,
                    metadata["line_count"],
                ),
                (
                    "segmented line count",
                    actual_segmented_line_count,
                    metadata["segmented_line_count"],
                ),
                (
                    "word count",
                    actual_word_count,
                    metadata["word_count"],
                ),
                (
                    "segmented word count",
                    actual_segmented_word_count,
                    metadata["segmented_word_count"],
                ),
            ]

            for name, actual, expected in checks:
                if actual != expected:
                    logger.warning(
                        f"{form_id}: {name} mismatch, "
                        f"XML={actual}, forms.txt={expected}"
                    )

        # Preserve line breaks when composing paragraph transcripts because
        # they encode the original layout of the handwritten form.

        paragraph_lines = []

        for line_element in line_elements:
            line_text = line_element.attrib.get("text", "").strip()

            if line_text:
                paragraph_lines.append(line_text)

        paragraph_text = "\n".join(paragraph_lines)

        if not paragraph_text:
            logger.warning(f"Paragraph text is empty: {form_id}")
            continue

        paragraph_bbox = _get_bbox(handwritten_part)

        if paragraph_bbox is None:
            logger.warning(f"Paragraph bounding box not found: {form_id}")
            continue

        output_paragraph = IMAGES / f"{normalized_form_id}.png"

        line_records = []
        word_records = []

        # Open and decode the full form only once, then crop all sample levels
        # from the in-memory grayscale image.

        try:
            with Image.open(form_source) as source_image:
                image = source_image.convert("L")

                paragraph_width, paragraph_height = _save_crop(
                    image=image,
                    bbox=paragraph_bbox,
                    output_path=output_paragraph,
                    padding=PARAGRAPH_PADDING,
                    level="paragraph",
                )
                paragraph_record = {
                    "id": normalized_form_id,
                    "image": output_paragraph.as_posix(),
                    "text": paragraph_text,
                    "writer_id": writer_id,
                    "level": "paragraph",
                    "width": paragraph_width,
                    "height": paragraph_height,
                }

                # Line crops are retained whenever their transcript and
                # component-derived bounding box are valid.

                for line_element in line_elements:
                    line_id = line_element.attrib.get("id")
                    line_text = line_element.attrib.get("text", "").strip()

                    if line_id is None or not line_text:
                        continue

                    line_bbox = _get_bbox(line_element)

                    if line_bbox is None:
                        logger.warning(f"Line bounding box not found: {line_id}")
                        continue

                    normalized_line_id = f"iam_{line_id.replace('-', '_')}"

                    output_line = IMAGES / f"{normalized_line_id}.png"

                    line_width, line_height = _save_crop(
                        image=image,
                        bbox=line_bbox,
                        output_path=output_line,
                        padding=LINE_PADDING,
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

                    # IAM explicitly marks unreliable word segmentation.
                    # Retain the line for line- and paragraph-level training,
                    # but do not emit potentially misaligned word crops.
                    if line_element.attrib.get("segmentation") != "ok":
                        segmentation_error_lines += 1
                        continue

                    for word_element in line_element.findall("./word"):
                        word_id = word_element.attrib.get("id")
                        word_text = word_element.attrib.get(
                            "text",
                            "",
                        ).strip()

                        if word_id is None or not word_text:
                            continue

                        word_bbox = _get_bbox(word_element)

                        if word_bbox is None:
                            logger.warning(f"Word bounding box not found: {word_id}")
                            continue

                        normalized_word_id = f"iam_{word_id.replace('-', '_')}"

                        output_word = IMAGES / f"{normalized_word_id}.png"

                        word_width, word_height = _save_crop(
                            image=image,
                            bbox=word_bbox,
                            output_path=output_word,
                            padding=WORD_PADDING,
                            level="word",
                        )

                        word_records.append(
                            {
                                "id": normalized_word_id,
                                "image": output_word.as_posix(),
                                "text": word_text,
                                "writer_id": writer_id,
                                "level": "word",
                                "width": word_width,
                                "height": word_height,
                            }
                        )

        except (OSError, ValueError) as error:
            logger.warning(f"Cannot process form {form_id}: {error}")

            # A form is an atomic unit in the manifest. Remove any crops that
            # were written before the failure so no orphaned files remain.
            output_paragraph.unlink(missing_ok=True)

            for record in line_records:
                Path(record["image"]).unlink(missing_ok=True)

            for record in word_records:
                Path(record["image"]).unlink(missing_ok=True)

            continue

        manifest.append(paragraph_record)
        manifest.extend(line_records)
        manifest.extend(word_records)

    # Serialize only fully accepted records after all forms have been handled.

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
    logger.info(f"Lines excluded from word level: " f"{segmentation_error_lines}")
    logger.info(f"Manifest: {MANIFEST}")
