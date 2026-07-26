"""Shared image normalization and width-only downscaling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PIL import Image

__all__ = [
    "MAX_WIDTH_BY_LEVEL",
    "downscale_to_width",
    "get_max_width",
    "save_normalized_image",
]

# Height is deliberately absent from this policy: unusually tall handwriting
# samples remain valid as long as their width is within the configured limit.
MAX_WIDTH_BY_LEVEL: Final[dict[str, int]] = {
    "paragraph": 1024,
    "line": 1024,
    "word": 512,
}


def get_max_width(level: str) -> int:
    """Return the configured width limit for a manifest sample level.

    Args:
        level: One of ``paragraph``, ``line``, or ``word``.

    Returns:
        The maximum permitted image width in pixels.

    Raises:
        ValueError: If ``level`` is not supported by the normalized manifest
            schema.
    """
    try:
        return MAX_WIDTH_BY_LEVEL[level]
    except KeyError as error:
        raise ValueError(f"Unsupported sample level: {level}") from error


def downscale_to_width(image: Image.Image, max_width: int) -> Image.Image:
    """Downscale an image by width while preserving its aspect ratio.

    Height is intentionally unconstrained. Images whose width is already at or
    below ``max_width`` are returned unchanged, which guarantees that this
    function never upscales a small image.

    Args:
        image: Source image to inspect or resize.
        max_width: Maximum permitted width in pixels.

    Returns:
        The original image when no resize is required, otherwise a new image
        resized with the high-quality Lanczos filter.

    Raises:
        ValueError: If ``max_width`` is not positive or the source image has
            invalid dimensions.
    """
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}")

    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image.size}")

    if image.width <= max_width:
        return image

    scale = max_width / image.width
    scaled_height = max(1, round(image.height * scale))

    return image.resize(
        (max_width, scaled_height),
        resample=Image.Resampling.LANCZOS,
    )


def save_normalized_image(
    image: Image.Image,
    output_path: Path,
    level: str,
) -> tuple[int, int]:
    """Convert, width-limit, and save one normalized dataset image.

    Paragraph and line images are capped at 1024 pixels wide; word images are
    capped at 512 pixels wide. Width is the only constrained dimension, so a
    tall image may remain taller than its width limit.

    Args:
        image: Source image or crop.
        output_path: Destination PNG path.
        level: Manifest sample level used to select the width limit.

    Returns:
        The final ``(width, height)`` written to ``output_path``.
    """
    max_width = get_max_width(level)
    grayscale = image if image.mode == "L" else image.convert("L")
    resized = downscale_to_width(grayscale, max_width)

    try:
        resized.save(
            output_path,
            format="PNG",
            compress_level=1,
        )
        return resized.size
    finally:
        # Close only images allocated inside this function. The caller owns the
        # source image and may need to reuse it for additional crops.
        if resized is not grayscale:
            resized.close()

        if grayscale is not image:
            grayscale.close()
