"""Image loading, normalization, and geometry transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np
import torch


@dataclass(frozen=True)
class FitInfo:
    """Affine transform used to fit an image into a fixed canvas."""

    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int
    fitted_width: int
    fitted_height: int


def load_grayscale(path: str | Path) -> Image.Image:
    """Load an image as 8-bit grayscale."""

    return Image.open(path).convert("L")


def fit_to_canvas(image: Image.Image, height: int, width: int, fill: int = 255) -> tuple[Image.Image, FitInfo]:
    """Resize an image with aspect ratio preserved and center-pad to a canvas."""

    ow, oh = image.size
    scale = min(width / max(1, ow), height / max(1, oh))
    nw, nh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
    resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (width, height), fill)
    pad_x = (width - nw) // 2
    pad_y = (height - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, FitInfo(scale, float(pad_x), float(pad_y), ow, oh, nw, nh)


def transform_box(box: list[float] | tuple[float, float, float, float], info: FitInfo) -> list[float]:
    """Apply a FitInfo affine transform to a pixel-space box."""

    x1, y1, x2, y2 = map(float, box)
    return [x1 * info.scale + info.pad_x, y1 * info.scale + info.pad_y, x2 * info.scale + info.pad_x, y2 * info.scale + info.pad_y]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL grayscale image to a float tensor in [-1, 1] without TypedStorage."""

    arr = np.asarray(image, dtype=np.uint8).copy()
    data = torch.from_numpy(arr).float() / 255.0
    return data.unsqueeze(0) * 2.0 - 1.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] tensor to a grayscale PIL image."""

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor[0]
    arr = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(arr, mode="L")
