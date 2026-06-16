"""Batch collation for dynamic-height paragraph canvases.

Images are not resized.  Each batch is padded to the maximum height/width in the
batch, rounded up to a multiple of the VAE downsample factor.  Box coordinates
are rescaled from sample-local normalized coordinates to the padded batch canvas.
"""

from __future__ import annotations

from typing import Any
import math
import torch
import torch.nn.functional as F

TEXT_KEYS = ["base", "modifier", "tone", "case", "type", "full"]
WHITE_VALUE = 1.0  # tensors are normalized to [-1, 1], so white background is +1


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _pad_1d(values: list[torch.Tensor], pad: int = 0) -> torch.Tensor:
    max_len = max(v.numel() for v in values)
    out = torch.full((len(values), max_len), pad, dtype=values[0].dtype)
    for i, v in enumerate(values):
        out[i, : v.numel()] = v
    return out


def _pad_2d(values: list[torch.Tensor], width: int) -> torch.Tensor:
    max_len = max(v.shape[0] for v in values)
    out = torch.zeros(len(values), max_len, width, dtype=values[0].dtype)
    for i, v in enumerate(values):
        if v.numel():
            out[i, : v.shape[0]] = v
    return out


def _pad_image_batch(images: list[torch.Tensor], multiple: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-left pad image tensors to a shared canvas.

    Returns the padded tensor and an ``[B, 2]`` tensor of original ``(H, W)``.
    """

    sizes = torch.tensor([[img.shape[-2], img.shape[-1]] for img in images], dtype=torch.long)
    max_h = _ceil_to_multiple(int(sizes[:, 0].max()), multiple)
    max_w = _ceil_to_multiple(int(sizes[:, 1].max()), multiple)
    padded = []
    for img in images:
        dh = max_h - img.shape[-2]
        dw = max_w - img.shape[-1]
        padded.append(F.pad(img, (0, dw, 0, dh), value=WHITE_VALUE))
    return torch.stack(padded), sizes


def _rescale_normalized_boxes(boxes: torch.Tensor, sizes: torch.Tensor, canvas_h: int, canvas_w: int) -> torch.Tensor:
    """Map sample-local normalized boxes to the padded batch canvas."""

    out = boxes.clone()
    if out.numel() == 0:
        return out
    sx = sizes[:, 1].float().view(-1, 1) / float(canvas_w)
    sy = sizes[:, 0].float().view(-1, 1) / float(canvas_h)
    out[:, :, 0] *= sx
    out[:, :, 2] *= sx
    out[:, :, 1] *= sy
    out[:, :, 3] *= sy
    return out


def paragraph_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate paragraph samples into padded tensors and masks."""

    text = {key: _pad_1d([b["text"][key] for b in batch], 0) for key in TEXT_KEYS}
    lengths = torch.tensor([b["text"]["full"].numel() for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    text_mask = torch.arange(max_len)[None, :] < lengths[:, None]

    archetypes = torch.zeros(len(batch), max_len, *batch[0]["archetypes"].shape[1:], dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["archetypes"].shape[0]
        archetypes[i, :n] = b["archetypes"]

    image, image_sizes = _pad_image_batch([b["image"] for b in batch], multiple=8)
    reference, reference_sizes = _pad_image_batch([b["reference"] for b in batch], multiple=8)
    canvas_h, canvas_w = image.shape[-2:]

    boxes = _pad_2d([b["boxes"] for b in batch], 4)
    line_boxes = _pad_2d([b["line_boxes"] for b in batch], 4)
    boxes = _rescale_normalized_boxes(boxes, image_sizes, canvas_h, canvas_w)
    line_boxes = _rescale_normalized_boxes(line_boxes, image_sizes, canvas_h, canvas_w)

    line_lengths = torch.tensor([b["line_boxes"].shape[0] for b in batch], dtype=torch.long)
    max_lines = int(line_lengths.max()) if len(line_lengths) else 0
    line_mask = torch.arange(max_lines)[None, :] < line_lengths[:, None] if max_lines else torch.zeros(len(batch), 0, dtype=torch.bool)

    return {
        "id": [b["id"] for b in batch],
        "image": image,
        "reference": reference,
        "image_size": image_sizes,
        "reference_size": reference_sizes,
        "text": text,
        "text_mask": text_mask,
        "lengths": lengths,
        "parts": [b["parts"] for b in batch],
        "archetypes": archetypes,
        "boxes": boxes,
        "line_ids": _pad_1d([b["line_ids"] for b in batch], 0),
        "line_boxes": line_boxes,
        "line_mask": line_mask,
        "line_lengths": line_lengths,
        "line_texts": [b["line_texts"] for b in batch],
        "writer": torch.stack([b["writer"] for b in batch]),
        "transcript": [b["transcript"] for b in batch],
        "meta": [b.get("meta", {}) for b in batch],
    }
