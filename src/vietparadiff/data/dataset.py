"""Paragraph dataset with dynamic bucketed canvases and episodic reference-target pairing.

Synthetic and real paragraph images are stored at their natural/bucket canvas size.
The dataset does not resize text to a fixed height.  It returns images and boxes
in the sample-local coordinate system; the collate function pads samples in a
batch and rescales normalized boxes to the padded batch canvas.  This keeps
paragraph height dynamic while still allowing efficient batched training.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from vietparadiff.utils.image import load_grayscale, pil_to_tensor

from .archetypes import ArchetypeRenderer
from .graphemes import VietnameseTokenizer
from .manifest import ManifestRow, read_jsonl


def infer_token_boxes(row: ManifestRow, parts_count: int) -> tuple[list[list[float]], list[int]]:
    """Return token boxes and line IDs using manifest boxes or a line-box fallback.

    The synthetic renderer writes one token annotation for every grapheme in the
    newline-inclusive transcript.  For real manifests that only contain line
    boxes, this fallback splits each line uniformly so training can still run,
    but paper experiments should prefer exact token boxes.
    """

    if row.tokens and len(row.tokens) >= parts_count:
        return [t.box for t in row.tokens[:parts_count]], [t.line_id for t in row.tokens[:parts_count]]
    boxes: list[list[float]] = []
    line_ids: list[int] = []
    idx = 0
    for li, line in enumerate(row.lines):
        chars = list(line.text)
        n = max(1, len(chars))
        x1, y1, x2, y2 = line.box
        w = (x2 - x1) / n
        for j, _ in enumerate(chars):
            if idx >= parts_count:
                break
            boxes.append([x1 + j * w, y1, x1 + (j + 1) * w, y2])
            line_ids.append(li)
            idx += 1
        if idx < parts_count:
            # Account for the newline token inserted between manifest lines.
            boxes.append([x2, y1, x2 + 1.0, y2])
            line_ids.append(li)
            idx += 1
    if not boxes:
        boxes = [[20.0, 20.0, 60.0, 60.0] for _ in range(parts_count)]
        line_ids = [0 for _ in range(parts_count)]
    while len(boxes) < parts_count:
        boxes.append(boxes[-1])
        line_ids.append(line_ids[-1])
    return boxes[:parts_count], line_ids[:parts_count]


class ParagraphDataset(Dataset):
    """Return one target paragraph and one same-writer reference paragraph.

    ``image_height`` and ``image_width`` are kept as optional arguments for
    backward-compatible CLI calls, but they are not used to resize samples.  The
    correct pipeline stores dynamic bucketed images on disk and pads them only in
    ``paragraph_collate``.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        tokenizer: VietnameseTokenizer,
        archetype_renderer: ArchetypeRenderer,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> None:
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"Manifest {manifest} is empty")
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.archetype_renderer = archetype_renderer
        self.by_writer: dict[str, list[int]] = defaultdict(list)
        self.writer_to_index = {w: i for i, w in enumerate(sorted({r.writer_id for r in self.rows}))}
        for i, row in enumerate(self.rows):
            self.by_writer[row.writer_id].append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, relative: str) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load a grayscale image without resizing.

        Returns a tensor in ``[-1, 1]`` and the original ``(height, width)``.
        """

        image = load_grayscale(self.root / relative)
        return pil_to_tensor(image), (image.height, image.width)

    @staticmethod
    def _normalize_box(box: list[float] | tuple[float, float, float, float], height: int, width: int) -> list[float]:
        x1, y1, x2, y2 = map(float, box)
        return [x1 / width, y1 / height, x2 / width, y2 / height]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        candidates = [i for i in self.by_writer[row.writer_id] if i != index]
        ref_index = random.choice(candidates) if candidates else index
        ref_row = self.rows[ref_index]

        image, image_size = self._load_image(row.image)
        reference, reference_size = self._load_image(ref_row.image)
        image_h, image_w = image_size

        encoded = self.tokenizer.encode(row.transcript)
        parts = encoded["parts"]
        archetypes = self.archetype_renderer.render_batch(parts)

        raw_boxes, line_ids = infer_token_boxes(row, len(parts))
        boxes = torch.tensor([self._normalize_box(b, image_h, image_w) for b in raw_boxes], dtype=torch.float32)
        line_boxes = torch.tensor([self._normalize_box(line.box, image_h, image_w) for line in row.lines], dtype=torch.float32) if row.lines else torch.zeros(0, 4)
        line_texts = [line.text for line in row.lines]

        return {
            "id": row.id,
            "image": image,
            "reference": reference,
            "image_size": torch.tensor([image_h, image_w], dtype=torch.long),
            "reference_size": torch.tensor([reference_size[0], reference_size[1]], dtype=torch.long),
            "text": {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items() if k != "parts"},
            "parts": parts,
            "archetypes": archetypes,
            "boxes": boxes,
            "line_ids": torch.tensor(line_ids[: len(parts)], dtype=torch.long),
            "line_boxes": line_boxes,
            "line_texts": line_texts,
            "writer": torch.tensor(self.writer_to_index[row.writer_id], dtype=torch.long),
            "transcript": row.transcript,
            "meta": row.meta,
        }
