"""Memory-aware samplers for dynamic paragraph canvases."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterator

from torch.utils.data import Sampler

from .manifest import ManifestRow


class PixelBudgetBatchSampler(Sampler[list[int]]):
    """Group samples by canvas size and keep each batch under a pixel budget.

    Dynamic-height paragraph training is only memory-safe if a batch does not
    accidentally combine large canvases.  This sampler uses the synthetic/real
    manifest metadata ``canvas_height`` and ``canvas_width`` to create batches
    with similar dimensions.  It respects both a maximum number of samples and a
    maximum total pixel count:

    ``sum_i(canvas_height_i * canvas_width_i) <= max_pixels``.

    The model still sees dynamic canvases; this sampler only controls which
    samples are padded together in one micro-batch.
    """

    def __init__(
        self,
        rows: list[ManifestRow],
        max_batch_size: int,
        max_pixels: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        self.rows = rows
        self.max_batch_size = int(max_batch_size)
        self.max_pixels = int(max_pixels)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self._batches = self._build_batches(random.Random(self.seed))

    @staticmethod
    def _size(row: ManifestRow) -> tuple[int, int]:
        meta = row.meta or {}
        h = int(meta.get("canvas_height") or meta.get("height") or 0)
        w = int(meta.get("canvas_width") or meta.get("width") or 0)
        if h <= 0 or w <= 0:
            # Safe fallback for older manifests.  Width is normally fixed at 1024.
            h = 512
            w = 1024
        return h, w

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic shuffle epoch."""

        self.epoch = int(epoch)

    def _build_batches(self, rng: random.Random) -> list[list[int]]:
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, row in enumerate(self.rows):
            buckets[self._size(row)].append(idx)

        all_batches: list[list[int]] = []
        for size in sorted(buckets):
            indices = buckets[size]
            if self.shuffle:
                rng.shuffle(indices)
            h, w = size
            pixels = max(1, h * w)
            batch: list[int] = []
            batch_pixels = 0
            for idx in indices:
                would_exceed_count = len(batch) >= self.max_batch_size
                would_exceed_pixels = batch and batch_pixels + pixels > self.max_pixels
                if would_exceed_count or would_exceed_pixels:
                    if len(batch) == self.max_batch_size or not self.drop_last:
                        all_batches.append(batch)
                    batch = []
                    batch_pixels = 0
                batch.append(idx)
                batch_pixels += pixels
            if batch and (len(batch) == self.max_batch_size or not self.drop_last):
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + 1009 * self.epoch)
        batches = self._build_batches(rng)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)
