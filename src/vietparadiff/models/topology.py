"""Diacritic topology detector and exact/weak topology target generation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from vietparadiff.data.graphemes import GraphemeParts
from .common import ResidualBlock

DIACRITIC_TYPES = ["breve", "circumflex", "horn", "stroke", "acute", "grave", "hook", "tilde", "dot"]
TYPE_TO_INDEX = {name: i for i, name in enumerate(DIACRITIC_TYPES)}


class DiacriticTopologyDetector(nn.Module):
    """Predict diacritic type heatmaps and a stroke/body structure map."""

    def __init__(self, in_ch: int = 1, base: int = 96, num_types: int = len(DIACRITIC_TYPES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(), ResidualBlock(base),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 2),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base),
        )
        self.heatmap = nn.Conv2d(base, num_types, 1)
        self.structure = nn.Conv2d(base, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.net(image)
        return {"heatmap": self.heatmap(feat), "structure": self.structure(feat)}


def _draw_gaussian(target: torch.Tensor, channel: int, x: torch.Tensor, y: torch.Tensor, sigma: float = 0.012) -> None:
    h, w = target.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=target.device), torch.linspace(0, 1, w, device=target.device), indexing="ij")
    g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    target[channel] = torch.maximum(target[channel], g)


def exact_topology_from_parts(parts_batch: list[list[GraphemeParts]], boxes: torch.Tensor, anchors: torch.Tensor, mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Create exact per-diacritic heatmap targets from grapheme parts and anchors."""

    target = boxes.new_zeros(boxes.shape[0], len(DIACRITIC_TYPES), height, width)
    for bi, parts in enumerate(parts_batch):
        for i, part in enumerate(parts[: boxes.shape[1]]):
            if not bool(mask[bi, i]):
                continue
            x1, y1, x2, y2 = boxes[bi, i]
            ux, uy, lx, ly = anchors[bi, i]
            if part.modifier in TYPE_TO_INDEX and part.modifier != "none":
                if part.modifier == "stroke":
                    _draw_gaussian(target[bi], TYPE_TO_INDEX["stroke"], (x1 + x2) * 0.5, y1 + (y2 - y1) * 0.45)
                else:
                    _draw_gaussian(target[bi], TYPE_TO_INDEX[part.modifier], ux, uy)
            if part.tone in TYPE_TO_INDEX and part.tone != "none":
                if part.tone == "dot":
                    _draw_gaussian(target[bi], TYPE_TO_INDEX["dot"], lx, ly)
                else:
                    # Place tone marks slightly above the structural modifier anchor.
                    _draw_gaussian(target[bi], TYPE_TO_INDEX[part.tone], ux, (uy - 0.018).clamp(0, 1))
    return target.clamp(0, 1)


def target_topology_from_layout(layout_fields: torch.Tensor) -> torch.Tensor:
    """Create weak topology heatmap targets from upper/lower layout fields."""

    upper = layout_fields[:, 1]
    lower = layout_fields[:, 2]
    b, h, w = upper.shape
    target = upper.new_zeros(b, len(DIACRITIC_TYPES), h, w)
    for name in ["breve", "circumflex", "horn", "acute", "grave", "hook", "tilde"]:
        target[:, TYPE_TO_INDEX[name]] = upper
    target[:, TYPE_TO_INDEX["dot"]] = lower
    return target.clamp(0, 1)


def topology_loss(pred: dict[str, torch.Tensor], target_heatmap: torch.Tensor, structure_target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Focal heatmap loss plus optional structure BCE loss."""

    logits = pred["heatmap"]
    target = F.interpolate(target_heatmap, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    prob = logits.sigmoid()
    focal = -(target * (1 - prob).pow(2) * torch.log(prob.clamp_min(1e-6)) + 0.25 * (1 - target) * prob.pow(2) * torch.log((1 - prob).clamp_min(1e-6))).mean()
    if structure_target is None:
        structure = pred["structure"].new_tensor(0.0)
    else:
        st = F.interpolate(structure_target, size=pred["structure"].shape[-2:], mode="bilinear", align_corners=False)
        structure = F.binary_cross_entropy_with_logits(pred["structure"], st)
    return {"loss": focal + structure, "topology/focal": focal, "topology/structure": structure}
