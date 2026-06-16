"""Autoregressive paragraph layout planner and differentiable soft rasterizer."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AutoregressiveLayoutPlanner(nn.Module):
    """Plan grapheme boxes, line breaks, and upper/lower diacritic anchors.

    A recurrent state tracks pen cursor and baseline.  Predicted EOL/EOP classes
    update the state during inference, while teacher forcing can use target boxes
    during training to reduce exposure bias.
    """

    def __init__(self, dim: int = 384, hidden_dim: int = 512, fields: int = 5) -> None:
        super().__init__()
        self.init = nn.Linear(dim * 3, hidden_dim)
        self.cell = nn.GRUCell(dim + 6, hidden_dim)
        self.geom = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))
        self.break_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.line_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))  # line height, left margin
        self.fields = fields

    def forward(
        self,
        content: torch.Tensor,
        mask: torch.Tensor,
        style_global: torch.Tensor,
        style_layout: torch.Tensor | None = None,
        target_boxes: torch.Tensor | None = None,
        target_line_ids: torch.Tensor | None = None,
        teacher_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        b, n, _ = content.shape
        device = content.device
        layout_summary = style_layout.mean(dim=1) if style_layout is not None else style_global
        h = torch.tanh(self.init(torch.cat([content[:, 0], style_global, layout_summary], dim=-1)))
        line_params = self.line_head(h)
        line_h = torch.sigmoid(line_params[:, 0]) * 0.10 + 0.075
        left_margin = torch.sigmoid(line_params[:, 1]) * 0.055 + 0.035
        cursor_x = left_margin.clone()
        baseline = torch.full((b,), 0.16, device=device)
        prev = torch.zeros(b, 6, device=device)
        boxes: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        break_logits: list[torch.Tensor] = []

        for i in range(n):
            h = self.cell(torch.cat([content[:, i], prev], dim=-1), h)
            raw = self.geom(h)
            width = torch.sigmoid(raw[:, 0]) * 0.055 + 0.010
            height = torch.sigmoid(raw[:, 1]) * 0.105 + 0.050
            advance = width + torch.sigmoid(raw[:, 2]) * 0.040
            base_delta = torch.tanh(raw[:, 3]) * 0.018
            upper_delta = torch.sigmoid(raw[:, 4]) * 0.050 + 0.014
            lower_delta = torch.sigmoid(raw[:, 5]) * 0.034 + 0.010
            y2 = (baseline + base_delta).clamp(0.03, 0.98)
            y1 = (y2 - height).clamp(0.0, 0.96)
            x1 = cursor_x.clamp(0.0, 0.985)
            x2 = (x1 + width).clamp(0.01, 1.0)
            pred_box = torch.stack([x1, y1, x2, y2], dim=-1)
            if target_boxes is not None and teacher_ratio > 0:
                use_teacher = torch.rand(b, device=device) < teacher_ratio
                box_used = torch.where(use_teacher[:, None], target_boxes[:, i], pred_box)
            else:
                box_used = pred_box
            logits = self.break_head(h)
            pred_break = logits.argmax(dim=-1)
            if target_line_ids is not None and teacher_ratio >= 0.999 and i + 1 < n:
                # In fully teacher-forced layout pretraining, use line-id transition as the state break.
                eol_target = target_line_ids[:, i + 1].ne(target_line_ids[:, i]) & mask[:, i] & mask[:, i + 1]
                eol = eol_target
            else:
                eol = pred_break.eq(1) | (box_used[:, 2] + advance > 0.965)
            eop = pred_break.eq(2) | (~mask[:, i])
            cursor_x = torch.where(eol | eop, left_margin, box_used[:, 2] + advance)
            baseline = torch.where(eol & ~eop, (baseline + line_h).clamp(max=0.96), baseline)
            prev = torch.stack([box_used[:, 0], box_used[:, 1], box_used[:, 2], box_used[:, 3], upper_delta, lower_delta], dim=-1)
            center_x = (x1 + x2) * 0.5
            boxes.append(pred_box)
            anchors.append(torch.stack([center_x, (y1 - upper_delta).clamp(0, 1), center_x, (y2 + lower_delta).clamp(0, 1)], dim=-1))
            break_logits.append(logits)

        boxes_t = torch.stack(boxes, dim=1) * mask[:, :, None]
        anchors_t = torch.stack(anchors, dim=1) * mask[:, :, None]
        return {"boxes": boxes_t, "anchors": anchors_t, "break_logits": torch.stack(break_logits, dim=1), "line_height": line_h, "left_margin": left_margin}


def break_targets_from_line_ids(line_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Derive CONTINUE/EOL/EOP labels from token line IDs and padding mask."""

    target = torch.zeros_like(line_ids)
    target[:, :-1] = (line_ids[:, 1:] != line_ids[:, :-1]).long()
    target[:, -1] = 2
    target = torch.where(mask, target, torch.full_like(target, 2))
    return target


def rasterize_layout(boxes: torch.Tensor, anchors: torch.Tensor, mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterize boxes and diacritic anchors into five soft layout fields."""

    device = boxes.device
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=device), torch.linspace(0, 1, width, device=device), indexing="ij")
    xx = xx[None, None]
    yy = yy[None, None]
    x1, y1, x2, y2 = [boxes[..., i][..., None, None] for i in range(4)]
    sharp = 90.0
    body = torch.sigmoid((xx - x1) * sharp) * torch.sigmoid((x2 - xx) * sharp) * torch.sigmoid((yy - y1) * sharp) * torch.sigmoid((y2 - yy) * sharp)
    body = (body * mask[:, :, None, None]).amax(dim=1)
    ux, uy, lx, ly = [anchors[..., i][..., None, None] for i in range(4)]
    sigma = 0.010
    upper = torch.exp(-((xx - ux) ** 2 + (yy - uy) ** 2) / (2 * sigma**2))
    lower = torch.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * sigma**2))
    upper = (upper * mask[:, :, None, None]).amax(dim=1)
    lower = (lower * mask[:, :, None, None]).amax(dim=1)
    baseline_y = y2 - (y2 - y1) * 0.12
    baseline = torch.exp(-((yy - baseline_y) ** 2) / (2 * (sigma * 0.75) ** 2))
    baseline = (baseline * mask[:, :, None, None]).amax(dim=1)
    whitespace = (1.0 - body).clamp(0, 1)
    return torch.stack([body, upper, lower, baseline, whitespace], dim=1)


def layout_losses(
    pred: dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    mask: torch.Tensor,
    line_ids: torch.Tensor | None = None,
    break_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute layout regression, break, overlap, and overflow losses."""

    valid = mask[..., None].float()
    box_loss = (F.smooth_l1_loss(pred["boxes"], target_boxes, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    boxes = pred["boxes"]
    left = torch.maximum(boxes[:, :-1, 0], boxes[:, 1:, 0])
    top = torch.maximum(boxes[:, :-1, 1], boxes[:, 1:, 1])
    right = torch.minimum(boxes[:, :-1, 2], boxes[:, 1:, 2])
    bottom = torch.minimum(boxes[:, :-1, 3], boxes[:, 1:, 3])
    inter = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    overlap = (inter * (mask[:, :-1] & mask[:, 1:]).float()).sum() / (mask[:, :-1] & mask[:, 1:]).float().sum().clamp_min(1)
    overflow = (boxes.clamp(0, 1) - boxes).abs().mean()
    losses = {"layout/box": box_loss, "layout/overlap": overlap, "layout/overflow": overflow}
    if line_ids is not None:
        targets = break_targets_from_line_ids(line_ids, mask)
        break_loss = F.cross_entropy(pred["break_logits"].transpose(1, 2), targets, reduction="none")
        break_loss = (break_loss * mask.float()).sum() / mask.float().sum().clamp_min(1)
        losses["layout/break"] = break_loss * break_weight
    return losses
