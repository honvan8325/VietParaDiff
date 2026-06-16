"""Line-wise Vietnamese handwriting recognizer used for auxiliary losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def crop_lines(image: torch.Tensor, line_boxes: torch.Tensor, out_h: int = 64, out_w: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably crop normalized line boxes from paragraph images."""

    b, c, _, _ = image.shape
    crops: list[torch.Tensor] = []
    owner: list[int] = []
    for bi in range(b):
        for box in line_boxes[bi]:
            if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
                continue
            xs = torch.linspace(box[0] * 2 - 1, box[2] * 2 - 1, out_w, device=image.device)
            ys = torch.linspace(box[1] * 2 - 1, box[3] * 2 - 1, out_h, device=image.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([xx, yy], dim=-1)[None]
            crops.append(F.grid_sample(image[bi : bi + 1], grid, align_corners=True))
            owner.append(bi)
    if not crops:
        return image.new_zeros(0, c, out_h, out_w), torch.zeros(0, dtype=torch.long, device=image.device)
    return torch.cat(crops, dim=0), torch.tensor(owner, dtype=torch.long, device=image.device)


class LineHTR(nn.Module):
    """CNN-BiGRU-CTC recognizer with full/base/modifier/tone heads."""

    def __init__(self, vocab_sizes: dict[str, int], hidden_dim: int = 384) -> None:
        super().__init__()
        self.full_classes = vocab_sizes["full"] + 1
        self.blank_id = vocab_sizes["full"]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, 3, padding=1), nn.SiLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1), nn.SiLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1), nn.SiLU(),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True, num_layers=3, dropout=0.1)
        self.full = nn.Linear(hidden_dim, self.full_classes)
        self.base = nn.Linear(hidden_dim, vocab_sizes["base"])
        self.modifier = nn.Linear(hidden_dim, vocab_sizes["modifier"])
        self.tone = nn.Linear(hidden_dim, vocab_sizes["tone"])

    def forward(self, line_images: torch.Tensor) -> dict[str, torch.Tensor]:
        if line_images.numel() == 0:
            raise ValueError("LineHTR received no line crops")
        feat = self.cnn(line_images)
        seq = feat.mean(dim=2).transpose(1, 2)
        seq, _ = self.rnn(seq)
        return {"seq": seq, "full": self.full(seq), "base": self.base(seq), "modifier": self.modifier(seq), "tone": self.tone(seq)}

    def ctc_loss(self, line_images: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        logits = self.forward(line_images)["full"].log_softmax(-1).transpose(0, 1)
        input_lengths = torch.full((line_images.shape[0],), logits.shape[0], dtype=torch.long, device=line_images.device)
        return F.ctc_loss(logits, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
