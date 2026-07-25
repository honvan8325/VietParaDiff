"""Auxiliary Vietnamese CNN-Conformer HTR teacher."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import HTRConfig


@dataclass(frozen=True, slots=True)
class HTROutput:
    raw_logits: Tensor
    base_logits: Tensor
    shape_logits: Tensor
    tone_logits: Tensor
    input_lengths: Tensor


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.conv_norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, 31, padding=15, groups=dim)
        self.conv_group_norm = nn.GroupNorm(1, dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.conv_dropout = nn.Dropout(dropout)
        self.ffn2_norm = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attended
        convolved = self.conv_norm(x).transpose(1, 2)
        convolved = F.glu(self.pointwise_in(convolved), dim=1)
        convolved = F.silu(self.conv_group_norm(self.depthwise(convolved)))
        convolved = self.conv_dropout(self.pointwise_out(convolved)).transpose(1, 2)
        x = x + convolved
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.output_norm(x)


class VietnameseHTR(nn.Module):
    """Line-level Vietnamese HTR teacher.

    Input phải là ảnh một dòng chữ ``[B, 1, 64, W]``.  Generated
    paragraphs phải được crop thành từng line bằng layout slots trước khi
    đưa vào teacher; module này không nhận trực tiếp paragraph nhiều dòng.
    """

    def __init__(self, config: HTRConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )
        self.position_embedding = nn.Parameter(torch.randn(2048, 256) * 0.02)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    config.model_dim,
                    config.num_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.raw_head = nn.Linear(256, config.raw_vocab_size)
        self.base_head = nn.Linear(256, config.base_vocab_size)
        self.shape_head = nn.Linear(256, config.shape_vocab_size)
        self.tone_head = nn.Linear(256, config.tone_vocab_size)

    def forward(
        self, images: Tensor, valid_widths: Tensor | None = None
    ) -> HTROutput:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"HTR images phải có shape [B,1,H,W], nhận {tuple(images.shape)}.")
        if images.shape[2] != 64:
            raise ValueError(
                "VietnameseHTR chỉ nhận ảnh một dòng cao 64px, "
                f"nhận height={images.shape[2]}."
            )
        if not images.is_floating_point():
            raise TypeError("HTR images phải có floating-point dtype.")
        batch, width = images.shape[0], images.shape[-1]
        if valid_widths is None:
            valid_widths = torch.full(
                (batch,), width, dtype=torch.long, device=images.device
            )
        if valid_widths.shape != (batch,) or valid_widths.dtype != torch.long:
            raise ValueError(f"valid_widths phải là torch.long shape [{batch}].")
        if valid_widths.device != images.device:
            raise ValueError("valid_widths và images phải cùng device.")
        if ((valid_widths <= 0) | (valid_widths > width)).any():
            raise ValueError("valid_widths phải nằm trong (0, image_width].")
        features = self.visual_encoder(images).mean(dim=2).transpose(1, 2)
        length = features.shape[1]
        if length > self.position_embedding.shape[0]:
            raise ValueError("HTR sequence vượt giới hạn 2048.")
        features = features + self.position_embedding[:length][None]
        input_lengths = (valid_widths + 3) // 4
        positions = torch.arange(length, device=images.device)[None]
        padding = positions >= input_lengths[:, None]
        features = features.masked_fill(padding[:, :, None], 0.0)
        for block in self.blocks:
            features = block(features, padding).masked_fill(padding[:, :, None], 0.0)
        return HTROutput(
            self.raw_head(features),
            self.base_head(features),
            self.shape_head(features),
            self.tone_head(features),
            input_lengths,
        )

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy HTR checkpoint: {path}")
        state: object = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and set(state) == {"model"}:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise ValueError("HTR checkpoint không phải state_dict hợp lệ.")
        self.load_state_dict(dict(state), strict=True)
