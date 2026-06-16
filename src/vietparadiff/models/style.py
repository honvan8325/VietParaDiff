"""One-shot factorized style encoder with separate style concept codebooks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import ResidualBlock, VectorQuantizer, group_norm


def haar_highpass(x: torch.Tensor) -> torch.Tensor:
    """Approximate wavelet-style high-pass response for stroke and diacritic details."""

    pooled = F.avg_pool2d(x, kernel_size=2, stride=1, padding=1)[:, :, : x.shape[-2], : x.shape[-1]]
    return (x - pooled).abs()


class FactorizedStyleEncoder(nn.Module):
    """Encode a single reference image into global/layout/stroke style tokens.

    The encoder uses four views: original spatial image, high-pass image, row-mask
    image, and column-mask image.  Three independent learned query sets attend to
    pooled spatial features, then each group is quantized by its own codebook.
    """

    def __init__(
        self,
        dim: int = 384,
        tokens_per_group: int = 16,
        codebook_size: int = 1024,
        writer_classes: int = 10000,
        base_channels: int = 96,
        transformer_layers: int = 4,
        heads: int = 8,
        pooled_grid: tuple[int, int] | list[int] = (8, 32),
    ) -> None:
        super().__init__()
        self.tokens_per_group = tokens_per_group
        self.pooled_grid = tuple(pooled_grid)
        self.backbone = nn.Sequential(
            nn.Conv2d(4, base_channels, 3, padding=1), nn.SiLU(), ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base_channels * 4),
            nn.Conv2d(base_channels * 4, dim, 3, padding=1), nn.SiLU(), group_norm(dim),
        )
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, norm_first=True, activation="gelu")
        self.spatial_transformer = nn.TransformerEncoder(enc_layer, transformer_layers)
        self.global_queries = nn.Parameter(torch.randn(tokens_per_group, dim) * 0.02)
        self.layout_queries = nn.Parameter(torch.randn(tokens_per_group, dim) * 0.02)
        self.stroke_queries = nn.Parameter(torch.randn(tokens_per_group, dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layout_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.stroke_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.global_q = VectorQuantizer(codebook_size, dim)
        self.layout_q = VectorQuantizer(codebook_size, dim)
        self.stroke_q = VectorQuantizer(codebook_size, dim)
        self.writer_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, writer_classes))

    def _stripe_mask(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Random row/column stripe masking used only during training."""

        if not self.training:
            return x
        mask = torch.ones_like(x)
        if mode == "row":
            every = max(2, x.shape[-2] // 8)
            offset = torch.randint(0, every, (), device=x.device).item()
            mask[:, :, offset::every, :] = 0
        else:
            every = max(2, x.shape[-1] // 8)
            offset = torch.randint(0, every, (), device=x.device).item()
            mask[:, :, :, offset::every] = 0
        return x * mask

    def _query(self, tokens: torch.Tensor, queries: torch.Tensor, attn: nn.MultiheadAttention) -> torch.Tensor:
        q = queries[None].expand(tokens.shape[0], -1, -1)
        out, _ = attn(q, tokens, tokens, need_weights=False)
        return out

    def forward(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial = reference
        high = haar_highpass(reference)
        row = self._stripe_mask(reference, "row")
        col = self._stripe_mask(reference, "col")
        views = torch.cat([spatial, high, row, col], dim=1)
        feat = self.backbone(views)
        pooled = F.adaptive_avg_pool2d(feat, self.pooled_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.spatial_transformer(tokens)
        g = self._query(tokens, self.global_queries, self.global_attn)
        l = self._query(tokens, self.layout_queries, self.layout_attn)
        s = self._query(tokens, self.stroke_queries, self.stroke_attn)
        gq, lq, sq = self.global_q(g), self.layout_q(l), self.stroke_q(s)
        global_style = gq["quantized"].mean(dim=1)
        return {
            "global": gq["quantized"],
            "layout": lq["quantized"],
            "stroke": sq["quantized"],
            "global_style": global_style,
            "all_tokens": torch.cat([gq["quantized"], lq["quantized"], sq["quantized"]], dim=1),
            "vq_loss": gq["loss"] + lq["loss"] + sq["loss"],
            "perplexity_global": gq["perplexity"],
            "perplexity_layout": lq["perplexity"],
            "perplexity_stroke": sq["perplexity"],
            "perplexity": (gq["perplexity"] + lq["perplexity"] + sq["perplexity"]) / 3,
            "writer_logits": self.writer_head(global_style),
        }
