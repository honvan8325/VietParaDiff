"""Common neural-network layers for VietParaDiff."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Create GroupNorm with a valid group count for any channel size."""

    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""

    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


class ResidualBlock(nn.Module):
    """Residual ConvNeXt-like block with optional FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int | None = None, cond_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        out_ch = out_ch or in_ch
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch * 2) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.cond is not None and cond is not None:
            scale, shift = self.cond(cond).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with commitment and codebook losses."""

    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        dist = flat.pow(2).sum(dim=1, keepdim=True) - 2 * flat @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(dim=1)[None, :]
        indices = dist.argmin(dim=1)
        quantized = self.codebook(indices).view_as(x)
        codebook_loss = (quantized - x.detach()).pow(2).mean()
        commit_loss = (x - quantized.detach()).pow(2).mean()
        loss = codebook_loss + self.beta * commit_loss
        quantized = x + (quantized - x).detach()
        counts = torch.bincount(indices, minlength=self.codebook.num_embeddings).float()
        probs = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-9).log()).sum())
        return {"quantized": quantized, "loss": loss, "indices": indices.view(x.shape[:-1]), "perplexity": perplexity}
