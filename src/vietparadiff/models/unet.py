"""Conditional latent diffusion U-Net and high-band stroke refiner."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import ResidualBlock, group_norm, sinusoidal_embedding


class SpatialCrossAttention(nn.Module):
    """Cross-attention from spatial latent positions to content/style context tokens."""

    def __init__(self, channels: int, context_dim: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = self.norm(x).flatten(2).transpose(1, 2)
        q = self.q(seq)
        k, v = self.kv(context).chunk(2, dim=-1)
        key_padding_mask = None if context_mask is None else ~context_mask
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.out(out).transpose(1, 2).view(b, c, h, w)
        return x + out


class ConditionalUNet(nn.Module):
    """Multi-scale U-Net for epsilon prediction in low-band latent diffusion."""

    def __init__(
        self,
        latent_ch: int = 8,
        layout_ch: int = 5,
        context_dim: int = 384,
        base: int = 128,
        time_dim: int = 384,
        channel_mults: tuple[int, ...] | list[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.layout_adapter = nn.Conv2d(layout_ch, latent_ch, 1)
        nn.init.zeros_(self.layout_adapter.weight)
        nn.init.zeros_(self.layout_adapter.bias)
        self.in_conv = nn.Conv2d(latent_ch, base, 3, padding=1)
        channels = [base * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base
        for ch in channels:
            blocks = nn.ModuleList([ResidualBlock(in_ch if j == 0 else ch, ch, time_dim) for j in range(num_res_blocks)])
            self.down_blocks.append(blocks)
            self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
            in_ch = ch
        mid_ch = channels[-1]
        self.mid1 = ResidualBlock(mid_ch, mid_ch, time_dim)
        self.attn = SpatialCrossAttention(mid_ch, context_dim, heads=heads)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, time_dim)
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for ch in reversed(channels):
            self.upsamples.append(nn.ConvTranspose2d(in_ch, ch, 4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList([ResidualBlock(ch + ch if j == 0 else ch, ch, time_dim) for j in range(num_res_blocks)]))
            in_ch = ch
        self.out_norm = group_norm(base)
        self.out = nn.Conv2d(base, latent_ch, 3, padding=1)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, layout: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        if layout.shape[-2:] != noisy.shape[-2:]:
            layout = F.interpolate(layout, size=noisy.shape[-2:], mode="bilinear", align_corners=False)
        x = noisy + self.layout_adapter(layout)
        t = self.time_mlp(sinusoidal_embedding(timestep, self.time_dim))
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, t)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, t)
        h = self.attn(h, context, context_mask)
        h = self.mid2(h, t)
        for up, blocks in zip(self.upsamples, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t)
        return self.out(F.silu(self.out_norm(h)))


class DiacriticStrokeRefiner(nn.Module):
    """Predict high-frequency latent from low latent, layout, and stroke style."""

    def __init__(self, low_ch: int = 8, high_ch: int = 8, layout_ch: int = 5, style_dim: int = 384, base: int = 128) -> None:
        super().__init__()
        self.style = nn.Linear(style_dim, base * 2)
        self.in_conv = nn.Conv2d(low_ch + layout_ch, base, 3, padding=1)
        self.blocks = nn.Sequential(ResidualBlock(base, base, base * 2), ResidualBlock(base, base, base * 2), ResidualBlock(base, base, base * 2))
        self.out = nn.Conv2d(base, high_ch, 3, padding=1)

    def forward(self, low_z: torch.Tensor, layout: torch.Tensor, stroke_style: torch.Tensor) -> torch.Tensor:
        if layout.shape[-2:] != low_z.shape[-2:]:
            layout = F.interpolate(layout, size=low_z.shape[-2:], mode="bilinear", align_corners=False)
        style = self.style(stroke_style.mean(dim=1))
        h = self.in_conv(torch.cat([low_z, layout], dim=1))
        for block in self.blocks:
            h = block(h, style)
        return self.out(h)
