"""AutoKL grayscale train từ đầu cho ảnh paragraph chữ viết tay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import AutoKLConfig


class DiagonalGaussianDistribution:
    def __init__(self, moments: Tensor) -> None:
        if moments.ndim != 4 or moments.shape[1] % 2:
            raise ValueError(
                f"moments phải có shape [B, 2*C, H, W], nhận {tuple(moments.shape)}."
            )
        self.mean, logvar = moments.chunk(2, dim=1)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar.float()).to(moments.dtype)
        self.var = torch.exp(self.logvar.float())

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean + self.std * noise

    def mode(self) -> Tensor:
        return self.mean

    def kl(self) -> Tensor:
        mean = self.mean.float()
        return 0.5 * (
            mean.square() + self.var - 1.0 - self.logvar.float()
        ).sum(dim=(1, 2, 3))

    def nll(self, sample: Tensor) -> Tensor:
        if sample.shape != self.mean.shape:
            raise ValueError(
                f"sample phải có shape {tuple(self.mean.shape)}, "
                f"nhận {tuple(sample.shape)}."
            )
        difference = sample.float() - self.mean.float()
        return 0.5 * (
            torch.log(2.0 * torch.pi * self.var)
            + difference.square() / self.var
        ).sum(dim=(1, 2, 3))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return residual + x


class AxialAttention(nn.Module):
    """Row attention rồi column attention tại AutoKL bottleneck."""

    def __init__(self, channels: int, heads: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.row_norm = nn.LayerNorm(channels)
        self.row_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.column_norm = nn.LayerNorm(channels)
        self.column_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        normalized = self.row_norm(rows)
        attended, _ = self.row_attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = (rows + attended).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        columns = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        normalized = self.column_norm(columns)
        attended, _ = self.column_attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = (columns + attended).reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        return residual + self.output(x)


class Encoder(nn.Module):
    def __init__(self, config: AutoKLConfig) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            config.base_channels * item for item in config.channel_multipliers
        )
        groups, dropout = config.group_norm_groups, config.dropout
        self.input_conv = nn.Conv2d(1, c1, 3, padding=1)
        self.level1 = nn.Sequential(
            ResBlock(c1, c1, groups, dropout), ResBlock(c1, c1, groups, dropout)
        )
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.level2 = nn.Sequential(
            ResBlock(c2, c2, groups, dropout), ResBlock(c2, c2, groups, dropout)
        )
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.level3 = nn.Sequential(
            ResBlock(c3, c3, groups, dropout), ResBlock(c3, c3, groups, dropout)
        )
        self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
        self.level4 = nn.Sequential(
            ResBlock(c4, c4, groups, dropout), ResBlock(c4, c4, groups, dropout)
        )
        self.middle1 = ResBlock(c4, c4, groups, dropout)
        self.attention = AxialAttention(c4, config.attention_heads, groups)
        self.middle2 = ResBlock(c4, c4, groups, dropout)
        self.output_norm = nn.GroupNorm(groups, c4)
        self.output_conv = nn.Conv2d(c4, 2 * config.latent_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.level1(self.input_conv(x))
        x = self.level2(self.down1(x))
        x = self.level3(self.down2(x))
        x = self.level4(self.down3(x))
        x = self.middle2(self.attention(self.middle1(x)))
        return self.output_conv(F.silu(self.output_norm(x)))


class Decoder(nn.Module):
    def __init__(self, config: AutoKLConfig) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            config.base_channels * item for item in config.channel_multipliers
        )
        groups, dropout = config.group_norm_groups, config.dropout
        self.input_conv = nn.Conv2d(config.latent_channels, c4, 3, padding=1)
        self.middle1 = ResBlock(c4, c4, groups, dropout)
        self.attention = AxialAttention(c4, config.attention_heads, groups)
        self.middle2 = ResBlock(c4, c4, groups, dropout)
        self.level4 = nn.Sequential(
            ResBlock(c4, c4, groups, dropout), ResBlock(c4, c4, groups, dropout)
        )
        self.up3 = nn.Conv2d(c4, c3, 3, padding=1)
        self.level3 = nn.Sequential(
            ResBlock(c3, c3, groups, dropout), ResBlock(c3, c3, groups, dropout)
        )
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.level2 = nn.Sequential(
            ResBlock(c2, c2, groups, dropout), ResBlock(c2, c2, groups, dropout)
        )
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.level1 = nn.Sequential(
            ResBlock(c1, c1, groups, dropout), ResBlock(c1, c1, groups, dropout)
        )
        self.output_norm = nn.GroupNorm(groups, c1)
        self.output_conv = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.middle2(self.attention(self.middle1(self.input_conv(x))))
        x = self.level4(x)
        x = self.level3(self.up3(F.interpolate(x, scale_factor=2.0, mode="nearest")))
        x = self.level2(self.up2(F.interpolate(x, scale_factor=2.0, mode="nearest")))
        x = self.level1(self.up1(F.interpolate(x, scale_factor=2.0, mode="nearest")))
        return torch.tanh(self.output_conv(F.silu(self.output_norm(x))))


@dataclass(frozen=True, slots=True)
class AutoKLOutput:
    reconstruction: Tensor
    posterior: DiagonalGaussianDistribution
    latent: Tensor


class HandwritingAutoKL(nn.Module):
    def __init__(self, config: AutoKLConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoKLConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def encode(self, images: Tensor) -> DiagonalGaussianDistribution:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"images phải có shape [B,1,H,W], nhận {tuple(images.shape)}.")
        if not images.is_floating_point():
            raise TypeError("images phải có floating-point dtype.")
        if images.shape[-2] % 8 or images.shape[-1] % 8:
            raise ValueError("H và W của images phải chia hết cho 8.")
        if not torch.isfinite(images).all():
            raise ValueError("images phải chứa toàn giá trị hữu hạn.")
        if images.min() < -1.0 or images.max() > 1.0:
            raise ValueError("images phải được normalize vào [-1, 1].")
        return DiagonalGaussianDistribution(self.encoder(images))

    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError(f"latents phải có shape [B,4,H,W], nhận {tuple(latents.shape)}.")
        if not latents.is_floating_point() or not torch.isfinite(latents).all():
            raise ValueError("latents phải là floating-point hữu hạn.")
        return self.decoder(latents)

    def forward(
        self,
        images: Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> AutoKLOutput:
        posterior = self.encode(images)
        latent = posterior.sample(generator) if sample_posterior else posterior.mode()
        return AutoKLOutput(self.decode(latent), posterior, latent)

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy AutoKL checkpoint: {path}")
        state: object = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and set(state) == {"model"}:
            state = state["model"]
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state.items()
        ):
            raise ValueError("Checkpoint phải là state_dict hoặc {'model': state_dict}.")
        self.load_state_dict(dict(state), strict=True)
