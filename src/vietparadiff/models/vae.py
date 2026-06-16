"""Dual-band variational autoencoder for paragraph images."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from .common import ResidualBlock, group_norm
from .style import haar_highpass


class DiagonalGaussian:
    """Diagonal Gaussian posterior wrapper used by the VAE."""

    def __init__(self, stats: torch.Tensor) -> None:
        self.mean, self.logvar = stats.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30, 20)

    def sample(self) -> torch.Tensor:
        return self.mean + torch.randn_like(self.mean) * torch.exp(0.5 * self.logvar)

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.pow(2) + self.logvar.exp() - self.logvar - 1.0).mean()


class Encoder(nn.Module):
    """Residual CNN encoder that downsamples images by 8x.

    The encoder supports activation checkpointing because paragraph canvases can
    be 1024 px wide and up to 1536 px tall.  Checkpointing recomputes internal
    activations during backward instead of storing every full-resolution feature
    map, which is the correct trade-off for paper-scale dynamic canvases.
    """

    def __init__(self, in_ch: int, latent_ch: int, base: int, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(), ResidualBlock(base),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 2),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 4),
            nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 4),
            group_norm(base * 4), nn.SiLU(), nn.Conv2d(base * 4, latent_ch * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> DiagonalGaussian:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            stats = checkpoint_sequential(self.net, segments=6, input=x, use_reentrant=False)
        else:
            stats = self.net(x)
        return DiagonalGaussian(stats)


class DualBandVAE(nn.Module):
    """VAE with low-frequency and high-frequency latent posteriors."""

    def __init__(self, in_ch: int = 1, latent_ch: int = 8, base: int = 128, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.low_encoder = Encoder(in_ch, latent_ch, base, gradient_checkpointing)
        self.high_encoder = Encoder(in_ch, latent_ch, base, gradient_checkpointing)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch * 2, base * 4, 3, padding=1), nn.SiLU(), ResidualBlock(base * 4),
            nn.ConvTranspose2d(base * 4, base * 4, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 4),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base * 2),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.SiLU(), ResidualBlock(base),
            nn.Conv2d(base, in_ch, 3, padding=1), nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> dict[str, DiagonalGaussian]:
        return {"low": self.low_encoder(x), "high": self.high_encoder(haar_highpass(x))}

    def decode(self, low_z: torch.Tensor, high_z: torch.Tensor) -> torch.Tensor:
        z = torch.cat([low_z, high_z], dim=1)
        if self.training and self.gradient_checkpointing and z.requires_grad:
            return checkpoint_sequential(self.decoder, segments=6, input=z, use_reentrant=False)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | DiagonalGaussian]:
        posts = self.encode(x)
        low_z = posts["low"].sample()
        high_z = posts["high"].sample()
        recon = self.decode(low_z, high_z)
        return {"recon": recon, "low_z": low_z, "high_z": high_z, "low_post": posts["low"], "high_post": posts["high"]}


def vae_loss(out: dict[str, torch.Tensor | DiagonalGaussian], target: torch.Tensor, kl_weight: float, freq_weight: float, orth_weight: float) -> dict[str, torch.Tensor]:
    """Compute pixel, KL, high-frequency, and low/high decorrelation losses."""

    pixel = F.l1_loss(out["recon"], target)
    freq = F.l1_loss(haar_highpass(out["recon"]), haar_highpass(target))
    kl = out["low_post"].kl() + out["high_post"].kl()
    low = F.normalize(out["low_z"].flatten(1), dim=1)
    high = F.normalize(out["high_z"].flatten(1), dim=1)
    orth = (low * high).sum(dim=1).abs().mean()
    total = pixel + kl_weight * kl + freq_weight * freq + orth_weight * orth
    return {"loss": total, "vae/pixel": pixel, "vae/kl": kl, "vae/frequency": freq, "vae/orthogonality": orth}
