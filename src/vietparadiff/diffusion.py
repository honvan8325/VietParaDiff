"""Shared cosine schedule and velocity-diffusion tensor operations."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_alpha_sigma(
    timesteps: Tensor,
    *,
    num_train_timesteps: int,
) -> tuple[Tensor, Tensor]:
    """Return midpoint cosine alpha/sigma values for integer timesteps."""
    if (
        timesteps.ndim != 1
        or timesteps.dtype != torch.long
        or num_train_timesteps < 2
    ):
        raise ValueError(
            "timesteps phải là long [B] và num_train_timesteps >= 2."
        )
    if (
        (timesteps < 0).any()
        or (timesteps >= num_train_timesteps).any()
    ):
        raise ValueError(
            f"timesteps phải nằm trong [0,{num_train_timesteps - 1}]."
        )
    offset = 0.008
    progress = (timesteps.float() + 0.5) / num_train_timesteps
    angles = (
        (progress + offset) / (1.0 + offset) * math.pi / 2.0
    )
    return torch.cos(angles), torch.sin(angles)


def velocity_target(
    clean_latents: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Construct the v-prediction target for a noised latent batch."""
    if clean_latents.shape != noise.shape:
        raise ValueError("clean_latents và noise phải cùng shape.")
    if clean_latents.ndim != 4:
        raise ValueError("clean_latents phải có shape [B,C,H,W].")
    expected = (clean_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(f"alpha/sigma phải có shape {expected}.")
    return (
        alpha[:, None, None, None].to(clean_latents.dtype) * noise
        - sigma[:, None, None, None].to(clean_latents.dtype)
        * clean_latents
    )


def add_diffusion_noise(
    clean_latents: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Mix clean latents and Gaussian noise at the supplied schedule point."""
    if clean_latents.shape != noise.shape or clean_latents.ndim != 4:
        raise ValueError(
            "clean_latents/noise phải cùng shape [B,C,H,W]."
        )
    expected = (clean_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(f"alpha/sigma phải có shape {expected}.")
    return (
        alpha[:, None, None, None].to(clean_latents.dtype)
        * clean_latents
        + sigma[:, None, None, None].to(clean_latents.dtype) * noise
    )


__all__ = [
    "add_diffusion_noise",
    "cosine_alpha_sigma",
    "velocity_target",
]
