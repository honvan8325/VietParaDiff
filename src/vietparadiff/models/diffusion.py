"""DDPM/DDIM utilities for low-band latent diffusion."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from improved DDPM."""

    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


class LatentDiffusion:
    """Noise schedule, training loss, and DDIM sampler for low latent diffusion."""

    def __init__(self, steps: int = 1000, device: torch.device | None = None, min_snr_gamma: float = 5.0) -> None:
        self.steps = int(steps)
        self.device = device or torch.device("cpu")
        self.betas = cosine_beta_schedule(self.steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.min_snr_gamma = float(min_snr_gamma)

    def to(self, device: torch.device) -> "LatentDiffusion":
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t).float()
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.alpha_bar.sqrt(), t, clean.shape) * clean + self.extract((1 - self.alpha_bar).sqrt(), t, clean.shape) * noise

    def predict_clean(self, noisy: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        return (noisy - self.extract((1 - self.alpha_bar).sqrt(), t, noisy.shape) * noise_pred) / self.extract(self.alpha_bar.sqrt(), t, noisy.shape).clamp_min(1e-6)

    def noise_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(noise_pred, noise, reduction="none").flatten(1).mean(dim=1)
        snr = self.alpha_bar[t] / (1 - self.alpha_bar[t]).clamp_min(1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr.clamp_min(1e-8)
        return (mse * weight).mean()

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple[int, ...],
        steps: int,
        context: torch.Tensor,
        layout: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        guidance: float = 5.0,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling with classifier-free guidance."""

        device = context.device
        self.to(device)
        z = torch.randn(shape, device=device)
        schedule = torch.linspace(self.steps - 1, 0, steps, device=device).long()
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        for idx, t_scalar in enumerate(schedule):
            t = torch.full((shape[0],), int(t_scalar.item()), dtype=torch.long, device=device)
            eps_cond = model(z, t, context, layout, context_mask)
            if guidance != 1.0:
                eps_uncond = model(z, t, torch.zeros_like(context), torch.zeros_like(layout), context_mask)
                eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            x0 = self.predict_clean(z, t, eps)
            if idx == len(schedule) - 1:
                z = x0
            else:
                t_next = schedule[idx + 1].repeat(shape[0])
                ab_next = self.extract(self.alpha_bar, t_next, z.shape)
                z = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return z
