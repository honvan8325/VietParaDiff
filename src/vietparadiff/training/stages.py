"""Training stage dispatch and module freezing policy."""

from __future__ import annotations

import torch

STAGES = {"vae", "htr", "style_layout", "topology", "diffusion"}


def set_trainable(model: torch.nn.Module, names: list[str]) -> None:
    """Enable gradients only for the selected top-level modules."""

    allowed = set(names)
    for module_name, module in model.named_children():
        requires = module_name in allowed
        for p in module.parameters():
            p.requires_grad = requires


def configure_stage(model, stage: str) -> None:
    """Freeze/unfreeze modules for the staged paper curriculum.

    The intended order is VAE -> HTR -> style_layout -> topology -> diffusion.
    Topology uses frozen content/style/layout checkpoints to create layout-aware
    targets and trains only the detector.  Diffusion trains only the U-Net and
    high-band refiner while consuming frozen VAE/style-layout/topology modules.
    """

    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}; expected one of {sorted(STAGES)}")
    if stage == "vae":
        set_trainable(model, ["vae"])
    elif stage == "htr":
        set_trainable(model, ["htr"])
    elif stage == "style_layout":
        set_trainable(model, ["content", "style", "layout"])
    elif stage == "topology":
        set_trainable(model, ["topology"])
    elif stage == "diffusion":
        set_trainable(model, ["unet", "refiner"])


def forward_stage(model, batch: dict, stage: str, global_step: int = 0) -> dict[str, torch.Tensor]:
    """Dispatch one batch to the correct stage-specific forward method."""

    if stage == "vae":
        return model.forward_vae(batch)
    if stage == "htr":
        return model.forward_htr(batch)
    if stage == "style_layout":
        return model.forward_style_layout(batch, global_step)
    if stage == "topology":
        return model.forward_topology(batch)
    if stage == "diffusion":
        return model.forward_diffusion(batch, global_step)
    raise ValueError(f"Unknown stage {stage}")
