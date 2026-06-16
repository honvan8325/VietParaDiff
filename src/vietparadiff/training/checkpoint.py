"""Checkpoint save/load helpers for full and stage-scoped training.

Stage checkpoints contain the whole model state plus optimizer/scaler state for
exact same-stage resume.  Dependency loading restores only selected top-level
modules so a checkpoint from one stage cannot overwrite unrelated modules with
random weights from that stage run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    cfg: dict,
    extra: dict | None = None,
    scaler: Any | None = None,
) -> None:
    """Save a reproducible training checkpoint.

    ``step`` is the number of completed optimizer updates, not micro-batches.
    ``optimizer`` and AMP ``scaler`` are stored when available so ``--resume``
    continues the same training run instead of merely loading weights.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": int(step), "cfg": cfg, "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> dict:
    """Load a complete checkpoint into a model and optionally restore states."""

    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def load_modules_from_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    module_names: list[str],
    strict: bool = True,
) -> dict[str, Any]:
    """Load only selected top-level modules from a checkpoint.

    Parameters
    ----------
    path:
        Checkpoint produced by ``save_checkpoint``.
    model:
        Destination VietParaDiff model.
    module_names:
        Top-level module names to restore, for example ``["vae"]`` or
        ``["content", "style", "layout"]``.
    strict:
        When true, every selected key from the checkpoint must match a key in the
        destination model.  Unselected modules are ignored by design.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dependency checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    source = payload["model"]
    prefixes = tuple(f"{name}." for name in module_names)
    filtered = {k: v for k, v in source.items() if k.startswith(prefixes)}
    if not filtered:
        raise RuntimeError(f"No parameters for modules {module_names} found in {path}")
    current = model.state_dict()
    unexpected = sorted(k for k in filtered if k not in current)
    if strict and unexpected:
        raise RuntimeError(f"Checkpoint {path} has unexpected selected keys: {unexpected[:20]}")
    current.update({k: v for k, v in filtered.items() if k in current})
    model.load_state_dict(current, strict=True)
    return {"path": str(path), "modules": module_names, "step": payload.get("step"), "loaded_keys": len(filtered)}
