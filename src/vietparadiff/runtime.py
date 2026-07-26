"""Shared device, precision, autocast, scaler, and seeding utilities."""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class RuntimePrecision:
    """Resolved execution device and numeric precision."""

    device: torch.device
    dtype: torch.dtype
    autocast_enabled: bool
    scaler_enabled: bool


def resolve_runtime(device: str, precision: str) -> RuntimePrecision:
    """Resolve an explicit or automatic device/precision pair."""
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS được yêu cầu nhưng không khả dụng.")

    if precision == "auto":
        if resolved_device.type == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32
    else:
        try:
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[precision]
        except KeyError as error:
            raise ValueError(
                "precision phải là auto, float32, float16 hoặc bfloat16."
            ) from error

    if resolved_device.type != "cuda" and dtype != torch.float32:
        raise ValueError("CPU/MPS chỉ hỗ trợ float32.")

    return RuntimePrecision(
        device=resolved_device,
        dtype=dtype,
        autocast_enabled=(
            resolved_device.type == "cuda" and dtype != torch.float32
        ),
        scaler_enabled=(
            resolved_device.type == "cuda" and dtype == torch.float16
        ),
    )


def create_grad_scaler(
    runtime: RuntimePrecision,
) -> torch.amp.GradScaler:
    """Create the CUDA gradient scaler required by the resolved runtime."""
    return torch.amp.GradScaler("cuda", enabled=runtime.scaler_enabled)


def autocast_context(runtime: RuntimePrecision) -> Any:
    """Return the appropriate autocast context for one forward pass."""
    if not runtime.autocast_enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=runtime.dtype,
        enabled=True,
    )


def seed_everything(seed: int) -> None:
    """Seed Python and Torch RNGs without silently accepting negative seeds."""
    if seed < 0:
        raise ValueError("seed không được âm.")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    """Capture Python/Torch accelerator RNG for epoch-boundary resume."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "mps": (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None
        ),
    }
    return state


def restore_rng_state(state: object) -> None:
    """Strictly restore a state created by :func:`capture_rng_state`."""
    if not isinstance(state, dict) or set(state) != {
        "python",
        "torch",
        "cuda",
        "mps",
    }:
        raise ValueError("RNG checkpoint sai schema.")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint CUDA RNG không thể restore trên host này.")
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Checkpoint MPS RNG không thể restore trên host này.")
        torch.mps.set_rng_state(state["mps"])


__all__ = [
    "RuntimePrecision",
    "autocast_context",
    "capture_rng_state",
    "create_grad_scaler",
    "resolve_runtime",
    "restore_rng_state",
    "seed_everything",
]
