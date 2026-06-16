"""Device selection utilities."""

from __future__ import annotations

import torch


def choose_device(preferred: str = "auto") -> torch.device:
    """Choose CUDA, MPS, or CPU."""

    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
