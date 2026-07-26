#!/usr/bin/env python3
"""Train the independent ArcFace writer-style metric."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from vietparadiff.runtime import seed_everything
from vietparadiff.training.writer import (
    load_writer_metric_config,
    train_writer_metric,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent ResNet-18 ArcFace writer metric."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/writer_metric/train.yaml"),
    )
    return parser.parse_args(argv)


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS được yêu cầu nhưng không khả dụng.")
    return device


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_writer_metric_config(args.config)
    seed_everything(config.seed)
    metrics = train_writer_metric(
        config,
        device=_device(config.device),
    )
    print(
        "Writer metric complete: "
        f"AUC={metrics['validation_auc']:.6f} "
        f"EER={metrics['validation_eer']:.6f}"
    )


if __name__ == "__main__":
    main()
