#!/usr/bin/env python3
"""Train the independent ArcFace writer-style metric."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.runtime import resolve_runtime, seed_everything
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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume strict epoch-boundary state from writer last.pt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_writer_metric_config(args.config)
    seed_everything(config.seed)
    runtime = resolve_runtime(config.device, config.precision)
    metrics = train_writer_metric(
        config,
        runtime=runtime,
        resume=args.resume,
    )
    print(
        "Writer metric complete: "
        f"AUC={metrics['validation_auc']:.6f} "
        f"EER={metrics['validation_eer']:.6f}"
    )


if __name__ == "__main__":
    main()
