#!/usr/bin/env python3
"""Aggregate three training seeds into paper tables."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.experiments import (
    aggregate_experiments,
    load_experiment_config,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate VietParaDiff multi-seed metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/paper.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/aggregate"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    aggregate_experiments(
        load_experiment_config(args.config),
        output_dir=args.output_dir,
    )
    print(f"Aggregated results: {args.output_dir}")


if __name__ == "__main__":
    main()
