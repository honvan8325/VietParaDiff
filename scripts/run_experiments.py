#!/usr/bin/env python3
"""Run the cumulative VietParaDiff ablation DAG for three seeds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.experiments import (
    PaperExperimentRunner,
    load_experiment_config,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cumulative A0-Full VietParaDiff experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/paper.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    runner = PaperExperimentRunner(
        load_experiment_config(args.config),
        allow_dirty=args.allow_dirty,
    )
    runner.run(dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
