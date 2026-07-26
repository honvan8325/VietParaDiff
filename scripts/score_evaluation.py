#!/usr/bin/env python3
"""Score fixed-pair generated PNGs without sampling again."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.evaluation.scoring import (
    EvaluationScorer,
    load_scoring_config,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute VietParaDiff paper metrics from fixed PNGs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietparadiff/metrics.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scorer = EvaluationScorer(load_scoring_config(args.config))
    summary = scorer.run(resume=args.resume)
    print(
        "Scoring complete: "
        f"pairs={summary['pair_count']} "
        f"samples={summary['sample_count']}"
    )


if __name__ == "__main__":
    main()
