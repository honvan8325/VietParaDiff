#!/usr/bin/env python3
"""Run a pinned external baseline through the common JSONL protocol."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.baselines import (
    ExternalBaselineRunner,
    load_external_baseline_config,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pinned One-DM or Paragraph LDM baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = ExternalBaselineRunner(
        load_external_baseline_config(args.config)
    ).run()
    print(
        "Baseline complete: "
        f"name={summary['baseline']} "
        f"pairs={summary['pair_count']} "
        f"samples={summary['sample_count']}"
    )


if __name__ == "__main__":
    main()
