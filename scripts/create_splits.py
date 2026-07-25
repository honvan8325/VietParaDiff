"""Create deterministic writer-disjoint training manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import SplitConfig, create_data_splits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create writer-disjoint AutoKL, HTR, and VietParaDiff manifests."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing the five normalized dataset manifests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/splits"),
        help="Destination directory (default: data/splits).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Writer fraction reserved for test in each corpus family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic writer and reference selection seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    counts = create_data_splits(
        SplitConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            test_fraction=args.test_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
