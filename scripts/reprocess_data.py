"""Command-line dispatcher for rebuilding one supported dataset."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    # Direct execution sets sys.path[0] to ``scripts`` rather than the project
    # root. Add the root so ``src`` imports work for both ``python script.py``
    # and ``python -m scripts.reprocess_data``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    build_cvl_dataset,
    build_iam_dataset,
    build_uithwdb_dataset,
    build_vnondb_dataset,
)

DATASET_BUILDERS: dict[str, Callable[[], None]] = {
    "cvl": build_cvl_dataset,
    "iam": build_iam_dataset,
    "uithwdb": build_uithwdb_dataset,
    "vnondb": build_vnondb_dataset,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the dataset name from command-line arguments.

    Args:
        argv: Optional explicit arguments for tests or programmatic use.
            ``None`` delegates to ``sys.argv``.

    Returns:
        A namespace whose ``dataset`` value is a key in ``DATASET_BUILDERS``.
    """
    parser = argparse.ArgumentParser(
        description="Rebuild one handwriting dataset from its raw data.",
    )
    parser.add_argument(
        "dataset",
        choices=DATASET_BUILDERS,
        help="Dataset to rebuild.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the selected dataset to its builder."""
    args = parse_args(argv)
    DATASET_BUILDERS[args.dataset]()


if __name__ == "__main__":
    main()
