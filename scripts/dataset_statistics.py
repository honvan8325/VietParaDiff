"""Calculate sample and writer counts from normalized JSONL manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.table import Table

DATASETS = ("cvl", "iam", "uithwdb", "vnondb")


class DatasetStatistics(TypedDict):
    """Schema for one dataset row in table and JSON output."""

    dataset: str
    writers: int
    paragraphs: int
    lines: int
    words: int
    total: int


def collect_statistics(dataset: str, manifest_path: Path) -> DatasetStatistics:
    """Stream one manifest and calculate its sample-level statistics.

    Blank lines are ignored. Every non-blank line must contain a JSON object;
    malformed JSON fails fast with the exact source line to make corrupt
    manifests easy to diagnose.

    Args:
        dataset: Display name for the dataset.
        manifest_path: Path to the dataset's JSONL manifest.

    Returns:
        Counts for unique writers, each supported sample level, and all
        non-blank manifest records.

    Raises:
        ValueError: If a non-blank line is not a valid JSON object.
    """
    level_counts: Counter[str] = Counter()
    writer_ids: set[str] = set()
    total = 0

    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            if not raw_line.strip():
                continue

            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {manifest_path}:{line_number}: {error.msg}"
                ) from error

            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected an object at {manifest_path}:{line_number}"
                )

            level = sample.get("level")
            writer_id = sample.get("writer_id")

            if isinstance(level, str):
                level_counts[level] += 1

            if isinstance(writer_id, str) and writer_id:
                writer_ids.add(writer_id)

            total += 1

    return {
        "dataset": dataset,
        "writers": len(writer_ids),
        "paragraphs": level_counts["paragraph"],
        "lines": level_counts["line"],
        "words": level_counts["word"],
        "total": total,
    }


def total_statistics(
    statistics: Sequence[DatasetStatistics],
) -> DatasetStatistics:
    """Sum individual dataset rows into one aggregate row.

    Writer counts are summed rather than deduplicated across corpora because
    each builder namespaces writer IDs with its dataset name.
    """
    return {
        "dataset": "total",
        "writers": sum(item["writers"] for item in statistics),
        "paragraphs": sum(item["paragraphs"] for item in statistics),
        "lines": sum(item["lines"] for item in statistics),
        "words": sum(item["words"] for item in statistics),
        "total": sum(item["total"] for item in statistics),
    }


def render_table(statistics: Sequence[DatasetStatistics]) -> None:
    """Render statistics as a human-readable Rich table."""
    table = Table(title="Dataset statistics")
    table.add_column("Dataset", style="cyan")
    table.add_column("Writers", justify="right")
    table.add_column("Paragraphs", justify="right")
    table.add_column("Lines", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Total", justify="right", style="green")

    for item in statistics:
        table.add_row(
            item["dataset"],
            f"{item['writers']:,}",
            f"{item['paragraphs']:,}",
            f"{item['lines']:,}",
            f"{item['words']:,}",
            f"{item['total']:,}",
        )

    Console().print(table)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse dataset filters, the data root, and the output format."""
    parser = argparse.ArgumentParser(
        description="Calculate dataset statistics from manifest files.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=DATASETS,
        help="Datasets to inspect. Defaults to all datasets.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing <dataset>/manifest.jsonl (default: data).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load selected manifests and print per-dataset and aggregate statistics."""
    args = parse_args(argv)
    datasets = args.datasets or DATASETS
    statistics = []

    for dataset in datasets:
        # Normalized datasets share the same directory contract, so selecting a
        # dataset changes only this path and requires no dataset-specific code.
        manifest_path = args.data_root / dataset / "manifest.jsonl"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        statistics.append(collect_statistics(dataset, manifest_path))

    statistics.append(total_statistics(statistics))

    if args.json:
        print(json.dumps(statistics, ensure_ascii=False, indent=2))
    else:
        render_table(statistics)


if __name__ == "__main__":
    main()
