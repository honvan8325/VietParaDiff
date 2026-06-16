"""CLI for writer-disjoint dataset splitting."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

from vietparadiff.data.manifest import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create writer-disjoint train/val/test splits")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    by_writer = defaultdict(list)
    for row in rows:
        by_writer[row.writer_id].append(row)
    writers = sorted(by_writer)
    rng = random.Random(args.seed)
    rng.shuffle(writers)
    n_test = max(1, int(len(writers) * args.test_ratio)) if len(writers) > 2 else 0
    n_val = max(1, int(len(writers) * args.val_ratio)) if len(writers) > 2 else 0
    test_w = set(writers[:n_test])
    val_w = set(writers[n_test:n_test + n_val])
    splits = {"train": [], "val": [], "test": []}
    for w, wr in by_writer.items():
        key = "test" if w in test_w else "val" if w in val_w else "train"
        splits[key].extend(wr)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, split_rows in splits.items():
        write_jsonl(out / f"{key}.jsonl", split_rows)
        print(key, len(split_rows))


if __name__ == "__main__":
    main()
