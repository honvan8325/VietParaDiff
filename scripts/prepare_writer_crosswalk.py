#!/usr/bin/env python3
"""Generate evidence for manual UIT-HWDB/VNOnDB writer review."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.data.splits import (
    generate_writer_crosswalk_candidates,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate evidence only; this command never approves "
            "a writer identity mapping."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/metadata/vietnamese_writer_crosswalk_candidates.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = generate_writer_crosswalk_candidates(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Wrote {len(payload['candidates'])} candidate records to "
        f"{args.output}. Human approval is still required."
    )


if __name__ == "__main__":
    main()
