#!/usr/bin/env python3
"""Audit every split manifest and referenced image."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.data.audit import DatasetAuditor


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full VietParaDiff dataset integrity audit."
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_audit.json"),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = DatasetAuditor(
        args.split_root,
        image_root=args.image_root,
        workers=args.workers,
    ).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"Audit complete: manifests={report['manifest_count']} "
        f"images={report['decoded_image_count']} "
        f"hard_errors={report['hard_error_count']} "
        f"rejections={report['expected_rejection_count']} "
        f"warnings={report['warning_count']}"
    )
    if int(report["hard_error_count"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
