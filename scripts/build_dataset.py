"""Command-line dispatcher for rebuilding one supported dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from vietparadiff.data import (
    build_cvl_dataset,
    build_iam_dataset,
    build_uithwdb_dataset,
)

DATASET_BUILDERS: dict[str, Callable[[], None]] = {
    "cvl": build_cvl_dataset,
    "iam": build_iam_dataset,
    "uithwdb": build_uithwdb_dataset,
}


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_atomic(dataset: str, builder: Callable[[], None]) -> None:
    """Run legacy constant-based builders in a same-filesystem staging dir."""
    module = importlib.import_module(builder.__module__)
    target = Path(module.OUT)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.build.",
            dir=target.parent,
        )
    )
    original = {
        "OUT": module.OUT,
        "IMAGES": module.IMAGES,
        "MANIFEST": module.MANIFEST,
    }
    module.OUT = staging
    module.IMAGES = staging / "images"
    module.MANIFEST = staging / "manifest.jsonl"
    try:
        builder()
        manifest = staging / "manifest.jsonl"
        records = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            image = Path(str(record["image"]))
            try:
                relative = image.relative_to(staging)
            except ValueError as error:
                raise ValueError(
                    f"{dataset} builder ghi image ngoài staging: {image}"
                ) from error
            record["image"] = (target / relative).as_posix()
            records.append(record)
        manifest.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        report_path = staging / "build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 2:
            raise ValueError(f"{dataset} build report sai schema.")
        hard_errors = report.get("hard_errors")
        hard_count = report.get("hard_error_count")
        if (
            not isinstance(hard_errors, list)
            or hard_count != len(hard_errors)
        ):
            raise ValueError(f"{dataset} build report sai hard-error counts.")
        if hard_count:
            failed_report = target.with_name(
                f"{target.name}.failed_build_report.json"
            )
            shutil.copy2(report_path, failed_report)
            raise ValueError(
                f"{dataset} build có {hard_count} hard error; "
                "giữ nguyên output tốt hiện tại; chi tiết tại "
                f"{failed_report}."
            )
        report["output_manifest"] = str(target / "manifest.jsonl")
        report["output_manifest_sha256"] = _manifest_sha256(manifest)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        backup = target.with_name(f".{target.name}.previous")
        if backup.exists():
            raise FileExistsError(f"Stale build backup tồn tại: {backup}")
        if target.exists():
            target.replace(backup)
        try:
            staging.replace(target)
        except BaseException:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        target.with_name(
            f"{target.name}.failed_build_report.json"
        ).unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for name, value in original.items():
            setattr(module, name, value)


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
    _run_atomic(args.dataset, DATASET_BUILDERS[args.dataset])


if __name__ == "__main__":
    main()
