"""Provenance-bound bridge to a baseline-specific worker in a pinned checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _tracked_backend(checkout: Path, backend: Path) -> Path:
    checkout = checkout.resolve()
    backend = backend.resolve()
    try:
        relative = backend.relative_to(checkout)
    except ValueError as error:
        raise ValueError("Backend adapter phải nằm trong pinned checkout.") from error
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "--error-unmatch",
            relative.as_posix(),
        ],
        check=True,
        capture_output=True,
    )
    if not backend.is_file():
        raise FileNotFoundError(f"Thiếu backend adapter: {backend}")
    return backend


def _validate_outputs(requests: Path, output_dir: Path) -> None:
    expected = {
        str(json.loads(line)["request_id"])
        for line in requests.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    manifest = output_dir / "outputs.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Backend không tạo {manifest}.")
    actual: set[str] = set()
    for number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        payload = json.loads(line)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"request_id", "image"}
            or not isinstance(payload["request_id"], str)
            or not isinstance(payload["image"], str)
        ):
            raise ValueError(f"outputs.jsonl dòng {number} sai schema.")
        if payload["request_id"] in actual:
            raise ValueError("Backend trả request_id trùng.")
        image = Path(payload["image"])
        if not image.is_absolute():
            image = manifest.parent / image
        if not image.is_file():
            raise FileNotFoundError(f"Backend output thiếu: {image}")
        actual.add(payload["request_id"])
    if actual != expected:
        raise ValueError(
            "Backend output IDs không khớp requests: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--backend-script", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    backend = _tracked_backend(args.checkout, args.backend_script)
    command = [
        sys.executable,
        str(backend),
        f"--requests={args.requests.resolve()}",
        f"--output-dir={args.output_dir.resolve()}",
        f"--checkpoint={args.checkpoint.resolve()}",
    ]
    subprocess.run(command, cwd=args.checkout, check=True)
    _validate_outputs(args.requests, args.output_dir)


if __name__ == "__main__":
    main()
