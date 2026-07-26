#!/usr/bin/env python3
"""Download torchvision weights explicitly and bind them to local hashes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torchvision
from torch import Tensor
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
)

from vietparadiff.artifacts import sha256_file


WEIGHTS = {
    "convnext_tiny_imagenet1k_v1": (
        "convnext_tiny_imagenet1k_v1.pt",
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    ),
    "resnet18_imagenet1k_v1": (
        "resnet18_imagenet1k_v1.pt",
        ResNet18_Weights.IMAGENET1K_V1,
    ),
}


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly download ConvNeXt-Tiny and ResNet-18 ImageNet "
            "weights for offline VietParaDiff runs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/vision"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing checkpoints only when explicitly requested.",
    )
    return parser.parse_args(argv)


def _save_state_dict(
    path: Path,
    state: Mapping[str, Tensor],
    *,
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} đã tồn tại; dùng --force để thay thế."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    records: dict[str, dict[str, str]] = {}
    for name, (filename, weights) in WEIGHTS.items():
        state = weights.get_state_dict(
            progress=True,
            check_hash=True,
        )
        path = output_dir / filename
        _save_state_dict(path, state, force=args.force)
        records[name] = {
            "filename": filename,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "torchvision_version": torchvision.__version__,
        "weights": records,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} đã tồn tại; dùng --force để thay thế."
        )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"Saved visual backbone contract: {manifest_path}")


if __name__ == "__main__":
    main()
