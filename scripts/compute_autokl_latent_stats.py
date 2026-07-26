#!/usr/bin/env python3
"""Compute frozen AutoKL posterior-mode latent statistics."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vietparadiff.artifacts import (
    save_latent_statistics,
    sha256_file,
)
from vietparadiff.data.pipeline import (
    AutoKLDataset,
    HeightBucketBatchSampler,
    collate_autokl,
)
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.runtime import (
    autocast_context,
    resolve_runtime,
    seed_everything,
)
from vietparadiff.training.generator import (
    LatentStatisticsAccumulator,
)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute scalar mean/std from posterior.mode() over AutoKL "
            "train paragraphs."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/autokl/train_paragraphs.jsonl"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/autokl/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/autokl/latent_statistics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size phải dương.")
    if args.num_workers < 0:
        parser.error("--num-workers không được âm.")
    if args.seed < 0:
        parser.error("--seed không được âm.")
    return args


def _loader(
    dataset: AutoKLDataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, object]]:
    sampler = HeightBucketBatchSampler(
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "collate_fn": collate_autokl,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    runtime = resolve_runtime(args.device, args.precision)
    dataset = AutoKLDataset(
        args.manifest,
        image_root=args.image_root,
    )
    loader = _loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=runtime.device,
    )
    model = HandwritingAutoKL()
    model.load_checkpoint(args.checkpoint)
    model.to(runtime.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    accumulator = LatentStatisticsAccumulator()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch.get("images")
            if not isinstance(images, torch.Tensor):
                raise TypeError("AutoKL stats batch images phải là Tensor.")
            images = images.to(
                runtime.device,
                non_blocking=runtime.device.type == "cuda",
            )
            with autocast_context(runtime):
                latents = model.encode(images).mode()
            accumulator.update(latents)
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"batch={batch_index}/{len(loader)} "
                    f"samples={accumulator.sample_count} "
                    f"elements={accumulator.count}"
                )
            del images, latents
            if runtime.device.type == "mps":
                torch.mps.empty_cache()

    statistics = accumulator.finalize(
        sha256_file(args.checkpoint)
    )
    save_latent_statistics(args.output, statistics)
    print(
        f"Saved {args.output}: mean={statistics.latent_mean:.9f}, "
        f"std={statistics.latent_std:.9f}, "
        f"scaling_factor={statistics.scaling_factor:.9f}, "
        f"samples={statistics.num_samples}"
    )


if __name__ == "__main__":
    main()
