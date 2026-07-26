#!/usr/bin/env python3
"""Train the base VietParaDiff velocity model without HTR loss."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.training import (
    HeightBucketBatchSampler,
    VietParaDiffCollator,
    VietParaDiffDataset,
)
from src.models.autokl import HandwritingAutoKL
from src.models.config import (
    AutoKLConfig,
    ParagraphUNetConfig,
    StyleEncoderConfig,
    TextEncoderConfig,
    VietParaDiffConfig,
)
from src.models.vietparadiff import VietParaDiff
from src.models.text import GraphemeVocabulary, ParagraphFormatter
from src.vietparadiff_training import (
    VietParaDiffLogger,
    VietParaDiffTrainer,
    artifact_hashes,
    create_grad_scaler,
    create_optimizer_and_scheduler,
    ensure_inference_static_artifacts,
    load_latent_statistics,
    load_vietparadiff_training_config,
    resolve_runtime,
    seed_everything,
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
            "Train base VietParaDiff with frozen AutoKL and velocity MSE."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietparadiff_pretrain.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume strictly from generator last.pt.",
    )
    return parser.parse_args(argv)


def _loader(
    dataset: VietParaDiffDataset,
    collator: VietParaDiffCollator,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, object]]:
    sampler = HeightBucketBatchSampler(
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)  # type: ignore[arg-type]


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _model_and_text_config(
    *,
    use_pretrained_backbone: bool,
    convnext_checkpoint: Path | None,
) -> tuple[VietParaDiffConfig, GraphemeVocabulary]:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    text = TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
    )
    config = VietParaDiffConfig(
        autokl=AutoKLConfig(),
        text=text,
        style=StyleEncoderConfig(
            use_pretrained_backbone=use_pretrained_backbone,
            convnext_checkpoint=convnext_checkpoint,
        ),
        unet=ParagraphUNetConfig(),
    )
    return config, vocabulary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_vietparadiff_training_config(args.config)
    seed_everything(config.seed)
    runtime = resolve_runtime(config.device, config.precision)
    model_config, vocabulary = _model_and_text_config(
        use_pretrained_backbone=config.style.use_pretrained_backbone,
        convnext_checkpoint=config.style.convnext_checkpoint,
    )
    formatter = ParagraphFormatter(model_config.text)
    dataset = VietParaDiffDataset(
        config.data.train_targets,
        mode="train",
        reference_manifest=config.data.train_references,
        image_root=config.data.image_root,
        formatter=formatter,
        seed=config.seed,
    )
    collator = VietParaDiffCollator(formatter, vocabulary)
    loader = _loader(
        dataset,
        collator,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        seed=config.seed,
        device=runtime.device,
    )
    if len(loader) <= 0:
        raise ValueError("Generator train loader không được rỗng.")

    statistics = load_latent_statistics(
        config.autokl.latent_statistics,
        expected_autokl_checkpoint=config.autokl.checkpoint,
    )
    hashes = artifact_hashes(config)
    autokl = HandwritingAutoKL(model_config.autokl)
    autokl.load_checkpoint(config.autokl.checkpoint)
    autokl.to(runtime.device).eval()
    model = VietParaDiff(model_config).to(runtime.device)

    steps_per_epoch = math.ceil(
        len(loader) / config.data.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.diffusion.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
        config.scheduler,
        total_steps=total_steps,
    )
    scaler = create_grad_scaler(runtime)
    output_dir = config.checkpoint.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config_payload = _jsonable(asdict(model_config))
    if not isinstance(model_config_payload, Mapping):
        raise RuntimeError("Model config phải serialize thành mapping.")
    ensure_inference_static_artifacts(
        output_dir,
        model_config_payload,
        vocabulary,
    )
    logger = VietParaDiffLogger(
        config.logging,
        output_dir,
        config.resolved_dict(),
    )
    trainer = VietParaDiffTrainer(
        model,
        autokl,
        statistics,
        optimizer,
        scheduler,
        scaler,
        config,
        runtime,
        hashes,
        model_config_payload,
        vocabulary,
        logger,
    )
    start_epoch = 0
    if args.resume is not None:
        state = trainer.resume(args.resume)
        start_epoch = state.epoch
        print(
            f"Resumed epoch={state.epoch}, "
            f"global_step={state.global_step}, "
            f"best_score={state.best_score:.8f}"
        )
    if start_epoch > config.diffusion.epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} vượt configured epochs "
            f"{config.diffusion.epochs}."
        )

    try:
        for epoch in range(start_epoch, config.diffusion.epochs):
            metrics = trainer.train_epoch(loader, epoch=epoch)
            improved = trainer.save_epoch_checkpoints(
                next_epoch=epoch + 1,
                train_score=metrics.velocity_mse,
            )
            print(
                f"epoch={epoch + 1}/{config.diffusion.epochs} "
                f"step={trainer.global_step} "
                f"velocity_mse={metrics.velocity_mse:.8f} "
                f"samples={metrics.sample_count} best={improved}"
            )
    finally:
        logger.close()
    print(
        "Base diffusion training complete. Downstream checkpoint: "
        f"{output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
