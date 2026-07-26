#!/usr/bin/env python3
"""Train and evaluate the four-head Vietnamese HTR teacher."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vietparadiff.data.pipeline import (
    HTRDataset,
    HTRVocabulary,
    WidthBucketBatchSampler,
    collate_htr,
)
from vietparadiff.training.htr import (
    HTRLogger,
    HTRTrainer,
    artifact_hashes,
    create_optimizer_and_scheduler,
    load_best_for_evaluation,
    load_htr_training_config,
    validate_htr_dataset,
)
from vietparadiff.models.config import HTRConfig
from vietparadiff.models.htr import VietnameseHTR
from vietparadiff.runtime import (
    create_grad_scaler,
    resolve_runtime,
    seed_everything,
)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the line-level Vietnamese HTR teacher.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/htr/train.yaml"),
        help="HTR YAML training configuration.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume strictly from outputs/htr/last.pt.",
    )
    return parser.parse_args(argv)


def build_or_load_train_vocabulary(
    train_lines_manifest: Path,
    train_words_manifest: Path,
    vocabulary_path: Path,
) -> HTRVocabulary:
    train_vocabulary = HTRVocabulary.build_from_manifests(
        (train_lines_manifest, train_words_manifest)
    )
    if vocabulary_path.exists():
        stored_vocabulary = HTRVocabulary.load(vocabulary_path)
        if stored_vocabulary != train_vocabulary:
            raise ValueError(
                "vocabulary.json không khớp train manifests hiện tại."
            )
        return stored_vocabulary
    train_vocabulary.save(vocabulary_path)
    return train_vocabulary


def _loader(
    dataset: HTRDataset,
    *,
    batch_size: int,
    bucket_width: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[dict[str, object]]:
    sampler = WidthBucketBatchSampler(
        dataset,
        batch_size,
        bucket_width=bucket_width,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "collate_fn": collate_htr,
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
    config = load_htr_training_config(args.config)
    seed_everything(config.seed)
    runtime = resolve_runtime(config.device, config.precision)
    vocabulary = build_or_load_train_vocabulary(
        config.data.train_lines,
        config.data.train_words,
        config.data.vocabulary,
    )

    datasets = {
        "train_lines": HTRDataset(
            config.data.train_lines,
            vocabulary,
            image_root=config.data.image_root,
        ),
        "train_words": HTRDataset(
            config.data.train_words,
            vocabulary,
            image_root=config.data.image_root,
        ),
        "test_lines": HTRDataset(
            config.data.test_lines,
            vocabulary,
            image_root=config.data.image_root,
        ),
        "test_words": HTRDataset(
            config.data.test_words,
            vocabulary,
            image_root=config.data.image_root,
        ),
    }
    for dataset_name, dataset in datasets.items():
        validate_htr_dataset(dataset, dataset_name)

    output_dir = config.checkpoint.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(output_dir / "vocabulary.json")
    model_config = HTRConfig(
        raw_vocab_size=len(vocabulary.raw_to_id),
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
    )
    model_config_payload = asdict(model_config)
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaders = {
        "train_lines": _loader(
            datasets["train_lines"],
            batch_size=config.data.line_batch_size,
            bucket_width=config.data.width_bucket_size,
            shuffle=True,
            seed=config.seed,
            num_workers=config.data.num_workers,
            device=runtime.device,
        ),
        "train_words": _loader(
            datasets["train_words"],
            batch_size=config.data.word_batch_size,
            bucket_width=config.data.width_bucket_size,
            shuffle=True,
            seed=config.seed + 1,
            num_workers=config.data.num_workers,
            device=runtime.device,
        ),
        "test_lines": _loader(
            datasets["test_lines"],
            batch_size=config.data.line_batch_size,
            bucket_width=config.data.width_bucket_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.data.num_workers,
            device=runtime.device,
        ),
        "test_words": _loader(
            datasets["test_words"],
            batch_size=config.data.word_batch_size,
            bucket_width=config.data.width_bucket_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.data.num_workers,
            device=runtime.device,
        ),
    }
    steps_per_epoch = (
        len(loaders["train_lines"]) + config.data.line_batches_per_step - 1
    ) // config.data.line_batches_per_step
    total_steps = steps_per_epoch * config.htr.epochs

    model = VietnameseHTR(model_config).to(runtime.device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
        config.scheduler,
        total_steps=total_steps,
    )
    scaler = create_grad_scaler(runtime)
    vocabulary_hash, manifest_hashes = artifact_hashes(config)
    logger = HTRLogger(
        config.logging,
        output_dir,
        config.resolved_dict(),
    )
    trainer = HTRTrainer(
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        runtime,
        vocabulary,
        vocabulary_hash,
        manifest_hashes,
        model_config_payload,
        logger,
    )
    start_epoch = 0
    if args.resume is not None:
        state = trainer.resume(args.resume)
        start_epoch = state.epoch
        print(
            f"Resumed epoch={state.epoch}, global_step={state.global_step}, "
            f"best_score={state.best_score:.6f}"
        )
    if start_epoch > config.htr.epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} vượt configured epochs "
            f"{config.htr.epochs}."
        )

    try:
        for epoch in range(start_epoch, config.htr.epochs):
            metrics = trainer.train_epoch(
                loaders["train_lines"],
                loaders["train_words"],
                epoch=epoch,
            )
            improved = trainer.save_epoch_checkpoints(
                next_epoch=epoch + 1,
                train_checkpoint_score=metrics.checkpoint_score,
            )
            print(
                f"epoch={epoch + 1}/{config.htr.epochs} "
                f"step={trainer.global_step} "
                f"line_total={metrics.line_total:.6f} "
                f"word_total={metrics.word_total:.6f} "
                f"selection={metrics.checkpoint_score:.6f} best={improved}"
            )

        best_path = output_dir / "best.pt"
        load_best_for_evaluation(model, best_path, runtime.device)
        print(f"Loaded best model for final test: {best_path}")
        line_metrics = trainer.evaluate(
            loaders["test_lines"],
            level="line",
            prediction_path=output_dir / "test_line_predictions.jsonl",
        )
        word_metrics = trainer.evaluate(
            loaders["test_words"],
            level="word",
            prediction_path=output_dir / "test_word_predictions.jsonl",
        )
        test_metrics = {
            "line": line_metrics.as_dict(),
            "word": word_metrics.as_dict(),
        }
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.log_scalars(
            {
                **{
                    f"test/line_{name}": value
                    for name, value in line_metrics.as_dict().items()
                },
                **{
                    f"test/word_{name}": value
                    for name, value in word_metrics.as_dict().items()
                },
            },
            step=trainer.global_step,
        )
        print(
            f"final_test line_CER={line_metrics.raw_cer:.6f} "
            f"line_WER={line_metrics.raw_wer:.6f} "
            f"word_CER={word_metrics.raw_cer:.6f} "
            f"word_accuracy={word_metrics.exact_word_accuracy:.6f}"
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
