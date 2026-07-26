"""Train Stage 1 handwriting AutoKL and evaluate once after training."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vietparadiff.training.autokl import (
    AutoKLLogger,
    AutoKLTrainer,
    create_optimizer_and_scheduler,
    load_best_for_evaluation,
    load_training_config,
)
from vietparadiff.data.pipeline import (
    AutoKLDataset,
    HeightBucketBatchSampler,
    collate_autokl,
)
from vietparadiff.models.autokl import HandwritingAutoKL
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
        description="Train Stage 1 HandwritingAutoKL from scratch.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/autokl/train.yaml"),
        help="YAML training configuration.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume strictly from last.pt.",
    )
    return parser.parse_args(argv)


def _loader(
    dataset: AutoKLDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[dict[str, object]]:
    sampler = HeightBucketBatchSampler(
        dataset,
        batch_size,
        shuffle=shuffle,
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
    config = load_training_config(args.config)
    seed_everything(config.seed)
    runtime = resolve_runtime(config.device, config.precision)
    output_dir = config.checkpoint.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AutoKLDataset(
        config.data.train_manifest,
        image_root=config.data.image_root,
    )
    test_dataset = AutoKLDataset(
        config.data.test_manifest,
        image_root=config.data.image_root,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.data.num_workers,
        device=runtime.device,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.data.num_workers,
        device=runtime.device,
    )

    model = HandwritingAutoKL().to(runtime.device)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
    )
    scaler = create_grad_scaler(runtime)
    logger = AutoKLLogger(
        config.logging,
        output_dir,
        config.resolved_dict(),
    )
    trainer = AutoKLTrainer(
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        runtime,
        logger,
    )
    start_epoch = 0
    if args.resume is not None:
        resume = trainer.resume(args.resume)
        start_epoch = resume.epoch
        print(
            f"Resumed epoch={resume.epoch}, "
            f"global_step={resume.global_step}, "
            f"best_score={resume.best_score:.6f}"
        )
    if start_epoch > config.autokl.epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} vượt configured epochs "
            f"{config.autokl.epochs}."
        )

    try:
        for epoch in range(start_epoch, config.autokl.epochs):
            metrics = trainer.train_epoch(train_loader, epoch=epoch)
            improved = trainer.save_epoch_checkpoints(
                next_epoch=epoch + 1,
                train_checkpoint_score=metrics.checkpoint_score,
            )
            print(
                f"epoch={epoch + 1}/{config.autokl.epochs} "
                f"step={trainer.global_step} "
                f"reconstruction={metrics.reconstruction_loss:.6f} "
                f"edge={metrics.edge_loss:.6f} "
                f"kl={metrics.kl_loss:.6f} "
                f"selection={metrics.checkpoint_score:.6f} "
                f"best={improved}"
            )

        best_path = output_dir / "best.pt"
        load_best_for_evaluation(model, best_path, runtime.device)
        print(f"Loaded best model for final test: {best_path}")

        test_metrics = trainer.evaluate(
            test_loader,
            render_dir=output_dir / "reconstructions",
        )
        print(
            "final_test "
            f"reconstruction={test_metrics.reconstruction_loss:.6f} "
            f"edge={test_metrics.edge_loss:.6f} "
            f"kl={test_metrics.kl_loss:.6f}"
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
