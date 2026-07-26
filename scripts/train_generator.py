#!/usr/bin/env python3
"""Train pretrain, Vietnamese fine-tune, or HTR-guided VietParaDiff."""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vietparadiff.data.pipeline import (
    HeightBucketBatchSampler,
    VietParaDiffCollator,
    VietParaDiffDataset,
)
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.models.config import (
    AutoKLConfig,
    ParagraphUNetConfig,
    StyleEncoderConfig,
    TextEncoderConfig,
    VietParaDiffConfig,
)
from vietparadiff.models.generator import VietParaDiff
from vietparadiff.models.grapheme import GraphemeVocabulary, ParagraphFormatter
from vietparadiff.artifacts import load_latent_statistics
from vietparadiff.inference.generator import (
    checkpoint_loading_config,
    load_inference_contract,
    load_model_config,
)
from vietparadiff.runtime import (
    create_grad_scaler,
    resolve_runtime,
    seed_everything,
)
from vietparadiff.training.generator import (
    DeterministicRealSyntheticBatchMixer,
    VietParaDiffLogger,
    VietParaDiffTrainer,
    artifact_hashes,
    create_optimizer_and_scheduler,
    ensure_inference_static_artifacts,
    load_vietparadiff_training_config,
)
from vietparadiff.training.htr_guidance import FrozenHTRTeacher


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
            "Train a stage-aware VietParaDiff run with strict artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietparadiff/pretrain.yaml"),
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
    behavior: object | None = None,
) -> tuple[VietParaDiffConfig, GraphemeVocabulary]:
    from vietparadiff.training.generator import ModelBehaviorConfig

    if behavior is None:
        behavior = ModelBehaviorConfig()
    if not isinstance(behavior, ModelBehaviorConfig):
        raise TypeError("behavior phải là ModelBehaviorConfig.")
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
            use_high_frequency_style=(
                behavior.use_high_frequency_style
            ),
        ),
        unet=ParagraphUNetConfig(
            use_shape_condition=behavior.use_shape_condition,
            use_tone_condition=behavior.use_tone_condition,
            use_local_style_tokens=(
                behavior.use_local_style_tokens
            ),
            use_harmonizer=behavior.use_harmonizer,
        ),
    )
    return config, vocabulary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_vietparadiff_training_config(args.config)
    seed_everything(config.seed)
    runtime = resolve_runtime(config.device, config.precision)
    parent_contract = None
    if config.stage == "pretrain":
        if config.style is None:
            raise RuntimeError("Pretrain style config bị thiếu.")
        stored_model_config, vocabulary = _model_and_text_config(
            use_pretrained_backbone=(
                config.style.use_pretrained_backbone
            ),
            convnext_checkpoint=config.style.convnext_checkpoint,
            behavior=config.behavior,
        )
        loading_model_config = stored_model_config
    else:
        if config.initialization is None:
            raise RuntimeError(
                "Derived stage initialization bị thiếu."
            )
        parent_contract = load_inference_contract(
            config.initialization.contract,
            generator_checkpoint=config.initialization.checkpoint,
            model_config=config.initialization.model_config,
            vocabulary=config.initialization.vocabulary,
            autokl_checkpoint=config.autokl.checkpoint,
            latent_statistics=config.autokl.latent_statistics,
        )
        if (
            parent_contract.num_train_timesteps
            != config.diffusion.num_train_timesteps
        ):
            raise ValueError(
                "Derived stage diffusion timesteps không khớp parent."
            )
        stored_model_config = load_model_config(
            config.initialization.model_config
        )
        loading_model_config = checkpoint_loading_config(
            stored_model_config
        )
        vocabulary = GraphemeVocabulary.load(
            config.initialization.vocabulary
        )
    expected_vocab_sizes = (
        len(vocabulary.base_to_id),
        len(vocabulary.shape_to_id),
        len(vocabulary.tone_to_id),
        len(vocabulary.case_to_id),
        len(vocabulary.class_to_id),
    )
    actual_vocab_sizes = (
        stored_model_config.text.base_vocab_size,
        stored_model_config.text.shape_vocab_size,
        stored_model_config.text.tone_vocab_size,
        stored_model_config.text.case_vocab_size,
        stored_model_config.text.class_vocab_size,
    )
    if actual_vocab_sizes != expected_vocab_sizes:
        raise ValueError(
            "Generator vocabulary không khớp model config: "
            f"expected={expected_vocab_sizes}, "
            f"actual={actual_vocab_sizes}."
        )
    formatter = ParagraphFormatter(stored_model_config.text)
    collator = VietParaDiffCollator(formatter, vocabulary)
    if config.stage == "pretrain":
        if config.data.train_targets is None:
            raise RuntimeError("Pretrain targets bị thiếu.")
        dataset = VietParaDiffDataset(
            config.data.train_targets,
            mode="train",
            reference_manifest=config.data.train_references,
            image_root=config.data.image_root,
            formatter=formatter,
            seed=config.seed,
        )
        loader: object = _loader(
            dataset,
            collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            seed=config.seed,
            device=runtime.device,
        )
    else:
        if config.data.real_targets is None:
            raise RuntimeError("Derived targets bị thiếu.")
        real_dataset = VietParaDiffDataset(
            config.data.real_targets,
            mode="train",
            reference_manifest=config.data.train_references,
            image_root=config.data.image_root,
            formatter=formatter,
            seed=config.seed,
        )
        real_loader = _loader(
            real_dataset,
            collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            seed=config.seed,
            device=runtime.device,
        )
        if config.data.use_synthetic_data:
            if config.data.synthetic_targets is None:
                raise RuntimeError(
                    "Synthetic data được bật nhưng manifest bị thiếu."
                )
            synthetic_dataset = VietParaDiffDataset(
                config.data.synthetic_targets,
                mode="train",
                reference_manifest=config.data.train_references,
                image_root=config.data.image_root,
                formatter=formatter,
                seed=config.seed + 1,
            )
            synthetic_loader = _loader(
                synthetic_dataset,
                collator,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                seed=config.seed + 1,
                device=runtime.device,
            )
            loader = DeterministicRealSyntheticBatchMixer(
                real_loader,
                synthetic_loader,
                real_batches_per_cycle=(
                    config.data.real_batches_per_cycle
                ),
                synthetic_batches_per_cycle=(
                    config.data.synthetic_batches_per_cycle
                ),
            )
        else:
            loader = real_loader
    if len(loader) <= 0:
        raise ValueError("Generator train loader không được rỗng.")

    statistics = load_latent_statistics(
        config.autokl.latent_statistics,
        expected_autokl_checkpoint=config.autokl.checkpoint,
    )
    hashes = artifact_hashes(config)
    autokl = HandwritingAutoKL(stored_model_config.autokl)
    autokl.load_checkpoint(config.autokl.checkpoint)
    autokl.to(runtime.device).eval()
    model = VietParaDiff(loading_model_config)
    if config.stage != "pretrain":
        if config.initialization is None:
            raise RuntimeError("Derived initialization bị thiếu.")
        model.load_checkpoint(config.initialization.checkpoint)
    model.to(runtime.device)

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
    model_config_payload = _jsonable(asdict(stored_model_config))
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
    htr_teacher = (
        FrozenHTRTeacher.load(
            config.guidance,
            device=runtime.device,
        )
        if config.guidance is not None
        else None
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
        htr_teacher,
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
    if config.stage == "htr_guided":
        try:
            probe_batch = next(iter(loader))
        except StopIteration as error:
            raise ValueError(
                "HTR-guidance structural probe thiếu training batch."
            ) from error
        probe = trainer.run_htr_guidance_structural_probe(probe_batch)
        print(
            "HTR guidance structural preflight passed: "
            f"lines={int(probe['line_count'])}, "
            f"htr_loss={probe['htr_loss']:.6f}, "
            f"slot_ink_coverage={probe['slot_ink_coverage']:.6f}, "
            f"generator_gradients="
            f"{int(probe['generator_gradient_count'])}"
        )

    try:
        for epoch in range(start_epoch, config.diffusion.epochs):
            metrics = trainer.train_epoch(  # type: ignore[arg-type]
                loader,
                epoch=epoch,
            )
            checkpoint_score = metrics.velocity_mse
            improved = trainer.save_epoch_checkpoints(
                next_epoch=epoch + 1,
                train_score=checkpoint_score,
                force_model_checkpoint=(
                    config.stage == "htr_guided"
                ),
            )
            print(
                f"epoch={epoch + 1}/{config.diffusion.epochs} "
                f"step={trainer.global_step} "
                f"total={metrics.total_loss:.8f} "
                f"velocity_mse={metrics.velocity_mse:.8f} "
                f"htr={metrics.htr_loss:.8f} "
                f"guided_lines={metrics.guided_line_count} "
                f"samples={metrics.sample_count} best={improved}"
            )
    finally:
        logger.close()
    print(
        f"{config.stage} training complete. Downstream checkpoint: "
        f"{output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
