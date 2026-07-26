"""Base velocity-diffusion training mechanics for VietParaDiff."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import wandb
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from vietparadiff.artifacts import (
    LatentStatistics,
    load_latent_statistics,
    save_latent_statistics,
    sha256_file,
    verify_visual_backbone,
)
from vietparadiff.diffusion import (
    add_diffusion_noise,
    cosine_alpha_sigma,
    velocity_target,
)
from vietparadiff.runtime import (
    RuntimePrecision,
    autocast_context,
    create_grad_scaler,
    resolve_runtime,
    seed_everything,
)
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.models.grapheme import GraphemeBatch, GraphemeVocabulary
from vietparadiff.models.generator import (
    VietParaDiff,
    VietParaDiffInput,
    VietParaDiffOutput,
)
from vietparadiff.training.htr_guidance import (
    FrozenHTRTeacher,
    HTRGuidanceConfig,
    HTRGuidanceResult,
    guidance_step_enabled,
    guidance_weight,
    predicted_clean_from_velocity,
)

HEIGHT_BUCKETS = (384, 512, 640, 768, 896, 1024, 1280)
TrainingStage = Literal["pretrain", "finetune", "htr_guided"]


@dataclass(frozen=True, slots=True)
class ModelBehaviorConfig:
    use_shape_condition: bool = True
    use_tone_condition: bool = True
    use_high_frequency_style: bool = True
    use_local_style_tokens: bool = True
    use_harmonizer: bool = True

    def __post_init__(self) -> None:
        values = (
            self.use_shape_condition,
            self.use_tone_condition,
            self.use_high_frequency_style,
            self.use_local_style_tokens,
            self.use_harmonizer,
        )
        if not all(isinstance(value, bool) for value in values):
            raise TypeError("Mọi model behavior flag phải là bool.")
        if self.use_tone_condition and not self.use_shape_condition:
            raise ValueError(
                "Tone conditioning yêu cầu shape conditioning."
            )


@dataclass(frozen=True, slots=True)
class VietParaDiffDataConfig:
    train_targets: Path | None
    train_references: Path
    image_root: Path
    num_workers: int
    batch_size: int
    gradient_accumulation_steps: int
    real_targets: Path | None = None
    synthetic_targets: Path | None = None
    real_batches_per_cycle: int = 3
    synthetic_batches_per_cycle: int = 1
    use_synthetic_data: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("data.num_workers không được âm.")
        if self.batch_size <= 0:
            raise ValueError("data.batch_size phải dương.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "data.gradient_accumulation_steps phải dương."
            )
        if self.real_batches_per_cycle <= 0:
            raise ValueError(
                "data.real_batches_per_cycle phải dương."
            )
        if self.synthetic_batches_per_cycle <= 0:
            raise ValueError(
                "data.synthetic_batches_per_cycle phải dương."
            )
        if not isinstance(self.use_synthetic_data, bool):
            raise TypeError("data.use_synthetic_data phải là bool.")


@dataclass(frozen=True, slots=True)
class FrozenAutoKLConfig:
    checkpoint: Path
    latent_statistics: Path


@dataclass(frozen=True, slots=True)
class StyleInitializationConfig:
    use_pretrained_backbone: bool
    convnext_checkpoint: Path | None
    backbone_contract: Path | None

    def __post_init__(self) -> None:
        if not self.use_pretrained_backbone:
            raise ValueError(
                "Base VietParaDiff phải khởi tạo ConvNeXt từ ImageNet."
            )
        if self.convnext_checkpoint is None:
            raise ValueError(
                "Pretrain yêu cầu ConvNeXt checkpoint local."
            )
        if self.backbone_contract is None:
            raise ValueError(
                "Pretrain yêu cầu visual backbone contract."
            )


@dataclass(frozen=True, slots=True)
class GeneratorInitializationConfig:
    checkpoint: Path
    contract: Path
    model_config: Path
    vocabulary: Path


@dataclass(frozen=True, slots=True)
class DiffusionStageConfig:
    epochs: int
    num_train_timesteps: int
    noise_schedule: Literal["cosine"]

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("diffusion.epochs phải dương.")
        if self.num_train_timesteps < 2:
            raise ValueError(
                "diffusion.num_train_timesteps phải >= 2."
            )
        if self.noise_schedule != "cosine":
            raise ValueError("Base trainer chỉ hỗ trợ cosine schedule.")


@dataclass(frozen=True, slots=True)
class GeneratorOptimizerConfig:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if self.name.lower() != "adamw":
            raise ValueError("optimizer.name phải là adamw.")
        if self.learning_rate <= 0.0:
            raise ValueError("optimizer.learning_rate phải dương.")
        if (
            len(self.betas) != 2
            or not all(0.0 <= beta < 1.0 for beta in self.betas)
        ):
            raise ValueError(
                "optimizer.betas phải gồm hai giá trị trong [0,1)."
            )
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay không được âm.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "optimizer.gradient_clip_norm phải dương."
            )


@dataclass(frozen=True, slots=True)
class GeneratorSchedulerConfig:
    warmup_steps: int
    minimum_learning_rate_ratio: float

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError("scheduler.warmup_steps phải dương.")
        if not 0.0 < self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError(
                "scheduler.minimum_learning_rate_ratio phải trong (0,1]."
            )


@dataclass(frozen=True, slots=True)
class GeneratorLoggingConfig:
    log_every_steps: int
    tensorboard: bool
    wandb: bool
    wandb_mode: Literal["online", "offline", "disabled"]
    wandb_project: str
    wandb_entity: str | None
    run_name: str | None

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0:
            raise ValueError("logging.log_every_steps phải dương.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "logging.wandb_mode phải là online/offline/disabled."
            )
        if not self.wandb_project:
            raise ValueError("logging.wandb_project không được rỗng.")


@dataclass(frozen=True, slots=True)
class GeneratorCheckpointConfig:
    output_dir: Path
    save_last: bool
    save_best: bool

    def __post_init__(self) -> None:
        if not self.save_last or not self.save_best:
            raise ValueError(
                "Generator phải lưu cả last.pt và best.pt."
            )


@dataclass(frozen=True, slots=True)
class VietParaDiffTrainingConfig:
    seed: int
    device: str
    precision: str
    data: VietParaDiffDataConfig
    autokl: FrozenAutoKLConfig
    style: StyleInitializationConfig | None
    diffusion: DiffusionStageConfig
    optimizer: GeneratorOptimizerConfig
    scheduler: GeneratorSchedulerConfig
    logging: GeneratorLoggingConfig
    checkpoint: GeneratorCheckpointConfig
    stage: TrainingStage = "pretrain"
    initialization: GeneratorInitializationConfig | None = None
    guidance: HTRGuidanceConfig | None = None
    behavior: ModelBehaviorConfig | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed không được âm.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device phải là auto/cuda/mps/cpu.")
        if self.precision not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError("precision không hợp lệ.")
        if self.stage not in {"pretrain", "finetune", "htr_guided"}:
            raise ValueError(
                "stage phải là pretrain/finetune/htr_guided."
            )
        if self.stage == "pretrain":
            if self.data.train_targets is None:
                raise ValueError(
                    "Pretrain yêu cầu data.train_targets."
                )
            if (
                self.data.real_targets is not None
                or self.data.synthetic_targets is not None
            ):
                raise ValueError(
                    "Pretrain không nhận real/synthetic targets."
                )
            if self.style is None:
                raise ValueError(
                    "Pretrain yêu cầu style initialization."
                )
            if self.initialization is not None or self.guidance is not None:
                raise ValueError(
                    "Pretrain không nhận initialization/guidance."
                )
        else:
            if self.data.train_targets is not None:
                raise ValueError(
                    "Derived stage không nhận data.train_targets."
                )
            if (
                self.data.real_targets is None
                or (
                    self.data.use_synthetic_data
                    and self.data.synthetic_targets is None
                )
            ):
                raise ValueError(
                    "Derived stage yêu cầu real_targets và synthetic_targets "
                    "khi use_synthetic_data=true."
                )
            if self.style is not None:
                raise ValueError(
                    "Derived stage đọc style topology từ parent artifact."
                )
            if self.behavior is not None:
                raise ValueError(
                    "Derived stage đọc behavior flags từ parent model config."
                )
            if self.initialization is None:
                raise ValueError(
                    "Derived stage yêu cầu initialization artifacts."
                )
            if self.stage == "htr_guided" and self.guidance is None:
                raise ValueError(
                    "htr_guided stage yêu cầu guidance config."
                )
            if (
                self.guidance is not None
                and self.guidance.maximum_timestep
                >= self.diffusion.num_train_timesteps
            ):
                raise ValueError(
                    "guidance.maximum_timestep phải nhỏ hơn "
                    "num_train_timesteps."
                )
            if self.stage == "finetune" and self.guidance is not None:
                raise ValueError(
                    "finetune stage không nhận guidance config."
                )

    def resolved_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {
                    str(key): convert(item)
                    for key, item in value.items()
                }
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        payload = convert(asdict(self))
        if not isinstance(payload, dict):
            raise RuntimeError("Resolved generator config phải là mapping.")
        return payload


def _config_section(
    raw: Mapping[str, object],
    name: str,
    keys: set[str],
) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(keys)}, nhận {actual}."
        )
    return dict(value)


def load_vietparadiff_training_config(
    path: Path,
) -> VietParaDiffTrainingConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Generator config root phải là mapping.")
    legacy_root = {
        "seed",
        "device",
        "precision",
        "data",
        "autokl",
        "style",
        "diffusion",
        "optimizer",
        "scheduler",
        "logging",
        "checkpoint",
    }
    stage_root = legacy_root | {
        "stage",
        "initialization",
        "guidance",
    }
    stage_root_with_behavior = stage_root | {"behavior"}
    if set(raw) == legacy_root:
        stage: TrainingStage = "pretrain"
        initialization_raw: object = None
        guidance_raw: object = None
    elif set(raw) == stage_root:
        stage_value = raw["stage"]
        if stage_value not in {"pretrain", "finetune", "htr_guided"}:
            raise ValueError(
                "config.stage phải là pretrain/finetune/htr_guided."
            )
        stage = stage_value
        initialization_raw = raw["initialization"]
        guidance_raw = raw["guidance"]
    elif set(raw) == stage_root_with_behavior:
        stage_value = raw["stage"]
        if stage_value not in {"pretrain", "finetune", "htr_guided"}:
            raise ValueError(
                "config.stage phải là pretrain/finetune/htr_guided."
            )
        stage = stage_value
        initialization_raw = raw["initialization"]
        guidance_raw = raw["guidance"]
    else:
        raise ValueError(
            "Generator config root keys phải khớp schema stage-aware."
        )
    behavior_raw = raw.get("behavior")
    if behavior_raw is None:
        behavior = None
    else:
        expected_behavior = {
            "use_shape_condition",
            "use_tone_condition",
            "use_high_frequency_style",
            "use_local_style_tokens",
            "use_harmonizer",
        }
        if (
            not isinstance(behavior_raw, Mapping)
            or set(behavior_raw) != expected_behavior
        ):
            raise ValueError("config.behavior sai schema.")
        if not all(
            isinstance(behavior_raw[key], bool)
            for key in expected_behavior
        ):
            raise TypeError("Mọi config.behavior flag phải là bool.")
        behavior = ModelBehaviorConfig(
            use_shape_condition=behavior_raw["use_shape_condition"],
            use_tone_condition=behavior_raw["use_tone_condition"],
            use_high_frequency_style=behavior_raw[
                "use_high_frequency_style"
            ],
            use_local_style_tokens=behavior_raw[
                "use_local_style_tokens"
            ],
            use_harmonizer=behavior_raw["use_harmonizer"],
        )
    common_data_keys = {
        "train_references",
        "image_root",
        "num_workers",
        "batch_size",
        "gradient_accumulation_steps",
    }
    data_keys = (
        common_data_keys | {"train_targets"}
        if stage == "pretrain"
        else common_data_keys
        | {
            "real_targets",
            "synthetic_targets",
            "real_batches_per_cycle",
            "synthetic_batches_per_cycle",
            "use_synthetic_data",
        }
    )
    data = _config_section(
        raw,
        "data",
        data_keys,
    )
    if stage != "pretrain" and not isinstance(
        data["use_synthetic_data"], bool
    ):
        raise TypeError("data.use_synthetic_data phải là bool.")
    autokl = _config_section(
        raw, "autokl", {"checkpoint", "latent_statistics"}
    )
    style_raw = raw.get("style")
    if stage == "pretrain":
        style = _config_section(
            raw,
            "style",
            {
                "use_pretrained_backbone",
                "convnext_checkpoint",
                "backbone_contract",
            },
        )
        style_config: StyleInitializationConfig | None = (
            StyleInitializationConfig(
                use_pretrained_backbone=style["use_pretrained_backbone"],
                convnext_checkpoint=(
                    None
                    if style["convnext_checkpoint"] is None
                    else Path(str(style["convnext_checkpoint"]))
                ),
                backbone_contract=(
                    None
                    if style["backbone_contract"] is None
                    else Path(str(style["backbone_contract"]))
                ),
            )
        )
        if not isinstance(style["use_pretrained_backbone"], bool):
            raise TypeError(
                "style.use_pretrained_backbone phải là bool."
            )
    else:
        if style_raw is not None:
            raise ValueError(
                "config.style phải là null ở derived stage."
            )
        style_config = None
    diffusion = _config_section(
        raw,
        "diffusion",
        {"epochs", "num_train_timesteps", "noise_schedule"},
    )
    optimizer = _config_section(
        raw,
        "optimizer",
        {
            "name",
            "learning_rate",
            "betas",
            "weight_decay",
            "gradient_clip_norm",
        },
    )
    scheduler = _config_section(
        raw,
        "scheduler",
        {"warmup_steps", "minimum_learning_rate_ratio"},
    )
    logging = _config_section(
        raw,
        "logging",
        {
            "log_every_steps",
            "tensorboard",
            "wandb",
            "wandb_mode",
            "wandb_project",
            "wandb_entity",
            "run_name",
        },
    )
    checkpoint = _config_section(
        raw,
        "checkpoint",
        {"output_dir", "save_last", "save_best"},
    )
    betas = optimizer["betas"]
    if (
        not isinstance(betas, Sequence)
        or isinstance(betas, (str, bytes))
        or len(betas) != 2
    ):
        raise ValueError("optimizer.betas phải là sequence dài 2.")

    initialization: GeneratorInitializationConfig | None
    if initialization_raw is None:
        initialization = None
    else:
        if not isinstance(initialization_raw, Mapping) or set(
            initialization_raw
        ) != {"checkpoint", "contract", "model_config", "vocabulary"}:
            raise ValueError(
                "config.initialization sai schema."
            )
        initialization = GeneratorInitializationConfig(
            checkpoint=Path(str(initialization_raw["checkpoint"])),
            contract=Path(str(initialization_raw["contract"])),
            model_config=Path(str(initialization_raw["model_config"])),
            vocabulary=Path(str(initialization_raw["vocabulary"])),
        )

    guidance: HTRGuidanceConfig | None
    if guidance_raw is None:
        guidance = None
    else:
        expected_guidance = {
            "checkpoint",
            "model_config",
            "vocabulary",
            "maximum_weight",
            "warmup_steps",
            "maximum_timestep",
            "every_n_optimizer_steps",
            "raw_weight",
            "base_weight",
            "shape_weight",
            "tone_weight",
        }
        if not isinstance(guidance_raw, Mapping) or set(
            guidance_raw
        ) != expected_guidance:
            raise ValueError("config.guidance sai schema.")
        guidance = HTRGuidanceConfig(
            checkpoint=Path(str(guidance_raw["checkpoint"])),
            model_config=Path(str(guidance_raw["model_config"])),
            vocabulary=Path(str(guidance_raw["vocabulary"])),
            maximum_weight=float(guidance_raw["maximum_weight"]),
            warmup_steps=int(guidance_raw["warmup_steps"]),
            maximum_timestep=int(guidance_raw["maximum_timestep"]),
            every_n_optimizer_steps=int(
                guidance_raw["every_n_optimizer_steps"]
            ),
            raw_weight=float(guidance_raw["raw_weight"]),
            base_weight=float(guidance_raw["base_weight"]),
            shape_weight=float(guidance_raw["shape_weight"]),
            tone_weight=float(guidance_raw["tone_weight"]),
        )

    return VietParaDiffTrainingConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        data=VietParaDiffDataConfig(
            train_targets=(
                Path(str(data["train_targets"]))
                if stage == "pretrain"
                else None
            ),
            train_references=Path(str(data["train_references"])),
            image_root=Path(str(data["image_root"])),
            num_workers=int(data["num_workers"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation_steps=int(
                data["gradient_accumulation_steps"]
            ),
            real_targets=(
                None
                if stage == "pretrain"
                else Path(str(data["real_targets"]))
            ),
            synthetic_targets=(
                None
                if (
                    stage == "pretrain"
                    or data["synthetic_targets"] is None
                )
                else Path(str(data["synthetic_targets"]))
            ),
            real_batches_per_cycle=(
                3
                if stage == "pretrain"
                else int(data["real_batches_per_cycle"])
            ),
            synthetic_batches_per_cycle=(
                1
                if stage == "pretrain"
                else int(data["synthetic_batches_per_cycle"])
            ),
            use_synthetic_data=(
                False
                if stage == "pretrain"
                else data["use_synthetic_data"]
            ),
        ),
        autokl=FrozenAutoKLConfig(
            checkpoint=Path(str(autokl["checkpoint"])),
            latent_statistics=Path(str(autokl["latent_statistics"])),
        ),
        style=style_config,
        diffusion=DiffusionStageConfig(
            epochs=int(diffusion["epochs"]),
            num_train_timesteps=int(
                diffusion["num_train_timesteps"]
            ),
            noise_schedule=str(  # type: ignore[arg-type]
                diffusion["noise_schedule"]
            ),
        ),
        optimizer=GeneratorOptimizerConfig(
            name=str(optimizer["name"]),
            learning_rate=float(optimizer["learning_rate"]),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_norm=float(
                optimizer["gradient_clip_norm"]
            ),
        ),
        scheduler=GeneratorSchedulerConfig(
            warmup_steps=int(scheduler["warmup_steps"]),
            minimum_learning_rate_ratio=float(
                scheduler["minimum_learning_rate_ratio"]
            ),
        ),
        logging=GeneratorLoggingConfig(
            log_every_steps=int(logging["log_every_steps"]),
            tensorboard=bool(logging["tensorboard"]),
            wandb=bool(logging["wandb"]),
            wandb_mode=str(logging["wandb_mode"]),  # type: ignore[arg-type]
            wandb_project=str(logging["wandb_project"]),
            wandb_entity=(
                None
                if logging["wandb_entity"] is None
                else str(logging["wandb_entity"])
            ),
            run_name=(
                None
                if logging["run_name"] is None
                else str(logging["run_name"])
            ),
        ),
        checkpoint=GeneratorCheckpointConfig(
            output_dir=Path(str(checkpoint["output_dir"])),
            save_last=bool(checkpoint["save_last"]),
            save_best=bool(checkpoint["save_best"]),
        ),
        stage=stage,
        initialization=initialization,
        guidance=guidance,
        behavior=behavior,
    )


class LatentStatisticsAccumulator:
    """Streaming population mean/std in float64 without retaining latents."""

    def __init__(self) -> None:
        self.count = 0
        self.sample_count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, latents: Tensor) -> None:
        if (
            latents.ndim != 4
            or latents.shape[1] != 4
            or not latents.is_floating_point()
        ):
            raise ValueError(
                "Latent statistics input phải là float [B,4,H,W]."
            )
        if not torch.isfinite(latents).all():
            raise ValueError("Latent statistics input chứa NaN/Inf.")
        values = latents.detach().to(device="cpu", dtype=torch.float64)
        batch_count = values.numel()
        batch_mean = float(values.mean())
        batch_m2 = float(
            (values - batch_mean).square().sum()
        )
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            combined = self.count + batch_count
            self.mean += delta * batch_count / combined
            self.m2 += (
                batch_m2
                + delta * delta * self.count * batch_count / combined
            )
        self.count += batch_count
        self.sample_count += latents.shape[0]

    def finalize(self, checkpoint_sha256: str) -> LatentStatistics:
        if self.count <= 1 or self.sample_count <= 0:
            raise ValueError(
                "Không đủ latent elements để tính statistics."
            )
        variance = self.m2 / self.count
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("Latent variance phải hữu hạn và dương.")
        std = math.sqrt(variance)
        return LatentStatistics(
            latent_mean=self.mean,
            latent_std=std,
            scaling_factor=1.0 / std,
            num_samples=self.sample_count,
            num_elements=self.count,
            autokl_checkpoint_sha256=checkpoint_sha256,
        )


def learning_rate_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if (
        step < 0
        or warmup_steps <= 0
        or total_steps <= 0
        or not 0.0 < minimum_ratio <= 1.0
    ):
        raise ValueError("Scheduler arguments không hợp lệ.")
    if step < warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(
        (step - warmup_steps) / (total_steps - warmup_steps),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def create_optimizer_and_scheduler(
    model: nn.Module,
    optimizer_config: GeneratorOptimizerConfig,
    scheduler_config: GeneratorSchedulerConfig,
    *,
    total_steps: int,
) -> tuple[Optimizer, LambdaLR]:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Generator không có trainable parameter.")
    optimizer = AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_factor(
            step,
            warmup_steps=scheduler_config.warmup_steps,
            total_steps=total_steps,
            minimum_ratio=scheduler_config.minimum_learning_rate_ratio,
        ),
    )
    return optimizer, scheduler


def _move_graphemes(
    graphemes: GraphemeBatch,
    device: torch.device,
) -> GraphemeBatch:
    return GraphemeBatch(
        graphemes.base_ids.to(device),
        graphemes.shape_ids.to(device),
        graphemes.tone_ids.to(device),
        graphemes.case_ids.to(device),
        graphemes.class_ids.to(device),
        graphemes.line_ids.to(device),
        graphemes.position_in_line_ids.to(device),
        graphemes.height_bucket_ids.to(device),
        graphemes.attention_mask.to(device),
    )


@dataclass(frozen=True, slots=True)
class DiffusionStepOutput:
    model_output: VietParaDiffOutput
    loss: Tensor
    velocity_loss: Tensor
    htr_result: HTRGuidanceResult | None
    htr_weight: float
    clean_latents: Tensor
    noisy_latents: Tensor
    target_velocity: Tensor
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True, slots=True)
class GeneratorEpochMetrics:
    total_loss: float
    velocity_mse: float
    htr_loss: float
    guided_line_count: int
    sample_count: int


class VietParaDiffLogger:
    def __init__(
        self,
        config: GeneratorLoggingConfig,
        output_dir: Path,
        resolved_config: Mapping[str, object],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = (
            SummaryWriter(output_dir / "tensorboard")
            if config.tensorboard
            else None
        )
        self.run = None
        if config.wandb:
            self.run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                mode=config.wandb_mode,
                config=dict(resolved_config),
                dir=str(output_dir),
            )

    def log_scalars(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
    ) -> None:
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.run is not None:
            self.run.finish()


def _set_loader_epoch(loader: object, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


class DeterministicRealSyntheticBatchMixer:
    """Yield every real batch once and insert deterministic synthetic batches."""

    def __init__(
        self,
        real_loader: Iterable[Mapping[str, object]] & Sized,
        synthetic_loader: Iterable[Mapping[str, object]] & Sized,
        *,
        real_batches_per_cycle: int = 3,
        synthetic_batches_per_cycle: int = 1,
    ) -> None:
        if len(real_loader) <= 0 or len(synthetic_loader) <= 0:
            raise ValueError(
                "Real và synthetic loaders phải không rỗng."
            )
        if real_batches_per_cycle <= 0 or synthetic_batches_per_cycle <= 0:
            raise ValueError("Mixer cycle counts phải dương.")
        self.real_loader = real_loader
        self.synthetic_loader = synthetic_loader
        self.real_batches_per_cycle = real_batches_per_cycle
        self.synthetic_batches_per_cycle = synthetic_batches_per_cycle
        self.epoch = 0

    def __len__(self) -> int:
        completed_cycles = (
            len(self.real_loader) // self.real_batches_per_cycle
        )
        return (
            len(self.real_loader)
            + completed_cycles * self.synthetic_batches_per_cycle
        )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Mixer epoch không được âm.")
        self.epoch = epoch
        _set_loader_epoch(self.real_loader, epoch)
        _set_loader_epoch(self.synthetic_loader, epoch)

    @staticmethod
    def _tag(
        batch: Mapping[str, object],
        source: Literal["real", "synthetic"],
    ) -> dict[str, object]:
        tagged = dict(batch)
        if "data_source" in tagged:
            raise ValueError("Batch input không được chứa data_source.")
        tagged["data_source"] = source
        return tagged

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        synthetic_iterator = iter(self.synthetic_loader)
        real_since_synthetic = 0
        for real_batch in self.real_loader:
            yield self._tag(real_batch, "real")
            real_since_synthetic += 1
            if real_since_synthetic < self.real_batches_per_cycle:
                continue
            real_since_synthetic = 0
            for _ in range(self.synthetic_batches_per_cycle):
                try:
                    synthetic_batch = next(synthetic_iterator)
                except StopIteration:
                    synthetic_iterator = iter(self.synthetic_loader)
                    try:
                        synthetic_batch = next(synthetic_iterator)
                    except StopIteration as error:
                        raise RuntimeError(
                            "Synthetic loader rỗng khi cycle."
                        ) from error
                yield self._tag(synthetic_batch, "synthetic")


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "mps": (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None
        ),
    }


def _restore_rng(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "torch",
        "cuda",
        "mps",
    }:
        raise ValueError("Generator checkpoint RNG state sai schema.")
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise ValueError("Torch RNG state phải là Tensor.")
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(torch_state)
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint có CUDA RNG nhưng CUDA không khả dụng."
            )
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
    mps_state = state["mps"]
    if mps_state is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Checkpoint có MPS RNG nhưng MPS không khả dụng."
            )
        if not isinstance(mps_state, Tensor):
            raise ValueError("MPS RNG state phải là Tensor.")
        torch.mps.set_rng_state(mps_state)


def _resume_config_signature(
    config: Mapping[str, object],
) -> dict[str, object]:
    stage = config.get("stage", "pretrain")
    if stage not in {"pretrain", "finetune", "htr_guided"}:
        raise ValueError("Resume config stage không hợp lệ.")
    names = ("data", "autokl", "diffusion", "optimizer", "scheduler")
    sections: dict[str, Mapping[str, object]] = {}
    for name in names:
        section = config.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(
                f"Resume config.{name} phải là mapping."
            )
        sections[name] = section
    if "seed" not in config or "precision" not in config:
        raise ValueError("Resume config thiếu seed/precision.")
    data = sections["data"]
    required_data = (
        {
            "train_targets",
            "train_references",
            "image_root",
            "batch_size",
            "gradient_accumulation_steps",
        }
        if stage == "pretrain"
        else {
            "real_targets",
            "synthetic_targets",
            "train_references",
            "image_root",
            "batch_size",
            "gradient_accumulation_steps",
            "real_batches_per_cycle",
            "synthetic_batches_per_cycle",
        }
    )
    if not required_data.issubset(data):
        raise ValueError("Resume config.data thiếu training fields.")
    signature = {
        "stage": stage,
        "seed": config["seed"],
        "precision": config["precision"],
        "data": {
            name: data[name]
            for name in sorted(required_data)
        },
        "autokl": dict(sections["autokl"]),
        "diffusion": dict(sections["diffusion"]),
        "optimizer": dict(sections["optimizer"]),
        "scheduler": dict(sections["scheduler"]),
    }
    if stage == "pretrain":
        style = config.get("style")
        if not isinstance(style, Mapping):
            raise ValueError("Resume pretrain config.style phải là mapping.")
        signature["style"] = dict(style)
    else:
        initialization = config.get("initialization")
        if not isinstance(initialization, Mapping):
            raise ValueError(
                "Resume derived config.initialization phải là mapping."
            )
        signature["initialization"] = dict(initialization)
        if stage == "htr_guided":
            guidance = config.get("guidance")
            if not isinstance(guidance, Mapping):
                raise ValueError(
                    "Resume htr_guided config.guidance phải là mapping."
                )
            signature["guidance"] = dict(guidance)
    return signature


def validate_resume_config(
    checkpoint_config: object,
    current_config: Mapping[str, object],
) -> None:
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint config phải là mapping.")
    old = _resume_config_signature(checkpoint_config)
    new = _resume_config_signature(current_config)
    mismatches = [
        name for name in old if old[name] != new[name]
    ]
    if mismatches:
        details = "; ".join(
            f"{name}: checkpoint={old[name]!r}, current={new[name]!r}"
            for name in mismatches
        )
        raise ValueError(
            f"Resume generator config không tương thích: {details}"
        )


def save_model_checkpoint(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict()}, temporary)
    temporary.replace(path)


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_inference_static_artifacts(
    output_dir: Path,
    model_config: Mapping[str, object],
    vocabulary: GraphemeVocabulary,
) -> tuple[Path, Path]:
    model_config_path = output_dir / "model_config.json"
    vocabulary_path = output_dir / "grapheme_vocabulary.json"
    if model_config_path.exists():
        stored = json.loads(model_config_path.read_text(encoding="utf-8"))
        if stored != dict(model_config):
            raise ValueError(
                "model_config.json hiện có không khớp model config."
            )
    else:
        _write_json_atomic(model_config_path, model_config)
    if vocabulary_path.exists():
        stored_vocabulary = GraphemeVocabulary.load(vocabulary_path)
        if stored_vocabulary.to_dict() != vocabulary.to_dict():
            raise ValueError(
                "grapheme_vocabulary.json hiện có không khớp vocabulary."
            )
    else:
        vocabulary.save(vocabulary_path)
    return model_config_path, vocabulary_path


def save_inference_contract(
    output_dir: Path,
    *,
    num_train_timesteps: int,
    artifact_sha256: Mapping[str, str],
) -> None:
    if not {
        "autokl_checkpoint",
        "latent_statistics",
    }.issubset(artifact_sha256):
        raise ValueError(
            "Inference contract thiếu AutoKL/latent artifacts."
        )
    for name in ("autokl_checkpoint", "latent_statistics"):
        value = artifact_sha256[name]
        if (
            len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise ValueError(
                f"Inference contract {name} SHA-256 không hợp lệ."
            )
    best_path = output_dir / "best.pt"
    model_config_path = output_dir / "model_config.json"
    vocabulary_path = output_dir / "grapheme_vocabulary.json"
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps phải >= 2.")
    contract = {
        "schema_version": 1,
        "prediction_type": "velocity",
        "noise_schedule": "cosine",
        "num_train_timesteps": num_train_timesteps,
        "neutral_layout": True,
        "generator_checkpoint_sha256": sha256_file(best_path),
        "model_config_sha256": sha256_file(model_config_path),
        "grapheme_vocabulary_sha256": sha256_file(vocabulary_path),
        "autokl_checkpoint_sha256": artifact_sha256[
            "autokl_checkpoint"
        ],
        "latent_statistics_sha256": artifact_sha256[
            "latent_statistics"
        ],
    }
    _write_json_atomic(
        output_dir / "inference_contract.json",
        contract,
    )


def expected_artifact_names(
    stage: TrainingStage,
    *,
    use_synthetic_data: bool = True,
) -> set[str]:
    if stage == "pretrain":
        return {
            "train_targets",
            "train_references",
            "autokl_checkpoint",
            "latent_statistics",
            "convnext_checkpoint",
            "visual_backbone_contract",
        }
    names = {
        "real_targets",
        "train_references",
        "autokl_checkpoint",
        "latent_statistics",
        "parent_checkpoint",
        "parent_contract",
        "parent_model_config",
        "parent_vocabulary",
    }
    if use_synthetic_data:
        names.add("synthetic_targets")
    if stage == "htr_guided":
        names |= {
            "htr_checkpoint",
            "htr_contract",
            "htr_model_config",
            "htr_vocabulary",
        }
    return names


def training_lineage(
    config: VietParaDiffTrainingConfig,
    artifact_sha256: Mapping[str, str],
) -> dict[str, object]:
    if set(artifact_sha256) != expected_artifact_names(
        config.stage,
        use_synthetic_data=config.data.use_synthetic_data,
    ):
        raise ValueError("Training lineage artifact schema sai.")
    manifests = (
        {
            "train_targets": artifact_sha256["train_targets"],
            "train_references": artifact_sha256["train_references"],
        }
        if config.stage == "pretrain"
        else {
            "real_targets": artifact_sha256["real_targets"],
            **(
                {
                    "synthetic_targets": artifact_sha256[
                        "synthetic_targets"
                    ]
                }
                if config.data.use_synthetic_data
                else {}
            ),
            "train_references": artifact_sha256[
                "train_references"
            ],
        }
    )
    parent = None
    mixing = None
    if config.stage != "pretrain":
        parent = {
            "checkpoint_sha256": artifact_sha256[
                "parent_checkpoint"
            ],
            "contract_sha256": artifact_sha256["parent_contract"],
            "model_config_sha256": artifact_sha256[
                "parent_model_config"
            ],
            "vocabulary_sha256": artifact_sha256[
                "parent_vocabulary"
            ],
        }
        mixing = {
            "use_synthetic_data": config.data.use_synthetic_data,
            "real_batches_per_cycle": (
                config.data.real_batches_per_cycle
            ),
            "synthetic_batches_per_cycle": (
                config.data.synthetic_batches_per_cycle
            ),
            "epoch_policy": "exhaust_real_once",
        }
    htr = None
    if config.stage == "htr_guided":
        htr = {
            "checkpoint_sha256": artifact_sha256["htr_checkpoint"],
            "contract_sha256": artifact_sha256["htr_contract"],
            "model_config_sha256": artifact_sha256[
                "htr_model_config"
            ],
            "vocabulary_sha256": artifact_sha256[
                "htr_vocabulary"
            ],
        }
    return {
        "schema_version": 1,
        "stage": config.stage,
        "parent": parent,
        "manifests": manifests,
        "mixing_schedule": mixing,
        "htr_teacher": htr,
        "checkpoint_selection": (
            "final_epoch"
            if config.stage == "htr_guided"
            else "minimum_train_velocity_mse"
        ),
        "autokl_checkpoint_sha256": artifact_sha256[
            "autokl_checkpoint"
        ],
        "latent_statistics_sha256": artifact_sha256[
            "latent_statistics"
        ],
    }


@dataclass(frozen=True, slots=True)
class ResumeState:
    epoch: int
    global_step: int
    best_score: float


class VietParaDiffTrainer:
    def __init__(
        self,
        model: VietParaDiff,
        autokl: HandwritingAutoKL,
        statistics: LatentStatistics,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        scaler: torch.amp.GradScaler,
        config: VietParaDiffTrainingConfig,
        runtime: RuntimePrecision,
        artifact_sha256: Mapping[str, str],
        model_config: Mapping[str, object],
        grapheme_vocabulary: GraphemeVocabulary,
        logger: VietParaDiffLogger | None = None,
        htr_teacher: FrozenHTRTeacher | None = None,
    ) -> None:
        if set(artifact_sha256) != expected_artifact_names(
            config.stage,
            use_synthetic_data=config.data.use_synthetic_data,
        ):
            raise ValueError("Generator artifact SHA-256 schema sai.")
        if (
            artifact_sha256["autokl_checkpoint"]
            != statistics.autokl_checkpoint_sha256
        ):
            raise ValueError(
                "Latent statistics không khớp frozen AutoKL artifact."
            )
        self.model = model
        self.autokl = autokl
        self.statistics = statistics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.runtime = runtime
        self.artifact_sha256 = dict(artifact_sha256)
        self.model_config = dict(model_config)
        self.grapheme_vocabulary = grapheme_vocabulary
        self.logger = logger
        self.htr_teacher = htr_teacher
        self.global_step = 0
        self.best_score = float("inf")
        self.lineage = training_lineage(config, artifact_sha256)
        if (config.stage == "htr_guided") != (
            self.htr_teacher is not None
        ):
            raise ValueError(
                "htr_teacher chỉ được cung cấp cho htr_guided stage."
            )
        for parameter in self.autokl.parameters():
            parameter.requires_grad_(False)
        self.autokl.eval()
        if self.htr_teacher is not None:
            self.htr_teacher.eval()
        self.optimizer.zero_grad(set_to_none=True)
        ensure_inference_static_artifacts(
            self.config.checkpoint.output_dir,
            self.model_config,
            self.grapheme_vocabulary,
        )
        lineage_path = (
            self.config.checkpoint.output_dir
            / "training_lineage.json"
        )
        if lineage_path.exists():
            stored_lineage = json.loads(
                lineage_path.read_text(encoding="utf-8")
            )
            if stored_lineage != self.lineage:
                raise ValueError(
                    "training_lineage.json hiện có không khớp run."
                )
        else:
            _write_json_atomic(lineage_path, self.lineage)

    def _clean_latents(self, images: Tensor) -> Tensor:
        self.autokl.eval()
        with torch.no_grad(), autocast_context(self.runtime):
            posterior = self.autokl.encode(images)
            latents = posterior.mode()
        expected = (
            images.shape[0],
            4,
            images.shape[2] // 8,
            images.shape[3] // 8,
        )
        if latents.shape != expected:
            raise ValueError(
                "Frozen AutoKL latent shape sai: "
                f"expected {expected}, actual {tuple(latents.shape)}."
            )
        return self.statistics.normalize(latents)

    def train_micro_batch(
        self,
        batch: Mapping[str, object],
        *,
        accumulation_divisor: int = 1,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> DiffusionStepOutput:
        if accumulation_divisor <= 0:
            raise ValueError("accumulation_divisor phải dương.")
        target_images = batch.get("target_images")
        reference_images = batch.get("reference_images")
        reference_valid_mask = batch.get("reference_valid_mask")
        graphemes = batch.get("graphemes")
        if (
            not isinstance(target_images, Tensor)
            or target_images.ndim != 4
            or target_images.shape[1] != 1
            or target_images.shape[-1] != 1024
        ):
            raise ValueError(
                "target_images phải có shape [B,1,H,1024]."
            )
        if (
            target_images.shape[-2] not in HEIGHT_BUCKETS
            or not target_images.is_floating_point()
            or not torch.isfinite(target_images).all()
            or target_images.min() < -1.0
            or target_images.max() > 1.0
        ):
            raise ValueError(
                "target_images phải dùng height bucket hợp lệ và giá trị "
                "float hữu hạn trong [-1,1]."
            )
        if (
            not isinstance(reference_images, Tensor)
            or reference_images.ndim != 4
            or reference_images.shape[:3]
            != (target_images.shape[0], 1, 256)
            or reference_images.shape[-1] > 1536
            or reference_images.shape[-1] % 32
        ):
            raise ValueError(
                "reference_images phải có shape [B,1,256,W], "
                "W<=1536 và chia hết cho 32."
            )
        if (
            not reference_images.is_floating_point()
            or not torch.isfinite(reference_images).all()
            or reference_images.min() < -1.0
            or reference_images.max() > 1.0
        ):
            raise ValueError(
                "reference_images phải là float hữu hạn trong [-1,1]."
            )
        if (
            not isinstance(reference_valid_mask, Tensor)
            or reference_valid_mask.dtype != torch.bool
            or reference_valid_mask.shape != reference_images.shape
            or not reference_valid_mask.flatten(1).any(dim=1).all()
        ):
            raise ValueError(
                "reference_valid_mask phải là bool Tensor cùng shape "
                "reference và mỗi sample phải có pixel hợp lệ."
            )
        if not isinstance(graphemes, GraphemeBatch):
            raise TypeError("graphemes phải là GraphemeBatch.")
        if graphemes.base_ids.shape[0] != target_images.shape[0]:
            raise ValueError(
                "graphemes và target_images phải cùng batch size."
            )
        device = self.runtime.device
        target_images = target_images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        reference_images = reference_images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        reference_valid_mask = reference_valid_mask.to(
            device,
            non_blocking=device.type == "cuda",
        )
        graphemes = _move_graphemes(graphemes, device)
        self.model.train()
        clean_latents = self._clean_latents(target_images)
        batch_size = clean_latents.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                self.config.diffusion.num_train_timesteps,
                (batch_size,),
                device=device,
            )
        else:
            timesteps = timesteps.to(device)
        alpha, sigma = cosine_alpha_sigma(
            timesteps,
            num_train_timesteps=(
                self.config.diffusion.num_train_timesteps
            ),
        )
        if noise is None:
            noise = torch.randn_like(clean_latents)
        else:
            noise = noise.to(
                device=device,
                dtype=clean_latents.dtype,
            )
            if (
                noise.shape != clean_latents.shape
                or not torch.isfinite(noise).all()
            ):
                raise ValueError(
                    "Provided noise phải cùng shape clean latents và hữu hạn."
                )
        noisy_latents = add_diffusion_noise(
            clean_latents,
            noise,
            alpha,
            sigma,
        )
        target_velocity = velocity_target(
            clean_latents,
            noise,
            alpha,
            sigma,
        )
        with autocast_context(self.runtime):
            style = self.model.encode_reference(
                reference_images,
                reference_valid_mask,
            )
            model_output = self.model(
                VietParaDiffInput(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    graphemes=graphemes,
                    style_condition=style,
                )
            )
            if model_output.predicted_velocity.shape != target_velocity.shape:
                raise ValueError(
                    "predicted_velocity shape sai: expected "
                    f"{tuple(target_velocity.shape)}, actual "
                    f"{tuple(model_output.predicted_velocity.shape)}."
                )
            velocity_loss = F.mse_loss(
                model_output.predicted_velocity.float(),
                target_velocity.float(),
            )
            htr_result: HTRGuidanceResult | None = None
            htr_weight = 0.0
            if (
                self.htr_teacher is not None
                and self.config.guidance is not None
                and guidance_step_enabled(
                    self.config.guidance,
                    self.global_step,
                )
            ):
                eligible = timesteps <= (
                    self.config.guidance.maximum_timestep
                )
                eligible_indices = torch.nonzero(
                    eligible,
                    as_tuple=False,
                ).flatten()
                if eligible_indices.numel() > 0:
                    canonical_slots = batch.get(
                        "canonical_line_slots"
                    )
                    target_texts = batch.get("target_texts")
                    target_ids = batch.get("target_ids")
                    if not isinstance(canonical_slots, Tensor):
                        raise TypeError(
                            "HTR guidance yêu cầu canonical_line_slots "
                            "Tensor."
                        )
                    if (
                        not isinstance(target_texts, Sequence)
                        or isinstance(target_texts, (str, bytes))
                        or len(target_texts) != batch_size
                        or not all(
                            isinstance(text, str)
                            for text in target_texts
                        )
                    ):
                        raise ValueError(
                            "HTR guidance yêu cầu target_texts cùng batch."
                        )
                    if target_ids is None:
                        target_ids = [
                            f"generated_{index}"
                            for index in range(batch_size)
                        ]
                    if (
                        not isinstance(target_ids, Sequence)
                        or isinstance(target_ids, (str, bytes))
                        or len(target_ids) != batch_size
                        or not all(
                            isinstance(target_id, str) and target_id
                            for target_id in target_ids
                        )
                    ):
                        raise ValueError(
                            "HTR guidance yêu cầu target_ids cùng batch."
                        )
                    selected = eligible_indices.detach().cpu().tolist()
                    predicted_clean = predicted_clean_from_velocity(
                        noisy_latents,
                        model_output.predicted_velocity,
                        alpha,
                        sigma,
                    ).index_select(0, eligible_indices)
                    decoded = self.autokl.decode(
                        self.statistics.denormalize(predicted_clean)
                    )
                    expected_decoded = (
                        eligible_indices.numel(),
                        1,
                        target_images.shape[-2],
                        1024,
                    )
                    if (
                        decoded.shape != expected_decoded
                        or not torch.isfinite(decoded).all()
                    ):
                        raise ValueError(
                            "Differentiable AutoKL decode shape/value sai: "
                            f"expected={expected_decoded}, "
                            f"actual={tuple(decoded.shape)}."
                        )
                    selected_slots = canonical_slots.to(
                        device=device,
                        dtype=decoded.dtype,
                    ).index_select(0, eligible_indices)
                    htr_result = self.htr_teacher(
                        decoded,
                        selected_slots,
                        [target_texts[index] for index in selected],
                        sample_ids=[
                            str(target_ids[index]) for index in selected
                        ],
                    )
                    htr_weight = guidance_weight(
                        self.config.guidance,
                        self.global_step,
                    )
            loss = velocity_loss
            if htr_result is not None:
                loss = loss + htr_weight * htr_result.losses.total.to(
                    velocity_loss.device
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("Generator total loss chứa NaN/Inf.")
        self.scaler.scale(loss / accumulation_divisor).backward()
        return DiffusionStepOutput(
            model_output=model_output,
            loss=loss,
            velocity_loss=velocity_loss,
            htr_result=htr_result,
            htr_weight=htr_weight,
            clean_latents=clean_latents,
            noisy_latents=noisy_latents,
            target_velocity=target_velocity,
            timesteps=timesteps,
            noise=noise,
        )

    def optimizer_step(self) -> float:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            self.config.optimizer.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Generator gradient norm không hữu hạn.")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(gradient_norm.detach().cpu())

    def run_htr_guidance_structural_probe(
        self,
        batch: Mapping[str, object],
    ) -> dict[str, float]:
        """Validate the differentiable guidance graph without optimizing.

        This is deliberately a structural gate: it checks shape, finite CTC,
        slots, ink coverage, and gradients. It never imposes a CER or visual
        quality threshold on an untrained generator.
        """
        if self.htr_teacher is None or self.config.guidance is None:
            raise RuntimeError(
                "Structural HTR probe chỉ hợp lệ cho htr_guided stage."
            )
        target_images = batch.get("target_images")
        slots = batch.get("canonical_line_slots")
        if not isinstance(target_images, Tensor) or not isinstance(
            slots, Tensor
        ):
            raise TypeError(
                "Structural HTR probe cần target_images và "
                "canonical_line_slots Tensor."
            )
        if target_images.shape[0] <= 0:
            raise ValueError("Structural HTR probe batch không được rỗng.")
        ink = ((1.0 - target_images.float()) / 2.0).clamp(0.0, 1.0)
        if not torch.isfinite(ink).all() or float(ink.sum()) <= 0.0:
            raise ValueError(
                "Structural HTR probe fixture phải có foreground hữu hạn."
            )
        slot_union = slots.float().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        slot_union = F.interpolate(
            slot_union,
            size=target_images.shape[-2:],
            mode="nearest",
        )
        slot_ink_coverage = float(
            ((ink * slot_union).sum() / ink.sum()).detach()
        )
        if (
            not math.isfinite(slot_ink_coverage)
            or not 0.0 <= slot_ink_coverage <= 1.0
        ):
            raise ValueError("Structural HTR slot/ink coverage sai.")

        old_step = self.global_step
        probe_step = (
            math.ceil(
                self.config.guidance.warmup_steps
                / self.config.guidance.every_n_optimizer_steps
            )
            * self.config.guidance.every_n_optimizer_steps
        )
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self.global_step = probe_step
            batch_size = target_images.shape[0]
            latent_shape = (
                batch_size,
                4,
                target_images.shape[-2] // 8,
                128,
            )
            output = self.train_micro_batch(
                batch,
                timesteps=torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=self.runtime.device,
                ),
                noise=torch.zeros(
                    latent_shape,
                    dtype=target_images.dtype,
                    device=self.runtime.device,
                ),
            )
            if output.htr_result is None:
                raise RuntimeError(
                    "Structural probe không đi qua HTR guidance branch."
                )
            trainable_gradients = [
                parameter.grad
                for parameter in self.model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not trainable_gradients or not all(
                torch.isfinite(gradient).all()
                for gradient in trainable_gradients
            ):
                raise FloatingPointError(
                    "Structural probe không tạo generator gradient hữu hạn."
                )
            if any(
                parameter.grad is not None
                for parameter in self.autokl.parameters()
            ):
                raise RuntimeError(
                    "Structural probe làm frozen AutoKL nhận gradient."
                )
            if any(
                parameter.grad is not None
                for parameter in self.htr_teacher.parameters()
            ):
                raise RuntimeError(
                    "Structural probe làm frozen HTR nhận gradient."
                )
            return {
                "line_count": float(output.htr_result.line_count),
                "htr_loss": float(
                    output.htr_result.losses.total.detach().cpu()
                ),
                "slot_ink_coverage": slot_ink_coverage,
                "generator_gradient_count": float(
                    len(trainable_gradients)
                ),
            }
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step = old_step
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def train_epoch(
        self,
        loader: Iterable[Mapping[str, object]],
        *,
        epoch: int,
    ) -> GeneratorEpochMetrics:
        if not hasattr(loader, "__len__") or len(loader) <= 0:  # type: ignore[arg-type]
            raise ValueError("Generator train loader không được rỗng.")
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)  # type: ignore[attr-defined]
        else:
            _set_loader_epoch(loader, epoch)
        total_batches = len(loader)  # type: ignore[arg-type]
        accumulation = self.config.data.gradient_accumulation_steps
        total_loss_sum = 0.0
        velocity_loss_sum = 0.0
        htr_loss_sum = 0.0
        guided_line_count = 0
        sample_count = 0
        group_total_losses: list[float] = []
        group_velocity_losses: list[float] = []
        group_htr_losses: list[float] = []
        group_htr_weights: list[float] = []
        group_guided_lines: list[int] = []
        group_sources: list[str] = []
        group_timesteps: list[float] = []
        group_latent_means: list[float] = []
        group_latent_stds: list[float] = []
        group_bucket = 0
        for batch_index, batch in enumerate(loader):
            if batch_index % accumulation == 0:
                group_size = min(
                    accumulation,
                    total_batches - batch_index,
                )
                group_total_losses.clear()
                group_velocity_losses.clear()
                group_htr_losses.clear()
                group_htr_weights.clear()
                group_guided_lines.clear()
                group_sources.clear()
                group_timesteps.clear()
                group_latent_means.clear()
                group_latent_stds.clear()
            output = self.train_micro_batch(
                batch,
                accumulation_divisor=group_size,
            )
            target_images = batch.get("target_images")
            output_height = batch.get("output_height")
            if not isinstance(target_images, Tensor):
                raise TypeError("target_images phải là Tensor.")
            if not isinstance(output_height, int):
                raise TypeError("output_height phải là int.")
            group_bucket = output_height
            batch_size = target_images.shape[0]
            total_loss_value = float(output.loss.detach().cpu())
            velocity_loss_value = float(
                output.velocity_loss.detach().cpu()
            )
            total_loss_sum += total_loss_value * batch_size
            velocity_loss_sum += velocity_loss_value * batch_size
            sample_count += batch_size
            group_total_losses.append(total_loss_value)
            group_velocity_losses.append(velocity_loss_value)
            group_htr_weights.append(output.htr_weight)
            if output.htr_result is not None:
                htr_value = float(
                    output.htr_result.losses.total.detach().cpu()
                )
                lines = output.htr_result.line_count
                htr_loss_sum += htr_value * lines
                guided_line_count += lines
                group_htr_losses.append(htr_value)
                group_guided_lines.append(lines)
            source = batch.get("data_source", "pretrain")
            if source not in {"pretrain", "real", "synthetic"}:
                raise ValueError("Batch data_source không hợp lệ.")
            group_sources.append(str(source))
            group_timesteps.append(
                float(output.timesteps.float().mean().detach().cpu())
            )
            group_latent_means.append(
                float(output.clean_latents.float().mean().detach().cpu())
            )
            group_latent_stds.append(
                float(output.clean_latents.float().std().detach().cpu())
            )
            group_end = (
                len(group_total_losses) == group_size
                or batch_index + 1 == total_batches
            )
            if not group_end:
                del output
                if self.runtime.device.type == "mps":
                    torch.mps.empty_cache()
                continue
            gradient_norm = self.optimizer_step()
            if (
                self.logger is not None
                and self.global_step
                % self.config.logging.log_every_steps
                == 0
            ):
                self.logger.log_scalars(
                    {
                        "train/total_loss": sum(group_total_losses)
                        / len(group_total_losses),
                        "train/velocity_mse": sum(
                            group_velocity_losses
                        )
                        / len(group_velocity_losses),
                        "train/htr_loss": (
                            sum(group_htr_losses)
                            / len(group_htr_losses)
                            if group_htr_losses
                            else 0.0
                        ),
                        "train/htr_weight": max(
                            group_htr_weights,
                            default=0.0,
                        ),
                        "train/guided_lines": float(
                            sum(group_guided_lines)
                        ),
                        "train/synthetic_batch_fraction": (
                            sum(
                                source == "synthetic"
                                for source in group_sources
                            )
                            / len(group_sources)
                        ),
                        "train/timestep_mean": sum(group_timesteps)
                        / len(group_timesteps),
                        "train/latent_mean": sum(group_latent_means)
                        / len(group_latent_means),
                        "train/latent_std": sum(group_latent_stds)
                        / len(group_latent_stds),
                        "train/learning_rate": float(
                            self.optimizer.param_groups[0]["lr"]
                        ),
                        "train/gradient_norm": gradient_norm,
                        "train/height_bucket": float(group_bucket),
                    },
                    step=self.global_step,
                )
            del output
            if self.runtime.device.type == "mps":
                torch.mps.empty_cache()
        if sample_count <= 0:
            raise ValueError("Generator train epoch không có sample.")
        return GeneratorEpochMetrics(
            total_loss=total_loss_sum / sample_count,
            velocity_mse=velocity_loss_sum / sample_count,
            htr_loss=(
                htr_loss_sum / guided_line_count
                if guided_line_count > 0
                else 0.0
            ),
            guided_line_count=guided_line_count,
            sample_count=sample_count,
        )

    def save_epoch_checkpoints(
        self,
        *,
        next_epoch: int,
        train_score: float,
        force_model_checkpoint: bool = False,
    ) -> bool:
        if next_epoch < 0 or not math.isfinite(train_score):
            raise ValueError("Checkpoint epoch/score không hợp lệ.")
        if not isinstance(force_model_checkpoint, bool):
            raise TypeError("force_model_checkpoint phải là bool.")
        improved = (
            force_model_checkpoint
            or train_score < self.best_score
        )
        if improved:
            self.best_score = train_score
        output_dir = self.config.checkpoint.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": (
                self.scaler.state_dict()
                if self.scaler.is_enabled()
                else None
            ),
            "epoch": next_epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "config": self.config.resolved_dict(),
            "artifact_sha256": self.artifact_sha256,
            "model_config": self.model_config,
            "grapheme_vocabulary": (
                self.grapheme_vocabulary.to_dict()
            ),
            "stage": self.config.stage,
            "lineage": self.lineage,
            "rng": _rng_state(),
        }
        temporary = output_dir / "last.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output_dir / "last.pt")
        if improved:
            save_model_checkpoint(output_dir / "best.pt", self.model)
            save_inference_contract(
                output_dir,
                num_train_timesteps=(
                    self.config.diffusion.num_train_timesteps
                ),
                artifact_sha256=self.artifact_sha256,
            )
        return improved

    def resume(self, path: Path) -> ResumeState:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        legacy_required = {
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "epoch",
            "global_step",
            "best_score",
            "config",
            "artifact_sha256",
            "model_config",
            "grapheme_vocabulary",
            "rng",
        }
        required = legacy_required | {"stage", "lineage"}
        if not isinstance(payload, Mapping):
            raise ValueError("Generator checkpoint phải là mapping.")
        if set(payload) == legacy_required:
            checkpoint_stage: object = "pretrain"
            checkpoint_lineage = training_lineage(
                self.config,
                payload["artifact_sha256"],  # type: ignore[arg-type]
            )
        elif set(payload) == required:
            checkpoint_stage = payload["stage"]
            checkpoint_lineage = payload["lineage"]
        else:
            raise ValueError(
                f"Generator checkpoint keys phải bằng {sorted(required)}."
            )
        if checkpoint_stage != self.config.stage:
            raise ValueError(
                "Resume bị từ chối: training stage đã thay đổi."
            )
        if checkpoint_lineage != self.lineage:
            raise ValueError(
                "Resume bị từ chối: training lineage đã thay đổi."
            )
        validate_resume_config(
            payload["config"],
            self.config.resolved_dict(),
        )
        if payload["artifact_sha256"] != self.artifact_sha256:
            raise ValueError(
                "Resume bị từ chối: training artifacts đã thay đổi."
            )
        if payload["model_config"] != self.model_config:
            raise ValueError(
                "Resume bị từ chối: model config đã thay đổi."
            )
        if (
            payload["grapheme_vocabulary"]
            != self.grapheme_vocabulary.to_dict()
        ):
            raise ValueError(
                "Resume bị từ chối: grapheme vocabulary đã thay đổi."
            )
        model_state = payload["model"]
        if not isinstance(model_state, Mapping):
            raise ValueError("Generator model state phải là mapping.")
        self.model.load_state_dict(dict(model_state), strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
        self.scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
        scaler_state = payload["scaler"]
        if self.scaler.is_enabled():
            if not isinstance(scaler_state, Mapping):
                raise ValueError("FP16 resume yêu cầu scaler state.")
            self.scaler.load_state_dict(dict(scaler_state))
        elif scaler_state is not None:
            raise ValueError(
                "Checkpoint có scaler nhưng runtime không dùng scaler."
            )
        _restore_rng(payload["rng"])
        self.global_step = int(payload["global_step"])
        self.best_score = float(payload["best_score"])
        return ResumeState(
            int(payload["epoch"]),
            self.global_step,
            self.best_score,
        )


def artifact_hashes(
    config: VietParaDiffTrainingConfig,
) -> dict[str, str]:
    common = {
        "train_references": sha256_file(config.data.train_references),
        "autokl_checkpoint": sha256_file(config.autokl.checkpoint),
        "latent_statistics": sha256_file(
            config.autokl.latent_statistics
        ),
    }
    if config.stage == "pretrain":
        if config.data.train_targets is None:
            raise RuntimeError("Pretrain train_targets bị thiếu.")
        if (
            config.style is None
            or config.style.convnext_checkpoint is None
            or config.style.backbone_contract is None
        ):
            raise RuntimeError("Pretrain visual artifacts bị thiếu.")
        convnext_hash = verify_visual_backbone(
            config.style.backbone_contract,
            name="convnext_tiny_imagenet1k_v1",
            checkpoint=config.style.convnext_checkpoint,
        )
        return {
            "train_targets": sha256_file(config.data.train_targets),
            "convnext_checkpoint": convnext_hash,
            "visual_backbone_contract": sha256_file(
                config.style.backbone_contract
            ),
            **common,
        }
    if (
        config.data.real_targets is None
        or config.initialization is None
    ):
        raise RuntimeError("Derived stage artifacts bị thiếu.")
    hashes = {
        "real_targets": sha256_file(config.data.real_targets),
        **common,
        "parent_checkpoint": sha256_file(
            config.initialization.checkpoint
        ),
        "parent_contract": sha256_file(
            config.initialization.contract
        ),
        "parent_model_config": sha256_file(
            config.initialization.model_config
        ),
        "parent_vocabulary": sha256_file(
            config.initialization.vocabulary
        ),
    }
    if config.data.use_synthetic_data:
        if config.data.synthetic_targets is None:
            raise RuntimeError("Synthetic target manifest bị thiếu.")
        hashes["synthetic_targets"] = sha256_file(
            config.data.synthetic_targets
        )
    if config.stage == "htr_guided":
        if config.guidance is None:
            raise RuntimeError("HTR guidance config bị thiếu.")
        hashes.update(
            {
                "htr_checkpoint": sha256_file(
                    config.guidance.checkpoint
                ),
                "htr_contract": sha256_file(
                    config.guidance.checkpoint.parent
                    / "inference_contract.json"
                ),
                "htr_model_config": sha256_file(
                    config.guidance.model_config
                ),
                "htr_vocabulary": sha256_file(
                    config.guidance.vocabulary
                ),
            }
        )
    return hashes


__all__ = [
    "DiffusionStageConfig",
    "DiffusionStepOutput",
    "DeterministicRealSyntheticBatchMixer",
    "FrozenAutoKLConfig",
    "GeneratorCheckpointConfig",
    "GeneratorInitializationConfig",
    "GeneratorEpochMetrics",
    "GeneratorLoggingConfig",
    "GeneratorOptimizerConfig",
    "GeneratorSchedulerConfig",
    "LatentStatistics",
    "LatentStatisticsAccumulator",
    "ModelBehaviorConfig",
    "ResumeState",
    "StyleInitializationConfig",
    "VietParaDiffDataConfig",
    "VietParaDiffLogger",
    "VietParaDiffTrainer",
    "VietParaDiffTrainingConfig",
    "add_diffusion_noise",
    "artifact_hashes",
    "cosine_alpha_sigma",
    "create_grad_scaler",
    "create_optimizer_and_scheduler",
    "ensure_inference_static_artifacts",
    "expected_artifact_names",
    "learning_rate_factor",
    "load_latent_statistics",
    "load_vietparadiff_training_config",
    "resolve_runtime",
    "save_latent_statistics",
    "save_inference_contract",
    "seed_everything",
    "sha256_file",
    "training_lineage",
    "validate_resume_config",
    "velocity_target",
]
