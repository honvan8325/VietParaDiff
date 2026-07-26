"""Base velocity-diffusion training mechanics for VietParaDiff."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
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

from src.autokl_training import (
    RuntimePrecision,
    autocast_context,
    create_grad_scaler,
    resolve_runtime,
    seed_everything,
)
from src.models.autokl import HandwritingAutoKL
from src.models.text import GraphemeBatch, GraphemeVocabulary
from src.models.vietparadiff import (
    VietParaDiff,
    VietParaDiffInput,
    VietParaDiffOutput,
)

HEIGHT_BUCKETS = (384, 512, 640, 768, 896, 1024, 1280)


@dataclass(frozen=True, slots=True)
class VietParaDiffDataConfig:
    train_targets: Path
    train_references: Path
    image_root: Path
    num_workers: int
    batch_size: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("data.num_workers không được âm.")
        if self.batch_size <= 0:
            raise ValueError("data.batch_size phải dương.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "data.gradient_accumulation_steps phải dương."
            )


@dataclass(frozen=True, slots=True)
class FrozenAutoKLConfig:
    checkpoint: Path
    latent_statistics: Path


@dataclass(frozen=True, slots=True)
class StyleInitializationConfig:
    use_pretrained_backbone: bool
    convnext_checkpoint: Path | None

    def __post_init__(self) -> None:
        if not self.use_pretrained_backbone:
            raise ValueError(
                "Base VietParaDiff phải khởi tạo ConvNeXt từ ImageNet."
            )


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
    style: StyleInitializationConfig
    diffusion: DiffusionStageConfig
    optimizer: GeneratorOptimizerConfig
    scheduler: GeneratorSchedulerConfig
    logging: GeneratorLoggingConfig
    checkpoint: GeneratorCheckpointConfig

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
    expected_root = {
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
    if set(raw) != expected_root:
        raise ValueError(
            f"Generator config root keys phải bằng {sorted(expected_root)}."
        )
    data = _config_section(
        raw,
        "data",
        {
            "train_targets",
            "train_references",
            "image_root",
            "num_workers",
            "batch_size",
            "gradient_accumulation_steps",
        },
    )
    autokl = _config_section(
        raw, "autokl", {"checkpoint", "latent_statistics"}
    )
    style = _config_section(
        raw,
        "style",
        {"use_pretrained_backbone", "convnext_checkpoint"},
    )
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
    return VietParaDiffTrainingConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        data=VietParaDiffDataConfig(
            train_targets=Path(str(data["train_targets"])),
            train_references=Path(str(data["train_references"])),
            image_root=Path(str(data["image_root"])),
            num_workers=int(data["num_workers"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation_steps=int(
                data["gradient_accumulation_steps"]
            ),
        ),
        autokl=FrozenAutoKLConfig(
            checkpoint=Path(str(autokl["checkpoint"])),
            latent_statistics=Path(str(autokl["latent_statistics"])),
        ),
        style=StyleInitializationConfig(
            use_pretrained_backbone=bool(
                style["use_pretrained_backbone"]
            ),
            convnext_checkpoint=(
                None
                if style["convnext_checkpoint"] is None
                else Path(str(style["convnext_checkpoint"]))
            ),
        ),
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
    )


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LatentStatistics:
    latent_mean: float
    latent_std: float
    scaling_factor: float
    num_samples: int
    num_elements: int
    autokl_checkpoint_sha256: str

    def __post_init__(self) -> None:
        numeric = (
            self.latent_mean,
            self.latent_std,
            self.scaling_factor,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Latent statistics phải hữu hạn.")
        if self.latent_std <= 0.0 or self.scaling_factor <= 0.0:
            raise ValueError("latent_std/scaling_factor phải dương.")
        if not math.isclose(
            self.scaling_factor,
            1.0 / self.latent_std,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "scaling_factor phải bằng chính xác 1 / latent_std."
            )
        if self.num_samples <= 0 or self.num_elements <= 1:
            raise ValueError("Latent statistics cần dữ liệu không rỗng.")
        if len(self.autokl_checkpoint_sha256) != 64:
            raise ValueError("AutoKL SHA-256 không hợp lệ.")

    def normalize(self, latents: Tensor) -> Tensor:
        if not latents.is_floating_point():
            raise TypeError("latents phải có floating-point dtype.")
        return (
            latents - latents.new_tensor(self.latent_mean)
        ) * latents.new_tensor(self.scaling_factor)

    def denormalize(self, scaled_latents: Tensor) -> Tensor:
        if not scaled_latents.is_floating_point():
            raise TypeError(
                "scaled_latents phải có floating-point dtype."
            )
        return (
            scaled_latents
            / scaled_latents.new_tensor(self.scaling_factor)
            + scaled_latents.new_tensor(self.latent_mean)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "scaling_factor": self.scaling_factor,
            "num_samples": self.num_samples,
            "num_elements": self.num_elements,
            "autokl_checkpoint_sha256": self.autokl_checkpoint_sha256,
        }


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


def save_latent_statistics(
    path: Path,
    statistics: LatentStatistics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            statistics.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_latent_statistics(
    path: Path,
    *,
    expected_autokl_checkpoint: Path,
) -> LatentStatistics:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy latent statistics: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "latent_mean",
        "latent_std",
        "scaling_factor",
        "num_samples",
        "num_elements",
        "autokl_checkpoint_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError(
            f"Latent statistics keys phải bằng {sorted(expected_keys)}."
        )
    statistics = LatentStatistics(
        latent_mean=float(payload["latent_mean"]),
        latent_std=float(payload["latent_std"]),
        scaling_factor=float(payload["scaling_factor"]),
        num_samples=int(payload["num_samples"]),
        num_elements=int(payload["num_elements"]),
        autokl_checkpoint_sha256=str(
            payload["autokl_checkpoint_sha256"]
        ),
    )
    actual_hash = sha256_file(expected_autokl_checkpoint)
    if statistics.autokl_checkpoint_sha256 != actual_hash:
        raise ValueError(
            "latent_statistics.json không thuộc AutoKL checkpoint hiện tại."
        )
    return statistics


def cosine_alpha_sigma(
    timesteps: Tensor,
    *,
    num_train_timesteps: int,
) -> tuple[Tensor, Tensor]:
    if (
        timesteps.ndim != 1
        or timesteps.dtype != torch.long
        or num_train_timesteps < 2
    ):
        raise ValueError(
            "timesteps phải là long [B] và num_train_timesteps >= 2."
        )
    if (
        (timesteps < 0).any()
        or (timesteps >= num_train_timesteps).any()
    ):
        raise ValueError(
            f"timesteps phải nằm trong [0,{num_train_timesteps - 1}]."
        )
    offset = 0.008
    progress = (timesteps.float() + 0.5) / num_train_timesteps
    angles = (
        (progress + offset) / (1.0 + offset) * math.pi / 2.0
    )
    return torch.cos(angles), torch.sin(angles)


def velocity_target(
    clean_latents: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    if clean_latents.shape != noise.shape:
        raise ValueError("clean_latents và noise phải cùng shape.")
    if clean_latents.ndim != 4:
        raise ValueError("clean_latents phải có shape [B,C,H,W].")
    expected = (clean_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(
            f"alpha/sigma phải có shape {expected}."
        )
    return (
        alpha[:, None, None, None].to(clean_latents.dtype) * noise
        - sigma[:, None, None, None].to(clean_latents.dtype)
        * clean_latents
    )


def add_diffusion_noise(
    clean_latents: Tensor,
    noise: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    if clean_latents.shape != noise.shape or clean_latents.ndim != 4:
        raise ValueError(
            "clean_latents/noise phải cùng shape [B,C,H,W]."
        )
    expected = (clean_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(
            f"alpha/sigma phải có shape {expected}."
        )
    return (
        alpha[:, None, None, None].to(clean_latents.dtype)
        * clean_latents
        + sigma[:, None, None, None].to(clean_latents.dtype) * noise
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
    clean_latents: Tensor
    noisy_latents: Tensor
    target_velocity: Tensor
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True, slots=True)
class GeneratorEpochMetrics:
    velocity_mse: float
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
    names = (
        "data",
        "autokl",
        "style",
        "diffusion",
        "optimizer",
        "scheduler",
    )
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
    required_data = {
        "train_targets",
        "train_references",
        "image_root",
        "batch_size",
        "gradient_accumulation_steps",
    }
    if not required_data.issubset(data):
        raise ValueError("Resume config.data thiếu training fields.")
    return {
        "seed": config["seed"],
        "precision": config["precision"],
        "data": {
            name: data[name]
            for name in sorted(required_data)
        },
        "autokl": dict(sections["autokl"]),
        "style": dict(sections["style"]),
        "diffusion": dict(sections["diffusion"]),
        "optimizer": dict(sections["optimizer"]),
        "scheduler": dict(sections["scheduler"]),
    }


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
    expected_artifacts = {
        "train_targets",
        "train_references",
        "autokl_checkpoint",
        "latent_statistics",
    }
    if set(artifact_sha256) != expected_artifacts:
        raise ValueError("Inference contract artifact schema sai.")
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
    ) -> None:
        if set(artifact_sha256) != {
            "train_targets",
            "train_references",
            "autokl_checkpoint",
            "latent_statistics",
        }:
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
        self.global_step = 0
        self.best_score = float("inf")
        for parameter in self.autokl.parameters():
            parameter.requires_grad_(False)
        self.autokl.eval()
        self.optimizer.zero_grad(set_to_none=True)
        ensure_inference_static_artifacts(
            self.config.checkpoint.output_dir,
            self.model_config,
            self.grapheme_vocabulary,
        )

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
            loss = F.mse_loss(
                model_output.predicted_velocity.float(),
                target_velocity.float(),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Velocity loss chứa NaN/Inf.")
        self.scaler.scale(loss / accumulation_divisor).backward()
        return DiffusionStepOutput(
            model_output=model_output,
            loss=loss,
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

    def train_epoch(
        self,
        loader: Iterable[Mapping[str, object]],
        *,
        epoch: int,
    ) -> GeneratorEpochMetrics:
        if not hasattr(loader, "__len__") or len(loader) <= 0:  # type: ignore[arg-type]
            raise ValueError("Generator train loader không được rỗng.")
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        total_batches = len(loader)  # type: ignore[arg-type]
        accumulation = self.config.data.gradient_accumulation_steps
        loss_sum = 0.0
        sample_count = 0
        group_losses: list[float] = []
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
                group_losses.clear()
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
            loss_value = float(output.loss.detach().cpu())
            loss_sum += loss_value * batch_size
            sample_count += batch_size
            group_losses.append(loss_value)
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
                len(group_losses) == group_size
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
                        "train/velocity_mse": sum(group_losses)
                        / len(group_losses),
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
            velocity_mse=loss_sum / sample_count,
            sample_count=sample_count,
        )

    def save_epoch_checkpoints(
        self,
        *,
        next_epoch: int,
        train_score: float,
    ) -> bool:
        if next_epoch < 0 or not math.isfinite(train_score):
            raise ValueError("Checkpoint epoch/score không hợp lệ.")
        improved = train_score < self.best_score
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
        required = {
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
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError(
                f"Generator checkpoint keys phải bằng {sorted(required)}."
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
    return {
        "train_targets": sha256_file(config.data.train_targets),
        "train_references": sha256_file(
            config.data.train_references
        ),
        "autokl_checkpoint": sha256_file(config.autokl.checkpoint),
        "latent_statistics": sha256_file(
            config.autokl.latent_statistics
        ),
    }


__all__ = [
    "DiffusionStageConfig",
    "DiffusionStepOutput",
    "FrozenAutoKLConfig",
    "GeneratorCheckpointConfig",
    "GeneratorEpochMetrics",
    "GeneratorLoggingConfig",
    "GeneratorOptimizerConfig",
    "GeneratorSchedulerConfig",
    "LatentStatistics",
    "LatentStatisticsAccumulator",
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
    "learning_rate_factor",
    "load_latent_statistics",
    "load_vietparadiff_training_config",
    "resolve_runtime",
    "save_latent_statistics",
    "save_inference_contract",
    "seed_everything",
    "sha256_file",
    "validate_resume_config",
    "velocity_target",
]
