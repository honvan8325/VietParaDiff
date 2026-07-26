"""Stage 1 training mechanics for the handwriting AutoKL."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import wandb
from src.models.autokl import AutoKLOutput, HandwritingAutoKL


HEIGHT_BUCKETS = (384, 512, 640, 768, 896, 1024, 1280)


@dataclass(frozen=True, slots=True)
class DataTrainingConfig:
    train_manifest: Path
    test_manifest: Path
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
class AutoKLStageConfig:
    epochs: int
    sample_posterior_train: bool
    sample_posterior_eval: bool

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("autokl.epochs phải dương.")
        if not self.sample_posterior_train:
            raise ValueError(
                "AutoKL training phải dùng posterior sampling."
            )
        if self.sample_posterior_eval:
            raise ValueError(
                "AutoKL evaluation phải dùng posterior mode."
            )


@dataclass(frozen=True, slots=True)
class OptimizerTrainingConfig:
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
            raise ValueError("optimizer.betas phải gồm hai giá trị trong [0,1).")
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay không được âm.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer.gradient_clip_norm phải dương.")


@dataclass(frozen=True, slots=True)
class AutoKLLossConfig:
    foreground_weight: float
    edge_weight: float
    kl_max_weight: float
    kl_warmup_steps: int

    def __post_init__(self) -> None:
        if self.foreground_weight < 0.0:
            raise ValueError("loss.foreground_weight không được âm.")
        if self.edge_weight < 0.0:
            raise ValueError("loss.edge_weight không được âm.")
        if self.kl_max_weight < 0.0:
            raise ValueError("loss.kl_max_weight không được âm.")
        if self.kl_warmup_steps <= 0:
            raise ValueError("loss.kl_warmup_steps phải dương.")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    log_every_steps: int
    image_every_steps: int
    tensorboard: bool
    wandb: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_mode: Literal["online", "offline", "disabled"]
    run_name: str | None

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0 or self.image_every_steps <= 0:
            raise ValueError("Logging intervals phải dương.")
        if not self.wandb_project:
            raise ValueError("logging.wandb_project không được rỗng.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "logging.wandb_mode phải là online, offline hoặc disabled."
            )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    output_dir: Path
    save_last: bool
    save_best: bool

    def __post_init__(self) -> None:
        if not self.save_last:
            raise ValueError("checkpoint.save_last phải bật.")
        if not self.save_best:
            raise ValueError("checkpoint.save_best phải bật.")


@dataclass(frozen=True, slots=True)
class AutoKLTrainingConfig:
    seed: int
    device: str
    precision: str
    data: DataTrainingConfig
    autokl: AutoKLStageConfig
    optimizer: OptimizerTrainingConfig
    loss: AutoKLLossConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed không được âm.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device phải là auto, cuda, mps hoặc cpu.")
        if self.precision not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError(
                "precision phải là auto, float32, float16 hoặc bfloat16."
            )

    def resolved_dict(self) -> dict[str, object]:
        payload = asdict(self)

        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        converted = convert(payload)
        if not isinstance(converted, dict):
            raise RuntimeError("Resolved training config phải là mapping.")
        return converted


def _mapping(
    payload: Mapping[str, object],
    key: str,
    expected_keys: set[str],
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} phải là mapping.")
    actual = set(value)
    if actual != expected_keys:
        raise ValueError(
            f"config.{key} keys phải bằng {sorted(expected_keys)}, "
            f"nhận {sorted(str(item) for item in actual)}."
        )
    return dict(value)


def load_training_config(path: Path) -> AutoKLTrainingConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Training config root phải là mapping.")
    expected_root = {
        "seed",
        "device",
        "precision",
        "data",
        "autokl",
        "optimizer",
        "loss",
        "logging",
        "checkpoint",
    }
    if set(raw) != expected_root:
        raise ValueError(
            f"Config root keys phải bằng {sorted(expected_root)}."
        )
    data = _mapping(
        raw,
        "data",
        {
            "train_manifest",
            "test_manifest",
            "image_root",
            "num_workers",
            "batch_size",
            "gradient_accumulation_steps",
        },
    )
    stage = _mapping(
        raw,
        "autokl",
        {"epochs", "sample_posterior_train", "sample_posterior_eval"},
    )
    optimizer = _mapping(
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
    loss = _mapping(
        raw,
        "loss",
        {
            "foreground_weight",
            "edge_weight",
            "kl_max_weight",
            "kl_warmup_steps",
        },
    )
    logging = _mapping(
        raw,
        "logging",
        {
            "log_every_steps",
            "image_every_steps",
            "tensorboard",
            "wandb",
            "wandb_project",
            "wandb_entity",
            "wandb_mode",
            "run_name",
        },
    )
    checkpoint = _mapping(
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
    return AutoKLTrainingConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        data=DataTrainingConfig(
            train_manifest=Path(str(data["train_manifest"])),
            test_manifest=Path(str(data["test_manifest"])),
            image_root=Path(str(data["image_root"])),
            num_workers=int(data["num_workers"]),
            batch_size=int(data["batch_size"]),
            gradient_accumulation_steps=int(
                data["gradient_accumulation_steps"]
            ),
        ),
        autokl=AutoKLStageConfig(
            epochs=int(stage["epochs"]),
            sample_posterior_train=bool(stage["sample_posterior_train"]),
            sample_posterior_eval=bool(stage["sample_posterior_eval"]),
        ),
        optimizer=OptimizerTrainingConfig(
            name=str(optimizer["name"]),
            learning_rate=float(optimizer["learning_rate"]),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        ),
        loss=AutoKLLossConfig(
            foreground_weight=float(loss["foreground_weight"]),
            edge_weight=float(loss["edge_weight"]),
            kl_max_weight=float(loss["kl_max_weight"]),
            kl_warmup_steps=int(loss["kl_warmup_steps"]),
        ),
        logging=LoggingConfig(
            log_every_steps=int(logging["log_every_steps"]),
            image_every_steps=int(logging["image_every_steps"]),
            tensorboard=bool(logging["tensorboard"]),
            wandb=bool(logging["wandb"]),
            wandb_project=str(logging["wandb_project"]),
            wandb_entity=(
                None
                if logging["wandb_entity"] is None
                else str(logging["wandb_entity"])
            ),
            wandb_mode=str(logging["wandb_mode"]),  # type: ignore[arg-type]
            run_name=(
                None
                if logging["run_name"] is None
                else str(logging["run_name"])
            ),
        ),
        checkpoint=CheckpointConfig(
            output_dir=Path(str(checkpoint["output_dir"])),
            save_last=bool(checkpoint["save_last"]),
            save_best=bool(checkpoint["save_best"]),
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimePrecision:
    device: torch.device
    dtype: torch.dtype
    autocast_enabled: bool
    scaler_enabled: bool


def resolve_runtime(device: str, precision: str) -> RuntimePrecision:
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS được yêu cầu nhưng không khả dụng.")

    if precision == "auto":
        if resolved_device.type == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32
    else:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
    if resolved_device.type != "cuda" and dtype != torch.float32:
        raise ValueError("CPU/MPS training chỉ hỗ trợ float32.")
    return RuntimePrecision(
        device=resolved_device,
        dtype=dtype,
        autocast_enabled=(
            resolved_device.type == "cuda" and dtype != torch.float32
        ),
        scaler_enabled=(
            resolved_device.type == "cuda" and dtype == torch.float16
        ),
    )


def kl_weight_at_step(
    global_step: int,
    *,
    max_weight: float,
    warmup_steps: int,
) -> float:
    if global_step < 0:
        raise ValueError("global_step không được âm.")
    if max_weight < 0.0 or warmup_steps <= 0:
        raise ValueError("KL max weight/warmup không hợp lệ.")
    return max_weight * min(global_step / warmup_steps, 1.0)


def laplacian(images: Tensor) -> Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"images phải có shape [B,1,H,W], nhận {tuple(images.shape)}."
        )
    kernel = images.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    return F.conv2d(images, kernel, padding=1)


@dataclass(frozen=True, slots=True)
class AutoKLLosses:
    total: Tensor
    reconstruction: Tensor
    edge: Tensor
    kl: Tensor
    kl_weight: float

    @property
    def checkpoint_score(self) -> Tensor:
        return self.reconstruction + 0.1 * self.edge


def compute_autokl_losses(
    output: AutoKLOutput,
    images: Tensor,
    config: AutoKLLossConfig,
    *,
    global_step: int,
) -> AutoKLLosses:
    if output.reconstruction.shape != images.shape:
        raise ValueError(
            "Reconstruction shape phải khớp images: "
            f"expected {tuple(images.shape)}, "
            f"actual {tuple(output.reconstruction.shape)}."
        )
    if output.latent.ndim != 4 or output.latent.shape[0] != images.shape[0]:
        raise ValueError("AutoKL latent phải là tensor [B,C,H,W].")
    target_ink = ((1.0 - images) / 2.0).clamp(0.0, 1.0)
    pixel_weights = 1.0 + config.foreground_weight * target_ink
    reconstruction = (
        pixel_weights * (output.reconstruction - images).abs()
    ).sum() / pixel_weights.sum()
    edge = F.l1_loss(
        laplacian(output.reconstruction),
        laplacian(images),
    )
    latent_elements = output.latent[0].numel()
    if latent_elements <= 0:
        raise ValueError("AutoKL latent không được rỗng.")
    kl = output.posterior.kl().mean() / latent_elements
    weight = kl_weight_at_step(
        global_step,
        max_weight=config.kl_max_weight,
        warmup_steps=config.kl_warmup_steps,
    )
    total = reconstruction + config.edge_weight * edge + weight * kl
    values = (total, reconstruction, edge, kl)
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("AutoKL loss chứa NaN hoặc Inf.")
    return AutoKLLosses(total, reconstruction, edge, kl, weight)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: OptimizerTrainingConfig,
) -> tuple[Optimizer, LambdaLR]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler


def create_grad_scaler(runtime: RuntimePrecision) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        "cuda",
        enabled=runtime.scaler_enabled,
    )


def autocast_context(runtime: RuntimePrecision) -> Any:
    if not runtime.autocast_enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=runtime.dtype,
        enabled=True,
    )


def _tensor_to_pil(tensor: Tensor) -> Image.Image:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("Render tensor phải có shape [1,H,W].")
    pixels = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .contiguous()
    )
    height, width = pixels.shape
    return Image.frombytes(
        "L",
        (width, height),
        bytes(pixels.untyped_storage()),
    )


def reconstruction_panel(
    images: Tensor,
    reconstructions: Tensor,
) -> Tensor:
    if images.shape != reconstructions.shape:
        raise ValueError("Input và reconstruction phải cùng shape.")
    input_view = images.detach().float().cpu().add(1.0).div(2.0)
    reconstruction_view = (
        reconstructions.detach().float().cpu().add(1.0).div(2.0)
    )
    error_view = (
        reconstructions.detach().float().cpu() - images.detach().float().cpu()
    ).abs().div(2.0).clamp(0.0, 1.0)
    return torch.cat(
        (input_view, reconstruction_view, error_view),
        dim=-1,
    )


def save_reconstruction_grid(
    images: Tensor,
    reconstructions: Tensor,
    path: Path,
    *,
    labels: Sequence[str] | None = None,
) -> None:
    panels = reconstruction_panel(images, reconstructions)
    panel_images = [_tensor_to_pil(panel) for panel in panels]
    if not panel_images:
        raise ValueError("Không có reconstruction để render.")
    label_height = 24
    width = max(image.width for image in panel_images)
    height = sum(image.height + label_height for image in panel_images)
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    try:
        y = 0
        for index, image in enumerate(panel_images):
            label = (
                labels[index]
                if labels is not None
                else f"sample {index}"
            )
            draw.text((8, y + 6), label, fill=0)
            canvas.paste(image, (0, y + label_height))
            y += image.height + label_height
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    finally:
        for image in panel_images:
            image.close()
        canvas.close()


class AutoKLLogger:
    def __init__(
        self,
        config: LoggingConfig,
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

    def log_images(
        self,
        panels: Mapping[int, Tensor],
        *,
        step: int,
    ) -> None:
        if self.writer is not None:
            for bucket, panel in panels.items():
                self.writer.add_images(
                    f"reconstruction/bucket_{bucket}",
                    panel,
                    step,
                )
        if self.run is not None:
            pil_images: list[Image.Image] = []
            payload: dict[str, list[wandb.Image]] = {}
            try:
                for bucket, panel in panels.items():
                    bucket_images = [
                        _tensor_to_pil(item)
                        for item in panel
                    ]
                    pil_images.extend(bucket_images)
                    payload[f"reconstruction/bucket_{bucket}"] = [
                        wandb.Image(
                            image,
                            caption=(
                                "input | deterministic reconstruction | error"
                            ),
                        )
                        for image in bucket_images
                    ]
                self.run.log(payload, step=step)
            finally:
                for image in pil_images:
                    image.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.run is not None:
            self.run.finish()


def _checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    global_step: int,
    best_score: float,
    resolved_config: Mapping[str, object],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "config": dict(resolved_config),
        "rng": {
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
        },
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    global_step: int,
    best_score: float,
    resolved_config: Mapping[str, object],
) -> None:
    if epoch < 0 or global_step < 0:
        raise ValueError("Checkpoint epoch/global_step không được âm.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            resolved_config=resolved_config,
        ),
        temporary,
    )
    temporary.replace(path)


def save_model_checkpoint(path: Path, model: nn.Module) -> None:
    """Atomically save the strict model-only checkpoint used downstream."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict()}, temporary)
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class ResumeState:
    epoch: int
    global_step: int
    best_score: float


def _resume_config_signature(
    config: Mapping[str, object],
) -> dict[str, object]:
    def section(name: str) -> Mapping[str, object]:
        value = config.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Checkpoint/config section {name!r} phải là mapping."
            )
        return value

    data = section("data")
    autokl = section("autokl")
    optimizer = section("optimizer")
    loss = section("loss")
    required = {
        "seed": config.get("seed"),
        "data.train_manifest": data.get("train_manifest"),
        "data.batch_size": data.get("batch_size"),
        "data.gradient_accumulation_steps": data.get(
            "gradient_accumulation_steps"
        ),
        "optimizer": dict(optimizer),
        "loss": dict(loss),
        "autokl.sample_posterior_train": autokl.get(
            "sample_posterior_train"
        ),
        "autokl.sample_posterior_eval": autokl.get(
            "sample_posterior_eval"
        ),
    }
    missing = [
        name
        for name, value in required.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Checkpoint/config thiếu resume fields: "
            f"{sorted(missing)}."
        )
    return required


def validate_resume_config(
    checkpoint_config: Mapping[str, object],
    current_config: Mapping[str, object],
) -> None:
    checkpoint_signature = _resume_config_signature(checkpoint_config)
    current_signature = _resume_config_signature(current_config)
    mismatches = [
        name
        for name in checkpoint_signature
        if checkpoint_signature[name] != current_signature[name]
    ]
    if mismatches:
        details = "; ".join(
            f"{name}: checkpoint={checkpoint_signature[name]!r}, "
            f"current={current_signature[name]!r}"
            for name in mismatches
        )
        raise ValueError(f"Resume config không tương thích: {details}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    current_config: Mapping[str, object],
) -> ResumeState:
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
        "rng",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError(
            f"Checkpoint keys phải bằng {sorted(required)}."
        )
    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint config phải là mapping.")
    validate_resume_config(checkpoint_config, current_config)
    model_state = payload["model"]
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint model state phải là mapping.")
    model.load_state_dict(dict(model_state), strict=True)
    optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
    scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
    scaler_state = payload["scaler"]
    if scaler.is_enabled():
        if not isinstance(scaler_state, Mapping):
            raise ValueError("CUDA FP16 resume yêu cầu scaler state.")
        scaler.load_state_dict(dict(scaler_state))
    elif scaler_state is not None:
        raise ValueError(
            "Checkpoint có scaler state nhưng runtime không dùng scaler."
        )
    rng = payload["rng"]
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "torch",
        "cuda",
        "mps",
    }:
        raise ValueError("Checkpoint RNG state sai schema.")
    torch_state = rng["torch"]
    if not isinstance(torch_state, Tensor):
        raise ValueError("Checkpoint torch RNG state phải là Tensor.")
    random.setstate(rng["python"])  # type: ignore[arg-type]
    torch.set_rng_state(torch_state)
    cuda_state = rng["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint có CUDA RNG state nhưng CUDA không khả dụng."
            )
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
    mps_state = rng["mps"]
    if mps_state is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Checkpoint có MPS RNG state nhưng MPS không khả dụng."
            )
        if not isinstance(mps_state, Tensor):
            raise ValueError("Checkpoint MPS RNG state phải là Tensor.")
        torch.mps.set_rng_state(mps_state)
    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_score=float(payload["best_score"]),
    )


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    total_loss: float
    reconstruction_loss: float
    edge_loss: float
    kl_loss: float
    checkpoint_score: float
    sample_count: int


class AutoKLTrainer:
    def __init__(
        self,
        model: HandwritingAutoKL,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        scaler: torch.amp.GradScaler,
        config: AutoKLTrainingConfig,
        runtime: RuntimePrecision,
        logger: AutoKLLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.runtime = runtime
        self.logger = logger
        self.global_step = 0
        self.best_score = float("inf")
        self.optimizer.zero_grad(set_to_none=True)

    def _forward(
        self,
        images: Tensor,
        *,
        sample_posterior: bool,
    ) -> tuple[AutoKLOutput, AutoKLLosses]:
        with autocast_context(self.runtime):
            output = self.model(
                images,
                sample_posterior=sample_posterior,
            )
            losses = compute_autokl_losses(
                output,
                images,
                self.config.loss,
                global_step=self.global_step,
            )
        return output, losses

    def train_step(
        self,
        images: Tensor,
        *,
        accumulation_divisor: int = 1,
    ) -> tuple[AutoKLOutput, AutoKLLosses]:
        if accumulation_divisor <= 0:
            raise ValueError("accumulation_divisor phải dương.")
        self.model.train()
        images = images.to(
            self.runtime.device,
            non_blocking=self.runtime.device.type == "cuda",
        )
        output, losses = self._forward(
            images,
            sample_posterior=self.config.autokl.sample_posterior_train,
        )
        self.scaler.scale(
            losses.total / accumulation_divisor
        ).backward()
        return output, losses

    def optimizer_step(self) -> float:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optimizer.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Gradient norm không hữu hạn.")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(gradient_norm.detach().cpu())

    @torch.no_grad()
    def deterministic_reconstruction(self, images: Tensor) -> Tensor:
        was_training = self.model.training
        self.model.eval()
        images = images.to(self.runtime.device)
        with autocast_context(self.runtime):
            reconstruction = self.model(
                images,
                sample_posterior=self.config.autokl.sample_posterior_eval,
            ).reconstruction
        if was_training:
            self.model.train()
        return reconstruction.detach().float().cpu()

    def train_epoch(
        self,
        loader: Any,
        *,
        epoch: int,
    ) -> EpochMetrics:
        if len(loader) <= 0:
            raise ValueError("Train loader không được rỗng.")
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        sums = {
            "total": 0.0,
            "reconstruction": 0.0,
            "edge": 0.0,
            "kl": 0.0,
            "checkpoint": 0.0,
        }
        sample_count = 0
        cached_images: dict[int, Tensor] = {}
        accumulation = self.config.data.gradient_accumulation_steps
        total_batches = len(loader)
        group_size = min(accumulation, total_batches)
        group_metrics: list[dict[str, float]] = []
        group_bucket = 0
        for batch_index, batch in enumerate(loader):
            images = batch.get("images")
            bucket = batch.get("height_bucket")
            if not isinstance(images, Tensor):
                raise TypeError("Train batch images phải là Tensor.")
            if not isinstance(bucket, int):
                raise TypeError("Train height_bucket phải là int.")
            if bucket not in HEIGHT_BUCKETS:
                raise ValueError(f"Height bucket không hợp lệ: {bucket}")
            if batch_index % accumulation == 0:
                group_size = min(
                    accumulation,
                    total_batches - batch_index,
                )
                group_metrics.clear()
            cached_images.setdefault(bucket, images[:1].detach().cpu())
            output, losses = self.train_step(
                images,
                accumulation_divisor=group_size,
            )
            group_bucket = bucket
            batch_size = images.shape[0]
            sample_count += batch_size
            metric = {
                "total": float(losses.total.detach().cpu()),
                "reconstruction": float(
                    losses.reconstruction.detach().cpu()
                ),
                "edge": float(losses.edge.detach().cpu()),
                "kl": float(losses.kl.detach().cpu()),
                "checkpoint": float(
                    losses.checkpoint_score.detach().cpu()
                ),
                "kl_weight": losses.kl_weight,
            }
            group_metrics.append(metric)
            for name in (
                "total",
                "reconstruction",
                "edge",
                "kl",
                "checkpoint",
            ):
                sums[name] += metric[name] * batch_size
            is_group_end = (
                len(group_metrics) == group_size
                or batch_index + 1 == total_batches
            )
            if not is_group_end:
                del output, losses
                if self.runtime.device.type == "mps":
                    torch.mps.empty_cache()
                continue
            gradient_norm = self.optimizer_step()
            if (
                self.logger is not None
                and self.global_step % self.config.logging.log_every_steps == 0
            ):
                count = len(group_metrics)
                last = group_metrics[-1]
                self.logger.log_scalars(
                    {
                        "train/total_loss": sum(
                            item["total"]
                            for item in group_metrics
                        )
                        / count,
                        "train/reconstruction_loss": sum(
                            item["reconstruction"]
                            for item in group_metrics
                        )
                        / count,
                        "train/edge_loss": sum(
                            item["edge"]
                            for item in group_metrics
                        )
                        / count,
                        "train/kl_loss": sum(
                            item["kl"]
                            for item in group_metrics
                        )
                        / count,
                        "train/kl_weight": last["kl_weight"],
                        "train/latent_mean": float(
                            output.latent.detach().float().mean().cpu()
                        ),
                        "train/latent_std": float(
                            output.latent.detach().float().std().cpu()
                        ),
                        "train/learning_rate": float(
                            self.optimizer.param_groups[0]["lr"]
                        ),
                        "train/gradient_norm": gradient_norm,
                        "train/height_bucket": float(group_bucket),
                    },
                    step=self.global_step,
                )
            if (
                self.logger is not None
                and self.global_step % self.config.logging.image_every_steps
                == 0
            ):
                panels = {
                    height: reconstruction_panel(
                        sample,
                        self.deterministic_reconstruction(sample),
                    )
                    for height, sample in sorted(cached_images.items())
                }
                self.logger.log_images(panels, step=self.global_step)
            del output, losses
            if self.runtime.device.type == "mps":
                torch.mps.empty_cache()
        if sample_count <= 0:
            raise RuntimeError("Train epoch không xử lý sample nào.")
        return EpochMetrics(
            total_loss=sums["total"] / sample_count,
            reconstruction_loss=sums["reconstruction"] / sample_count,
            edge_loss=sums["edge"] / sample_count,
            kl_loss=sums["kl"] / sample_count,
            checkpoint_score=sums["checkpoint"] / sample_count,
            sample_count=sample_count,
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: Any,
        *,
        render_dir: Path | None = None,
    ) -> EpochMetrics:
        self.model.eval()
        sums = {
            "total": 0.0,
            "reconstruction": 0.0,
            "edge": 0.0,
            "kl": 0.0,
            "checkpoint": 0.0,
        }
        sample_count = 0
        rendered: set[int] = set()
        for batch in loader:
            images = batch.get("images")
            bucket = batch.get("height_bucket")
            sample_ids = batch.get("sample_ids")
            if not isinstance(images, Tensor) or not isinstance(bucket, int):
                raise TypeError("Evaluation batch sai schema.")
            images = images.to(self.runtime.device)
            output, losses = self._forward(
                images,
                sample_posterior=self.config.autokl.sample_posterior_eval,
            )
            batch_size = images.shape[0]
            sample_count += batch_size
            sums["total"] += float(losses.total.cpu()) * batch_size
            sums["reconstruction"] += (
                float(losses.reconstruction.cpu()) * batch_size
            )
            sums["edge"] += float(losses.edge.cpu()) * batch_size
            sums["kl"] += float(losses.kl.cpu()) * batch_size
            sums["checkpoint"] += (
                float(losses.checkpoint_score.cpu()) * batch_size
            )
            if render_dir is not None and bucket not in rendered:
                labels = (
                    [str(item) for item in sample_ids]
                    if isinstance(sample_ids, Sequence)
                    else None
                )
                save_reconstruction_grid(
                    images.detach().float().cpu(),
                    output.reconstruction.detach().float().cpu(),
                    render_dir / f"bucket_{bucket}.png",
                    labels=labels,
                )
                rendered.add(bucket)
            del output, losses, images
            if self.runtime.device.type == "mps":
                torch.mps.empty_cache()
        if sample_count <= 0:
            raise ValueError("Evaluation loader không được rỗng.")
        metrics = EpochMetrics(
            total_loss=sums["total"] / sample_count,
            reconstruction_loss=sums["reconstruction"] / sample_count,
            edge_loss=sums["edge"] / sample_count,
            kl_loss=sums["kl"] / sample_count,
            checkpoint_score=sums["checkpoint"] / sample_count,
            sample_count=sample_count,
        )
        if self.logger is not None:
            self.logger.log_scalars(
                {
                    "test/total_loss": metrics.total_loss,
                    "test/reconstruction_loss": metrics.reconstruction_loss,
                    "test/edge_loss": metrics.edge_loss,
                    "test/kl_loss": metrics.kl_loss,
                },
                step=self.global_step,
            )
        return metrics

    def save_epoch_checkpoints(
        self,
        *,
        next_epoch: int,
        train_checkpoint_score: float,
    ) -> bool:
        output_dir = self.config.checkpoint.output_dir
        improved = train_checkpoint_score < self.best_score
        if improved:
            self.best_score = train_checkpoint_score
        resolved = self.config.resolved_dict()
        save_checkpoint(
            output_dir / "last.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch=next_epoch,
            global_step=self.global_step,
            best_score=self.best_score,
            resolved_config=resolved,
        )
        if improved:
            save_model_checkpoint(
                output_dir / "best.pt",
                self.model,
            )
        return improved

    def resume(self, path: Path) -> ResumeState:
        state = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            current_config=self.config.resolved_dict(),
        )
        self.global_step = state.global_step
        self.best_score = state.best_score
        return state


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed không được âm.")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
