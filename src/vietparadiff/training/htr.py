"""Training mechanics for the four-head Vietnamese HTR teacher."""

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

from vietparadiff.artifacts import sha256_file
from vietparadiff.data.pipeline import HTRDataset, HTRVocabulary
from vietparadiff.models.htr import HTROutput, VietnameseHTR
from vietparadiff.runtime import (
    RuntimePrecision,
    autocast_context,
    create_grad_scaler,
    resolve_runtime,
    seed_everything,
)


@dataclass(frozen=True, slots=True)
class HTRDataConfig:
    train_lines: Path
    train_words: Path
    test_lines: Path
    test_words: Path
    vocabulary: Path
    image_root: Path
    num_workers: int
    line_batch_size: int
    word_batch_size: int
    width_bucket_size: int
    line_batches_per_step: int
    word_batches_per_step: int

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("data.num_workers không được âm.")
        positive = {
            "line_batch_size": self.line_batch_size,
            "word_batch_size": self.word_batch_size,
            "width_bucket_size": self.width_bucket_size,
            "line_batches_per_step": self.line_batches_per_step,
            "word_batches_per_step": self.word_batches_per_step,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"data.{name} phải dương.")
        if (self.line_batches_per_step, self.word_batches_per_step) != (3, 1):
            raise ValueError("HTR training schedule phải là 3 line : 1 word.")


@dataclass(frozen=True, slots=True)
class HTRStageConfig:
    epochs: int

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("htr.epochs phải dương.")


@dataclass(frozen=True, slots=True)
class HTROptimizerConfig:
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
        if len(self.betas) != 2 or not all(0.0 <= value < 1.0 for value in self.betas):
            raise ValueError("optimizer.betas phải gồm hai giá trị trong [0,1).")
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay không được âm.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer.gradient_clip_norm phải dương.")


@dataclass(frozen=True, slots=True)
class HTRSchedulerConfig:
    warmup_steps: int
    minimum_learning_rate_ratio: float

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError("scheduler.warmup_steps phải dương.")
        if not 0.0 < self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError(
                "scheduler.minimum_learning_rate_ratio phải nằm trong (0,1]."
            )


@dataclass(frozen=True, slots=True)
class HTRLossConfig:
    raw_weight: float = 1.0
    base_weight: float = 0.5
    shape_weight: float = 0.25
    tone_weight: float = 0.25

    def __post_init__(self) -> None:
        if (self.raw_weight, self.base_weight, self.shape_weight, self.tone_weight) != (
            1.0,
            0.5,
            0.25,
            0.25,
        ):
            raise ValueError("HTR loss weights phải là 1.0/0.5/0.25/0.25.")


@dataclass(frozen=True, slots=True)
class HTRAugmentationConfig:
    ink_intensity_jitter: float
    gaussian_noise_std: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.ink_intensity_jitter <= 0.25:
            raise ValueError(
                "augmentation.ink_intensity_jitter phải trong [0,0.25]."
            )
        if not 0.0 <= self.gaussian_noise_std <= 0.1:
            raise ValueError(
                "augmentation.gaussian_noise_std phải trong [0,0.1]."
            )


@dataclass(frozen=True, slots=True)
class HTRLoggingConfig:
    log_every_steps: int
    decode_every_steps: int
    tensorboard: bool
    wandb: bool
    wandb_mode: Literal["online", "offline", "disabled"]
    wandb_project: str

    def __post_init__(self) -> None:
        if self.log_every_steps <= 0 or self.decode_every_steps <= 0:
            raise ValueError("Logging intervals phải dương.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode phải là online, offline hoặc disabled.")
        if not self.wandb_project:
            raise ValueError("wandb_project không được rỗng.")


@dataclass(frozen=True, slots=True)
class HTRCheckpointConfig:
    output_dir: Path
    save_last: bool
    save_best: bool

    def __post_init__(self) -> None:
        if not self.save_last or not self.save_best:
            raise ValueError("HTR phải lưu cả last.pt và best.pt.")


@dataclass(frozen=True, slots=True)
class HTRTrainingConfig:
    seed: int
    device: str
    precision: str
    data: HTRDataConfig
    htr: HTRStageConfig
    optimizer: HTROptimizerConfig
    scheduler: HTRSchedulerConfig
    loss: HTRLossConfig
    augmentation: HTRAugmentationConfig
    logging: HTRLoggingConfig
    checkpoint: HTRCheckpointConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed không được âm.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device phải là auto, cuda, mps hoặc cpu.")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("precision không hợp lệ.")

    def resolved_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        payload = convert(asdict(self))
        if not isinstance(payload, dict):
            raise RuntimeError("Resolved HTR config phải là mapping.")
        return payload


def _resume_config_signature(
    config: Mapping[str, object],
) -> dict[str, object]:
    data = config.get("data")
    htr = config.get("htr")
    if not isinstance(data, Mapping):
        raise ValueError("Resume config.data phải là mapping.")
    if not isinstance(htr, Mapping):
        raise ValueError("Resume config.htr phải là mapping.")
    required_data = {
        "image_root",
        "line_batch_size",
        "word_batch_size",
        "width_bucket_size",
        "line_batches_per_step",
        "word_batches_per_step",
    }
    missing_data = required_data - set(data)
    if missing_data:
        raise ValueError(
            "Resume config.data thiếu keys "
            f"{sorted(missing_data)}."
        )
    if "epochs" not in htr:
        raise ValueError("Resume config.htr thiếu key 'epochs'.")
    for name in ("seed", "precision"):
        if name not in config:
            raise ValueError(f"Resume config thiếu key '{name}'.")
    for name in ("optimizer", "scheduler", "loss", "augmentation"):
        if not isinstance(config.get(name), Mapping):
            raise ValueError(
                f"Resume config.{name} phải là mapping."
            )

    return {
        "seed": config["seed"],
        "precision": config["precision"],
        "image_root": data["image_root"],
        "line_batch_size": data["line_batch_size"],
        "word_batch_size": data["word_batch_size"],
        "width_bucket_size": data["width_bucket_size"],
        "line_batches_per_step": data["line_batches_per_step"],
        "word_batches_per_step": data["word_batches_per_step"],
        "epochs": htr["epochs"],
        "optimizer": dict(config["optimizer"]),  # type: ignore[arg-type]
        "scheduler": dict(config["scheduler"]),  # type: ignore[arg-type]
        "loss": dict(config["loss"]),  # type: ignore[arg-type]
        "augmentation": dict(config["augmentation"]),  # type: ignore[arg-type]
    }


def validate_resume_config(
    checkpoint_config: object,
    current_config: Mapping[str, object],
) -> None:
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint config phải là mapping.")
    checkpoint_signature = _resume_config_signature(checkpoint_config)
    current_signature = _resume_config_signature(current_config)
    if checkpoint_signature != current_signature:
        mismatches = [
            name
            for name in checkpoint_signature
            if checkpoint_signature[name] != current_signature[name]
        ]
        details = "; ".join(
            f"{name}: checkpoint={checkpoint_signature[name]!r}, "
            f"current={current_signature[name]!r}"
            for name in mismatches
        )
        raise ValueError(
            f"Resume config không tương thích: {details}"
        )


def _section(
    raw: Mapping[str, object], name: str, keys: set[str]
) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(keys)}, nhận {actual}."
        )
    return dict(value)


def load_htr_training_config(path: Path) -> HTRTrainingConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HTR config root phải là mapping.")
    root_keys = {
        "seed", "device", "precision", "data", "htr", "optimizer",
        "scheduler", "loss", "logging", "checkpoint",
        "augmentation",
    }
    if set(raw) != root_keys:
        raise ValueError(f"HTR config root keys phải bằng {sorted(root_keys)}.")
    data = _section(raw, "data", {
        "train_lines", "train_words", "test_lines", "test_words",
        "vocabulary", "image_root", "num_workers", "line_batch_size",
        "word_batch_size", "width_bucket_size", "line_batches_per_step",
        "word_batches_per_step",
    })
    htr = _section(raw, "htr", {"epochs"})
    optimizer = _section(raw, "optimizer", {
        "name", "learning_rate", "betas", "weight_decay", "gradient_clip_norm",
    })
    scheduler = _section(
        raw, "scheduler", {"warmup_steps", "minimum_learning_rate_ratio"}
    )
    loss = _section(
        raw, "loss", {"raw_weight", "base_weight", "shape_weight", "tone_weight"}
    )
    augmentation = _section(
        raw,
        "augmentation",
        {"ink_intensity_jitter", "gaussian_noise_std"},
    )
    logging = _section(raw, "logging", {
        "log_every_steps", "decode_every_steps", "tensorboard", "wandb",
        "wandb_mode", "wandb_project",
    })
    checkpoint = _section(raw, "checkpoint", {
        "output_dir", "save_last", "save_best",
    })
    betas = optimizer["betas"]
    if (
        not isinstance(betas, Sequence)
        or isinstance(betas, (str, bytes))
        or len(betas) != 2
    ):
        raise ValueError("optimizer.betas phải là sequence dài 2.")
    return HTRTrainingConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        data=HTRDataConfig(
            train_lines=Path(str(data["train_lines"])),
            train_words=Path(str(data["train_words"])),
            test_lines=Path(str(data["test_lines"])),
            test_words=Path(str(data["test_words"])),
            vocabulary=Path(str(data["vocabulary"])),
            image_root=Path(str(data["image_root"])),
            num_workers=int(data["num_workers"]),
            line_batch_size=int(data["line_batch_size"]),
            word_batch_size=int(data["word_batch_size"]),
            width_bucket_size=int(data["width_bucket_size"]),
            line_batches_per_step=int(data["line_batches_per_step"]),
            word_batches_per_step=int(data["word_batches_per_step"]),
        ),
        htr=HTRStageConfig(epochs=int(htr["epochs"])),
        optimizer=HTROptimizerConfig(
            name=str(optimizer["name"]),
            learning_rate=float(optimizer["learning_rate"]),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        ),
        scheduler=HTRSchedulerConfig(
            warmup_steps=int(scheduler["warmup_steps"]),
            minimum_learning_rate_ratio=float(
                scheduler["minimum_learning_rate_ratio"]
            ),
        ),
        loss=HTRLossConfig(
            raw_weight=float(loss["raw_weight"]),
            base_weight=float(loss["base_weight"]),
            shape_weight=float(loss["shape_weight"]),
            tone_weight=float(loss["tone_weight"]),
        ),
        augmentation=HTRAugmentationConfig(
            ink_intensity_jitter=float(
                augmentation["ink_intensity_jitter"]
            ),
            gaussian_noise_std=float(
                augmentation["gaussian_noise_std"]
            ),
        ),
        logging=HTRLoggingConfig(
            log_every_steps=int(logging["log_every_steps"]),
            decode_every_steps=int(logging["decode_every_steps"]),
            tensorboard=bool(logging["tensorboard"]),
            wandb=bool(logging["wandb"]),
            wandb_mode=str(logging["wandb_mode"]),  # type: ignore[arg-type]
            wandb_project=str(logging["wandb_project"]),
        ),
        checkpoint=HTRCheckpointConfig(
            output_dir=Path(str(checkpoint["output_dir"])),
            save_last=bool(checkpoint["save_last"]),
            save_best=bool(checkpoint["save_best"]),
        ),
    )


def minimum_ctc_steps(target: Tensor) -> int:
    if target.ndim != 1:
        raise ValueError("target phải là [N].")
    if target.numel() == 0:
        raise ValueError("CTC target không được rỗng.")
    if target.dtype != torch.long:
        raise TypeError("CTC target phải có dtype torch.long.")
    if (target == 0).any():
        raise ValueError("Active CTC target không được chứa blank ID 0.")
    repeats = int((target[1:] == target[:-1]).sum().item())
    return target.numel() + repeats


def validate_htr_dataset(
    dataset: HTRDataset,
    dataset_name: str,
) -> None:
    """Preflight every sample against the fixed HTR/CTC width contract."""
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = sample.get("sample_id")
        valid_width = sample.get("valid_width")
        if not isinstance(sample_id, str) or not sample_id:
            raise TypeError(
                f"HTR sample ID không hợp lệ: dataset={dataset_name}, "
                f"index={index}."
            )
        if not isinstance(valid_width, int) or valid_width <= 0:
            raise TypeError(
                "HTR valid_width phải là số nguyên dương: "
                f"dataset={dataset_name}, sample={sample_id}, "
                f"actual={valid_width!r}."
            )
        input_length = (valid_width + 3) // 4
        if input_length > 2048:
            raise ValueError(
                "HTR width infeasible: "
                f"dataset={dataset_name}, sample={sample_id}, "
                f"valid_width={valid_width}, "
                f"input_length={input_length}, maximum=2048."
            )
        for head in (
            "raw_targets",
            "base_targets",
            "shape_targets",
            "tone_targets",
        ):
            target = sample.get(head)
            if not isinstance(target, Tensor):
                raise TypeError(
                    "HTR target phải là Tensor: "
                    f"dataset={dataset_name}, sample={sample_id}, "
                    f"head={head}, actual={type(target).__name__}."
                )
            required = minimum_ctc_steps(target)
            if required > input_length:
                raise ValueError(
                    "HTR CTC target infeasible: "
                    f"dataset={dataset_name}, sample={sample_id}, "
                    f"head={head}, required={required}, "
                    f"input_length={input_length}."
                )


def _long_tensor(batch: Mapping[str, object], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor) or value.dtype != torch.long:
        raise TypeError(f"{name} phải là torch.long Tensor.")
    return value


_TARGET_NAMES = ("raw_targets", "base_targets", "shape_targets", "tone_targets")


def validate_ctc_feasibility(
    output: HTROutput, batch: Mapping[str, object]
) -> None:
    target_lengths = _long_tensor(batch, "target_lengths")
    if target_lengths.ndim != 1:
        raise ValueError("target_lengths phải có shape [B].")
    batch_size = target_lengths.shape[0]
    if output.input_lengths.shape != (batch_size,):
        raise ValueError(f"input_lengths phải có shape [{batch_size}].")
    sample_ids = batch.get("sample_ids")
    if (
        not isinstance(sample_ids, Sequence)
        or isinstance(sample_ids, (str, bytes))
        or len(sample_ids) != batch_size
    ):
        raise ValueError(f"sample_ids phải là sequence dài {batch_size}.")
    for name in _TARGET_NAMES:
        targets = _long_tensor(batch, name)
        if targets.ndim != 2 or targets.shape[0] != batch_size:
            raise ValueError(f"{name} phải có shape [B,S].")
        if (target_lengths < 1).any() or (target_lengths > targets.shape[1]).any():
            raise ValueError(f"target_lengths không hợp lệ cho {name}.")
        for index, sample_id in enumerate(sample_ids):
            length = int(target_lengths[index].item())
            input_length = int(output.input_lengths[index].item())
            required = minimum_ctc_steps(targets[index, :length])
            if required > input_length:
                raise ValueError(
                    f"CTC infeasible: sample={sample_id}, head={name}, "
                    f"required={required}, input_length={input_length}."
                )


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    if logits.ndim != 3:
        raise ValueError("logits phải có shape [B,T,V].")
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    if logits.device.type == "mps":
        # PyTorch does not currently implement aten::_ctc_loss on MPS.
        # Tensor.to("cpu") keeps the autograd edge, so CTC gradients still
        # propagate back through the MPS logits and the HTR network.
        log_probs = log_probs.cpu()
        targets = targets.cpu()
        input_lengths = input_lengths.cpu()
        target_lengths = target_lengths.cpu()
    loss = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("CTC loss chứa NaN hoặc Inf.")
    return loss


@dataclass(frozen=True, slots=True)
class HTRLosses:
    total: Tensor
    raw: Tensor
    base: Tensor
    shape: Tensor
    tone: Tensor


def compute_htr_losses(
    output: HTROutput,
    batch: Mapping[str, object],
    config: HTRLossConfig,
) -> HTRLosses:
    validate_ctc_feasibility(output, batch)
    lengths = _long_tensor(batch, "target_lengths")
    raw = ctc_loss(
        output.raw_logits, _long_tensor(batch, "raw_targets"),
        output.input_lengths, lengths,
    )
    base = ctc_loss(
        output.base_logits, _long_tensor(batch, "base_targets"),
        output.input_lengths, lengths,
    )
    shape = ctc_loss(
        output.shape_logits, _long_tensor(batch, "shape_targets"),
        output.input_lengths, lengths,
    )
    tone = ctc_loss(
        output.tone_logits, _long_tensor(batch, "tone_targets"),
        output.input_lengths, lengths,
    )
    total = (
        config.raw_weight * raw
        + config.base_weight * base
        + config.shape_weight * shape
        + config.tone_weight * tone
    )
    if not torch.isfinite(total):
        raise FloatingPointError("HTR total loss chứa NaN hoặc Inf.")
    return HTRLosses(total, raw, base, shape, tone)


def micro_batch_loss_weight(
    level: Literal["line", "word"],
    *,
    line_batch_count: int,
    word_batch_count: int,
) -> float:
    if line_batch_count <= 0 or word_batch_count <= 0:
        raise ValueError("Mỗi optimizer step phải có line và word batch.")
    if level == "line":
        return 0.75 / line_batch_count
    if level == "word":
        return 0.25 / word_batch_count
    raise ValueError(f"HTR sample level không hợp lệ: {level!r}.")


def greedy_ctc_decode(
    logits: Tensor, input_lengths: Tensor, *, blank_id: int = 0
) -> list[list[int]]:
    if logits.ndim != 3:
        raise ValueError("logits phải có shape [B,T,V].")
    if input_lengths.shape != (logits.shape[0],):
        raise ValueError(f"input_lengths phải có shape [{logits.shape[0]}].")
    if (
        (input_lengths < 0).any()
        or (input_lengths > logits.shape[1]).any()
    ):
        raise ValueError("input_lengths nằm ngoài sequence logits.")
    predictions = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for row, length in zip(
        predictions, input_lengths.detach().cpu().tolist(), strict=True
    ):
        sequence: list[int] = []
        previous: int | None = None
        for token in row[:length].detach().cpu().tolist():
            if token != blank_id and token != previous:
                sequence.append(token)
            previous = token
        decoded.append(sequence)
    return decoded


def edit_distance(
    reference: Sequence[object], hypothesis: Sequence[object]
) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + int(ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def _inverse(vocabulary: Mapping[str, int]) -> dict[int, str]:
    return {value: key for key, value in vocabulary.items()}


def _decode_tokens(ids: Sequence[int], inverse: Mapping[int, str]) -> list[str]:
    return [inverse.get(token, "<unk>") for token in ids]


@dataclass(frozen=True, slots=True)
class HTRMetrics:
    total_loss: float
    raw_ctc: float
    base_ctc: float
    shape_ctc: float
    tone_ctc: float
    raw_cer: float
    base_error_rate: float
    shape_error_rate: float
    tone_error_rate: float
    raw_oov_rate: float
    base_oov_rate: float
    shape_oov_rate: float
    tone_oov_rate: float
    raw_wer: float | None = None
    exact_word_accuracy: float | None = None

    def as_dict(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, value in asdict(self).items():
            if value is not None:
                values[name] = float(value)
        return values


class _MetricAccumulator:
    def __init__(self, level: Literal["line", "word"], vocabulary: HTRVocabulary) -> None:
        self.level = level
        self.inverse = {
            "raw": _inverse(vocabulary.raw_to_id),
            "base": _inverse(vocabulary.base_to_id),
            "shape": _inverse(vocabulary.shape_to_id),
            "tone": _inverse(vocabulary.tone_to_id),
        }
        self.samples = 0
        self.loss_sums = {name: 0.0 for name in ("total", "raw", "base", "shape", "tone")}
        self.edits = {name: 0 for name in ("raw", "base", "shape", "tone")}
        self.tokens = {name: 0 for name in ("raw", "base", "shape", "tone")}
        self.oov = {name: 0 for name in ("raw", "base", "shape", "tone")}
        self.word_edits = 0
        self.word_count = 0
        self.exact = 0

    def update(
        self,
        losses: HTRLosses,
        output: HTROutput,
        batch: Mapping[str, object],
    ) -> list[dict[str, object]]:
        lengths = _long_tensor(batch, "target_lengths").detach().cpu()
        batch_size = lengths.numel()
        self.samples += batch_size
        for name in self.loss_sums:
            value = losses.total if name == "total" else getattr(losses, name)
            self.loss_sums[name] += float(value.detach().item()) * batch_size
        decoded = {
            "raw": greedy_ctc_decode(output.raw_logits, output.input_lengths),
            "base": greedy_ctc_decode(output.base_logits, output.input_lengths),
            "shape": greedy_ctc_decode(output.shape_logits, output.input_lengths),
            "tone": greedy_ctc_decode(output.tone_logits, output.input_lengths),
        }
        sample_ids = batch["sample_ids"]
        texts = batch["texts"]
        levels = batch["sample_levels"]
        if not all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in (sample_ids, texts, levels)
        ):
            raise TypeError("HTR metadata batch phải là sequences.")
        records: list[dict[str, object]] = []
        for index in range(batch_size):
            length = int(lengths[index].item())
            record: dict[str, object] = {
                "sample_id": str(sample_ids[index]),
                "sample_level": str(levels[index]),
                "ground_truth": str(texts[index]),
            }
            for head, target_name in zip(
                ("raw", "base", "shape", "tone"), _TARGET_NAMES, strict=True
            ):
                reference = (
                    _long_tensor(batch, target_name)[index, :length]
                    .detach().cpu().tolist()
                )
                hypothesis = decoded[head][index]
                self.edits[head] += edit_distance(reference, hypothesis)
                self.tokens[head] += len(reference)
                self.oov[head] += sum(token == 1 for token in reference)
                ref_tokens = _decode_tokens(reference, self.inverse[head])
                hyp_tokens = _decode_tokens(hypothesis, self.inverse[head])
                record[f"{head}_target"] = ref_tokens
                record[f"{head}_prediction"] = hyp_tokens
                if head == "raw":
                    reference_text = "".join(ref_tokens)
                    hypothesis_text = "".join(hyp_tokens)
                    record["raw_prediction_text"] = hypothesis_text
                    if self.level == "line":
                        ref_words = reference_text.split()
                        hyp_words = hypothesis_text.split()
                        self.word_edits += edit_distance(ref_words, hyp_words)
                        self.word_count += len(ref_words)
                    else:
                        self.exact += int(reference == hypothesis)
            records.append(record)
        return records

    def finish(self) -> HTRMetrics:
        if self.samples == 0:
            raise ValueError("Không có sample HTR để tính metrics.")
        rates = {
            name: self.edits[name] / max(self.tokens[name], 1)
            for name in self.edits
        }
        oov = {
            name: self.oov[name] / max(self.tokens[name], 1)
            for name in self.oov
        }
        return HTRMetrics(
            total_loss=self.loss_sums["total"] / self.samples,
            raw_ctc=self.loss_sums["raw"] / self.samples,
            base_ctc=self.loss_sums["base"] / self.samples,
            shape_ctc=self.loss_sums["shape"] / self.samples,
            tone_ctc=self.loss_sums["tone"] / self.samples,
            raw_cer=rates["raw"],
            base_error_rate=rates["base"],
            shape_error_rate=rates["shape"],
            tone_error_rate=rates["tone"],
            raw_oov_rate=oov["raw"],
            base_oov_rate=oov["base"],
            shape_oov_rate=oov["shape"],
            tone_oov_rate=oov["tone"],
            raw_wer=(
                self.word_edits / max(self.word_count, 1)
                if self.level == "line" else None
            ),
            exact_word_accuracy=(
                self.exact / self.samples if self.level == "word" else None
            ),
        )


def learning_rate_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if step < 0 or warmup_steps <= 0 or total_steps <= 0:
        raise ValueError("Scheduler steps không hợp lệ.")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio phải nằm trong (0,1].")
    if step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return minimum_ratio
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def create_optimizer_and_scheduler(
    model: nn.Module,
    optimizer_config: HTROptimizerConfig,
    scheduler_config: HTRSchedulerConfig,
    *,
    total_steps: int,
) -> tuple[Optimizer, LambdaLR]:
    optimizer = AdamW(
        model.parameters(),
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


def _move_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    moved: dict[str, object] = {}
    for name, value in batch.items():
        moved[name] = value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
    return moved


def artifact_hashes(config: HTRTrainingConfig) -> tuple[str, dict[str, str]]:
    vocabulary_hash = sha256_file(config.data.vocabulary)
    manifests = {
        "train_lines": sha256_file(config.data.train_lines),
        "train_words": sha256_file(config.data.train_words),
        "test_lines": sha256_file(config.data.test_lines),
        "test_words": sha256_file(config.data.test_words),
    }
    return vocabulary_hash, manifests


class HTRLogger:
    def __init__(
        self,
        config: HTRLoggingConfig,
        output_dir: Path,
        resolved_config: Mapping[str, object],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = (
            SummaryWriter(output_dir / "tensorboard") if config.tensorboard else None
        )
        self.run = (
            wandb.init(
                project=config.wandb_project,
                mode=config.wandb_mode,
                config=dict(resolved_config),
                dir=str(output_dir),
            )
            if config.wandb
            else None
        )

    def log_scalars(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def log_prediction(self, record: Mapping[str, object], *, step: int) -> None:
        text = json.dumps(dict(record), ensure_ascii=False)
        if self.writer is not None:
            self.writer.add_text("train/decoded_prediction", text, step)
        if self.run is not None:
            self.run.log({"train/decoded_prediction": text}, step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.run is not None:
            self.run.finish()


@dataclass(frozen=True, slots=True)
class HTREpochMetrics:
    line_total: float
    word_total: float

    @property
    def checkpoint_score(self) -> float:
        return self.line_total + 0.25 * self.word_total


@dataclass(frozen=True, slots=True)
class ResumeState:
    epoch: int
    global_step: int
    best_score: float


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None
        ),
    }


def _restore_rng(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {"python", "torch", "cuda", "mps"}:
        raise ValueError("Checkpoint RNG state sai schema.")
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise ValueError("Torch RNG state phải là Tensor.")
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(torch_state)
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint có CUDA RNG nhưng CUDA không khả dụng.")
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Checkpoint có MPS RNG nhưng MPS không khả dụng.")
        if not isinstance(state["mps"], Tensor):
            raise ValueError("MPS RNG state phải là Tensor.")
        torch.mps.set_rng_state(state["mps"])


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


def ensure_htr_static_artifacts(
    output_dir: Path,
    vocabulary: HTRVocabulary,
    model_config: Mapping[str, object],
) -> None:
    vocabulary_path = output_dir / "vocabulary.json"
    model_config_path = output_dir / "model_config.json"
    if vocabulary_path.exists():
        stored_vocabulary = HTRVocabulary.load(vocabulary_path)
        if stored_vocabulary != vocabulary:
            raise ValueError(
                "HTR output vocabulary.json không khớp run."
            )
    else:
        temporary = vocabulary_path.with_suffix(".json.tmp")
        vocabulary.save(temporary)
        temporary.replace(vocabulary_path)
    if model_config_path.exists():
        stored_config = json.loads(
            model_config_path.read_text(encoding="utf-8")
        )
        if stored_config != dict(model_config):
            raise ValueError(
                "HTR output model_config.json không khớp run."
            )
    else:
        _write_json_atomic(model_config_path, model_config)


def save_htr_inference_contract(output_dir: Path) -> None:
    checkpoint = output_dir / "best.pt"
    model_config = output_dir / "model_config.json"
    vocabulary = output_dir / "vocabulary.json"
    contract = {
        "schema_version": 1,
        "htr_checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": sha256_file(model_config),
        "vocabulary_sha256": sha256_file(vocabulary),
    }
    _write_json_atomic(
        output_dir / "inference_contract.json",
        contract,
    )


def htr_resolved_config_sha256(config: HTRTrainingConfig) -> str:
    payload = json.dumps(
        config.resolved_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_htr_training_contract(
    output_dir: Path,
    config: HTRTrainingConfig,
    manifest_sha256: Mapping[str, str],
) -> None:
    required = {"train_lines", "train_words"}
    if not required.issubset(manifest_sha256):
        raise ValueError(
            "HTR training contract thiếu train manifest hashes."
        )
    for name in sorted(required):
        digest = manifest_sha256[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                f"HTR training contract {name} SHA-256 không hợp lệ."
            )
    contract = {
        "schema_version": 1,
        "resolved_config_sha256": htr_resolved_config_sha256(
            config
        ),
        "seed": config.seed,
        "augmentation": asdict(config.augmentation),
        "train_lines_sha256": manifest_sha256["train_lines"],
        "train_words_sha256": manifest_sha256["train_words"],
        "htr_checkpoint_sha256": sha256_file(
            output_dir / "best.pt"
        ),
        "checkpoint_selection": "minimum_train_loss",
    }
    _write_json_atomic(
        output_dir / "training_contract.json",
        contract,
    )


def validate_htr_training_contract(
    config: HTRTrainingConfig,
) -> dict[str, object]:
    output_dir = config.checkpoint.output_dir
    path = output_dir / "training_contract.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy HTR training contract: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "resolved_config_sha256",
        "seed",
        "augmentation",
        "train_lines_sha256",
        "train_words_sha256",
        "htr_checkpoint_sha256",
        "checkpoint_selection",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_keys
        or payload["schema_version"] != 1
    ):
        raise ValueError("HTR training contract sai schema.")
    expected_values: dict[str, object] = {
        "resolved_config_sha256": htr_resolved_config_sha256(
            config
        ),
        "seed": config.seed,
        "augmentation": asdict(config.augmentation),
        "train_lines_sha256": sha256_file(
            config.data.train_lines
        ),
        "train_words_sha256": sha256_file(
            config.data.train_words
        ),
        "htr_checkpoint_sha256": sha256_file(
            output_dir / "best.pt"
        ),
        "checkpoint_selection": "minimum_train_loss",
    }
    for name, expected in expected_values.items():
        if payload[name] != expected:
            raise ValueError(
                f"HTR training contract mismatch tại {name}."
            )
    return dict(payload)


class HTRTrainer:
    def __init__(
        self,
        model: VietnameseHTR,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        scaler: torch.amp.GradScaler,
        config: HTRTrainingConfig,
        runtime: RuntimePrecision,
        vocabulary: HTRVocabulary,
        vocabulary_sha256: str,
        manifest_sha256: Mapping[str, str],
        model_config: Mapping[str, object],
        logger: HTRLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.runtime = runtime
        self.vocabulary = vocabulary
        self.vocabulary_sha256 = vocabulary_sha256
        self.manifest_sha256 = dict(manifest_sha256)
        self.model_config = dict(model_config)
        self.logger = logger
        self.global_step = 0
        self.best_score = math.inf
        ensure_htr_static_artifacts(
            self.config.checkpoint.output_dir,
            self.vocabulary,
            self.model_config,
        )

    def _augment_images(self, images: Tensor) -> Tensor:
        config = self.config.augmentation
        if (
            config.ink_intensity_jitter == 0.0
            and config.gaussian_noise_std == 0.0
        ):
            return images
        ink = ((1.0 - images) / 2.0).clamp(0.0, 1.0)
        if config.ink_intensity_jitter > 0.0:
            lower = 1.0 - config.ink_intensity_jitter
            upper = 1.0 + config.ink_intensity_jitter
            factor = torch.empty(
                images.shape[0],
                1,
                1,
                1,
                device=images.device,
                dtype=images.dtype,
            ).uniform_(lower, upper)
            ink = ink * factor
        if config.gaussian_noise_std > 0.0:
            ink = ink + torch.randn_like(ink) * (
                config.gaussian_noise_std * ink.sqrt()
            )
        return (1.0 - 2.0 * ink.clamp(0.0, 1.0)).clamp(-1.0, 1.0)

    def _forward(
        self,
        batch: Mapping[str, object],
        *,
        augment: bool = False,
    ) -> tuple[HTROutput, HTRLosses]:
        images = batch.get("images")
        valid_widths = batch.get("valid_widths")
        if not isinstance(images, Tensor) or not isinstance(valid_widths, Tensor):
            raise TypeError("HTR batch thiếu images/valid_widths Tensor.")
        if augment:
            images = self._augment_images(images)
        with autocast_context(self.runtime):
            output = self.model(images, valid_widths)
            losses = compute_htr_losses(output, batch, self.config.loss)
        return output, losses

    def train_epoch(
        self,
        line_loader: Iterable[Mapping[str, object]],
        word_loader: Iterable[Mapping[str, object]],
        *,
        epoch: int,
    ) -> HTREpochMetrics:
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        for loader in (line_loader, word_loader):
            sampler = getattr(loader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        self.model.train()
        line_iterator = iter(line_loader)
        word_iterator = iter(word_loader)
        line_loss_sum = 0.0
        word_loss_sum = 0.0
        line_batches = 0
        word_batches = 0
        while True:
            lines: list[Mapping[str, object]] = []
            for _ in range(self.config.data.line_batches_per_step):
                try:
                    lines.append(next(line_iterator))
                except StopIteration:
                    break
            if not lines:
                break
            words: list[Mapping[str, object]] = []
            for _ in range(self.config.data.word_batches_per_step):
                try:
                    words.append(next(word_iterator))
                except StopIteration:
                    word_iterator = iter(word_loader)
                    try:
                        words.append(next(word_iterator))
                    except StopIteration as error:
                        raise ValueError("Word loader không được rỗng.") from error
            micro_batches = [
                *[("line", batch) for batch in lines],
                *[("word", batch) for batch in words],
            ]
            self.optimizer.zero_grad(set_to_none=True)
            aggregate = {name: 0.0 for name in ("total", "raw", "base", "shape", "tone")}
            decode_record: dict[str, object] | None = None
            should_decode = (
                self.logger is not None
                and (self.global_step + 1)
                % self.config.logging.decode_every_steps
                == 0
            )
            for level, host_batch in micro_batches:
                batch = _move_batch(host_batch, self.runtime.device)
                output, losses = self._forward(batch, augment=True)
                loss_weight = micro_batch_loss_weight(
                    level,
                    line_batch_count=len(lines),
                    word_batch_count=len(words),
                )
                self.scaler.scale(losses.total * loss_weight).backward()
                for name in aggregate:
                    value = losses.total if name == "total" else getattr(losses, name)
                    aggregate[name] += (
                        float(value.detach().item()) * loss_weight
                    )
                if level == "line":
                    line_loss_sum += float(losses.total.detach().item())
                    line_batches += 1
                else:
                    word_loss_sum += float(losses.total.detach().item())
                    word_batches += 1
                if should_decode and decode_record is None:
                    accumulator = _MetricAccumulator(
                        "line" if level == "line" else "word", self.vocabulary
                    )
                    decode_record = accumulator.update(losses, output, batch)[0]
                del output, losses, batch
                if self.runtime.device.type == "mps":
                    torch.mps.empty_cache()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("HTR gradient norm chứa NaN hoặc Inf.")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            if self.logger is not None and self.global_step % self.config.logging.log_every_steps == 0:
                self.logger.log_scalars({
                    "train/total_loss": aggregate["total"],
                    "train/raw_ctc": aggregate["raw"],
                    "train/base_ctc": aggregate["base"],
                    "train/shape_ctc": aggregate["shape"],
                    "train/tone_ctc": aggregate["tone"],
                    "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "train/gradient_norm": float(gradient_norm.detach().item()),
                    "train/line_batches": float(len(lines)),
                    "train/word_batches": float(len(words)),
                }, step=self.global_step)
            if (
                self.logger is not None
                and decode_record is not None
                and self.global_step % self.config.logging.decode_every_steps == 0
            ):
                self.logger.log_prediction(decode_record, step=self.global_step)
        if line_batches == 0 or word_batches == 0:
            raise ValueError("Line và word loaders đều phải sinh ít nhất một batch.")
        return HTREpochMetrics(
            line_total=line_loss_sum / line_batches,
            word_total=word_loss_sum / word_batches,
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Mapping[str, object]],
        *,
        level: Literal["line", "word"],
        prediction_path: Path,
    ) -> HTRMetrics:
        self.model.eval()
        accumulator = _MetricAccumulator(level, self.vocabulary)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for host_batch in loader:
                batch = _move_batch(host_batch, self.runtime.device)
                output, losses = self._forward(batch)
                for record in accumulator.update(losses, output, batch):
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                del output, losses, batch
                if self.runtime.device.type == "mps":
                    torch.mps.empty_cache()
        temporary.replace(prediction_path)
        return accumulator.finish()

    def save_epoch_checkpoints(
        self, *, next_epoch: int, train_checkpoint_score: float
    ) -> bool:
        if next_epoch < 0 or not math.isfinite(train_checkpoint_score):
            raise ValueError("Checkpoint epoch/score không hợp lệ.")
        improved = train_checkpoint_score < self.best_score
        if improved:
            self.best_score = train_checkpoint_score
        output_dir = self.config.checkpoint.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "epoch": next_epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "config": self.config.resolved_dict(),
            "vocabulary_sha256": self.vocabulary_sha256,
            "manifest_sha256": self.manifest_sha256,
            "model_config": self.model_config,
            "rng": _rng_state(),
        }
        temporary = output_dir / "last.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output_dir / "last.pt")
        if improved:
            save_model_checkpoint(output_dir / "best.pt", self.model)
            save_htr_training_contract(
                output_dir,
                self.config,
                self.manifest_sha256,
            )
            save_htr_inference_contract(output_dir)
        return improved

    def resume(self, path: Path) -> ResumeState:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model", "optimizer", "scheduler", "scaler", "epoch", "global_step",
            "best_score", "config", "vocabulary_sha256", "manifest_sha256",
            "model_config", "rng",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError(f"HTR checkpoint keys phải bằng {sorted(required)}.")
        validate_resume_config(
            payload["config"],
            self.config.resolved_dict(),
        )
        if payload["vocabulary_sha256"] != self.vocabulary_sha256:
            raise ValueError("Resume bị từ chối: vocabulary SHA-256 đã thay đổi.")
        if payload["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("Resume bị từ chối: manifest SHA-256 đã thay đổi.")
        if payload["model_config"] != self.model_config:
            raise ValueError("Resume bị từ chối: model config đã thay đổi.")
        model_state = payload["model"]
        if not isinstance(model_state, Mapping):
            raise ValueError("Checkpoint model state phải là mapping.")
        self.model.load_state_dict(dict(model_state), strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
        self.scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
        scaler_state = payload["scaler"]
        if self.scaler.is_enabled():
            if not isinstance(scaler_state, Mapping):
                raise ValueError("FP16 resume yêu cầu scaler state.")
            self.scaler.load_state_dict(dict(scaler_state))
        elif scaler_state is not None:
            raise ValueError("Checkpoint có scaler nhưng runtime không dùng scaler.")
        _restore_rng(payload["rng"])
        self.global_step = int(payload["global_step"])
        self.best_score = float(payload["best_score"])
        return ResumeState(int(payload["epoch"]), self.global_step, self.best_score)


def load_best_for_evaluation(
    model: VietnameseHTR, path: Path, device: torch.device
) -> None:
    model.load_checkpoint(path)
    model.to(device)
