"""Differentiable frozen-HTR guidance for generated paragraph lines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vietparadiff.artifacts import sha256_file
from vietparadiff.data.pipeline import HTRVocabulary
from vietparadiff.models.config import HTRConfig
from vietparadiff.models.htr import VietnameseHTR
from vietparadiff.training.htr import (
    HTRLossConfig,
    HTRLosses,
    compute_htr_losses,
)


@dataclass(frozen=True, slots=True)
class HTRGuidanceConfig:
    checkpoint: Path
    model_config: Path
    vocabulary: Path
    maximum_weight: float
    warmup_steps: int
    maximum_timestep: int
    every_n_optimizer_steps: int
    raw_weight: float
    base_weight: float
    shape_weight: float
    tone_weight: float

    def __post_init__(self) -> None:
        if self.maximum_weight <= 0.0:
            raise ValueError("guidance.maximum_weight phải dương.")
        if self.warmup_steps <= 0:
            raise ValueError("guidance.warmup_steps phải dương.")
        if self.maximum_timestep < 0:
            raise ValueError(
                "guidance.maximum_timestep không được âm."
            )
        if self.every_n_optimizer_steps <= 0:
            raise ValueError(
                "guidance.every_n_optimizer_steps phải dương."
            )
        weights = (
            self.raw_weight,
            self.base_weight,
            self.shape_weight,
            self.tone_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError(
                "HTR head weights phải không âm và có tổng dương."
            )
        if weights != (1.0, 0.5, 0.25, 0.25):
            raise ValueError(
                "HTR guidance head weights phải là "
                "1.0/0.5/0.25/0.25."
            )

    @property
    def loss_config(self) -> HTRLossConfig:
        return HTRLossConfig(
            raw_weight=self.raw_weight,
            base_weight=self.base_weight,
            shape_weight=self.shape_weight,
            tone_weight=self.tone_weight,
        )


@dataclass(frozen=True, slots=True)
class HTRGuidanceResult:
    losses: HTRLosses
    line_count: int


def guidance_weight(
    config: HTRGuidanceConfig,
    global_step: int,
) -> float:
    if global_step < 0:
        raise ValueError("global_step không được âm.")
    progress = min(global_step / config.warmup_steps, 1.0)
    return config.maximum_weight * progress


def guidance_step_enabled(
    config: HTRGuidanceConfig,
    global_step: int,
) -> bool:
    if global_step < 0:
        raise ValueError("global_step không được âm.")
    return (
        global_step % config.every_n_optimizer_steps == 0
        and guidance_weight(config, global_step) > 0.0
    )


def predicted_clean_from_velocity(
    noisy_latents: Tensor,
    predicted_velocity: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> Tensor:
    if (
        noisy_latents.shape != predicted_velocity.shape
        or noisy_latents.ndim != 4
    ):
        raise ValueError(
            "noisy_latents/predicted_velocity phải cùng shape [B,C,H,W]."
        )
    expected = (noisy_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(f"alpha/sigma phải có shape {expected}.")
    return (
        alpha[:, None, None, None].to(noisy_latents.dtype)
        * noisy_latents
        - sigma[:, None, None, None].to(noisy_latents.dtype)
        * predicted_velocity
    )


def load_htr_model_config(path: Path) -> HTRConfig:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy HTR model config: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {field.name for field in fields(HTRConfig)}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        actual = (
            sorted(payload)
            if isinstance(payload, Mapping)
            else type(payload).__name__
        )
        raise ValueError(
            f"HTR model config keys phải bằng {sorted(expected)}, "
            f"nhận {actual}."
        )
    return HTRConfig(**dict(payload))  # type: ignore[arg-type]


def validate_htr_inference_contract(
    config: HTRGuidanceConfig,
) -> Path:
    path = config.checkpoint.parent / "inference_contract.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy HTR inference contract: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "htr_checkpoint_sha256",
        "model_config_sha256",
        "vocabulary_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("HTR inference contract sai schema.")
    if payload["schema_version"] != 1:
        raise ValueError(
            "HTR inference contract schema_version phải bằng 1."
        )
    artifacts = {
        "HTR checkpoint": (
            config.checkpoint,
            payload["htr_checkpoint_sha256"],
        ),
        "HTR model config": (
            config.model_config,
            payload["model_config_sha256"],
        ),
        "HTR vocabulary": (
            config.vocabulary,
            payload["vocabulary_sha256"],
        ),
    }
    for name, (artifact, expected_hash) in artifacts.items():
        if not isinstance(expected_hash, str):
            raise TypeError(f"{name} hash phải là string.")
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                f"HTR inference contract từ chối {name}: "
                f"expected={expected_hash}, actual={actual_hash}."
            )
    return path


class GeneratedLineRouter(nn.Module):
    """Crop canonical generated line bands with differentiable grid sampling."""

    output_height = 64
    output_width = 1024
    maximum_lines = 8

    def forward(
        self,
        images: Tensor,
        canonical_line_slots: Tensor,
        target_texts: Sequence[str],
        vocabulary: HTRVocabulary,
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> dict[str, object]:
        if (
            images.ndim != 4
            or images.shape[1] != 1
            or images.shape[-1] != self.output_width
            or not images.is_floating_point()
            or not torch.isfinite(images).all()
        ):
            raise ValueError(
                "Generated images phải là float hữu hạn [B,1,H,1024]."
            )
        expected_slots = (
            images.shape[0],
            self.maximum_lines,
            images.shape[-2] // 8,
            self.output_width // 8,
        )
        if (
            canonical_line_slots.shape != expected_slots
            or not canonical_line_slots.is_floating_point()
            or not torch.isfinite(canonical_line_slots).all()
            or canonical_line_slots.min() < 0.0
            or canonical_line_slots.max() > 1.0
        ):
            raise ValueError(
                "canonical_line_slots phải có shape "
                f"{expected_slots} và giá trị trong [0,1], nhận "
                f"{tuple(canonical_line_slots.shape)}."
            )
        if (canonical_line_slots.sum(dim=1) > 1.0).any():
            raise ValueError(
                "canonical_line_slots không được overlap giữa các dòng."
            )
        batch_size = images.shape[0]
        if (
            isinstance(target_texts, (str, bytes))
            or len(target_texts) != batch_size
        ):
            raise ValueError(
                f"target_texts phải có {batch_size} phần tử."
            )
        if sample_ids is None:
            sample_ids = tuple(
                f"generated_{index}" for index in range(batch_size)
            )
        if (
            isinstance(sample_ids, (str, bytes))
            or len(sample_ids) != batch_size
            or not all(
                isinstance(sample_id, str) and sample_id
                for sample_id in sample_ids
            )
        ):
            raise ValueError(
                f"sample_ids phải gồm {batch_size} string không rỗng."
            )

        line_images: list[Tensor] = []
        line_targets: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        line_sample_ids: list[str] = []
        image_height = images.shape[-2]
        x_pixels = torch.arange(
            self.output_width,
            device=images.device,
            dtype=images.dtype,
        )
        x_grid = (
            2.0 * (x_pixels + 0.5) / self.output_width - 1.0
        )

        for batch_index, text in enumerate(target_texts):
            if not isinstance(text, str):
                raise TypeError("Mỗi target_text phải là string.")
            lines = text.split("\n")
            if len(lines) > self.maximum_lines:
                raise ValueError(
                    f"Target có {len(lines)} dòng, vượt "
                    f"{self.maximum_lines}."
                )
            active = canonical_line_slots[batch_index].flatten(1).sum(1) > 0
            expected_active = torch.zeros(
                self.maximum_lines,
                dtype=torch.bool,
                device=active.device,
            )
            for line_index, line in enumerate(lines):
                expected_active[line_index] = bool(line)
            if not torch.equal(active, expected_active):
                raise ValueError(
                    "Active canonical line slots không khớp các dòng "
                    f"transcript không rỗng tại sample {sample_ids[batch_index]}."
                )
            for line_index, line in enumerate(lines):
                if not line:
                    continue
                slot = canonical_line_slots[batch_index, line_index]
                active_rows = torch.nonzero(
                    slot.sum(dim=1) > 0,
                    as_tuple=False,
                ).flatten()
                if active_rows.numel() <= 0:
                    raise RuntimeError("Active line slot không có row.")
                y0 = int(active_rows[0].item()) * 8
                y1 = min(
                    (int(active_rows[-1].item()) + 1) * 8,
                    image_height,
                )
                if y0 < 0 or y1 <= y0 or y1 > image_height:
                    raise ValueError(
                        f"Canonical y-range không hợp lệ: [{y0},{y1})."
                    )
                y_pixels = torch.linspace(
                    y0 + 0.5,
                    y1 - 0.5,
                    self.output_height,
                    device=images.device,
                    dtype=images.dtype,
                )
                y_grid = 2.0 * y_pixels / image_height - 1.0
                grid_y, grid_x = torch.meshgrid(
                    y_grid,
                    x_grid,
                    indexing="ij",
                )
                grid = torch.stack((grid_x, grid_y), dim=-1)[None]
                cropped = F.grid_sample(
                    images[batch_index : batch_index + 1],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                if cropped.shape != (
                    1,
                    1,
                    self.output_height,
                    self.output_width,
                ):
                    raise RuntimeError(
                        "Generated line router trả shape sai."
                    )
                line_images.append(cropped[0])
                line_targets.append(vocabulary.encode(line))
                line_sample_ids.append(
                    f"{sample_ids[batch_index]}:line_{line_index}"
                )

        if not line_images:
            raise ValueError(
                "HTR guidance batch không có dòng transcript hoạt động."
            )
        lengths = torch.tensor(
            [targets[0].numel() for targets in line_targets],
            dtype=torch.long,
            device=images.device,
        )
        if (lengths <= 0).any():
            raise ValueError("HTR guidance không nhận transcript rỗng.")
        maximum_length = int(lengths.max().item())

        def padded_target(head_index: int) -> Tensor:
            output = torch.zeros(
                len(line_targets),
                maximum_length,
                dtype=torch.long,
                device=images.device,
            )
            for index, targets in enumerate(line_targets):
                target = targets[head_index].to(images.device)
                if target.numel() != int(lengths[index].item()):
                    raise ValueError(
                        "Bốn HTR target heads phải cùng chiều dài."
                    )
                output[index, : target.numel()] = target
            return output

        return {
            "images": torch.stack(line_images),
            "valid_widths": torch.full(
                (len(line_images),),
                self.output_width,
                dtype=torch.long,
                device=images.device,
            ),
            "raw_targets": padded_target(0),
            "base_targets": padded_target(1),
            "shape_targets": padded_target(2),
            "tone_targets": padded_target(3),
            "target_lengths": lengths,
            "sample_ids": line_sample_ids,
        }


class FrozenHTRTeacher(nn.Module):
    """Strictly loaded line-level HTR with gradients only to its images."""

    def __init__(
        self,
        model: VietnameseHTR,
        vocabulary: HTRVocabulary,
        config: HTRGuidanceConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.vocabulary = vocabulary
        self.config = config
        self.router = GeneratedLineRouter()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @classmethod
    def load(
        cls,
        config: HTRGuidanceConfig,
        *,
        device: torch.device,
    ) -> FrozenHTRTeacher:
        validate_htr_inference_contract(config)
        model_config = load_htr_model_config(config.model_config)
        vocabulary = HTRVocabulary.load(config.vocabulary)
        expected_sizes = (
            len(vocabulary.raw_to_id),
            len(vocabulary.base_to_id),
            len(vocabulary.shape_to_id),
            len(vocabulary.tone_to_id),
        )
        actual_sizes = (
            model_config.raw_vocab_size,
            model_config.base_vocab_size,
            model_config.shape_vocab_size,
            model_config.tone_vocab_size,
        )
        if actual_sizes != expected_sizes:
            raise ValueError(
                "HTR vocabulary sizes không khớp model config: "
                f"expected={expected_sizes}, actual={actual_sizes}."
            )
        model = VietnameseHTR(model_config)
        model.load_checkpoint(config.checkpoint)
        model.to(device).eval()
        return cls(model, vocabulary, config)

    def train(self, mode: bool = True) -> FrozenHTRTeacher:
        del mode
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        generated_images: Tensor,
        canonical_line_slots: Tensor,
        target_texts: Sequence[str],
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> HTRGuidanceResult:
        self.model.eval()
        batch = self.router(
            generated_images,
            canonical_line_slots,
            target_texts,
            self.vocabulary,
            sample_ids=sample_ids,
        )
        images = batch["images"]
        valid_widths = batch["valid_widths"]
        if not isinstance(images, Tensor) or not isinstance(
            valid_widths, Tensor
        ):
            raise RuntimeError("Generated HTR batch sai tensor contract.")
        output = self.model(images, valid_widths)
        losses = compute_htr_losses(
            output,
            batch,
            self.config.loss_config,
        )
        return HTRGuidanceResult(
            losses=losses,
            line_count=images.shape[0],
        )


__all__ = [
    "FrozenHTRTeacher",
    "GeneratedLineRouter",
    "HTRGuidanceConfig",
    "HTRGuidanceResult",
    "guidance_step_enabled",
    "guidance_weight",
    "load_htr_model_config",
    "predicted_clean_from_velocity",
    "validate_htr_inference_contract",
]
