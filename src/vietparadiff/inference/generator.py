"""Deterministic v-prediction sampling and paragraph generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml
from torch import Tensor

from vietparadiff.artifacts import (
    InferenceContract,
    LatentStatistics,
    sha256_file,
)
from vietparadiff.diffusion import cosine_alpha_sigma
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.models.config import (
    AutoKLConfig,
    ParagraphUNetConfig,
    StyleEncoderConfig,
    TextEncoderConfig,
    VietParaDiffConfig,
)
from vietparadiff.models.style import StyleCondition
from vietparadiff.models.grapheme import (
    FormattedParagraph,
    GraphemeBatch,
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.generator import VietParaDiff, VietParaDiffInput


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    num_inference_steps: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps phải >= 2.")
        if self.seed < 0:
            raise ValueError("seed không được âm.")


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    image: Tensor
    scaled_latent: Tensor
    latent: Tensor
    formatted_text: FormattedParagraph


@dataclass(frozen=True, slots=True)
class GenerationModelConfig:
    checkpoint: Path
    contract: Path
    model_config: Path
    vocabulary: Path


@dataclass(frozen=True, slots=True)
class GenerationAutoKLConfig:
    checkpoint: Path
    latent_statistics: Path


@dataclass(frozen=True, slots=True)
class GenerationDiffusionConfig:
    num_inference_steps: int

    def __post_init__(self) -> None:
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps phải >= 2.")


@dataclass(frozen=True, slots=True)
class GenerationInputConfig:
    reference_height: int
    maximum_reference_width: int

    def __post_init__(self) -> None:
        if (
            self.reference_height != 256
            or self.maximum_reference_width != 1536
        ):
            raise ValueError(
                "Inference reference phải cao 256 và rộng tối đa 1536."
            )


@dataclass(frozen=True, slots=True)
class GenerationOutputConfig:
    directory: Path


@dataclass(frozen=True, slots=True)
class VietParaDiffGenerationConfig:
    seed: int
    device: str
    precision: str
    model: GenerationModelConfig
    autokl: GenerationAutoKLConfig
    diffusion: GenerationDiffusionConfig
    input: GenerationInputConfig
    output: GenerationOutputConfig

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


def _section(
    raw: Mapping[str, object],
    name: str,
    expected_keys: set[str],
) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(expected_keys)}, "
            f"nhận {actual}."
        )
    return dict(value)


def load_generation_config(path: Path) -> VietParaDiffGenerationConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Generation config root phải là mapping.")
    root_keys = {
        "seed",
        "device",
        "precision",
        "model",
        "autokl",
        "diffusion",
        "input",
        "output",
    }
    if set(raw) != root_keys:
        raise ValueError(
            f"Generation config root keys phải bằng {sorted(root_keys)}."
        )
    model = _section(
        raw,
        "model",
        {"checkpoint", "contract", "model_config", "vocabulary"},
    )
    autokl = _section(
        raw, "autokl", {"checkpoint", "latent_statistics"}
    )
    diffusion = _section(
        raw,
        "diffusion",
        {"num_inference_steps"},
    )
    input_config = _section(
        raw,
        "input",
        {"reference_height", "maximum_reference_width"},
    )
    output = _section(raw, "output", {"directory"})
    return VietParaDiffGenerationConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        model=GenerationModelConfig(
            checkpoint=Path(str(model["checkpoint"])),
            contract=Path(str(model["contract"])),
            model_config=Path(str(model["model_config"])),
            vocabulary=Path(str(model["vocabulary"])),
        ),
        autokl=GenerationAutoKLConfig(
            checkpoint=Path(str(autokl["checkpoint"])),
            latent_statistics=Path(str(autokl["latent_statistics"])),
        ),
        diffusion=GenerationDiffusionConfig(
            num_inference_steps=int(
                diffusion["num_inference_steps"]
            ),
        ),
        input=GenerationInputConfig(
            reference_height=int(input_config["reference_height"]),
            maximum_reference_width=int(
                input_config["maximum_reference_width"]
            ),
        ),
        output=GenerationOutputConfig(
            directory=Path(str(output["directory"])),
        ),
    )


def load_inference_contract(
    path: Path,
    *,
    generator_checkpoint: Path,
    model_config: Path,
    vocabulary: Path,
    autokl_checkpoint: Path,
    latent_statistics: Path,
) -> InferenceContract:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy inference contract: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {field.name for field in fields(InferenceContract)}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        actual = (
            sorted(payload)
            if isinstance(payload, Mapping)
            else type(payload).__name__
        )
        raise ValueError(
            f"Inference contract keys phải bằng {sorted(expected)}, "
            f"nhận {actual}."
        )
    if (
        not isinstance(payload["schema_version"], int)
        or isinstance(payload["schema_version"], bool)
        or not isinstance(payload["num_train_timesteps"], int)
        or isinstance(payload["num_train_timesteps"], bool)
        or not isinstance(payload["neutral_layout"], bool)
        or not all(
            isinstance(payload[name], str)
            for name in (
                "prediction_type",
                "noise_schedule",
                "generator_checkpoint_sha256",
                "model_config_sha256",
                "grapheme_vocabulary_sha256",
                "autokl_checkpoint_sha256",
                "latent_statistics_sha256",
            )
        )
    ):
        raise TypeError("Inference contract field types không hợp lệ.")
    contract = InferenceContract(
        schema_version=payload["schema_version"],
        prediction_type=payload["prediction_type"],
        noise_schedule=payload["noise_schedule"],
        num_train_timesteps=payload["num_train_timesteps"],
        neutral_layout=payload["neutral_layout"],
        generator_checkpoint_sha256=payload[
            "generator_checkpoint_sha256"
        ],
        model_config_sha256=payload["model_config_sha256"],
        grapheme_vocabulary_sha256=payload[
            "grapheme_vocabulary_sha256"
        ],
        autokl_checkpoint_sha256=payload[
            "autokl_checkpoint_sha256"
        ],
        latent_statistics_sha256=payload[
            "latent_statistics_sha256"
        ],
    )
    artifacts = {
        "generator checkpoint": (
            generator_checkpoint,
            contract.generator_checkpoint_sha256,
        ),
        "model config": (
            model_config,
            contract.model_config_sha256,
        ),
        "grapheme vocabulary": (
            vocabulary,
            contract.grapheme_vocabulary_sha256,
        ),
        "AutoKL checkpoint": (
            autokl_checkpoint,
            contract.autokl_checkpoint_sha256,
        ),
        "latent statistics": (
            latent_statistics,
            contract.latent_statistics_sha256,
        ),
    }
    for name, (artifact_path, expected_hash) in artifacts.items():
        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Inference contract từ chối {name}: "
                f"expected SHA-256 {expected_hash}, actual {actual_hash}."
            )
    return contract


def _strict_dataclass_payload(
    payload: object,
    dataclass_type: type[object],
    name: str,
) -> dict[str, object]:
    expected = {field.name for field in fields(dataclass_type)}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        actual = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
        raise ValueError(
            f"model_config.{name} keys phải bằng {sorted(expected)}, "
            f"nhận {actual}."
        )
    return dict(payload)


def load_model_config(path: Path) -> VietParaDiffConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    root_keys = {"autokl", "text", "style", "unet"}
    if not isinstance(payload, Mapping) or set(payload) != root_keys:
        raise ValueError(
            f"Model config root keys phải bằng {sorted(root_keys)}."
        )
    autokl = _strict_dataclass_payload(
        payload["autokl"], AutoKLConfig, "autokl"
    )
    text = _strict_dataclass_payload(
        payload["text"], TextEncoderConfig, "text"
    )
    style = _strict_dataclass_payload(
        payload["style"], StyleEncoderConfig, "style"
    )
    unet = _strict_dataclass_payload(
        payload["unet"], ParagraphUNetConfig, "unet"
    )
    autokl["channel_multipliers"] = tuple(
        int(item) for item in autokl["channel_multipliers"]  # type: ignore[union-attr]
    )
    text["height_buckets"] = tuple(
        int(item) for item in text["height_buckets"]  # type: ignore[union-attr]
    )
    unet["channels"] = tuple(
        int(item) for item in unet["channels"]  # type: ignore[union-attr]
    )
    checkpoint = style["convnext_checkpoint"]
    style["convnext_checkpoint"] = (
        None if checkpoint is None else Path(str(checkpoint))
    )
    return VietParaDiffConfig(
        autokl=AutoKLConfig(**autokl),  # type: ignore[arg-type]
        text=TextEncoderConfig(**text),  # type: ignore[arg-type]
        style=StyleEncoderConfig(**style),  # type: ignore[arg-type]
        unet=ParagraphUNetConfig(**unet),  # type: ignore[arg-type]
    )


def checkpoint_loading_config(
    stored: VietParaDiffConfig,
) -> VietParaDiffConfig:
    """Preserve topology while avoiding a redundant ImageNet download."""

    return VietParaDiffConfig(
        autokl=stored.autokl,
        text=stored.text,
        style=StyleEncoderConfig(
            reference_height=stored.style.reference_height,
            max_reference_width=stored.style.max_reference_width,
            stem_channels=stored.style.stem_channels,
            feature_dim=stored.style.feature_dim,
            local_token_count=stored.style.local_token_count,
            local_attention_heads=stored.style.local_attention_heads,
            foreground_threshold=stored.style.foreground_threshold,
            use_pretrained_backbone=False,
            convnext_checkpoint=None,
        ),
        unet=stored.unet,
    )


def build_sampling_timesteps(
    num_train_timesteps: int,
    num_inference_steps: int,
    *,
    device: torch.device,
) -> Tensor:
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps phải >= 2.")
    if not 2 <= num_inference_steps <= num_train_timesteps:
        raise ValueError(
            "num_inference_steps phải nằm trong "
            "[2, num_train_timesteps]."
        )
    ascending = torch.linspace(
        0,
        num_train_timesteps - 1,
        steps=num_inference_steps,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.long)
    timesteps = ascending.flip(0)
    if not torch.all(timesteps[:-1] > timesteps[1:]):
        raise RuntimeError(
            "Sampling timetable phải giảm nghiêm ngặt."
        )
    return timesteps


def velocity_to_clean_and_noise(
    noisy_latents: Tensor,
    predicted_velocity: Tensor,
    alpha: Tensor,
    sigma: Tensor,
) -> tuple[Tensor, Tensor]:
    if noisy_latents.shape != predicted_velocity.shape:
        raise ValueError(
            "noisy_latents và predicted_velocity phải cùng shape."
        )
    if noisy_latents.ndim != 4 or not noisy_latents.is_floating_point():
        raise ValueError("noisy_latents phải là float [B,C,H,W].")
    expected = (noisy_latents.shape[0],)
    if alpha.shape != expected or sigma.shape != expected:
        raise ValueError(
            f"alpha và sigma phải có shape {expected}."
        )
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            noisy_latents,
            predicted_velocity,
            alpha,
            sigma,
        )
    ):
        raise ValueError("Velocity conversion input chứa NaN/Inf.")
    alpha_view = alpha[:, None, None, None].to(
        dtype=noisy_latents.dtype,
        device=noisy_latents.device,
    )
    sigma_view = sigma[:, None, None, None].to(
        dtype=noisy_latents.dtype,
        device=noisy_latents.device,
    )
    predicted_clean = (
        alpha_view * noisy_latents
        - sigma_view * predicted_velocity
    )
    predicted_noise = (
        sigma_view * noisy_latents
        + alpha_view * predicted_velocity
    )
    return predicted_clean, predicted_noise


def _validate_sampling_condition(
    graphemes: GraphemeBatch,
    style_condition: StyleCondition,
    *,
    device: torch.device,
) -> int:
    if not isinstance(graphemes, GraphemeBatch):
        raise TypeError("graphemes phải là GraphemeBatch.")
    if not isinstance(style_condition, StyleCondition):
        raise TypeError("style_condition phải là StyleCondition.")
    batch_size = graphemes.base_ids.shape[0]
    if batch_size <= 0 or graphemes.base_ids.device != device:
        raise ValueError(
            "graphemes phải có batch dương và nằm trên sampling device."
        )
    if (
        style_condition.local_tokens.shape[0] != batch_size
        or style_condition.global_style.shape[0] != batch_size
        or style_condition.local_tokens.device != device
        or style_condition.global_style.device != device
    ):
        raise ValueError(
            "style_condition phải cùng batch size và sampling device."
        )
    return batch_size


@torch.inference_mode()
def sample_scaled_latent(
    model: VietParaDiff,
    graphemes: GraphemeBatch,
    style_condition: StyleCondition,
    *,
    latent_height: int,
    latent_width: int,
    config: SamplingConfig,
    num_train_timesteps: int,
    device: torch.device,
) -> Tensor:
    if latent_height <= 0 or latent_height % 8:
        raise ValueError(
            "latent_height phải dương và chia hết cho 8."
        )
    if latent_width != 128:
        raise ValueError("latent_width phải bằng 128.")
    if not isinstance(config, SamplingConfig):
        raise TypeError("config phải là SamplingConfig.")
    batch_size = _validate_sampling_condition(
        graphemes,
        style_condition,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    current = torch.randn(
        batch_size,
        4,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
    )
    schedule = build_sampling_timesteps(
        num_train_timesteps,
        config.num_inference_steps,
        device=device,
    )
    model.eval()
    for index, timestep_value in enumerate(schedule):
        timestep_int = int(timestep_value.item())
        timesteps = torch.full(
            (batch_size,),
            timestep_int,
            dtype=torch.long,
            device=device,
        )
        alpha, sigma = cosine_alpha_sigma(
            timesteps,
            num_train_timesteps=num_train_timesteps,
        )
        output = model(
            VietParaDiffInput(
                noisy_latents=current,
                timesteps=timesteps,
                graphemes=graphemes,
                style_condition=style_condition,
            )
        )
        if output.predicted_velocity.shape != current.shape:
            raise ValueError(
                "predicted_velocity shape phải bằng latent shape."
            )
        if not torch.isfinite(output.predicted_velocity).all():
            raise FloatingPointError(
                f"Model sinh NaN/Inf velocity tại timestep {timestep_int}."
            )
        predicted_clean, predicted_noise = (
            velocity_to_clean_and_noise(
                current,
                output.predicted_velocity,
                alpha,
                sigma,
            )
        )
        if index + 1 < schedule.numel():
            next_timesteps = torch.full(
                (batch_size,),
                int(schedule[index + 1].item()),
                dtype=torch.long,
                device=device,
            )
            alpha_next, sigma_next = cosine_alpha_sigma(
                next_timesteps,
                num_train_timesteps=num_train_timesteps,
            )
        else:
            alpha_next = torch.ones(
                batch_size,
                device=device,
            )
            sigma_next = torch.zeros(
                batch_size,
                device=device,
            )
        current = (
            alpha_next[:, None, None, None].to(current.dtype)
            * predicted_clean
            + sigma_next[:, None, None, None].to(current.dtype)
            * predicted_noise
        )
        if not torch.isfinite(current).all():
            raise FloatingPointError(
                f"Sampling sinh NaN/Inf tại timestep {timestep_int}."
            )
    return current


@torch.inference_mode()
def decode_scaled_latent(
    autokl: HandwritingAutoKL,
    statistics: LatentStatistics,
    scaled_latent: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        scaled_latent.ndim != 4
        or scaled_latent.shape[1] != 4
        or not scaled_latent.is_floating_point()
        or not torch.isfinite(scaled_latent).all()
    ):
        raise ValueError(
            "scaled_latent phải là float hữu hạn [B,4,H,W]."
        )
    autokl.eval()
    latent = statistics.denormalize(scaled_latent)
    image = autokl.decode(latent)
    expected = (
        scaled_latent.shape[0],
        1,
        scaled_latent.shape[2] * 8,
        scaled_latent.shape[3] * 8,
    )
    if (
        image.shape != expected
        or not image.is_floating_point()
        or not torch.isfinite(image).all()
    ):
        raise ValueError(
            "AutoKL decoder output không hợp lệ: "
            f"expected {expected}, actual {tuple(image.shape)}."
        )
    return latent, image.clamp(-1.0, 1.0)


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


@torch.inference_mode()
def generate_paragraph(
    model: VietParaDiff,
    autokl: HandwritingAutoKL,
    statistics: LatentStatistics,
    formatter: ParagraphFormatter,
    vocabulary: GraphemeVocabulary,
    *,
    text: str,
    reference_image: Tensor,
    reference_valid_mask: Tensor,
    sampling_config: SamplingConfig,
    num_train_timesteps: int,
    device: torch.device,
) -> GenerationOutput:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Generation text UTF-8 không được rỗng.")
    if (
        reference_image.ndim != 4
        or reference_image.shape[0:3] != (1, 1, 256)
        or reference_image.shape[-1] > 1536
        or reference_image.shape[-1] % 32
        or not reference_image.is_floating_point()
        or not torch.isfinite(reference_image).all()
        or reference_image.min() < -1.0
        or reference_image.max() > 1.0
    ):
        raise ValueError(
            "reference_image phải là [1,1,256,W], W<=1536, "
            "chia hết 32 và nằm trong [-1,1]."
        )
    if (
        reference_valid_mask.dtype != torch.bool
        or reference_valid_mask.shape != reference_image.shape
        or not reference_valid_mask.any()
    ):
        raise ValueError(
            "reference_valid_mask phải là bool mask không rỗng cùng shape."
        )
    reference_image = reference_image.to(device)
    reference_valid_mask = reference_valid_mask.to(device)
    model.eval()
    autokl.eval()
    style = model.encode_reference(
        reference_image,
        reference_valid_mask,
    )
    if not torch.equal(
        style.layout_scales,
        torch.ones_like(style.layout_scales),
    ):
        raise ValueError(
            "Base inference chỉ hỗ trợ neutral layout scales."
        )
    formatted = formatter.format(text)
    if formatted.output_height not in formatter.config.height_buckets:
        raise RuntimeError("Formatter trả output height ngoài contract.")
    text_batch = vocabulary.encode_batch(
        [formatted],
        device=device,
    )
    scaled_latent = sample_scaled_latent(
        model,
        text_batch.graphemes,
        style,
        latent_height=formatted.output_height // 8,
        latent_width=formatter.config.canvas_width // 8,
        config=sampling_config,
        num_train_timesteps=num_train_timesteps,
        device=device,
    )
    latent, image = decode_scaled_latent(
        autokl,
        statistics,
        scaled_latent,
    )
    expected = (1, 1, formatted.output_height, 1024)
    if image.shape != expected:
        raise RuntimeError(
            f"Generated image phải có shape {expected}, "
            f"nhận {tuple(image.shape)}."
        )
    return GenerationOutput(
        image=image,
        scaled_latent=scaled_latent,
        latent=latent,
        formatted_text=formatted,
    )


__all__ = [
    "GenerationAutoKLConfig",
    "GenerationDiffusionConfig",
    "GenerationInputConfig",
    "InferenceContract",
    "GenerationModelConfig",
    "GenerationOutput",
    "GenerationOutputConfig",
    "SamplingConfig",
    "VietParaDiffGenerationConfig",
    "build_sampling_timesteps",
    "checkpoint_loading_config",
    "decode_scaled_latent",
    "generate_paragraph",
    "load_generation_config",
    "load_inference_contract",
    "load_model_config",
    "sample_scaled_latent",
    "velocity_to_clean_and_noise",
]
