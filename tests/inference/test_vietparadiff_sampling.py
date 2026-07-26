from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vietparadiff.artifacts import LatentStatistics, sha256_file
from vietparadiff.diffusion import (
    add_diffusion_noise,
    cosine_alpha_sigma,
    velocity_target,
)
from vietparadiff.models.config import (
    AutoKLConfig,
    ParagraphUNetConfig,
    StyleEncoderConfig,
    TextEncoderConfig,
    VietParaDiffConfig,
)
from vietparadiff.models.style import StyleCondition
from vietparadiff.models.grapheme import (
    GraphemeBatch,
    GraphemeCondition,
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.generator import (
    VietParaDiffInput,
    VietParaDiffOutput,
)
from vietparadiff.inference.generator import (
    SamplingConfig,
    build_sampling_timesteps,
    checkpoint_loading_config,
    decode_scaled_latent,
    generate_paragraph,
    load_generation_config,
    load_inference_contract,
    load_model_config,
    sample_scaled_latent,
    velocity_to_clean_and_noise,
)
class TinySamplingModel(nn.Module):
    def __init__(self, *, nan_velocity: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.1))
        self.nan_velocity = nan_velocity
        self.forward_calls = 0
        self.reference_calls = 0
        self.last_input: Tensor | None = None
        self.last_velocity: Tensor | None = None
        self.last_timesteps: Tensor | None = None

    def encode_reference(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> StyleCondition:
        del valid_mask
        self.reference_calls += 1
        batch = images.shape[0]
        global_style = images.mean((1, 2, 3))[:, None].expand(-1, 4)
        return StyleCondition(
            local_tokens=global_style[:, None],
            global_style=global_style,
            layout_scales=torch.ones(
                batch, 3, device=images.device
            ),
            valid_feature_mask=torch.ones(
                batch,
                1,
                1,
                1,
                dtype=torch.bool,
                device=images.device,
            ),
        )

    def forward(self, batch: VietParaDiffInput) -> VietParaDiffOutput:
        self.forward_calls += 1
        predicted = batch.noisy_latents * self.anchor
        if self.nan_velocity:
            predicted = torch.full_like(predicted, float("nan"))
        self.last_input = batch.noisy_latents.detach().clone()
        self.last_velocity = predicted.detach().clone()
        self.last_timesteps = batch.timesteps.detach().clone()
        batch_size, length = batch.graphemes.base_ids.shape
        context = torch.zeros(
            batch_size,
            length,
            4,
            device=batch.noisy_latents.device,
        )
        condition = GraphemeCondition(
            base_context=context,
            shape_context=context,
            tone_context=context,
            attention_mask=batch.graphemes.attention_mask,
            line_ids=batch.graphemes.line_ids,
        )
        return VietParaDiffOutput(
            predicted_velocity=predicted,
            grapheme_condition=condition,
            style_condition=batch.style_condition,
            line_tokens=torch.zeros(
                batch_size,
                8,
                4,
                device=batch.noisy_latents.device,
            ),
            diagnostics={"tiny": predicted.nan_to_num().square().mean()},
        )


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_latent: Tensor | None = None

    def decode(self, latent: Tensor) -> Tensor:
        self.last_latent = latent.detach().clone()
        return F.interpolate(
            latent[:, :1],
            scale_factor=8.0,
            mode="nearest",
        )


def _text_components() -> tuple[
    ParagraphFormatter,
    GraphemeVocabulary,
    GraphemeBatch,
]:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    config = TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
        dropout=0.0,
    )
    formatter = ParagraphFormatter(config)
    paragraph = formatter.format("a")
    graphemes = vocabulary.encode_batch([paragraph]).graphemes
    return formatter, vocabulary, graphemes


def _style() -> StyleCondition:
    return StyleCondition(
        local_tokens=torch.zeros(1, 1, 4),
        global_style=torch.zeros(1, 4),
        layout_scales=torch.ones(1, 3),
        valid_feature_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )


def _statistics() -> LatentStatistics:
    return LatentStatistics(
        latent_mean=0.25,
        latent_std=2.0,
        scaling_factor=0.5,
        num_samples=2,
        num_elements=16,
        autokl_checkpoint_sha256="a" * 64,
    )


def _contract_artifacts(
    tmp_path: Path,
    *,
    num_train_timesteps: int = 1000,
) -> dict[str, Path]:
    paths = {
        "checkpoint": tmp_path / "best.pt",
        "contract": tmp_path / "inference_contract.json",
        "model_config": tmp_path / "model_config.json",
        "vocabulary": tmp_path / "grapheme_vocabulary.json",
        "autokl": tmp_path / "autokl.pt",
        "statistics": tmp_path / "latent_statistics.json",
    }
    paths["checkpoint"].write_bytes(b"generator checkpoint")
    paths["model_config"].write_text(
        json.dumps({"model": "config"}),
        encoding="utf-8",
    )
    GraphemeVocabulary.default_vietnamese().save(
        paths["vocabulary"]
    )
    paths["autokl"].write_bytes(b"autokl checkpoint")
    paths["statistics"].write_text(
        json.dumps({"latent": "statistics"}),
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "prediction_type": "velocity",
        "noise_schedule": "cosine",
        "num_train_timesteps": num_train_timesteps,
        "neutral_layout": True,
        "generator_checkpoint_sha256": sha256_file(
            paths["checkpoint"]
        ),
        "model_config_sha256": sha256_file(paths["model_config"]),
        "grapheme_vocabulary_sha256": sha256_file(
            paths["vocabulary"]
        ),
        "autokl_checkpoint_sha256": sha256_file(paths["autokl"]),
        "latent_statistics_sha256": sha256_file(
            paths["statistics"]
        ),
    }
    paths["contract"].write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    return paths


def _load_contract(paths: dict[str, Path]):
    return load_inference_contract(
        paths["contract"],
        generator_checkpoint=paths["checkpoint"],
        model_config=paths["model_config"],
        vocabulary=paths["vocabulary"],
        autokl_checkpoint=paths["autokl"],
        latent_statistics=paths["statistics"],
    )


def test_sampling_timesteps_are_strictly_descending() -> None:
    timesteps = build_sampling_timesteps(
        1000, 50, device=torch.device("cpu")
    )
    assert torch.all(timesteps[:-1] > timesteps[1:])


def test_sampling_timesteps_include_train_boundaries() -> None:
    timesteps = build_sampling_timesteps(
        1000, 50, device=torch.device("cpu")
    )
    assert timesteps[0].item() == 999
    assert timesteps[-1].item() == 0


def test_velocity_inverse_recovers_clean_and_noise() -> None:
    clean = torch.randn(2, 4, 8, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([100, 800], dtype=torch.long)
    alpha, sigma = cosine_alpha_sigma(
        timesteps,
        num_train_timesteps=1000,
    )
    noisy = add_diffusion_noise(clean, noise, alpha, sigma)
    velocity = velocity_target(clean, noise, alpha, sigma)
    recovered_clean, recovered_noise = velocity_to_clean_and_noise(
        noisy,
        velocity,
        alpha,
        sigma,
    )
    assert torch.allclose(recovered_clean, clean, atol=1e-5)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_final_clean_endpoint_has_alpha_one_sigma_zero() -> None:
    _, _, graphemes = _text_components()
    model = TinySamplingModel()
    result = sample_scaled_latent(
        model,  # type: ignore[arg-type]
        graphemes,
        _style(),
        latent_height=48,
        latent_width=128,
        config=SamplingConfig(num_inference_steps=2, seed=3),
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert model.last_input is not None
    assert model.last_velocity is not None
    assert model.last_timesteps is not None
    alpha, sigma = cosine_alpha_sigma(
        model.last_timesteps,
        num_train_timesteps=1000,
    )
    predicted_clean, _ = velocity_to_clean_and_noise(
        model.last_input,
        model.last_velocity,
        alpha,
        sigma,
    )
    assert torch.equal(result, predicted_clean)


def test_sampler_is_deterministic_for_same_seed() -> None:
    _, _, graphemes = _text_components()
    kwargs = {
        "latent_height": 48,
        "latent_width": 128,
        "config": SamplingConfig(num_inference_steps=3, seed=11),
        "num_train_timesteps": 1000,
        "device": torch.device("cpu"),
    }
    first = sample_scaled_latent(
        TinySamplingModel(),  # type: ignore[arg-type]
        graphemes,
        _style(),
        **kwargs,
    )
    second = sample_scaled_latent(
        TinySamplingModel(),  # type: ignore[arg-type]
        graphemes,
        _style(),
        **kwargs,
    )
    assert torch.equal(first, second)


def test_different_seed_changes_initial_latent() -> None:
    _, _, graphemes = _text_components()
    model = TinySamplingModel()
    first = sample_scaled_latent(
        model,  # type: ignore[arg-type]
        graphemes,
        _style(),
        latent_height=48,
        latent_width=128,
        config=SamplingConfig(num_inference_steps=2, seed=1),
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    second = sample_scaled_latent(
        model,  # type: ignore[arg-type]
        graphemes,
        _style(),
        latent_height=48,
        latent_width=128,
        config=SamplingConfig(num_inference_steps=2, seed=2),
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert not torch.equal(first, second)


@pytest.mark.parametrize("latent_height", [48, 160])
def test_sampler_preserves_required_latent_shapes(
    latent_height: int,
) -> None:
    _, _, graphemes = _text_components()
    result = sample_scaled_latent(
        TinySamplingModel(),  # type: ignore[arg-type]
        graphemes,
        _style(),
        latent_height=latent_height,
        latent_width=128,
        config=SamplingConfig(num_inference_steps=2, seed=7),
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert result.shape == (1, 4, latent_height, 128)


def test_sampler_rejects_nan_velocity() -> None:
    _, _, graphemes = _text_components()
    with pytest.raises(
        FloatingPointError,
        match="velocity tại timestep",
    ):
        sample_scaled_latent(
            TinySamplingModel(nan_velocity=True),  # type: ignore[arg-type]
            graphemes,
            _style(),
            latent_height=48,
            latent_width=128,
            config=SamplingConfig(num_inference_steps=2, seed=7),
            num_train_timesteps=1000,
            device=torch.device("cpu"),
        )


def test_decode_denormalizes_before_autokl() -> None:
    decoder = TinyDecoder()
    scaled = torch.ones(1, 4, 2, 128)
    latent, image = decode_scaled_latent(
        decoder,  # type: ignore[arg-type]
        _statistics(),
        scaled,
    )
    assert decoder.last_latent is not None
    assert torch.equal(decoder.last_latent, latent)
    assert torch.allclose(latent, torch.full_like(latent, 2.25))
    assert image.shape == (1, 1, 16, 1024)
    assert image.max() == 1.0


def test_generate_encodes_reference_once_and_returns_canvas() -> None:
    formatter, vocabulary, _ = _text_components()
    model = TinySamplingModel()
    result = generate_paragraph(
        model,  # type: ignore[arg-type]
        TinyDecoder(),  # type: ignore[arg-type]
        _statistics(),
        formatter,
        vocabulary,
        text="a",
        reference_image=torch.zeros(1, 1, 256, 32),
        reference_valid_mask=torch.ones(
            1, 1, 256, 32, dtype=torch.bool
        ),
        sampling_config=SamplingConfig(
            num_inference_steps=2,
            seed=13,
        ),
        num_train_timesteps=1000,
        device=torch.device("cpu"),
    )
    assert model.reference_calls == 1
    assert model.forward_calls == 2
    assert result.image.shape == (1, 1, 384, 1024)
    assert result.scaled_latent.shape == (1, 4, 48, 128)
    assert result.latent.shape == result.scaled_latent.shape
    assert result.formatted_text.output_height == 384


def test_cfg_is_not_exposed_without_condition_dropout() -> None:
    assert {field.name for field in fields(SamplingConfig)} == {
        "num_inference_steps",
        "seed",
    }
    parameters = inspect.signature(sample_scaled_latent).parameters
    assert not any(
        name in parameters
        for name in (
            "guidance_scale",
            "cfg_scale",
            "eta",
            "num_candidates",
        )
    )


def test_generation_yaml_loads_locked_contract() -> None:
    config = load_generation_config(
        Path("configs/vietparadiff/generate.yaml")
    )
    assert config.diffusion.num_inference_steps == 50
    assert not hasattr(config.diffusion, "num_train_timesteps")
    assert config.model.contract.name == "inference_contract.json"
    assert config.model.vocabulary.name == "grapheme_vocabulary.json"
    assert config.input.reference_height == 256
    assert config.input.maximum_reference_width == 1536


def test_model_config_loads_strictly_without_inference_download(
    tmp_path: Path,
) -> None:
    _, vocabulary, _ = _text_components()
    stored = VietParaDiffConfig(
        autokl=AutoKLConfig(),
        text=TextEncoderConfig(
            base_vocab_size=len(vocabulary.base_to_id),
            shape_vocab_size=len(vocabulary.shape_to_id),
            tone_vocab_size=len(vocabulary.tone_to_id),
            case_vocab_size=len(vocabulary.case_to_id),
            class_vocab_size=len(vocabulary.class_to_id),
        ),
        style=StyleEncoderConfig(
            use_pretrained_backbone=True,
            convnext_checkpoint=None,
        ),
        unet=ParagraphUNetConfig(),
    )
    path = tmp_path / "model_config.json"
    path.write_text(
        json.dumps(asdict(stored)),
        encoding="utf-8",
    )
    loaded = load_model_config(path)
    assert loaded == stored
    checkpoint_config = checkpoint_loading_config(loaded)
    assert not checkpoint_config.style.use_pretrained_backbone
    assert checkpoint_config.style.convnext_checkpoint is None
    assert checkpoint_config.text == stored.text
    assert checkpoint_config.unet == stored.unet

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["style"]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model_config.style keys"):
        load_model_config(path)


def test_generation_rejects_changed_generator_checkpoint(
    tmp_path: Path,
) -> None:
    paths = _contract_artifacts(tmp_path)
    assert _load_contract(paths).num_train_timesteps == 1000
    paths["checkpoint"].write_bytes(b"different generator")
    with pytest.raises(
        ValueError,
        match="từ chối generator checkpoint",
    ):
        _load_contract(paths)


def test_generation_rejects_changed_model_config(
    tmp_path: Path,
) -> None:
    paths = _contract_artifacts(tmp_path)
    paths["model_config"].write_text(
        json.dumps({"model": "other run"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="từ chối model config"):
        _load_contract(paths)


def test_generation_rejects_changed_vocabulary(
    tmp_path: Path,
) -> None:
    paths = _contract_artifacts(tmp_path)
    vocabulary = GraphemeVocabulary.load(paths["vocabulary"])
    payload = vocabulary.to_dict()
    base = payload["base_to_id"]
    first = next(token for token, index in base.items() if index == 2)
    second = next(token for token, index in base.items() if index == 3)
    base[first], base[second] = base[second], base[first]
    GraphemeVocabulary.from_dict(payload).save(paths["vocabulary"])
    with pytest.raises(
        ValueError,
        match="từ chối grapheme vocabulary",
    ):
        _load_contract(paths)


def test_generation_uses_training_num_timesteps_from_contract(
    tmp_path: Path,
) -> None:
    paths = _contract_artifacts(
        tmp_path,
        num_train_timesteps=777,
    )
    contract = _load_contract(paths)
    assert contract.num_train_timesteps == 777
    generation = load_generation_config(
        Path("configs/vietparadiff/generate.yaml")
    )
    assert not hasattr(generation.diffusion, "num_train_timesteps")


def test_same_vocab_sizes_with_different_ids_are_rejected(
    tmp_path: Path,
) -> None:
    paths = _contract_artifacts(tmp_path)
    original = GraphemeVocabulary.load(paths["vocabulary"])
    payload = original.to_dict()
    base = payload["base_to_id"]
    base["a"], base["b"] = base["b"], base["a"]
    changed = GraphemeVocabulary.from_dict(payload)
    assert len(changed.base_to_id) == len(original.base_to_id)
    changed.save(paths["vocabulary"])
    with pytest.raises(
        ValueError,
        match="từ chối grapheme vocabulary",
    ):
        _load_contract(paths)


def test_vocabulary_save_load_preserves_exact_mapping(
    tmp_path: Path,
) -> None:
    vocabulary = GraphemeVocabulary.default_vietnamese()
    path = tmp_path / "vocabulary.json"
    vocabulary.save(path)
    restored = GraphemeVocabulary.load(path)
    assert restored.to_dict() == vocabulary.to_dict()


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("duplicate", "ID trùng"),
        ("gap", "liên tục từ 0"),
        ("boolean", "integer ID"),
        ("missing_pad", "<pad>=0"),
    ],
)
def test_vocabulary_loader_rejects_invalid_id_mapping(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    payload = GraphemeVocabulary.default_vietnamese().to_dict()
    base = payload["base_to_id"]
    if mutation == "duplicate":
        base["a"] = base["b"]
    elif mutation == "gap":
        base["a"] = 10_000
    elif mutation == "boolean":
        base["a"] = True  # type: ignore[assignment]
    elif mutation == "missing_pad":
        del base["<pad>"]
        base["replacement"] = 0
    path = tmp_path / "invalid_vocabulary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=error):
        GraphemeVocabulary.load(path)


def test_formatter_preserves_hard_newline_during_auto_wrap() -> None:
    formatter, _, _ = _text_components()
    formatted = formatter.format("a\nb")
    assert formatted.lines == ("a", "b")
    assert any(
        grapheme.class_name == "newline"
        for grapheme in formatted.graphemes
    )
