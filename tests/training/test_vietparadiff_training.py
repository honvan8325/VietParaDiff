from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vietparadiff.artifacts import (
    LatentStatistics,
    load_latent_statistics,
    save_latent_statistics,
    sha256_file,
)
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
    VietParaDiff,
    VietParaDiffInput,
    VietParaDiffOutput,
)
from vietparadiff.training.generator import (
    DiffusionStageConfig,
    FrozenAutoKLConfig,
    GeneratorCheckpointConfig,
    GeneratorLoggingConfig,
    GeneratorOptimizerConfig,
    GeneratorSchedulerConfig,
    LatentStatisticsAccumulator,
    StyleInitializationConfig,
    VietParaDiffDataConfig,
    VietParaDiffTrainer,
    VietParaDiffTrainingConfig,
    create_optimizer_and_scheduler,
    learning_rate_factor,
    load_vietparadiff_training_config,
)
from vietparadiff.runtime import RuntimePrecision, create_grad_scaler


class _Posterior:
    def __init__(self, latent: Tensor) -> None:
        self.latent = latent
        self.mode_calls = 0

    def mode(self) -> Tensor:
        self.mode_calls += 1
        return self.latent


class TinyAutoKL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.decode_calls = 0
        self.last_posterior: _Posterior | None = None

    def encode(self, images: Tensor) -> _Posterior:
        latent = F.avg_pool2d(images, kernel_size=8).repeat(1, 4, 1, 1)
        posterior = _Posterior(latent * self.scale)
        self.last_posterior = posterior
        return posterior

    def decode(self, latents: Tensor) -> Tensor:
        del latents
        self.decode_calls += 1
        raise AssertionError("Base diffusion trainer không được decode.")


class TinyVietParaDiff(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(16, 4)
        self.style_projection = nn.Linear(1, 4)
        self.output = nn.Conv2d(4, 4, 1)
        self.reference_calls = 0

    def encode_reference(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> StyleCondition:
        self.reference_calls += 1
        weights = valid_mask.to(images.dtype)
        mean = (images * weights).sum((1, 2, 3), keepdim=True)
        mean = mean / weights.sum((1, 2, 3), keepdim=True).clamp_min(1)
        global_style = self.style_projection(mean.flatten(1))
        return StyleCondition(
            local_tokens=global_style[:, None],
            global_style=global_style,
            layout_scales=torch.ones(
                images.shape[0], 3, device=images.device
            ),
            valid_feature_mask=torch.ones(
                images.shape[0],
                1,
                1,
                1,
                dtype=torch.bool,
                device=images.device,
            ),
        )

    def forward(self, batch: VietParaDiffInput) -> VietParaDiffOutput:
        token_features = self.text_embedding(
            batch.graphemes.base_ids
        )
        mask = batch.graphemes.attention_mask[:, :, None]
        text = (
            (token_features * mask).sum(1)
            / mask.sum(1).clamp_min(1)
        )
        condition = (
            text + batch.style_condition.global_style
        )[:, :, None, None]
        predicted = self.output(batch.noisy_latents + condition)
        context = GraphemeCondition(
            base_context=token_features,
            shape_context=token_features,
            tone_context=token_features,
            attention_mask=batch.graphemes.attention_mask,
            line_ids=batch.graphemes.line_ids,
        )
        return VietParaDiffOutput(
            predicted_velocity=predicted,
            grapheme_condition=context,
            style_condition=batch.style_condition,
            line_tokens=condition.flatten(2).transpose(1, 2),
            diagnostics={"tiny": predicted.square().mean()},
        )


def _config(tmp_path: Path) -> VietParaDiffTrainingConfig:
    return VietParaDiffTrainingConfig(
        seed=42,
        device="cpu",
        precision="float32",
        data=VietParaDiffDataConfig(
            train_targets=tmp_path / "targets.jsonl",
            train_references=tmp_path / "references.jsonl",
            image_root=tmp_path,
            num_workers=0,
            batch_size=1,
            gradient_accumulation_steps=1,
        ),
        autokl=FrozenAutoKLConfig(
            checkpoint=tmp_path / "autokl.pt",
            latent_statistics=tmp_path / "latent_statistics.json",
        ),
        style=StyleInitializationConfig(
            use_pretrained_backbone=True,
            convnext_checkpoint=None,
        ),
        diffusion=DiffusionStageConfig(
            epochs=2,
            num_train_timesteps=10,
            noise_schedule="cosine",
        ),
        optimizer=GeneratorOptimizerConfig(
            name="adamw",
            learning_rate=1e-3,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            gradient_clip_norm=1.0,
        ),
        scheduler=GeneratorSchedulerConfig(
            warmup_steps=1,
            minimum_learning_rate_ratio=0.1,
        ),
        logging=GeneratorLoggingConfig(
            log_every_steps=1,
            tensorboard=False,
            wandb=False,
            wandb_mode="disabled",
            wandb_project="test",
            wandb_entity=None,
            run_name=None,
        ),
        checkpoint=GeneratorCheckpointConfig(
            output_dir=tmp_path / "checkpoints",
            save_last=True,
            save_best=True,
        ),
    )


def _graphemes(batch_size: int) -> GraphemeBatch:
    ids = torch.tensor([[2, 3]], dtype=torch.long).expand(
        batch_size, -1
    )
    return GraphemeBatch(
        base_ids=ids,
        shape_ids=torch.full_like(ids, 2),
        tone_ids=torch.full_like(ids, 2),
        case_ids=torch.full_like(ids, 2),
        class_ids=torch.full_like(ids, 2),
        line_ids=torch.zeros_like(ids),
        position_in_line_ids=torch.tensor(
            [[0, 1]], dtype=torch.long
        ).expand(batch_size, -1),
        height_bucket_ids=torch.zeros(
            batch_size, dtype=torch.long
        ),
        attention_mask=torch.ones(
            batch_size, 2, dtype=torch.bool
        ),
    )


def _batch(height: int = 384) -> dict[str, object]:
    return {
        "target_images": torch.randn(1, 1, height, 1024).clamp(-1, 1),
        "reference_images": torch.randn(1, 1, 256, 32).clamp(-1, 1),
        "reference_valid_mask": torch.ones(
            1, 1, 256, 32, dtype=torch.bool
        ),
        "graphemes": _graphemes(1),
        "output_height": height,
    }


def _trainer(
    tmp_path: Path,
    config: VietParaDiffTrainingConfig | None = None,
    vocabulary: GraphemeVocabulary | None = None,
) -> tuple[VietParaDiffTrainer, TinyVietParaDiff, TinyAutoKL]:
    config = _config(tmp_path) if config is None else config
    model = TinyVietParaDiff()
    autokl = TinyAutoKL()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
        config.scheduler,
        total_steps=4,
    )
    runtime = RuntimePrecision(
        torch.device("cpu"), torch.float32, False, False
    )
    statistics = LatentStatistics(
        latent_mean=0.1,
        latent_std=2.0,
        scaling_factor=0.5,
        num_samples=2,
        num_elements=16,
        autokl_checkpoint_sha256="a" * 64,
    )
    trainer = VietParaDiffTrainer(
        model,  # type: ignore[arg-type]
        autokl,  # type: ignore[arg-type]
        statistics,
        optimizer,
        scheduler,
        create_grad_scaler(runtime),
        config,
        runtime,
        {
            "train_targets": "b" * 64,
            "train_references": "c" * 64,
            "autokl_checkpoint": "a" * 64,
            "latent_statistics": "d" * 64,
        },
        {"tiny": True},
        (
            GraphemeVocabulary.default_vietnamese()
            if vocabulary is None
            else vocabulary
        ),
    )
    return trainer, model, autokl


def test_pretrain_yaml_loads_locked_base_contract() -> None:
    config = load_vietparadiff_training_config(
        Path("configs/vietparadiff/pretrain.yaml")
    )
    assert config.data.train_targets.name == "pretrain_targets.jsonl"
    assert config.diffusion.num_train_timesteps == 1000
    assert config.diffusion.noise_schedule == "cosine"
    assert config.style.use_pretrained_backbone


def test_latent_statistics_streaming_matches_direct_population_std() -> None:
    first = torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 4)
    second = torch.arange(32, 64, dtype=torch.float32).reshape(
        1, 4, 2, 4
    )
    accumulator = LatentStatisticsAccumulator()
    accumulator.update(first)
    accumulator.update(second)
    statistics = accumulator.finalize("a" * 64)
    combined = torch.cat((first.flatten(), second.flatten())).double()
    assert statistics.latent_mean == pytest.approx(
        float(combined.mean())
    )
    assert statistics.latent_std == pytest.approx(
        float(combined.std(correction=0))
    )
    assert statistics.num_samples == 2
    assert statistics.num_elements == 64


def test_latent_statistics_round_trip_and_checkpoint_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {}}, checkpoint)
    statistics = LatentStatistics(
        0.25,
        2.0,
        0.5,
        3,
        24,
        sha256_file(checkpoint),
    )
    path = tmp_path / "latent_statistics.json"
    save_latent_statistics(path, statistics)
    restored = load_latent_statistics(
        path,
        expected_autokl_checkpoint=checkpoint,
    )
    latent = torch.randn(2, 4, 2, 3)
    assert torch.allclose(
        restored.denormalize(restored.normalize(latent)),
        latent,
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="không thuộc AutoKL"):
        load_latent_statistics(
            path,
            expected_autokl_checkpoint=checkpoint,
        )


def test_cosine_schedule_never_uses_exact_endpoints() -> None:
    timesteps = torch.tensor([0, 999], dtype=torch.long)
    alpha, sigma = cosine_alpha_sigma(
        timesteps, num_train_timesteps=1000
    )
    assert 0.0 < sigma[0] < 1.0
    assert 0.0 < alpha[0] < 1.0
    assert 0.0 < alpha[-1] < 1.0
    assert 0.0 < sigma[-1] < 1.0
    assert torch.allclose(
        alpha.square() + sigma.square(),
        torch.ones_like(alpha),
        atol=1e-6,
    )


def test_velocity_equations_use_schedule_coefficients() -> None:
    alpha = torch.tensor([0.8, 0.6])
    sigma = torch.tensor([0.6, 0.8])
    clean = torch.ones(2, 4, 1, 1)
    noise = torch.full_like(clean, 2.0)
    noisy = add_diffusion_noise(clean, noise, alpha, sigma)
    target = velocity_target(clean, noise, alpha, sigma)
    assert torch.allclose(noisy[0], torch.full_like(noisy[0], 2.0))
    assert torch.allclose(target[0], torch.full_like(target[0], 1.0))
    assert torch.allclose(noisy[1], torch.full_like(noisy[1], 2.2))
    assert torch.allclose(target[1], torch.full_like(target[1], 0.4))


def test_scheduler_warmup_and_cosine_floor() -> None:
    assert learning_rate_factor(
        0, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == 0.0
    assert learning_rate_factor(
        2, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == 1.0
    assert learning_rate_factor(
        10, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == pytest.approx(0.1)


@pytest.mark.parametrize("height", [384, 1280])
def test_train_step_contract_freezes_autokl_and_never_decodes(
    tmp_path: Path,
    height: int,
) -> None:
    trainer, model, autokl = _trainer(tmp_path)
    output = trainer.train_micro_batch(
        _batch(height),
        timesteps=torch.tensor([5], dtype=torch.long),
        noise=torch.zeros(1, 4, height // 8, 128),
    )
    assert output.model_output.predicted_velocity.shape == (
        1,
        4,
        height // 8,
        128,
    )
    assert torch.isfinite(output.loss)
    assert autokl.last_posterior is not None
    assert autokl.last_posterior.mode_calls == 1
    assert autokl.decode_calls == 0
    assert not autokl.training
    assert all(
        not parameter.requires_grad
        for parameter in autokl.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in autokl.parameters()
    )
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_optimizer_updates_generator_parameter(tmp_path: Path) -> None:
    trainer, model, _ = _trainer(tmp_path)
    before = model.output.weight.detach().clone()
    trainer.train_micro_batch(_batch())
    trainer.optimizer_step()
    trainer.train_micro_batch(_batch())
    trainer.optimizer_step()
    assert not torch.equal(before, model.output.weight)


def test_checkpoint_resume_is_strict_and_best_is_model_only(
    tmp_path: Path,
) -> None:
    trainer, model, _ = _trainer(tmp_path)
    trainer.global_step = 7
    trainer.save_epoch_checkpoints(next_epoch=1, train_score=0.3)
    expected_python = random.random()
    expected_torch = torch.rand(1)
    best = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(best) == {"model"}
    clone = TinyVietParaDiff()
    clone.load_state_dict(best["model"], strict=True)
    assert torch.equal(clone.output.weight, model.output.weight)
    output_dir = tmp_path / "checkpoints"
    contract = json.loads(
        (output_dir / "inference_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["schema_version"] == 1
    assert contract["prediction_type"] == "velocity"
    assert contract["noise_schedule"] == "cosine"
    assert contract["num_train_timesteps"] == 10
    assert contract["neutral_layout"] is True
    assert contract["generator_checkpoint_sha256"] == sha256_file(
        output_dir / "best.pt"
    )
    assert contract["model_config_sha256"] == sha256_file(
        output_dir / "model_config.json"
    )
    assert contract["grapheme_vocabulary_sha256"] == sha256_file(
        output_dir / "grapheme_vocabulary.json"
    )
    assert contract["autokl_checkpoint_sha256"] == "a" * 64
    assert contract["latent_statistics_sha256"] == "d" * 64

    resumed, _, _ = _trainer(tmp_path)
    state = resumed.resume(tmp_path / "checkpoints" / "last.pt")
    assert (state.epoch, state.global_step) == (1, 7)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(1), expected_torch)

    changed = replace(
        _config(tmp_path),
        optimizer=replace(
            _config(tmp_path).optimizer,
            learning_rate=2e-3,
        ),
    )
    incompatible, incompatible_model, _ = _trainer(tmp_path, changed)
    before = incompatible_model.output.weight.detach().clone()
    with pytest.raises(
        ValueError,
        match="Resume generator config không tương thích: optimizer",
    ):
        incompatible.resume(tmp_path / "checkpoints" / "last.pt")
    assert torch.equal(before, incompatible_model.output.weight)


def test_resume_rejects_changed_artifact_before_model_load(
    tmp_path: Path,
) -> None:
    trainer, _, _ = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(next_epoch=1, train_score=0.2)
    resumed, model, _ = _trainer(tmp_path)
    resumed.artifact_sha256["train_targets"] = "changed"
    before = model.output.weight.detach().clone()
    with pytest.raises(ValueError, match="training artifacts"):
        resumed.resume(tmp_path / "checkpoints" / "last.pt")
    assert torch.equal(before, model.output.weight)


def test_resume_rejects_same_size_vocabulary_with_changed_ids(
    tmp_path: Path,
) -> None:
    trainer, _, _ = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(next_epoch=1, train_score=0.2)
    payload = trainer.grapheme_vocabulary.to_dict()
    base = payload["base_to_id"]
    base["a"], base["b"] = base["b"], base["a"]
    changed = GraphemeVocabulary.from_dict(payload)
    changed.save(
        tmp_path / "checkpoints" / "grapheme_vocabulary.json"
    )
    resumed, model, _ = _trainer(
        tmp_path,
        vocabulary=changed,
    )
    before = model.output.weight.detach().clone()
    with pytest.raises(ValueError, match="grapheme vocabulary"):
        resumed.resume(tmp_path / "checkpoints" / "last.pt")
    assert torch.equal(before, model.output.weight)


def test_invalid_batch_is_rejected_before_autokl_encode(
    tmp_path: Path,
) -> None:
    trainer, _, autokl = _trainer(tmp_path)
    batch = _batch()
    batch["reference_valid_mask"] = torch.ones(
        1, 1, 256, 32, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="reference_valid_mask"):
        trainer.train_micro_batch(batch)
    assert autokl.last_posterior is None


def test_full_generator_intended_parameters_receive_gradient() -> None:
    torch.manual_seed(29)
    vocabulary = GraphemeVocabulary.default_vietnamese()
    text_config = TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
        dropout=0.0,
    )
    model = VietParaDiff(
        VietParaDiffConfig(
            autokl=AutoKLConfig(),
            text=text_config,
            style=StyleEncoderConfig(
                use_pretrained_backbone=False
            ),
            unet=ParagraphUNetConfig(dropout=0.0),
        )
    ).train()
    paragraph = ParagraphFormatter(text_config).format(
        "a\nb",
        preserve_physical_lines=True,
        output_height=384,
    )
    graphemes = vocabulary.encode_batch([paragraph]).graphemes
    reference = torch.rand(1, 1, 256, 32) * 2.0 - 1.0
    style = model.encode_reference(
        reference,
        torch.ones_like(reference, dtype=torch.bool),
    )
    output = model(
        VietParaDiffInput(
            noisy_latents=torch.randn(1, 4, 48, 128),
            timesteps=torch.tensor([500], dtype=torch.long),
            graphemes=graphemes,
            style_condition=style,
        )
    )
    output.predicted_velocity.square().mean().backward()

    raw_conv = model.style_encoder.raw_stem[0]
    hf_conv = model.style_encoder.hf_stem[0]
    assert isinstance(raw_conv, nn.Conv2d)
    assert isinstance(hf_conv, nn.Conv2d)
    shape_adapter = model.unet.encoder_high.shape_adapter
    tone_adapter = model.unet.encoder_high.tone_adapter
    assert shape_adapter is not None
    assert tone_adapter is not None
    expected = {
        "text_base_embedding": model.text_encoder.base_embedding.weight,
        "style_raw_stem": raw_conv.weight,
        "style_hf_stem": hf_conv.weight,
        "style_fusion": model.style_encoder.fusion_gate.weight,
        "style_shared_trunk": next(
            model.style_encoder.shared_trunk.parameters()
        ),
        "style_local_tokens": model.style_encoder.local_queries,
        "style_global": model.style_encoder.global_mlp[0].weight,
        "unet_base_attention": (
            model.unet.encoder_high.main_attention.cross_attention.q_proj_weight
        ),
        "shape_adapter": (
            shape_adapter.output_projection.weight
        ),
        "tone_adapter": (
            tone_adapter.output_projection.weight
        ),
        "style_film": (
            model.unet.encoder_high.resblocks[0].style_film.weight
        ),
        "harmonizer": model.unet.harmonizer.output_projection.weight,
        "unet_output": model.unet.output_conv.weight,
    }
    for name, parameter in expected.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if name != "style_global":
            assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.count_nonzero(
        expected["style_global"].grad
    ) == 0


def test_statistics_json_schema_is_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {}}, checkpoint)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"latent_mean": 0.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="Latent statistics keys"):
        load_latent_statistics(
            path,
            expected_autokl_checkpoint=checkpoint,
        )
