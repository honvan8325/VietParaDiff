from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vietparadiff.data.pipeline import HTRVocabulary
from vietparadiff.artifacts import LatentStatistics, sha256_file
from vietparadiff.models.grapheme import (
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.config import TextEncoderConfig
from vietparadiff.models.style import StyleCondition
from vietparadiff.models.htr import HTROutput
from vietparadiff.runtime import RuntimePrecision, create_grad_scaler
from vietparadiff.training.generator import (
    DiffusionStageConfig,
    FrozenAutoKLConfig,
    GeneratorCheckpointConfig,
    GeneratorInitializationConfig,
    GeneratorLoggingConfig,
    GeneratorOptimizerConfig,
    GeneratorSchedulerConfig,
    VietParaDiffDataConfig,
    VietParaDiffTrainer,
    VietParaDiffTrainingConfig,
    create_optimizer_and_scheduler,
)
from vietparadiff.training.htr_guidance import (
    FrozenHTRTeacher,
    GeneratedLineRouter,
    HTRGuidanceConfig,
    guidance_step_enabled,
    guidance_weight,
    predicted_clean_from_velocity,
    validate_htr_inference_contract,
)


def _config() -> HTRGuidanceConfig:
    return HTRGuidanceConfig(
        checkpoint=Path("best.pt"),
        model_config=Path("model_config.json"),
        vocabulary=Path("vocabulary.json"),
        maximum_weight=0.05,
        warmup_steps=5000,
        maximum_timestep=250,
        every_n_optimizer_steps=4,
        raw_weight=1.0,
        base_weight=0.5,
        shape_weight=0.25,
        tone_weight=0.25,
    )


def _vocabulary() -> HTRVocabulary:
    return HTRVocabulary(
        raw_to_id={"<blank>": 0, "<unk>": 1, "a": 2, "b": 3},
        base_to_id={"<blank>": 0, "<unk>": 1, "a": 2, "b": 3},
        shape_to_id={"<blank>": 0, "<unk>": 1, "none": 2},
        tone_to_id={"<blank>": 0, "<unk>": 1, "none": 2},
    )


def _slots() -> Tensor:
    slots = torch.zeros(1, 8, 48, 128)
    slots[0, 0, 6:20, 6:122] = 1.0
    slots[0, 2, 23:37, 6:122] = 1.0
    return slots


def test_velocity_inversion_recovers_clean_latent() -> None:
    clean = torch.randn(2, 4, 3, 5)
    noise = torch.randn_like(clean)
    alpha = torch.tensor([0.8, 0.6])
    sigma = torch.tensor([0.6, 0.8])
    noisy = (
        alpha[:, None, None, None] * clean
        + sigma[:, None, None, None] * noise
    )
    velocity = (
        alpha[:, None, None, None] * noise
        - sigma[:, None, None, None] * clean
    )
    recovered = predicted_clean_from_velocity(
        noisy,
        velocity,
        alpha,
        sigma,
    )
    assert torch.allclose(recovered, clean, atol=1e-6)


def test_guidance_schedule_is_warmed_and_sparse() -> None:
    config = _config()
    assert guidance_weight(config, 0) == 0.0
    assert guidance_weight(config, 2500) == pytest.approx(0.025)
    assert guidance_weight(config, 5000) == pytest.approx(0.05)
    assert not guidance_step_enabled(config, 0)
    assert guidance_step_enabled(config, 4)
    assert not guidance_step_enabled(config, 5)


def test_generated_line_router_is_differentiable_and_skips_empty_lines() -> None:
    images = torch.randn(
        1,
        1,
        384,
        1024,
        requires_grad=True,
    )
    batch = GeneratedLineRouter()(
        images,
        _slots(),
        ["a\n\nb"],
        _vocabulary(),
        sample_ids=["sample"],
    )
    routed = batch["images"]
    assert isinstance(routed, Tensor)
    assert routed.shape == (2, 1, 64, 1024)
    assert batch["sample_ids"] == [
        "sample:line_0",
        "sample:line_2",
    ]
    routed.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad) > 0


class _TinyHTR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(1, 4) for _ in range(4)])

    def forward(
        self,
        images: Tensor,
        valid_widths: Tensor,
    ) -> HTROutput:
        features = F.avg_pool2d(images, (64, 4)).squeeze(2).transpose(1, 2)
        lengths = (valid_widths + 3) // 4
        logits = [head(features) for head in self.heads]
        return HTROutput(*logits, lengths)


class _Posterior:
    def __init__(self, latent: Tensor) -> None:
        self.latent = latent

    def mode(self) -> Tensor:
        return self.latent


class _DifferentiableAutoKL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.decode_calls = 0

    def encode(self, images: Tensor) -> _Posterior:
        latent = F.avg_pool2d(images, 8).repeat(1, 4, 1, 1)
        return _Posterior(latent + self.anchor)

    def decode(self, latents: Tensor) -> Tensor:
        self.decode_calls += 1
        return torch.tanh(
            F.interpolate(
                latents[:, :1],
                scale_factor=8,
                mode="nearest",
            )
            + self.anchor
        )


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Conv2d(4, 4, 1)

    def encode_reference(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> StyleCondition:
        del valid_mask
        return StyleCondition(
            local_tokens=torch.zeros(
                images.shape[0], 1, 4, device=images.device
            ),
            global_style=torch.zeros(
                images.shape[0], 4, device=images.device
            ),
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

    def forward(self, batch):
        return type(
            "_Output",
            (),
            {
                "predicted_velocity": self.output(
                    batch.noisy_latents
                )
            },
        )()


def test_frozen_teacher_keeps_image_gradient_and_no_parameter_gradient() -> None:
    teacher = FrozenHTRTeacher(  # type: ignore[arg-type]
        _TinyHTR(),
        _vocabulary(),
        _config(),
    )
    images = torch.randn(
        1,
        1,
        384,
        1024,
        requires_grad=True,
    )
    result = teacher(
        images,
        _slots(),
        ["a\n\nb"],
        sample_ids=["sample"],
    )
    assert result.line_count == 2
    result.losses.total.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert all(
        parameter.grad is None
        for parameter in teacher.model.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in teacher.model.parameters()
    )


def test_htr_guided_trainer_backpropagates_only_to_generator(
    tmp_path: Path,
) -> None:
    guidance = _config()
    config = VietParaDiffTrainingConfig(
        seed=42,
        device="cpu",
        precision="float32",
        data=VietParaDiffDataConfig(
            train_targets=None,
            train_references=tmp_path / "references.jsonl",
            image_root=tmp_path,
            num_workers=0,
            batch_size=1,
            gradient_accumulation_steps=1,
            real_targets=tmp_path / "real.jsonl",
            synthetic_targets=tmp_path / "synthetic.jsonl",
        ),
        autokl=FrozenAutoKLConfig(
            tmp_path / "autokl.pt",
            tmp_path / "stats.json",
        ),
        style=None,
        diffusion=DiffusionStageConfig(1, 1000, "cosine"),
        optimizer=GeneratorOptimizerConfig(
            "adamw",
            1e-3,
            (0.9, 0.99),
            0.0,
            1.0,
        ),
        scheduler=GeneratorSchedulerConfig(1, 0.1),
        logging=GeneratorLoggingConfig(
            1,
            False,
            False,
            "disabled",
            "test",
            None,
            None,
        ),
        checkpoint=GeneratorCheckpointConfig(
            tmp_path / "checkpoints",
            True,
            True,
        ),
        stage="htr_guided",
        initialization=GeneratorInitializationConfig(
            tmp_path / "parent.pt",
            tmp_path / "parent_contract.json",
            tmp_path / "parent_model_config.json",
            tmp_path / "parent_vocabulary.json",
        ),
        guidance=guidance,
    )
    generator = _TinyGenerator()
    autokl = _DifferentiableAutoKL()
    optimizer, scheduler = create_optimizer_and_scheduler(
        generator,
        config.optimizer,
        config.scheduler,
        total_steps=2,
    )
    runtime = RuntimePrecision(
        torch.device("cpu"),
        torch.float32,
        False,
        False,
    )
    teacher = FrozenHTRTeacher(  # type: ignore[arg-type]
        _TinyHTR(),
        _vocabulary(),
        guidance,
    )
    artifacts = {
        "real_targets": "1" * 64,
        "synthetic_targets": "2" * 64,
        "train_references": "3" * 64,
        "autokl_checkpoint": "a" * 64,
        "latent_statistics": "4" * 64,
        "parent_checkpoint": "5" * 64,
        "parent_contract": "6" * 64,
        "parent_model_config": "7" * 64,
        "parent_vocabulary": "8" * 64,
        "htr_checkpoint": "9" * 64,
        "htr_contract": "d" * 64,
        "htr_model_config": "b" * 64,
        "htr_vocabulary": "c" * 64,
    }
    statistics = LatentStatistics(
        0.0,
        1.0,
        1.0,
        1,
        2,
        "a" * 64,
    )
    generator_vocabulary = GraphemeVocabulary.default_vietnamese()
    trainer = VietParaDiffTrainer(
        generator,  # type: ignore[arg-type]
        autokl,  # type: ignore[arg-type]
        statistics,
        optimizer,
        scheduler,
        create_grad_scaler(runtime),
        config,
        runtime,
        artifacts,
        {"tiny": True},
        generator_vocabulary,
        htr_teacher=teacher,
    )
    text_config = TextEncoderConfig(
        base_vocab_size=len(generator_vocabulary.base_to_id),
        shape_vocab_size=len(generator_vocabulary.shape_to_id),
        tone_vocab_size=len(generator_vocabulary.tone_to_id),
        case_vocab_size=len(generator_vocabulary.case_to_id),
        class_vocab_size=len(generator_vocabulary.class_to_id),
    )
    formatted = ParagraphFormatter(text_config).format(
        "a",
        preserve_physical_lines=True,
        output_height=384,
    )
    text_batch = generator_vocabulary.encode_batch([formatted])
    trainer.global_step = 4
    output = trainer.train_micro_batch(
        {
            "target_images": torch.randn(
                1, 1, 384, 1024
            ).clamp(-1, 1),
            "reference_images": torch.randn(
                1, 1, 256, 32
            ).clamp(-1, 1),
            "reference_valid_mask": torch.ones(
                1, 1, 256, 32, dtype=torch.bool
            ),
            "graphemes": text_batch.graphemes,
            "canonical_line_slots": (
                text_batch.canonical_line_slots
            ),
            "target_texts": ["a"],
            "target_ids": ["sample"],
            "output_height": 384,
        },
        timesteps=torch.tensor([100], dtype=torch.long),
        noise=torch.zeros(1, 4, 48, 128),
    )
    assert output.htr_result is not None
    assert output.htr_weight > 0.0
    assert autokl.decode_calls == 1
    assert generator.output.weight.grad is not None
    assert torch.isfinite(generator.output.weight.grad).all()
    assert all(
        parameter.grad is None
        for parameter in teacher.model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in autokl.parameters()
    )


def test_router_rejects_slot_transcript_mismatch() -> None:
    slots = _slots()
    slots[0, 2].zero_()
    with pytest.raises(ValueError, match="không khớp"):
        GeneratedLineRouter()(
            torch.randn(1, 1, 384, 1024),
            slots,
            ["a\n\nb"],
            _vocabulary(),
        )


def test_htr_contract_rejects_same_size_changed_vocabulary(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    model_config = tmp_path / "model_config.json"
    vocabulary_path = tmp_path / "vocabulary.json"
    checkpoint.write_bytes(b"checkpoint")
    model_config.write_text('{"model": "htr"}\n', encoding="utf-8")
    _vocabulary().save(vocabulary_path)
    contract = {
        "schema_version": 1,
        "htr_checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": sha256_file(model_config),
        "vocabulary_sha256": sha256_file(vocabulary_path),
    }
    (tmp_path / "inference_contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    config = HTRGuidanceConfig(
        checkpoint,
        model_config,
        vocabulary_path,
        0.05,
        5000,
        250,
        4,
        1.0,
        0.5,
        0.25,
        0.25,
    )
    assert validate_htr_inference_contract(config) == (
        tmp_path / "inference_contract.json"
    )
    changed = _vocabulary()
    changed.raw_to_id["a"], changed.raw_to_id["b"] = (
        changed.raw_to_id["b"],
        changed.raw_to_id["a"],
    )
    changed.save(vocabulary_path)
    with pytest.raises(ValueError, match="HTR vocabulary"):
        validate_htr_inference_contract(config)
