from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.nn import functional as F

from vietparadiff.artifacts import LatentStatistics
from vietparadiff.data.pipeline import ReferenceImageProcessor
from vietparadiff.evaluation.fixed_pairs import (
    EvaluationAutoKLConfig,
    EvaluationConfig,
    EvaluationDataConfig,
    EvaluationDiffusionConfig,
    EvaluationInputConfig,
    EvaluationModelConfig,
    EvaluationOutputConfig,
    FixedPairEvaluator,
    load_evaluation_config,
    load_test_pairs,
    stable_sample_seed,
)
from vietparadiff.models.config import TextEncoderConfig
from vietparadiff.models.grapheme import (
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.style import StyleCondition


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.reference_calls = 0

    def encode_reference(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> StyleCondition:
        del valid_mask
        self.reference_calls += 1
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
        return SimpleNamespace(
            predicted_velocity=torch.zeros_like(
                batch.noisy_latents
            )
            + self.anchor
        )


class _TinyAutoKL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def decode(self, latents: Tensor) -> Tensor:
        return torch.tanh(
            F.interpolate(
                latents[:, :1],
                scale_factor=8,
                mode="nearest",
            )
            + self.anchor
        )


def _write_image(path: Path, height: int) -> None:
    image = Image.new("L", (1024, height), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((32, height // 2, 700, height // 2), fill=0, width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    image.close()


def _records(tmp_path: Path) -> Path:
    _write_image(tmp_path / "target_short.png", 384)
    _write_image(tmp_path / "target_tall.png", 1280)
    _write_image(tmp_path / "reference.png", 64)
    records = [
        {
            "pair_id": "pair_b",
            "canonical_writer_id": "writer_b",
            "target_id": "target_b",
            "target_image": "target_tall.png",
            "target_text": "\n".join(["a"] * 8),
            "reference_id": "reference_b",
            "reference_image": "reference.png",
        },
        {
            "pair_id": "pair_a",
            "canonical_writer_id": "writer_a",
            "target_id": "target_a",
            "target_image": "target_short.png",
            "target_text": "a",
            "reference_id": "reference_a",
            "reference_image": "reference.png",
        },
    ]
    path = tmp_path / "test_pairs.jsonl"
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, manifest: Path) -> EvaluationConfig:
    return EvaluationConfig(
        base_seed=42,
        device="cpu",
        precision="float32",
        model=EvaluationModelConfig(
            checkpoint=tmp_path / "best.pt",
            contract=tmp_path / "inference_contract.json",
            model_config=tmp_path / "model_config.json",
            vocabulary=tmp_path / "vocabulary.json",
        ),
        autokl=EvaluationAutoKLConfig(
            checkpoint=tmp_path / "autokl.pt",
            latent_statistics=tmp_path / "stats.json",
        ),
        data=EvaluationDataConfig(
            test_pairs=manifest,
            image_root=tmp_path,
            samples_per_pair=3,
        ),
        diffusion=EvaluationDiffusionConfig(
            num_inference_steps=2
        ),
        input=EvaluationInputConfig(256, 1536),
        output=EvaluationOutputConfig(
            directory=tmp_path / "evaluation",
            contact_sheet_pairs=8,
        ),
    )


def _evaluator(
    tmp_path: Path,
) -> tuple[FixedPairEvaluator, _TinyGenerator]:
    manifest = _records(tmp_path)
    vocabulary = GraphemeVocabulary.default_vietnamese()
    text_config = TextEncoderConfig(
        base_vocab_size=len(vocabulary.base_to_id),
        shape_vocab_size=len(vocabulary.shape_to_id),
        tone_vocab_size=len(vocabulary.tone_to_id),
        case_vocab_size=len(vocabulary.case_to_id),
        class_vocab_size=len(vocabulary.class_to_id),
    )
    model = _TinyGenerator()
    statistics = LatentStatistics(
        latent_mean=0.0,
        latent_std=1.0,
        scaling_factor=1.0,
        num_samples=1,
        num_elements=2,
        autokl_checkpoint_sha256="a" * 64,
    )
    artifacts = {
        "generator_checkpoint": "a" * 64,
        "inference_contract": "b" * 64,
        "model_config": "c" * 64,
        "grapheme_vocabulary": "d" * 64,
        "autokl_checkpoint": "e" * 64,
        "latent_statistics": "f" * 64,
        "test_pairs": "0" * 64,
    }
    evaluator = FixedPairEvaluator(
        model,  # type: ignore[arg-type]
        _TinyAutoKL(),  # type: ignore[arg-type]
        statistics,
        ParagraphFormatter(text_config),
        vocabulary,
        ReferenceImageProcessor(),
        _config(tmp_path, manifest),
        num_train_timesteps=10,
        device=torch.device("cpu"),
        artifact_sha256=artifacts,
    )
    return evaluator, model


def test_evaluation_yaml_loads_locked_three_seed_contract() -> None:
    config = load_evaluation_config(
        Path("configs/vietparadiff/evaluate.yaml")
    )
    assert config.data.samples_per_pair == 3
    assert config.model.checkpoint.parent.name == "htr_guided"


def test_stable_seed_is_order_independent() -> None:
    first = stable_sample_seed(42, "pair_a", 0)
    assert first == stable_sample_seed(42, "pair_a", 0)
    assert first != stable_sample_seed(42, "pair_a", 1)
    assert first != stable_sample_seed(42, "pair_b", 0)


def test_fixed_pair_evaluation_generates_384_and_1280_and_resumes(
    tmp_path: Path,
) -> None:
    evaluator, model = _evaluator(tmp_path)
    summary = evaluator.run()
    assert summary["pair_count"] == 2
    assert summary["sample_count"] == 6
    assert model.reference_calls == 2
    records = [
        json.loads(line)
        for line in (
            tmp_path / "evaluation" / "results.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert {record["output_height"] for record in records} == {
        384,
        1280,
    }
    assert len(
        list((tmp_path / "evaluation" / "samples").rglob("*.png"))
    ) == 6
    assert (
        tmp_path
        / "evaluation"
        / "contact_sheets"
        / "sheet_0000.png"
    ).is_file()
    resumed = evaluator.run(resume=True)
    assert resumed["sample_count"] == 6
    assert model.reference_calls == 2
    resumed_records = (
        tmp_path / "evaluation" / "results.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(resumed_records) == 6


def test_resume_rejects_changed_artifact_contract(
    tmp_path: Path,
) -> None:
    evaluator, _ = _evaluator(tmp_path)
    evaluator.run()
    evaluator.artifact_sha256["test_pairs"] = "1" * 64
    with pytest.raises(ValueError, match="resume contract"):
        evaluator.run(resume=True)


def test_test_pair_manifest_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    manifest = _records(tmp_path)
    first = manifest.read_text(encoding="utf-8").splitlines()[0]
    manifest.write_text(first + "\n" + first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bị trùng"):
        load_test_pairs(manifest)
