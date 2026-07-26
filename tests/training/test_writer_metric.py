"""Writer verifier architecture and training primitive tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from vietparadiff.models import (
    ArcFaceHead,
    WriterEncoderConfig,
    WriterStyleEncoder,
)
from vietparadiff.training.writer import split_writer_ids
from vietparadiff.training.writer import (
    WriterMetricBackboneConfig,
    WriterMetricCheckpointConfig,
    WriterMetricDataConfig,
    WriterMetricLogger,
    WriterMetricLoggingConfig,
    WriterMetricOptimizerConfig,
    WriterMetricTrainingConfig,
    _save_writer_last,
    _writer_scheduler,
)
from vietparadiff.runtime import RuntimePrecision, create_grad_scaler
from torch.optim import AdamW


def test_arcface_target_margin_matches_formula() -> None:
    head = ArcFaceHead(256, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1.0
        head.weight[1, 1] = 1.0
    embeddings = torch.zeros(2, 256)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    logits = head(embeddings, torch.tensor([0, 1]))
    clamped = torch.tensor(1.0 - 1e-7)
    expected = 30.0 * (
        clamped * torch.cos(torch.tensor(0.5))
        - torch.sqrt(1.0 - clamped.square())
        * torch.sin(torch.tensor(0.5))
    )
    assert float(logits[0, 0].detach()) == pytest.approx(
        float(expected), rel=1e-5
    )
    assert float(logits[1, 1].detach()) == pytest.approx(
        float(expected), rel=1e-5
    )
    assert abs(float(logits[0, 1].detach())) < 1e-4


def test_writer_split_is_deterministic_and_writer_disjoint() -> None:
    records = [
        {
            "id": f"{writer}-{sample}",
            "canonical_writer_id": f"writer-{writer:02d}",
        }
        for writer in range(20)
        for sample in range(2)
    ]
    first = split_writer_ids(
        records,
        seed=42,
        validation_fraction=0.1,
    )
    second = split_writer_ids(
        list(reversed(records)),
        seed=42,
        validation_fraction=0.1,
    )
    assert first == second
    train, validation = first
    assert len(train) == 18
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)


def test_writer_encoder_has_finite_gradient_and_strict_roundtrip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    config = WriterEncoderConfig()
    model = WriterStyleEncoder(config).train()
    images = torch.randn(2, 1, 128, 64).clamp(-1.0, 1.0)
    embeddings = model(images)
    assert embeddings.shape == (2, 256)
    assert torch.allclose(
        embeddings.norm(dim=1),
        torch.ones(2),
        atol=1e-5,
    )
    loss = F.mse_loss(
        embeddings,
        torch.roll(embeddings.detach(), shifts=1, dims=0),
    )
    loss.backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()

    checkpoint = tmp_path / "writer.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = WriterStyleEncoder(config)
    restored.load_checkpoint(checkpoint)
    assert {
        key: tuple(value.shape)
        for key, value in restored.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }


def test_writer_metric_tensorboard_and_wandb_offline_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_SILENT", "true")
    logger = WriterMetricLogger(
        WriterMetricLoggingConfig(
            tensorboard=True,
            wandb=True,
            wandb_project="vietparadiff-writer-test",
            wandb_entity=None,
            wandb_mode="offline",
        ),
        tmp_path,
        {"seed": 42},
    )
    logger.log({"train/loss": 1.0}, 1)
    logger.close()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    assert (tmp_path / "wandb").is_dir()


def test_writer_last_checkpoint_binds_data_and_backbone_hashes(
    tmp_path: Path,
) -> None:
    config = WriterMetricTrainingConfig(
        seed=42,
        device="cpu",
        precision="float32",
        selection_protocol="writer_disjoint_internal_validation",
        data=WriterMetricDataConfig(
            tmp_path / "lines.jsonl",
            tmp_path / "paragraphs.jsonl",
            tmp_path,
            0,
            2,
            2,
            0.1,
        ),
        backbone=WriterMetricBackboneConfig(
            tmp_path / "resnet.pt",
            tmp_path / "vision.json",
        ),
        optimizer=WriterMetricOptimizerConfig(
            1e-4,
            1e-4,
            2,
            1.0,
            0,
        ),
        checkpoint=WriterMetricCheckpointConfig(tmp_path, True, True),
        logging=WriterMetricLoggingConfig(
            False,
            False,
            "test",
            None,
            "disabled",
        ),
    )
    model = WriterStyleEncoder(WriterEncoderConfig())
    head = ArcFaceHead(256, 2)
    optimizer = AdamW([*model.parameters(), *head.parameters()])
    scheduler = _writer_scheduler(optimizer, config.optimizer)
    runtime = RuntimePrecision(
        torch.device("cpu"),
        torch.float32,
        False,
        False,
    )
    hashes = {
        "line_manifest_sha256": "1" * 64,
        "paragraph_manifest_sha256": "2" * 64,
        "resnet18_checkpoint_sha256": "3" * 64,
        "visual_backbone_contract_sha256": "4" * 64,
    }
    checkpoint = tmp_path / "last.pt"
    _save_writer_last(
        checkpoint,
        model=model,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=create_grad_scaler(runtime),
        next_epoch=1,
        best_eer=0.2,
        best_auc=0.8,
        train_writers=("writer-a", "writer-b"),
        validation_writers=("writer-c",),
        writer_to_id={"writer-a": 0, "writer-b": 1},
        config=config,
        artifact_hashes=hashes,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["artifact_hashes"] == hashes
