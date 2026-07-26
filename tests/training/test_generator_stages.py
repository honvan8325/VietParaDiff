from __future__ import annotations

from pathlib import Path

import pytest

from vietparadiff.training.generator import (
    DeterministicRealSyntheticBatchMixer,
    training_lineage,
    load_vietparadiff_training_config,
)


class _EpochLoader:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.epochs: list[int] = []
        self.dataset = self
        self.batch_sampler = _Sampler(self.epochs)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        for name in self.names:
            yield {"name": name}

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _Sampler:
    def __init__(self, epochs: list[int]) -> None:
        self.epochs = epochs

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_stage_configs_are_strict_and_locked() -> None:
    pretrain = load_vietparadiff_training_config(
        Path("configs/vietparadiff/pretrain.yaml")
    )
    finetune = load_vietparadiff_training_config(
        Path("configs/vietparadiff/finetune.yaml")
    )
    guided = load_vietparadiff_training_config(
        Path("configs/vietparadiff/htr_guided.yaml")
    )
    assert pretrain.stage == "pretrain"
    assert finetune.stage == "finetune"
    assert finetune.diffusion.epochs == 50
    assert finetune.optimizer.learning_rate == pytest.approx(5e-5)
    assert finetune.scheduler.warmup_steps == 2000
    assert guided.stage == "htr_guided"
    assert guided.diffusion.epochs == 10
    assert guided.optimizer.learning_rate == pytest.approx(2e-5)
    assert guided.guidance is not None
    assert guided.guidance.maximum_weight == pytest.approx(0.05)
    assert guided.guidance.warmup_steps == 5000
    assert guided.guidance.maximum_timestep == 250
    assert guided.guidance.every_n_optimizer_steps == 4


def test_mixer_exhausts_real_once_and_cycles_synthetic() -> None:
    real = _EpochLoader(["r0", "r1", "r2", "r3", "r4", "r5", "r6"])
    synthetic = _EpochLoader(["s0"])
    mixer = DeterministicRealSyntheticBatchMixer(
        real,
        synthetic,
        real_batches_per_cycle=3,
        synthetic_batches_per_cycle=1,
    )
    mixer.set_epoch(7)
    batches = list(mixer)
    assert [
        (batch["name"], batch["data_source"])
        for batch in batches
    ] == [
        ("r0", "real"),
        ("r1", "real"),
        ("r2", "real"),
        ("s0", "synthetic"),
        ("r3", "real"),
        ("r4", "real"),
        ("r5", "real"),
        ("s0", "synthetic"),
        ("r6", "real"),
    ]
    assert len(mixer) == 9
    assert real.epochs == [7, 7]
    assert synthetic.epochs == [7, 7]


def test_mixer_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match="không rỗng"):
        DeterministicRealSyntheticBatchMixer(
            _EpochLoader([]),
            _EpochLoader(["s"]),
        )


def test_finetune_lineage_records_parent_manifests_and_mixing() -> None:
    config = load_vietparadiff_training_config(
        Path("configs/vietparadiff/finetune.yaml")
    )
    artifacts = {
        "real_targets": "1" * 64,
        "synthetic_targets": "2" * 64,
        "train_references": "3" * 64,
        "autokl_checkpoint": "4" * 64,
        "latent_statistics": "5" * 64,
        "parent_checkpoint": "6" * 64,
        "parent_contract": "7" * 64,
        "parent_model_config": "8" * 64,
        "parent_vocabulary": "9" * 64,
    }
    lineage = training_lineage(config, artifacts)
    assert lineage["stage"] == "finetune"
    assert lineage["parent"] == {
        "checkpoint_sha256": "6" * 64,
        "contract_sha256": "7" * 64,
        "model_config_sha256": "8" * 64,
        "vocabulary_sha256": "9" * 64,
    }
    assert lineage["manifests"] == {
        "real_targets": "1" * 64,
        "synthetic_targets": "2" * 64,
        "train_references": "3" * 64,
    }
    assert lineage["mixing_schedule"] == {
        "use_synthetic_data": True,
        "real_batches_per_cycle": 3,
        "synthetic_batches_per_cycle": 1,
        "epoch_policy": "exhaust_real_once",
    }
