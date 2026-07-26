from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from vietparadiff.data.pipeline import HTRVocabulary
from vietparadiff.training.htr import (
    HTRCheckpointConfig,
    HTRDataConfig,
    HTREpochMetrics,
    HTRLossConfig,
    HTRLoggingConfig,
    HTROptimizerConfig,
    HTRSchedulerConfig,
    HTRStageConfig,
    HTRTrainer,
    HTRTrainingConfig,
    compute_htr_losses,
    create_optimizer_and_scheduler,
    ctc_loss,
    greedy_ctc_decode,
    learning_rate_factor,
    load_best_for_evaluation,
    micro_batch_loss_weight,
    minimum_ctc_steps,
    save_model_checkpoint,
    validate_htr_dataset,
)
from vietparadiff.models.config import HTRConfig
from vietparadiff.models.htr import HTROutput, VietnameseHTR
from vietparadiff.runtime import RuntimePrecision, create_grad_scaler


class TinyHTR(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(1, vocab_size)
            for name in ("raw", "base", "shape", "tone")
        })

    def forward(self, images: Tensor, valid_widths: Tensor) -> HTROutput:
        steps = images.shape[-1] // 4
        feature = images.mean(dim=2)
        feature = feature[:, :, : steps * 4].reshape(
            images.shape[0], 1, steps, 4
        ).mean(dim=-1).transpose(1, 2)
        lengths = (valid_widths + 3) // 4
        return HTROutput(
            self.projections["raw"](feature),
            self.projections["base"](feature),
            self.projections["shape"](feature),
            self.projections["tone"](feature),
            lengths,
        )


def _vocabulary() -> HTRVocabulary:
    mapping = {"<blank>": 0, "<unk>": 1, "a": 2, "b": 3}
    attribute = {"<blank>": 0, "<unk>": 1, "none": 2, "x": 3}
    return HTRVocabulary(mapping, mapping.copy(), attribute, attribute.copy())


def _batch(
    *,
    sample_id: str = "sample-1",
    level: str = "line",
    target: Tensor | None = None,
    width: int = 32,
) -> dict[str, object]:
    active = (
        torch.tensor([2, 3], dtype=torch.long)
        if target is None
        else target.to(torch.long)
    )
    return {
        "images": torch.randn(1, 1, 64, width),
        "valid_widths": torch.tensor([width], dtype=torch.long),
        "texts": ["ab"],
        "raw_targets": active[None],
        "base_targets": active[None].clone(),
        "shape_targets": torch.full_like(active[None], 2),
        "tone_targets": torch.full_like(active[None], 2),
        "target_lengths": torch.tensor([active.numel()], dtype=torch.long),
        "sample_ids": [sample_id],
        "sample_levels": [level],
    }


def _output(batch: dict[str, object], vocab_size: int = 8) -> HTROutput:
    images = batch["images"]
    valid_widths = batch["valid_widths"]
    assert isinstance(images, Tensor)
    assert isinstance(valid_widths, Tensor)
    steps = images.shape[-1] // 4
    lengths = (valid_widths + 3) // 4
    logits = [
        torch.randn(1, steps, vocab_size, requires_grad=True)
        for _ in range(4)
    ]
    return HTROutput(*logits, lengths)


def _config(tmp_path: Path) -> HTRTrainingConfig:
    return HTRTrainingConfig(
        seed=42,
        device="cpu",
        precision="float32",
        data=HTRDataConfig(
            train_lines=tmp_path / "train_lines.jsonl",
            train_words=tmp_path / "train_words.jsonl",
            test_lines=tmp_path / "test_lines.jsonl",
            test_words=tmp_path / "test_words.jsonl",
            vocabulary=tmp_path / "vocabulary.json",
            image_root=tmp_path,
            num_workers=0,
            line_batch_size=1,
            word_batch_size=1,
            width_bucket_size=256,
            line_batches_per_step=3,
            word_batches_per_step=1,
        ),
        htr=HTRStageConfig(epochs=2),
        optimizer=HTROptimizerConfig(
            name="adamw",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            weight_decay=0.0,
            gradient_clip_norm=1.0,
        ),
        scheduler=HTRSchedulerConfig(
            warmup_steps=1,
            minimum_learning_rate_ratio=0.1,
        ),
        loss=HTRLossConfig(),
        logging=HTRLoggingConfig(
            log_every_steps=1,
            decode_every_steps=1,
            tensorboard=False,
            wandb=False,
            wandb_mode="disabled",
            wandb_project="test",
        ),
        checkpoint=HTRCheckpointConfig(
            output_dir=tmp_path / "checkpoints",
            save_last=True,
            save_best=True,
        ),
    )


def _trainer(
    tmp_path: Path,
    config: HTRTrainingConfig | None = None,
) -> tuple[HTRTrainer, TinyHTR]:
    config = _config(tmp_path) if config is None else config
    model = TinyHTR()
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
        config.scheduler,
        total_steps=4,
    )
    runtime = RuntimePrecision(
        torch.device("cpu"), torch.float32, False, False
    )
    trainer = HTRTrainer(
        model,  # type: ignore[arg-type]
        optimizer,
        scheduler,
        create_grad_scaler(runtime),
        config,
        runtime,
        _vocabulary(),
        "vocab-hash",
        {
            "train_lines": "a",
            "train_words": "b",
            "test_lines": "c",
            "test_words": "d",
        },
        {"vocab_size": 8},
    )
    return trainer, model


def test_real_vietnamese_htr_forward_backward_is_finite() -> None:
    torch.manual_seed(3)
    model = VietnameseHTR(
        HTRConfig(
            raw_vocab_size=8,
            base_vocab_size=8,
            shape_vocab_size=8,
            tone_vocab_size=8,
            dropout=0.0,
        )
    )
    batch = _batch(width=32)
    output = model(
        batch["images"],  # type: ignore[arg-type]
        batch["valid_widths"],  # type: ignore[arg-type]
    )
    losses = compute_htr_losses(output, batch, HTRLossConfig())
    losses.total.backward()
    for head in (
        model.raw_head, model.base_head, model.shape_head, model.tone_head
    ):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()


def test_all_four_ctc_losses_have_gradients() -> None:
    batch = _batch()
    output = _output(batch)
    losses = compute_htr_losses(output, batch, HTRLossConfig())
    losses.total.backward()
    for logits in (
        output.raw_logits,
        output.base_logits,
        output.shape_logits,
        output.tone_logits,
    ):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_padded_target_tail_does_not_change_loss() -> None:
    torch.manual_seed(2)
    logits = torch.randn(1, 8, 8)
    lengths = torch.tensor([8], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)
    first = torch.tensor([[2, 3, 0]], dtype=torch.long)
    second = torch.tensor([[2, 3, 7]], dtype=torch.long)
    assert torch.equal(
        ctc_loss(logits, first, lengths, target_lengths),
        ctc_loss(logits, second, lengths, target_lengths),
    )


def test_active_blank_target_is_rejected() -> None:
    batch = _batch(target=torch.tensor([2, 0]))
    with pytest.raises(ValueError, match="blank ID 0"):
        compute_htr_losses(_output(batch), batch, HTRLossConfig())


def test_minimum_ctc_steps_counts_repeated_labels() -> None:
    assert minimum_ctc_steps(torch.tensor([2, 2, 2, 3])) == 6


def test_infeasible_ctc_reports_sample_and_head() -> None:
    batch = _batch(
        sample_id="hard-line",
        target=torch.tensor([2, 2, 2]),
        width=16,
    )
    with pytest.raises(
        ValueError,
        match=r"sample=hard-line, head=raw_targets, required=5, input_length=4",
    ):
        compute_htr_losses(_output(batch), batch, HTRLossConfig())


def test_ctc_explicitly_uses_zero_infinity_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.nn.functional.ctc_loss
    observed: dict[str, object] = {}

    def wrapped(*args: object, **kwargs: object) -> Tensor:
        observed["zero_infinity"] = kwargs.get("zero_infinity")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.nn.functional, "ctc_loss", wrapped)
    batch = _batch()
    compute_htr_losses(_output(batch), batch, HTRLossConfig())
    assert observed == {"zero_infinity": False}


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host.",
)
def test_mps_ctc_cpu_operator_path_preserves_gradients() -> None:
    logits = torch.randn(
        1,
        8,
        5,
        device="mps",
        requires_grad=True,
    )
    loss = ctc_loss(
        logits,
        torch.tensor([[2, 3]], device="mps"),
        torch.tensor([8], device="mps"),
        torch.tensor([2], device="mps"),
    )

    loss.backward()

    assert loss.device.type == "cpu"
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_greedy_decoder_collapses_repeat_and_resets_after_blank() -> None:
    tokens = torch.tensor([[0, 2, 2, 0, 2]])
    logits = torch.nn.functional.one_hot(tokens, num_classes=4).float()
    assert greedy_ctc_decode(logits, torch.tensor([5])) == [[2, 2]]


def test_training_schedule_is_three_lines_to_one_word(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    line_batches = [_batch(sample_id=f"line-{index}") for index in range(3)]
    word_batches = [_batch(sample_id="word", level="word")]
    metrics = trainer.train_epoch(line_batches, word_batches, epoch=0)
    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(metrics.checkpoint_score))


def test_partial_final_step_preserves_three_to_one_gradient_weight() -> None:
    assert micro_batch_loss_weight(
        "line", line_batch_count=3, word_batch_count=1
    ) == pytest.approx(0.25)
    assert micro_batch_loss_weight(
        "word", line_batch_count=3, word_batch_count=1
    ) == pytest.approx(0.25)
    assert micro_batch_loss_weight(
        "line", line_batch_count=1, word_batch_count=1
    ) == pytest.approx(0.75)
    assert micro_batch_loss_weight(
        "word", line_batch_count=1, word_batch_count=1
    ) == pytest.approx(0.25)


def test_width_preflight_reports_dataset_and_sample() -> None:
    class OversizedDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            target = torch.tensor([2], dtype=torch.long)
            return {
                "sample_id": "too-wide-line",
                "valid_width": 8193,
                "raw_targets": target,
                "base_targets": target.clone(),
                "shape_targets": target.clone(),
                "tone_targets": target.clone(),
            }

    with pytest.raises(
        ValueError,
        match=(
            r"dataset=train_lines, sample=too-wide-line, "
            r"valid_width=8193, input_length=2049, maximum=2048"
        ),
    ):
        validate_htr_dataset(  # type: ignore[arg-type]
            OversizedDataset(),
            "train_lines",
        )


def test_dataset_preflight_rejects_infeasible_repeated_ctc_target() -> None:
    class InfeasibleDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            target = torch.tensor([2, 3], dtype=torch.long)
            return {
                "sample_id": "repeated-shape",
                "valid_width": 16,
                "raw_targets": target,
                "base_targets": target.clone(),
                "shape_targets": torch.tensor(
                    [2, 2, 2], dtype=torch.long
                ),
                "tone_targets": target.clone(),
            }

    with pytest.raises(
        ValueError,
        match=(
            r"dataset=train_lines, sample=repeated-shape, "
            r"head=shape_targets, required=5, input_length=4"
        ),
    ):
        validate_htr_dataset(  # type: ignore[arg-type]
            InfeasibleDataset(),
            "train_lines",
        )


def test_vocabulary_builds_only_from_train_and_test_is_oov(
    tmp_path: Path,
) -> None:
    train_line = tmp_path / "line.jsonl"
    train_word = tmp_path / "word.jsonl"
    train_line.write_text(
        json.dumps({"id": "l", "text": "a", "level": "line"}) + "\n"
    )
    train_word.write_text(
        json.dumps({"id": "w", "text": "b", "level": "word"}) + "\n"
    )
    vocabulary = HTRVocabulary.build_from_manifests((train_line, train_word))
    size = len(vocabulary.raw_to_id)
    raw, *_ = vocabulary.encode("z")
    assert raw.tolist() == [1]
    assert len(vocabulary.raw_to_id) == size
    assert "z" not in vocabulary.raw_to_id


def test_optimizer_updates_parameter(tmp_path: Path) -> None:
    trainer, model = _trainer(tmp_path)
    before = model.projections["raw"].weight.detach().clone()
    lines = [_batch(sample_id=f"line-{index}") for index in range(6)]
    trainer.train_epoch(lines, [_batch(level="word")], epoch=0)
    assert not torch.equal(before, model.projections["raw"].weight)


def test_warmup_cosine_scheduler_contract() -> None:
    assert learning_rate_factor(
        0, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == 0.0
    assert learning_rate_factor(
        2, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == 1.0
    assert learning_rate_factor(
        10, warmup_steps=2, total_steps=10, minimum_ratio=0.1
    ) == pytest.approx(0.1)
    assert learning_rate_factor(
        5, warmup_steps=10, total_steps=5, minimum_ratio=0.1
    ) == pytest.approx(0.5)


def test_last_resume_restores_step_rng_and_rejects_changed_hash(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    trainer.global_step = 7
    trainer.save_epoch_checkpoints(
        next_epoch=2, train_checkpoint_score=0.5
    )
    expected_python = random.random()
    expected_torch = torch.rand(1)
    resumed, _ = _trainer(tmp_path)
    state = resumed.resume(tmp_path / "checkpoints" / "last.pt")
    assert (state.epoch, state.global_step) == (2, 7)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(1), expected_torch)
    resumed.manifest_sha256["train_lines"] = "changed"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        resumed.resume(tmp_path / "checkpoints" / "last.pt")


@pytest.mark.parametrize("changed_section", ["optimizer", "scheduler"])
def test_resume_rejects_changed_training_config_before_loading_state(
    tmp_path: Path,
    changed_section: str,
) -> None:
    source_config = _config(tmp_path)
    trainer, _ = _trainer(tmp_path, source_config)
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.3,
    )
    if changed_section == "optimizer":
        changed_config = replace(
            source_config,
            optimizer=replace(
                source_config.optimizer,
                learning_rate=2e-3,
            ),
        )
    else:
        changed_config = replace(
            source_config,
            scheduler=replace(
                source_config.scheduler,
                warmup_steps=2,
            ),
        )
    resumed, model = _trainer(tmp_path, changed_config)
    before = model.projections["raw"].weight.detach().clone()

    with pytest.raises(
        ValueError,
        match=rf"Resume config không tương thích: {changed_section}:",
    ):
        resumed.resume(tmp_path / "checkpoints" / "last.pt")

    assert torch.equal(before, model.projections["raw"].weight)


def test_best_checkpoint_is_model_only_and_loads_strict(
    tmp_path: Path,
) -> None:
    trainer, model = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(
        next_epoch=1, train_checkpoint_score=0.2
    )
    state = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(state) == {"model"}
    clone = TinyHTR()
    clone.load_state_dict(state["model"], strict=True)
    assert torch.equal(
        clone.projections["raw"].weight,
        model.projections["raw"].weight,
    )


def test_final_evaluation_loader_uses_model_api(tmp_path: Path) -> None:
    config = HTRConfig(
        raw_vocab_size=4,
        base_vocab_size=4,
        shape_vocab_size=4,
        tone_vocab_size=4,
    )
    model = VietnameseHTR(config)
    path = tmp_path / "best.pt"
    save_model_checkpoint(path, model)
    clone = VietnameseHTR(config)
    load_best_for_evaluation(clone, path, torch.device("cpu"))
    assert torch.equal(clone.raw_head.weight, model.raw_head.weight)


def test_checkpoint_selection_uses_train_line_and_word_only() -> None:
    metrics = HTREpochMetrics(line_total=2.0, word_total=4.0)
    assert metrics.checkpoint_score == 3.0


def test_line_and_word_metrics_are_reported_separately(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    line = trainer.evaluate(
        [_batch(level="line")],
        level="line",
        prediction_path=tmp_path / "line.jsonl",
    )
    word = trainer.evaluate(
        [_batch(level="word")],
        level="word",
        prediction_path=tmp_path / "word.jsonl",
    )
    assert line.raw_wer is not None and line.exact_word_accuracy is None
    assert word.raw_wer is None and word.exact_word_accuracy is not None
    assert set(line.as_dict()) != set(word.as_dict())
