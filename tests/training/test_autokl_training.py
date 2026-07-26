from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from vietparadiff.training.autokl import (
    AutoKLLogger,
    AutoKLLossConfig,
    AutoKLStageConfig,
    AutoKLTrainer,
    AutoKLTrainingConfig,
    CheckpointConfig,
    DataTrainingConfig,
    LoggingConfig,
    OptimizerTrainingConfig,
    compute_autokl_losses,
    create_optimizer_and_scheduler,
    kl_weight_at_step,
    load_best_for_evaluation,
    save_model_checkpoint,
)
from vietparadiff.models.autokl import (
    AutoKLOutput,
    DiagonalGaussianDistribution,
    HandwritingAutoKL,
)
from vietparadiff.runtime import create_grad_scaler, resolve_runtime


class TinyAutoKL(nn.Module):
    """Small real Gaussian autoencoder used to test trainer mechanics."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(
            1,
            8,
            kernel_size=8,
            stride=8,
        )
        self.decoder = nn.ConvTranspose2d(
            4,
            1,
            kernel_size=8,
            stride=8,
        )
        self.sample_flags: list[bool] = []

    def forward(
        self,
        images: Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> AutoKLOutput:
        self.sample_flags.append(sample_posterior)
        posterior = DiagonalGaussianDistribution(self.encoder(images))
        latent = (
            posterior.sample(generator)
            if sample_posterior
            else posterior.mode()
        )
        return AutoKLOutput(
            torch.tanh(self.decoder(latent)),
            posterior,
            latent,
        )


def _config(
    tmp_path: Path,
    *,
    accumulation: int = 1,
) -> AutoKLTrainingConfig:
    return AutoKLTrainingConfig(
        seed=42,
        device="cpu",
        precision="float32",
        data=DataTrainingConfig(
            train_manifest=tmp_path / "train.jsonl",
            test_manifest=tmp_path / "test.jsonl",
            image_root=tmp_path,
            num_workers=0,
            batch_size=1,
            gradient_accumulation_steps=accumulation,
        ),
        autokl=AutoKLStageConfig(
            epochs=2,
            sample_posterior_train=True,
            sample_posterior_eval=False,
        ),
        optimizer=OptimizerTrainingConfig(
            name="adamw",
            learning_rate=2e-4,
            betas=(0.9, 0.99),
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
        ),
        loss=AutoKLLossConfig(
            foreground_weight=4.0,
            edge_weight=0.1,
            kl_max_weight=1e-4,
            kl_warmup_steps=10_000,
        ),
        logging=LoggingConfig(
            log_every_steps=1,
            image_every_steps=100,
            tensorboard=False,
            wandb=False,
            wandb_project="test",
            wandb_entity=None,
            wandb_mode="disabled",
            run_name=None,
        ),
        checkpoint=CheckpointConfig(
            output_dir=tmp_path / "checkpoints",
            save_last=True,
            save_best=True,
        ),
    )


def _trainer(
    tmp_path: Path,
    config: AutoKLTrainingConfig | None = None,
) -> tuple[AutoKLTrainer, TinyAutoKL]:
    torch.manual_seed(3)
    model = TinyAutoKL()
    config = config or _config(tmp_path)
    runtime = resolve_runtime("cpu", "float32")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        config.optimizer,
    )
    trainer = AutoKLTrainer(
        model,  # type: ignore[arg-type]
        optimizer,
        scheduler,
        create_grad_scaler(runtime),
        config,
        runtime,
    )
    return trainer, model


def test_train_step_has_finite_gradients_and_updates_parameters(
    tmp_path: Path,
) -> None:
    trainer, model = _trainer(tmp_path)
    images = torch.rand(1, 1, 16, 24).mul(2.0).sub(1.0)
    before = model.encoder.weight.detach().clone()

    output, losses = trainer.train_step(images)

    assert output.reconstruction.shape == images.shape
    assert torch.isfinite(losses.total)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    trainer.optimizer_step()

    assert trainer.global_step == 1
    assert not torch.equal(before, model.encoder.weight.detach())


def test_real_handwriting_autokl_forward_backward_is_finite() -> None:
    torch.manual_seed(5)
    model = HandwritingAutoKL()
    images = torch.rand(1, 1, 16, 32).mul(2.0).sub(1.0)

    output = model(images, sample_posterior=True)
    losses = compute_autokl_losses(
        output,
        images,
        AutoKLLossConfig(4.0, 0.1, 1e-4, 10_000),
        global_step=0,
    )
    losses.total.backward()

    assert output.reconstruction.shape == images.shape
    assert output.latent.shape == (1, 4, 2, 4)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_kl_is_normalized_by_latent_element_count() -> None:
    config = AutoKLLossConfig(4.0, 0.1, 1e-4, 10_000)
    observed: list[float] = []
    for height in (2, 7):
        moments = torch.zeros(1, 8, height, 5)
        moments[:, :4] = 1.0
        posterior = DiagonalGaussianDistribution(moments)
        latent = posterior.mode()
        images = torch.zeros(1, 1, height * 8, 40)
        output = AutoKLOutput(images.clone(), posterior, latent)
        observed.append(
            float(
                compute_autokl_losses(
                    output,
                    images,
                    config,
                    global_step=0,
                ).kl
            )
        )

    assert observed[0] == pytest.approx(0.5)
    assert observed[1] == pytest.approx(0.5)


def test_kl_warmup_start_middle_and_end() -> None:
    arguments = {"max_weight": 1e-4, "warmup_steps": 10_000}
    assert kl_weight_at_step(0, **arguments) == 0.0
    assert kl_weight_at_step(5_000, **arguments) == pytest.approx(5e-5)
    assert kl_weight_at_step(10_000, **arguments) == pytest.approx(1e-4)
    assert kl_weight_at_step(20_000, **arguments) == pytest.approx(1e-4)


def test_evaluation_uses_posterior_mode(tmp_path: Path) -> None:
    trainer, model = _trainer(tmp_path)
    images = torch.zeros(1, 1, 16, 24)
    loader = [
        {
            "images": images,
            "height_bucket": 384,
            "sample_ids": ["sample"],
        }
    ]

    first = trainer.deterministic_reconstruction(images)
    second = trainer.deterministic_reconstruction(images)
    metrics = trainer.evaluate(loader)

    assert torch.equal(first, second)
    assert metrics.sample_count == 1
    assert model.sample_flags == [False, False, False]


def test_resume_restores_global_step_model_and_optimizer(
    tmp_path: Path,
) -> None:
    trainer, model = _trainer(tmp_path)
    images = torch.rand(1, 1, 16, 24).mul(2.0).sub(1.0)
    trainer.train_step(images)
    trainer.optimizer_step()
    trainer.global_step = 9
    trainer.best_score = 0.25
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.25,
    )
    expected = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    restored, restored_model = _trainer(tmp_path)
    state = restored.resume(tmp_path / "checkpoints" / "last.pt")

    assert state.epoch == 1
    assert state.global_step == 9
    assert state.best_score == pytest.approx(0.25)
    for key, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[key])
    assert restored.optimizer.state_dict()["state"]


def test_resume_rejects_training_critical_config_change(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.25,
    )
    incompatible = replace(
        _config(tmp_path),
        loss=AutoKLLossConfig(5.0, 0.1, 1e-4, 10_000),
    )
    restored, _ = _trainer(tmp_path, incompatible)

    with pytest.raises(
        ValueError,
        match="Resume config không tương thích.*loss",
    ):
        restored.resume(tmp_path / "checkpoints" / "last.pt")


def test_resume_allows_runtime_only_config_changes(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    trainer.global_step = 7
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.25,
    )
    current = _config(tmp_path)
    compatible = replace(
        current,
        autokl=replace(current.autokl, epochs=10),
        data=replace(current.data, num_workers=3),
        logging=replace(
            current.logging,
            tensorboard=True,
            wandb=True,
            wandb_mode="offline",
        ),
        checkpoint=replace(
            current.checkpoint,
            output_dir=tmp_path / "different-output",
        ),
    )
    restored, _ = _trainer(tmp_path, compatible)

    state = restored.resume(tmp_path / "checkpoints" / "last.pt")

    assert state.epoch == 1
    assert state.global_step == 7


def test_best_checkpoint_uses_only_train_epoch_score(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    assert trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.4,
    )
    best = tmp_path / "checkpoints" / "best.pt"
    first_modified = best.stat().st_mtime_ns

    # A hypothetical excellent test score is deliberately not accepted by
    # this API; only the worse train score below can affect selection.
    assert not trainer.save_epoch_checkpoints(
        next_epoch=2,
        train_checkpoint_score=0.6,
    )

    assert trainer.best_score == pytest.approx(0.4)
    assert best.stat().st_mtime_ns == first_modified


def test_best_checkpoint_is_model_only_and_loads_strictly(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.2,
    )

    state = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert set(state) == {"model"}
    model = TinyAutoKL()
    model.load_state_dict(state["model"], strict=True)


def test_best_checkpoint_is_loaded_for_final_evaluation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    source = HandwritingAutoKL()
    path = tmp_path / "best.pt"
    save_model_checkpoint(path, source)
    expected = source.encoder.input_conv.weight.detach().clone()
    restored = HandwritingAutoKL()
    with torch.no_grad():
        restored.encoder.input_conv.weight.zero_()

    load_best_for_evaluation(restored, path, torch.device("cpu"))

    assert torch.equal(restored.encoder.input_conv.weight, expected)


def test_last_checkpoint_contains_all_rng_backends(
    tmp_path: Path,
) -> None:
    trainer, _ = _trainer(tmp_path)
    trainer.save_epoch_checkpoints(
        next_epoch=1,
        train_checkpoint_score=0.2,
    )

    state = torch.load(
        tmp_path / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert set(state["rng"]) == {"python", "torch", "cuda", "mps"}
    if torch.backends.mps.is_available():
        assert isinstance(state["rng"]["mps"], Tensor)
    else:
        assert state["rng"]["mps"] is None


@pytest.mark.parametrize("height", [384, 1280])
def test_bucket_contracts_run_at_full_canvas_shape(height: int) -> None:
    torch.manual_seed(height)
    model = TinyAutoKL().eval()
    images = torch.ones(1, 1, height, 1024)

    with torch.no_grad():
        output = model(images, sample_posterior=False)

    assert output.reconstruction.shape == images.shape
    assert output.latent.shape == (1, 4, height // 8, 128)
    assert torch.isfinite(output.reconstruction).all()
    assert torch.isfinite(output.latent).all()


def test_loss_is_finite_for_nearly_white_images() -> None:
    model = TinyAutoKL()
    images = torch.full((1, 1, 16, 24), 0.9999)
    output = model(images)

    losses = compute_autokl_losses(
        output,
        images,
        AutoKLLossConfig(4.0, 0.1, 1e-4, 10_000),
        global_step=10_000,
    )

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.reconstruction)
    assert torch.isfinite(losses.edge)
    assert torch.isfinite(losses.kl)


def test_tensorboard_and_wandb_logging_paths(tmp_path: Path) -> None:
    logging = LoggingConfig(
        log_every_steps=1,
        image_every_steps=1,
        tensorboard=True,
        wandb=True,
        wandb_project="vietparadiff-test",
        wandb_entity=None,
        wandb_mode="disabled",
        run_name="logger-contract",
    )
    output_dir = tmp_path / "logging"
    logger = AutoKLLogger(logging, output_dir, {"seed": 42})

    logger.log_scalars({"train/total_loss": 1.25}, step=1)
    logger.log_images(
        {384: torch.rand(1, 1, 8, 24)},
        step=1,
    )
    logger.close()

    events = list((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert len(events) == 1
    assert events[0].stat().st_size > 0
