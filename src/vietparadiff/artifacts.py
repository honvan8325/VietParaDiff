"""Shared checkpoint-adjacent artifact schemas and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of an existing artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LatentStatistics:
    """Scalar normalization statistics bound to one AutoKL checkpoint."""

    latent_mean: float
    latent_std: float
    scaling_factor: float
    num_samples: int
    num_elements: int
    autokl_checkpoint_sha256: str

    def __post_init__(self) -> None:
        numeric = (
            self.latent_mean,
            self.latent_std,
            self.scaling_factor,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Latent statistics phải hữu hạn.")
        if self.latent_std <= 0.0 or self.scaling_factor <= 0.0:
            raise ValueError("latent_std/scaling_factor phải dương.")
        if not math.isclose(
            self.scaling_factor,
            1.0 / self.latent_std,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "scaling_factor phải bằng chính xác 1 / latent_std."
            )
        if self.num_samples <= 0 or self.num_elements <= 1:
            raise ValueError("Latent statistics cần dữ liệu không rỗng.")
        if len(self.autokl_checkpoint_sha256) != 64:
            raise ValueError("AutoKL SHA-256 không hợp lệ.")

    def normalize(self, latents: Tensor) -> Tensor:
        if not latents.is_floating_point():
            raise TypeError("latents phải có floating-point dtype.")
        return (
            latents - latents.new_tensor(self.latent_mean)
        ) * latents.new_tensor(self.scaling_factor)

    def denormalize(self, scaled_latents: Tensor) -> Tensor:
        if not scaled_latents.is_floating_point():
            raise TypeError(
                "scaled_latents phải có floating-point dtype."
            )
        return (
            scaled_latents
            / scaled_latents.new_tensor(self.scaling_factor)
            + scaled_latents.new_tensor(self.latent_mean)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "scaling_factor": self.scaling_factor,
            "num_samples": self.num_samples,
            "num_elements": self.num_elements,
            "autokl_checkpoint_sha256": self.autokl_checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class InferenceContract:
    """Hashes and diffusion settings required to reproduce one trained run."""

    schema_version: int
    prediction_type: str
    noise_schedule: str
    num_train_timesteps: int
    neutral_layout: bool
    generator_checkpoint_sha256: str
    model_config_sha256: str
    grapheme_vocabulary_sha256: str
    autokl_checkpoint_sha256: str
    latent_statistics_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Inference contract schema_version phải bằng 1.")
        if self.prediction_type != "velocity":
            raise ValueError(
                "Inference contract prediction_type phải là velocity."
            )
        if self.noise_schedule != "cosine":
            raise ValueError(
                "Inference contract noise_schedule phải là cosine."
            )
        if self.num_train_timesteps < 2:
            raise ValueError(
                "Inference contract num_train_timesteps phải >= 2."
            )
        if self.neutral_layout is not True:
            raise ValueError(
                "Base inference contract phải dùng neutral_layout=true."
            )
        hashes = (
            self.generator_checkpoint_sha256,
            self.model_config_sha256,
            self.grapheme_vocabulary_sha256,
            self.autokl_checkpoint_sha256,
            self.latent_statistics_sha256,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("Inference contract chứa SHA-256 không hợp lệ.")


def save_latent_statistics(
    path: Path,
    statistics: LatentStatistics,
) -> None:
    """Atomically save normalized latent statistics as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            statistics.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_latent_statistics(
    path: Path,
    *,
    expected_autokl_checkpoint: Path,
) -> LatentStatistics:
    """Load statistics and verify their exact AutoKL checkpoint binding."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy latent statistics: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "latent_mean",
        "latent_std",
        "scaling_factor",
        "num_samples",
        "num_elements",
        "autokl_checkpoint_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError(
            f"Latent statistics keys phải bằng {sorted(expected_keys)}."
        )
    statistics = LatentStatistics(
        latent_mean=float(payload["latent_mean"]),
        latent_std=float(payload["latent_std"]),
        scaling_factor=float(payload["scaling_factor"]),
        num_samples=int(payload["num_samples"]),
        num_elements=int(payload["num_elements"]),
        autokl_checkpoint_sha256=str(
            payload["autokl_checkpoint_sha256"]
        ),
    )
    actual_hash = sha256_file(expected_autokl_checkpoint)
    if statistics.autokl_checkpoint_sha256 != actual_hash:
        raise ValueError(
            "latent_statistics.json không thuộc AutoKL checkpoint hiện tại."
        )
    return statistics


__all__ = [
    "InferenceContract",
    "LatentStatistics",
    "load_latent_statistics",
    "save_latent_statistics",
    "sha256_file",
]
