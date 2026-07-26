"""Independent grayscale writer-verification encoder and ArcFace head."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18

from .config import WriterEncoderConfig


def _state_dict(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet-18 checkpoint: {path}"
        )
    state: object = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(state, Mapping) and set(state) == {"model"}:
        state = state["model"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise ValueError(
            "ResNet-18 checkpoint phải là torchvision state_dict "
            "hoặc {'model': state_dict}."
        )
    return dict(state)


class WriterStyleEncoder(nn.Module):
    """ResNet-18 writer encoder independent from the generator."""

    def __init__(
        self,
        config: WriterEncoderConfig,
        *,
        imagenet_checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        source = resnet18(weights=None)
        if imagenet_checkpoint is not None:
            source.load_state_dict(
                _state_dict(imagenet_checkpoint),
                strict=True,
            )
        rgb_stem = source.conv1
        if rgb_stem.in_channels != 3:
            raise RuntimeError("torchvision ResNet-18 stem contract thay đổi.")
        gray_stem = nn.Conv2d(
            1,
            rgb_stem.out_channels,
            kernel_size=rgb_stem.kernel_size,
            stride=rgb_stem.stride,
            padding=rgb_stem.padding,
            bias=False,
        )
        with torch.no_grad():
            gray_stem.weight.copy_(
                rgb_stem.weight.detach().mean(dim=1, keepdim=True)
            )
        source.conv1 = gray_stem
        feature_dim = source.fc.in_features
        source.fc = nn.Identity()
        self.backbone = source
        self.projection = nn.Linear(
            feature_dim,
            config.embedding_dim,
            bias=False,
        )
        self.neck = nn.BatchNorm1d(config.embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        expected = (
            images.ndim == 4
            and images.shape[1] == self.config.input_channels
            and images.shape[2] == self.config.input_height
            and 0 < images.shape[3] <= self.config.max_width
        )
        if not expected:
            raise ValueError(
                "Writer images phải có shape [B,1,128,W<=1024], "
                f"nhận {tuple(images.shape)}."
            )
        if not images.is_floating_point():
            raise TypeError("Writer images phải là floating tensor.")
        if not torch.isfinite(images).all():
            raise ValueError("Writer images phải hữu hạn.")
        if images.min() < -1.0 or images.max() > 1.0:
            raise ValueError("Writer images phải normalize vào [-1,1].")
        features = self.backbone(images)
        embeddings = self.neck(self.projection(features))
        return F.normalize(embeddings, dim=1)

    def load_checkpoint(self, path: Path) -> None:
        state = _state_dict(path)
        self.load_state_dict(state, strict=True)


class ArcFaceHead(nn.Module):
    """Writer-classification ArcFace logits for normalized embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        if embedding_dim != 256:
            raise ValueError("ArcFace embedding_dim phải bằng 256.")
        if num_classes < 2:
            raise ValueError("ArcFace cần ít nhất hai writer classes.")
        if scale != 30.0 or margin != 0.5:
            raise ValueError("ArcFace khóa scale=30 và margin=0.5.")
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != 256:
            raise ValueError("embeddings phải có shape [B,256].")
        if labels.shape != (embeddings.shape[0],):
            raise ValueError("labels phải có shape [B].")
        if labels.dtype != torch.long:
            raise TypeError("ArcFace labels phải là torch.long.")
        if labels.min() < 0 or labels.max() >= self.weight.shape[0]:
            raise ValueError("ArcFace labels vượt class range.")
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
        phi = (
            cosine * torch.cos(cosine.new_tensor(self.margin))
            - sine * torch.sin(cosine.new_tensor(self.margin))
        )
        one_hot = F.one_hot(
            labels,
            num_classes=self.weight.shape[0],
        ).to(cosine.dtype)
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


__all__ = ["ArcFaceHead", "WriterStyleEncoder"]
