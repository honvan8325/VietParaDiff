"""Dual-frequency one-shot style encoder với shared ConvNeXt-Tiny trunk."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from .config import StyleEncoderConfig


@dataclass(frozen=True, slots=True)
class StyleCondition:
    local_tokens: Tensor
    global_style: Tensor
    layout_scales: Tensor
    valid_feature_mask: Tensor


class DualFrequencyStyleEncoder(nn.Module):
    """Hai grayscale stems, gated fusion và một shared ConvNeXt trunk."""

    def __init__(self, config: StyleEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_pretrained_backbone and config.convnext_checkpoint is None:
            source = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            source = convnext_tiny(weights=None)
            if config.use_pretrained_backbone:
                path = config.convnext_checkpoint
                if path is None or not path.is_file():
                    raise FileNotFoundError(
                        f"Không tìm thấy ConvNeXt checkpoint: {path}"
                    )
                state: object = torch.load(
                    path, map_location="cpu", weights_only=True
                )
                if isinstance(state, Mapping) and set(state) == {"model"}:
                    state = state["model"]
                if not isinstance(state, Mapping) or not all(
                    isinstance(key, str) and isinstance(value, Tensor)
                    for key, value in state.items()
                ):
                    raise ValueError(
                        "ConvNeXt checkpoint phải là torchvision state_dict "
                        "hoặc {'model': state_dict}."
                    )
                source.load_state_dict(dict(state), strict=True)

        raw_stem = copy.deepcopy(source.features[0])
        hf_stem = copy.deepcopy(source.features[0])
        rgb_conv = source.features[0][0]
        if not isinstance(rgb_conv, nn.Conv2d) or rgb_conv.in_channels != 3:
            raise RuntimeError("torchvision ConvNeXt-Tiny stem contract đã thay đổi.")
        gray_weight = rgb_conv.weight.detach().mean(dim=1, keepdim=True)
        for stem in (raw_stem, hf_stem):
            gray_conv = nn.Conv2d(
                1,
                rgb_conv.out_channels,
                rgb_conv.kernel_size,
                rgb_conv.stride,
                rgb_conv.padding,
                bias=rgb_conv.bias is not None,
            )
            with torch.no_grad():
                gray_conv.weight.copy_(gray_weight)
                if rgb_conv.bias is not None:
                    gray_conv.bias.copy_(rgb_conv.bias)
            stem[0] = gray_conv
        self.raw_stem = raw_stem
        self.hf_stem = hf_stem
        self.shared_trunk = source.features[1:]
        self.fusion_gate = nn.Conv2d(
            2 * config.stem_channels, config.stem_channels, 1
        )
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -4.0)
        self.local_queries = nn.Parameter(
            torch.randn(config.local_token_count, config.feature_dim) * 0.02
        )
        self.local_attention = nn.MultiheadAttention(
            config.feature_dim,
            config.local_attention_heads,
            batch_first=True,
        )
        self.local_norm = nn.LayerNorm(config.feature_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * config.feature_dim, config.feature_dim),
            nn.SiLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
        )
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel, persistent=True)

    def forward(
        self,
        reference_images: Tensor,
        reference_valid_mask: Tensor,
    ) -> StyleCondition:
        if (
            reference_images.ndim != 4
            or reference_images.shape[1] != 1
            or reference_images.shape[2] != self.config.reference_height
        ):
            raise ValueError(
                "reference_images phải có shape [B,1,256,W], "
                f"nhận {tuple(reference_images.shape)}."
            )
        if (
            reference_images.shape[-1] <= 0
            or reference_images.shape[-1] > self.config.max_reference_width
            or reference_images.shape[-1] % 32
        ):
            raise ValueError("Reference width phải dương, <=1536 và chia hết cho 32.")
        if reference_valid_mask.shape != reference_images.shape:
            raise ValueError("reference_valid_mask phải cùng shape với reference_images.")
        if reference_valid_mask.dtype != torch.bool:
            raise TypeError("reference_valid_mask phải có dtype torch.bool.")
        if not reference_images.is_floating_point():
            raise TypeError("reference_images phải có floating-point dtype.")
        if not torch.isfinite(reference_images).all():
            raise ValueError("reference_images phải chứa toàn giá trị hữu hạn.")
        if reference_images.min() < -1.0 or reference_images.max() > 1.0:
            raise ValueError("reference_images phải được normalize vào [-1, 1].")
        if not reference_valid_mask.flatten(1).any(dim=1).all():
            raise ValueError("Mỗi reference phải có ít nhất một valid pixel.")

        images = torch.where(
            reference_valid_mask,
            reference_images,
            torch.ones_like(reference_images),
        )
        laplacian = F.conv2d(
            images, self.laplacian_kernel.to(images.dtype), padding=1
        ).abs()
        threshold = self.config.foreground_threshold * 2.0 - 1.0
        foreground = (images < threshold) & reference_valid_mask
        high_frequency = laplacian * foreground.to(images.dtype)
        count = foreground.float().sum((1, 2, 3), keepdim=True).clamp_min(1.0)
        rms = torch.sqrt(
            high_frequency.float().square().sum((1, 2, 3), keepdim=True)
            / count
        ).clamp_min(1e-6)
        high_frequency = high_frequency / rms.to(high_frequency.dtype)

        raw = self.raw_stem(images)
        high = self.hf_stem(high_frequency)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((raw, high), dim=1)))
        features = self.shared_trunk(raw + gate * high)
        valid = (
            F.adaptive_max_pool2d(
                reference_valid_mask.float(), features.shape[-2:]
            )
            > 0
        )
        features = features * valid.to(features.dtype)
        batch = features.shape[0]
        tokens = features.flatten(2).transpose(1, 2)
        mask = valid.flatten(1)
        queries = self.local_queries[None].expand(batch, -1, -1)
        local, _ = self.local_attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        local = self.local_norm(local + queries)

        float_features, weights = features.float(), valid.float()
        count = weights.sum((2, 3), keepdim=True).clamp_min(1.0)
        mean = (float_features * weights).sum((2, 3), keepdim=True) / count
        variance = (
            (float_features - mean).square() * weights
        ).sum((2, 3), keepdim=True) / count
        global_style = self.global_mlp(
            torch.cat(
                (mean.flatten(1), torch.sqrt(variance + 1e-6).flatten(1)),
                dim=1,
            ).to(features.dtype)
        )
        layout_scales = torch.ones(
            global_style.shape[0],
            3,
            dtype=global_style.dtype,
            device=global_style.device,
        )
        return StyleCondition(local, global_style, layout_scales, valid)

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy style checkpoint: {path}")
        state: object = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and set(state) == {"model"}:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise ValueError("Style checkpoint không phải state_dict hợp lệ.")
        self.load_state_dict(dict(state), strict=True)
