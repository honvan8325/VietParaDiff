"""Scale-separated paragraph latent diffusion model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ParagraphUNetConfig, VietParaDiffConfig
from .style import DualFrequencyStyleEncoder, StyleCondition
from .text import FactorizedGraphemeEncoder, GraphemeBatch, GraphemeCondition


@dataclass(frozen=True, slots=True)
class VietParaDiffInput:
    noisy_latents: Tensor
    timesteps: Tensor
    graphemes: GraphemeBatch
    style_condition: StyleCondition
    line_slot_masks: Tensor


@dataclass(frozen=True, slots=True)
class VietParaDiffOutput:
    predicted_velocity: Tensor
    grapheme_condition: GraphemeCondition
    style_condition: StyleCondition
    line_tokens: Tensor
    diagnostics: dict[str, Tensor]


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(128, device=timesteps.device, dtype=torch.float32)
            / 128
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class Adaptive2DPositionEmbedding(nn.Module):
    def __init__(self, channels: int, base_height: int, base_width: int) -> None:
        super().__init__()
        self.rows = nn.Parameter(torch.randn(1, channels, base_height, 1) * 0.02)
        self.columns = nn.Parameter(torch.randn(1, channels, 1, base_width) * 0.02)

    def forward(self, features: Tensor) -> Tensor:
        height, width = features.shape[-2:]
        rows = F.interpolate(
            self.rows, size=(height, 1), mode="bilinear", align_corners=False
        )
        columns = F.interpolate(
            self.columns, size=(1, width), mode="bilinear", align_corners=False
        )
        return features + (rows + columns).to(features.dtype)


class UNetResBlock(nn.Module):
    """Time-conditioned ResBlock với global-style AdaGN/FiLM."""

    def __init__(self, in_channels: int, out_channels: int, config: ParagraphUNetConfig) -> None:
        super().__init__()
        groups = config.group_norm_groups
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(config.time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.style_film = nn.Linear(config.context_dim, 2 * out_channels)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        nn.init.zeros_(self.style_film.weight)
        nn.init.zeros_(self.style_film.bias)

    def forward(
        self, features: Tensor, timestep: Tensor, global_style: Tensor
    ) -> tuple[Tensor, Tensor]:
        residual = self.skip(features)
        features = self.conv1(F.silu(self.norm1(features)))
        features = features + self.time_projection(timestep)[:, :, None, None].to(features.dtype)
        scale, shift = self.style_film(global_style).to(features.dtype).chunk(2, dim=1)
        features = self.norm2(features)
        features = features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        features = self.conv2(self.dropout(F.silu(features)))
        norm = torch.cat((scale, shift), dim=1).float().square().mean().sqrt()
        return residual + features, norm


class SpatialTransformer(nn.Module):
    """Row, axial hoặc global self-attention rồi content/style cross-attention."""

    def __init__(
        self,
        channels: int,
        config: ParagraphUNetConfig,
        mode: Literal["row", "axial", "global"],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.position = Adaptive2DPositionEmbedding(
            channels, config.position_base_height, config.position_base_width
        )
        if mode == "global":
            self.global_norm: nn.LayerNorm | None = nn.LayerNorm(channels)
            self.global_attention: nn.MultiheadAttention | None = nn.MultiheadAttention(
                channels, config.attention_heads, dropout=config.dropout, batch_first=True
            )
        else:
            self.global_norm = None
            self.global_attention = None
        if mode in {"row", "axial"}:
            self.row_norm: nn.LayerNorm | None = nn.LayerNorm(channels)
            self.row_attention: nn.MultiheadAttention | None = nn.MultiheadAttention(
                channels, config.attention_heads, dropout=config.dropout, batch_first=True
            )
        else:
            self.row_norm = None
            self.row_attention = None
        if mode == "axial":
            self.column_norm: nn.LayerNorm | None = nn.LayerNorm(channels)
            self.column_attention: nn.MultiheadAttention | None = nn.MultiheadAttention(
                channels, config.attention_heads, dropout=config.dropout, batch_first=True
            )
        else:
            self.column_norm = None
            self.column_attention = None
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels,
            config.attention_heads,
            dropout=config.dropout,
            kdim=config.context_dim,
            vdim=config.context_dim,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, features: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        features = self.position(features)
        batch, channels, height, width = features.shape
        if self.mode == "global":
            if self.global_norm is None or self.global_attention is None:
                raise RuntimeError("Global attention modules bị thiếu.")
            tokens = features.flatten(2).transpose(1, 2)
            normalized = self.global_norm(tokens)
            attended, _ = self.global_attention(
                normalized, normalized, normalized, need_weights=False
            )
            features = (tokens + attended).transpose(1, 2).reshape(
                batch, channels, height, width
            )
        else:
            if self.row_norm is None or self.row_attention is None:
                raise RuntimeError("Row attention modules bị thiếu.")
            rows = features.permute(0, 2, 3, 1).reshape(
                batch * height, width, channels
            )
            normalized = self.row_norm(rows)
            attended, _ = self.row_attention(
                normalized, normalized, normalized, need_weights=False
            )
            features = (rows + attended).reshape(
                batch, height, width, channels
            ).permute(0, 3, 1, 2)
            if self.mode == "axial":
                if self.column_norm is None or self.column_attention is None:
                    raise RuntimeError("Column attention modules bị thiếu.")
                columns = features.permute(0, 3, 2, 1).reshape(
                    batch * width, height, channels
                )
                normalized = self.column_norm(columns)
                attended, _ = self.column_attention(
                    normalized, normalized, normalized, need_weights=False
                )
                features = (columns + attended).reshape(
                    batch, width, height, channels
                ).permute(0, 3, 2, 1)
        tokens = features.flatten(2).transpose(1, 2)
        attended, _ = self.cross_attention(
            self.cross_norm(tokens),
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class GraphemeResidualAdapter(nn.Module):
    """Zero-projection shape/tone adapter với bounded style gain."""

    def __init__(self, channels: int, config: ParagraphUNetConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            config.attention_heads,
            dropout=config.dropout,
            kdim=config.context_dim,
            vdim=config.context_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(channels, channels)
        self.style_gain = nn.Linear(config.context_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        features: Tensor,
        context: Tensor,
        mask: Tensor,
        global_style: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = features.shape
        queries = features.flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.query_norm(queries),
            context,
            context,
            key_padding_mask=~mask,
            need_weights=False,
        )
        residual = self.output_projection(attended)
        gain = 1.0 + 0.1 * torch.tanh(self.style_gain(global_style))
        residual = residual * gain[:, None]
        residual = residual.transpose(1, 2).reshape(batch, channels, height, width)
        return residual, residual.float().square().mean().sqrt()


class ConditionedLevel(nn.Module):
    def __init__(
        self,
        channels: int,
        config: ParagraphUNetConfig,
        *,
        attention_mode: Literal["row", "axial", "global"],
        use_shape: bool,
        use_tone: bool,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [UNetResBlock(channels, channels, config) for _ in range(2)]
        )
        self.main_attention = SpatialTransformer(channels, config, attention_mode)
        self.shape_adapter = GraphemeResidualAdapter(channels, config) if use_shape else None
        self.tone_adapter = GraphemeResidualAdapter(channels, config) if use_tone else None

    def forward(
        self,
        features: Tensor,
        timestep: Tensor,
        grapheme: GraphemeCondition,
        style: StyleCondition,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        film_norms: list[Tensor] = []
        for block in self.resblocks:
            features, norm = block(features, timestep, style.global_style)
            film_norms.append(norm)
        style_mask = torch.ones(
            features.shape[0],
            style.local_tokens.shape[1],
            dtype=torch.bool,
            device=features.device,
        )
        main_context = torch.cat((grapheme.base_context, style.local_tokens), dim=1)
        main_mask = torch.cat((grapheme.attention_mask, style_mask), dim=1)
        features = self.main_attention(features, main_context, main_mask)
        shape_norms: list[Tensor] = []
        if self.shape_adapter is not None:
            residual, norm = self.shape_adapter(
                features,
                grapheme.shape_context,
                grapheme.attention_mask,
                style.global_style,
            )
            features = features + residual
            shape_norms.append(norm)
        tone_norms: list[Tensor] = []
        if self.tone_adapter is not None:
            residual, norm = self.tone_adapter(
                features,
                grapheme.tone_context,
                grapheme.attention_mask,
                style.global_style,
            )
            features = features + residual
            tone_norms.append(norm)
        return features, shape_norms, tone_norms, film_norms


class InterLineStyleHarmonizer(nn.Module):
    """Transformer chỉ trao đổi masked line statistics ở deepest feature."""

    def __init__(self, channels: int, config: ParagraphUNetConfig) -> None:
        super().__init__()
        self.max_lines = config.max_lines
        self.statistics_projection = nn.Linear(2 * channels, config.harmonizer_dim)
        self.style_projection = nn.Linear(config.context_dim, config.harmonizer_dim)
        self.line_position = nn.Parameter(
            torch.randn(config.max_lines, config.harmonizer_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            config.harmonizer_dim,
            config.harmonizer_heads,
            2048,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.harmonizer_layers, enable_nested_tensor=False
        )
        self.output_projection = nn.Linear(config.harmonizer_dim, 2 * channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, features: Tensor, line_masks: Tensor, global_style: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        masks = F.interpolate(
            line_masks.float(), size=features.shape[-2:], mode="nearest"
        )
        areas = masks.sum((-2, -1))
        active = areas > 1e-6
        selected = torch.nonzero(active.sum(dim=1) > 1, as_tuple=False).flatten()
        all_tokens = features.new_zeros(
            features.shape[0], self.max_lines, self.statistics_projection.out_features
        )
        if selected.numel() == 0:
            return features, all_tokens, features.sum() * 0.0
        selected_features = features.index_select(0, selected)
        selected_masks = masks.index_select(0, selected)
        selected_active = active.index_select(0, selected)
        selected_areas = selected_masks.sum((-2, -1)).clamp_min(1e-6)
        values = selected_features.float()
        weights = selected_masks.float()
        means = torch.einsum("bchw,blhw->blc", values, weights)
        means = means / selected_areas[:, :, None]
        centered = values[:, None] - means[:, :, :, None, None]
        variances = (
            centered.square() * weights[:, :, None]
        ).sum((-2, -1)) / selected_areas[:, :, None]
        tokens = self.statistics_projection(
            torch.cat((means, torch.sqrt(variances + 1e-6)), dim=-1).to(
                self.statistics_projection.weight.dtype
            )
        )
        tokens = (
            tokens
            + self.style_projection(global_style.index_select(0, selected))[:, None]
            + self.line_position[None].to(tokens.dtype)
        )
        tokens = tokens.masked_fill(~selected_active[:, :, None], 0.0)
        tokens = self.transformer(tokens, src_key_padding_mask=~selected_active)
        tokens = tokens.masked_fill(~selected_active[:, :, None], 0.0)
        scale, shift = self.output_projection(tokens).chunk(2, dim=-1)
        scale, shift = scale.to(features.dtype), shift.to(features.dtype)
        delta = (
            scale[:, :, :, None, None] * selected_features[:, None]
            + shift[:, :, :, None, None]
        ) * selected_masks[:, :, None].to(features.dtype)
        selected_delta = delta.sum(dim=1)
        all_delta = torch.zeros_like(features).index_copy(0, selected, selected_delta)
        all_tokens = all_tokens.index_copy(0, selected, tokens.to(all_tokens.dtype))
        return features + all_delta, all_tokens, all_delta.float().square().mean().sqrt()


class ParagraphUNet(nn.Module):
    """Bốn-level direct paragraph latent U-Net dự đoán velocity."""

    def __init__(self, config: ParagraphUNetConfig) -> None:
        super().__init__()
        self.config = config
        high, medium, low, deep = config.channels
        self.input_conv = nn.Conv2d(4, high, 3, padding=1)
        self.timestep_embedding = TimestepEmbedding(1024)
        self.encoder_high = ConditionedLevel(high, config, attention_mode="row", use_shape=True, use_tone=True)
        self.down_high = nn.Conv2d(high, medium, 3, stride=2, padding=1)
        self.encoder_medium = ConditionedLevel(medium, config, attention_mode="axial", use_shape=True, use_tone=False)
        self.down_medium = nn.Conv2d(medium, low, 3, stride=2, padding=1)
        self.encoder_low = ConditionedLevel(low, config, attention_mode="axial", use_shape=False, use_tone=False)
        self.down_low = nn.Conv2d(low, deep, 3, stride=2, padding=1)
        self.encoder_deep = ConditionedLevel(deep, config, attention_mode="global", use_shape=False, use_tone=False)
        self.harmonizer = InterLineStyleHarmonizer(deep, config)
        self.middle1 = UNetResBlock(deep, deep, config)
        self.middle_attention = SpatialTransformer(deep, config, "global")
        self.middle2 = UNetResBlock(deep, deep, config)
        self.deep_skip = nn.Conv2d(2 * deep, deep, 1)
        self.decoder_deep = ConditionedLevel(deep, config, attention_mode="global", use_shape=False, use_tone=False)
        self.up_deep = nn.Conv2d(deep, low, 3, padding=1)
        self.low_skip = nn.Conv2d(2 * low, low, 1)
        self.decoder_low = ConditionedLevel(low, config, attention_mode="axial", use_shape=False, use_tone=False)
        self.up_low = nn.Conv2d(low, medium, 3, padding=1)
        self.medium_skip = nn.Conv2d(2 * medium, medium, 1)
        self.decoder_medium = ConditionedLevel(medium, config, attention_mode="axial", use_shape=True, use_tone=False)
        self.up_medium = nn.Conv2d(medium, high, 3, padding=1)
        self.high_skip = nn.Conv2d(2 * high, high, 1)
        self.decoder_high = ConditionedLevel(high, config, attention_mode="row", use_shape=True, use_tone=True)
        self.output_norm = nn.GroupNorm(config.group_norm_groups, high)
        self.output_conv = nn.Conv2d(high, 4, 3, padding=1)

    def forward(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        grapheme: GraphemeCondition,
        style: StyleCondition,
        line_slot_masks: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if noisy_latents.ndim != 4 or noisy_latents.shape[1] != 4:
            raise ValueError(f"noisy_latents phải có shape [B,4,H,W], nhận {tuple(noisy_latents.shape)}.")
        if not noisy_latents.is_floating_point() or not torch.isfinite(noisy_latents).all():
            raise ValueError("noisy_latents phải là floating-point hữu hạn.")
        batch, _, height, width = noisy_latents.shape
        if height % 8 or width % 8:
            raise ValueError("Latent H và W phải chia hết cho 8.")
        if width != 128:
            raise ValueError(f"Latent width phải bằng 128, nhận {width}.")
        if timesteps.shape != (batch,):
            raise ValueError(f"timesteps phải có shape [{batch}].")
        if timesteps.device != noisy_latents.device:
            raise ValueError("timesteps và noisy_latents phải cùng device.")
        expected_masks = (batch, self.config.max_lines, height, width)
        if line_slot_masks.shape != expected_masks:
            raise ValueError(
                f"line_slot_masks phải có shape {expected_masks}, "
                f"nhận {tuple(line_slot_masks.shape)}."
            )
        masks = line_slot_masks.float()
        if not torch.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
            raise ValueError("line_slot_masks phải hữu hạn trong [0,1].")
        if (masks.sum(dim=1) > 1.0001).any():
            raise ValueError("Các line-slot masks không được overlap.")
        if not (masks.sum((-2, -1)) > 0).any(dim=1).all():
            raise ValueError("Mỗi paragraph phải có ít nhất một active line mask.")

        timestep = self.timestep_embedding(timesteps)
        shape_norms: list[Tensor] = []
        tone_norms: list[Tensor] = []
        film_norms: list[Tensor] = []

        high, shape, tone, film = self.encoder_high(
            self.input_conv(noisy_latents), timestep, grapheme, style
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)
        high_skip = high

        medium, shape, tone, film = self.encoder_medium(
            self.down_high(high), timestep, grapheme, style
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)
        medium_skip = medium

        low, shape, tone, film = self.encoder_low(
            self.down_medium(medium), timestep, grapheme, style
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)
        low_skip = low

        deep, shape, tone, film = self.encoder_deep(
            self.down_low(low), timestep, grapheme, style
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)
        deep, line_tokens, harmonizer_norm = self.harmonizer(deep, masks, style.global_style)
        deep_skip = deep

        deep, norm = self.middle1(deep, timestep, style.global_style)
        film_norms.append(norm)
        style_mask = torch.ones(
            batch, style.local_tokens.shape[1], dtype=torch.bool, device=deep.device
        )
        context = torch.cat((grapheme.base_context, style.local_tokens), dim=1)
        context_mask = torch.cat((grapheme.attention_mask, style_mask), dim=1)
        deep = self.middle_attention(deep, context, context_mask)
        deep, norm = self.middle2(deep, timestep, style.global_style)
        film_norms.append(norm)

        deep, shape, tone, film = self.decoder_deep(
            self.deep_skip(torch.cat((deep, deep_skip), dim=1)),
            timestep,
            grapheme,
            style,
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)

        low = self.up_deep(F.interpolate(deep, scale_factor=2.0, mode="nearest"))
        low, shape, tone, film = self.decoder_low(
            self.low_skip(torch.cat((low, low_skip), dim=1)),
            timestep,
            grapheme,
            style,
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)

        medium = self.up_low(F.interpolate(low, scale_factor=2.0, mode="nearest"))
        medium, shape, tone, film = self.decoder_medium(
            self.medium_skip(torch.cat((medium, medium_skip), dim=1)),
            timestep,
            grapheme,
            style,
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)

        high = self.up_medium(F.interpolate(medium, scale_factor=2.0, mode="nearest"))
        high, shape, tone, film = self.decoder_high(
            self.high_skip(torch.cat((high, high_skip), dim=1)),
            timestep,
            grapheme,
            style,
        )
        shape_norms.extend(shape)
        tone_norms.extend(tone)
        film_norms.extend(film)

        velocity = self.output_conv(F.silu(self.output_norm(high)))
        diagnostics = {
            "shape_residual_norm": torch.stack(shape_norms).mean(),
            "tone_residual_norm": torch.stack(tone_norms).mean(),
            "style_film_norm": torch.stack(film_norms).mean(),
            "harmonizer_delta_norm": harmonizer_norm,
        }
        return velocity, line_tokens, diagnostics


class VietParaDiff(nn.Module):
    """Generator inference graph; không chứa AutoKL decoder, sampler hoặc HTR.

    Reference style phải được tạo trước bằng :meth:`encode_reference`. Caller
    dùng ``style.layout_scales`` để chạy ``ParagraphFormatter``, rồi truyền
    chính ``StyleCondition`` đó cùng grapheme IDs và line-slot masks vào
    :meth:`forward`. Contract hai bước này tránh encode reference hai lần và
    đảm bảo formatter thực sự được reference-calibrated.
    """

    def __init__(self, config: VietParaDiffConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = FactorizedGraphemeEncoder(config.text)
        self.style_encoder = DualFrequencyStyleEncoder(config.style)
        self.unet = ParagraphUNet(config.unet)

    def encode_reference(
        self,
        reference_images: Tensor,
        reference_valid_mask: Tensor,
    ) -> StyleCondition:
        """Encode reference một lần trước discrete paragraph formatting."""
        return self.style_encoder(reference_images, reference_valid_mask)

    def forward(self, batch: VietParaDiffInput) -> VietParaDiffOutput:
        if not isinstance(batch, VietParaDiffInput):
            raise TypeError("batch phải là VietParaDiffInput.")
        batch_size = batch.noisy_latents.shape[0]
        if batch.graphemes.base_ids.shape[0] != batch_size:
            raise ValueError("graphemes và noisy_latents phải cùng batch size.")
        if not isinstance(batch.style_condition, StyleCondition):
            raise TypeError("style_condition phải là StyleCondition.")
        if (
            batch.style_condition.local_tokens.shape[0] != batch_size
            or batch.style_condition.global_style.shape[0] != batch_size
        ):
            raise ValueError("style_condition và noisy_latents phải cùng batch size.")
        grapheme = self.text_encoder(batch.graphemes)
        style = batch.style_condition
        velocity, line_tokens, diagnostics = self.unet(
            batch.noisy_latents,
            batch.timesteps,
            grapheme,
            style,
            batch.line_slot_masks,
        )
        return VietParaDiffOutput(
            velocity, grapheme, style, line_tokens, diagnostics
        )

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy VietParaDiff checkpoint: {path}")
        state: object = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and set(state) == {"model"}:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint phải là state_dict hoặc {'model': state_dict}.")
        self.load_state_dict(dict(state), strict=True)
