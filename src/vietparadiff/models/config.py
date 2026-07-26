"""Cấu hình kiến trúc cho VietParaDiff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class AutoKLConfig:
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    latent_channels: int = 4
    downsample_factor: int = 8
    group_norm_groups: int = 8
    attention_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        channels = tuple(
            self.base_channels * multiplier
            for multiplier in self.channel_multipliers
        )
        if self.input_channels != 1 or self.output_channels != 1:
            raise ValueError("AutoKL chỉ hỗ trợ ảnh grayscale 1 channel.")
        if self.base_channels != 32:
            raise ValueError("AutoKL base_channels phải bằng 32.")
        if self.channel_multipliers != (1, 2, 4, 8):
            raise ValueError("AutoKL multipliers phải là (1, 2, 4, 8).")
        if self.num_res_blocks != 2:
            raise ValueError("AutoKL phải có 2 ResBlocks mỗi level.")
        if self.latent_channels != 4 or self.downsample_factor != 8:
            raise ValueError("AutoKL phải dùng latent 4 channel và downsample 8.")
        if any(channel % self.group_norm_groups for channel in channels):
            raise ValueError("Mọi AutoKL channel phải chia hết cho GroupNorm.")
        if channels[-1] % self.attention_heads:
            raise ValueError("Bottleneck phải chia hết cho attention_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("AutoKL dropout phải nằm trong [0, 1).")


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    base_vocab_size: int
    shape_vocab_size: int
    tone_vocab_size: int
    case_vocab_size: int
    class_vocab_size: int
    component_embedding_dim: int = 128
    model_dim: int = 512
    context_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_graphemes: int = 768
    max_lines: int = 8
    max_position_in_line: int = 128
    canvas_width: int = 1024
    height_buckets: tuple[int, ...] = (
        384,
        512,
        640,
        768,
        896,
        1024,
        1280,
    )

    def __post_init__(self) -> None:
        sizes = (
            self.base_vocab_size,
            self.shape_vocab_size,
            self.tone_vocab_size,
            self.case_vocab_size,
            self.class_vocab_size,
        )
        if any(size < 2 for size in sizes):
            raise ValueError("Mỗi vocabulary phải có ít nhất 2 token.")
        if (
            self.component_embedding_dim != 128
            or self.model_dim != 512
            or self.context_dim != 768
        ):
            raise ValueError("Text dims phải là embedding=128, model=512, context=768.")
        if self.num_layers != 6 or self.num_heads != 8 or self.ffn_dim != 2048:
            raise ValueError("Text Transformer phải là 6 layers, 8 heads, FFN 2048.")
        if self.max_graphemes != 768 or self.max_lines != 8:
            raise ValueError("Text phải hỗ trợ 768 graphemes và tối đa 8 dòng.")
        if self.max_position_in_line != 128 or self.canvas_width != 1024:
            raise ValueError("Position-in-line phải là 128 và canvas width là 1024.")
        if self.height_buckets != (384, 512, 640, 768, 896, 1024, 1280):
            raise ValueError("Paragraph height buckets không đúng method.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Text dropout phải nằm trong [0, 1).")


@dataclass(frozen=True, slots=True)
class StyleEncoderConfig:
    reference_height: int = 256
    max_reference_width: int = 1536
    stem_channels: int = 96
    feature_dim: int = 768
    local_token_count: int = 16
    local_attention_heads: int = 12
    foreground_threshold: float = 0.95
    use_pretrained_backbone: bool = True
    convnext_checkpoint: Path | None = None
    use_high_frequency_style: bool = True

    def __post_init__(self) -> None:
        if self.reference_height != 256 or self.max_reference_width != 1536:
            raise ValueError("Reference phải cao 256 và rộng tối đa 1536.")
        if self.stem_channels != 96 or self.feature_dim != 768:
            raise ValueError("Style encoder phải dùng stem 96 và feature 768.")
        if self.local_token_count != 16 or self.local_attention_heads != 12:
            raise ValueError("Style encoder phải dùng 16 queries và 12 heads.")
        if not 0.0 < self.foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold phải nằm trong (0, 1].")
        if self.convnext_checkpoint is not None and not isinstance(
            self.convnext_checkpoint, Path
        ):
            raise TypeError("convnext_checkpoint phải là pathlib.Path hoặc None.")
        if not isinstance(self.use_high_frequency_style, bool):
            raise TypeError("use_high_frequency_style phải là bool.")


@dataclass(frozen=True, slots=True)
class ParagraphUNetConfig:
    latent_channels: int = 4
    channels: tuple[int, ...] = (128, 256, 512, 768)
    num_res_blocks: int = 2
    context_dim: int = 768
    time_embedding_dim: int = 1024
    attention_heads: int = 8
    group_norm_groups: int = 32
    dropout: float = 0.0
    position_base_height: int = 160
    position_base_width: int = 128
    max_lines: int = 8
    harmonizer_dim: int = 512
    harmonizer_layers: int = 2
    harmonizer_heads: int = 8
    prediction_type: Literal["v"] = "v"
    use_shape_condition: bool = True
    use_tone_condition: bool = True
    use_local_style_tokens: bool = True
    use_harmonizer: bool = True

    def __post_init__(self) -> None:
        if self.latent_channels != 4 or self.channels != (128, 256, 512, 768):
            raise ValueError("U-Net phải dùng latent=4 và channels=(128,256,512,768).")
        if self.num_res_blocks != 2 or self.context_dim != 768:
            raise ValueError("U-Net phải có 2 ResBlocks và context 768.")
        if self.time_embedding_dim != 1024 or self.attention_heads != 8:
            raise ValueError("U-Net phải dùng time dim 1024 và 8 heads.")
        if any(channel % self.group_norm_groups for channel in self.channels):
            raise ValueError("Mọi U-Net channel phải chia hết cho GroupNorm.")
        if self.max_lines != 8:
            raise ValueError("U-Net chỉ hỗ trợ tối đa 8 dòng.")
        if (
            self.harmonizer_dim,
            self.harmonizer_layers,
            self.harmonizer_heads,
        ) != (512, 2, 8):
            raise ValueError("Harmonizer phải dùng dim=512, 2 layers, 8 heads.")
        if self.prediction_type != "v":
            raise ValueError("VietParaDiff phải dùng v-prediction.")
        flags = (
            self.use_shape_condition,
            self.use_tone_condition,
            self.use_local_style_tokens,
            self.use_harmonizer,
        )
        if not all(isinstance(flag, bool) for flag in flags):
            raise TypeError("Mọi U-Net behavior flag phải là bool.")
        if self.use_tone_condition and not self.use_shape_condition:
            raise ValueError(
                "Tone conditioning yêu cầu shape conditioning được bật."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("U-Net dropout phải nằm trong [0, 1).")


@dataclass(frozen=True, slots=True)
class HTRConfig:
    raw_vocab_size: int
    base_vocab_size: int
    shape_vocab_size: int
    tone_vocab_size: int
    input_channels: int = 1
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    width_downsample_factor: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if any(
            size < 2
            for size in (
                self.raw_vocab_size,
                self.base_vocab_size,
                self.shape_vocab_size,
                self.tone_vocab_size,
            )
        ):
            raise ValueError("Mỗi HTR vocabulary phải có ít nhất 2 token.")
        if self.input_channels != 1:
            raise ValueError("HTR chỉ hỗ trợ ảnh grayscale.")
        if (self.model_dim, self.num_layers, self.num_heads, self.ffn_dim) != (
            256,
            4,
            4,
            1024,
        ):
            raise ValueError("HTR phải dùng dim=256, 4 layers, 4 heads, FFN=1024.")
        if self.width_downsample_factor != 4:
            raise ValueError("HTR phải downsample width theo hệ số 4.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("HTR dropout phải nằm trong [0, 1).")


@dataclass(frozen=True, slots=True)
class WriterEncoderConfig:
    input_channels: int = 1
    embedding_dim: int = 256
    input_height: int = 128
    max_width: int = 1024
    backbone: Literal["resnet18"] = "resnet18"

    def __post_init__(self) -> None:
        if self.input_channels != 1:
            raise ValueError("Writer encoder chỉ nhận grayscale 1 channel.")
        if self.embedding_dim != 256:
            raise ValueError("Writer embedding dimension phải bằng 256.")
        if self.input_height != 128 or self.max_width != 1024:
            raise ValueError(
                "Writer metric input phải cao 128, rộng tối đa 1024."
            )
        if self.backbone != "resnet18":
            raise ValueError("Writer metric backbone phải là resnet18.")


@dataclass(frozen=True, slots=True)
class VietParaDiffConfig:
    autokl: AutoKLConfig
    text: TextEncoderConfig
    style: StyleEncoderConfig
    unet: ParagraphUNetConfig

    def __post_init__(self) -> None:
        if self.autokl.latent_channels != self.unet.latent_channels:
            raise ValueError("AutoKL và U-Net phải dùng cùng latent channels.")
        if not (
            self.text.context_dim
            == self.style.feature_dim
            == self.unet.context_dim
        ):
            raise ValueError("Text, style và U-Net context phải cùng dimension.")
        if self.text.max_lines != self.unet.max_lines:
            raise ValueError("Text formatter và U-Net phải dùng cùng max_lines.")
