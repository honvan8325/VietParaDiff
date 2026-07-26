"""Unicode factorization, paragraph formatting và grapheme encoding."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import TextEncoderConfig


SHAPE_MARKS = {"\u0306": "breve", "\u0302": "circumflex", "\u031b": "horn"}
TONE_MARKS = {
    "\u0301": "acute",
    "\u0300": "grave",
    "\u0309": "hook_above",
    "\u0303": "tilde",
    "\u0323": "dot_below",
}


@dataclass(frozen=True, slots=True)
class FactorizedGrapheme:
    surface: str
    base: str
    shape: str
    tone: str
    case: str
    class_name: str


class VietnameseGraphemeFactorizer:
    def factorize(self, text: str) -> list[FactorizedGrapheme]:
        if not isinstance(text, str):
            raise TypeError("text phải là str.")
        if not text:
            raise ValueError("text không được rỗng.")
        decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
        clusters: list[str] = []
        current = ""
        for character in decomposed:
            if character == "\n":
                if current:
                    clusters.append(unicodedata.normalize("NFC", current))
                    current = ""
                clusters.append("\n")
            elif unicodedata.combining(character) == 0:
                if current:
                    clusters.append(unicodedata.normalize("NFC", current))
                current = character
            else:
                if not current:
                    raise ValueError("Combining mark không có ký tự cơ sở.")
                current += character
        if current:
            clusters.append(unicodedata.normalize("NFC", current))

        result: list[FactorizedGrapheme] = []
        for surface in clusters:
            if surface == "\n":
                result.append(FactorizedGrapheme("\n", "\n", "none", "none", "none", "newline"))
            elif surface.isspace():
                result.append(FactorizedGrapheme(" ", " ", "none", "none", "none", "space"))
            elif surface.isdigit():
                result.append(FactorizedGrapheme(surface, surface, "none", "none", "none", "digit"))
            elif unicodedata.category(surface[0]).startswith(("P", "S")):
                result.append(FactorizedGrapheme(surface, surface, "none", "none", "none", "punctuation"))
            elif surface.isalpha():
                case = "upper" if surface[0].isupper() else "lower"
                shape, tone = "none", "none"
                if surface.lower() == "đ":
                    base, shape = "d", "bar"
                else:
                    nfd = unicodedata.normalize("NFD", surface)
                    base = nfd[0].lower()
                    for mark in nfd[1:]:
                        shape = SHAPE_MARKS.get(mark, shape)
                        tone = TONE_MARKS.get(mark, tone)
                result.append(FactorizedGrapheme(surface, base, shape, tone, case, "letter"))
            else:
                result.append(FactorizedGrapheme(surface, surface, "none", "none", "none", "special"))
        return result


@dataclass(frozen=True, slots=True)
class FormattedParagraph:
    graphemes: tuple[FactorizedGrapheme, ...]
    line_ids: tuple[int, ...]
    positions_in_line: tuple[int, ...]
    lines: tuple[str, ...]
    height_bucket_id: int
    output_height: int
    line_slot_mask: Tensor


class ParagraphFormatter:
    """Deterministic formatter calibrated by three bounded style scalars."""

    def __init__(self, config: TextEncoderConfig) -> None:
        self.config = config
        self.factorizer = VietnameseGraphemeFactorizer()

    def format(
        self,
        text: str,
        *,
        character_width_scale: float = 1.0,
        word_gap_scale: float = 1.0,
        line_gap_scale: float = 1.0,
        preserve_physical_lines: bool = False,
        output_height: int | None = None,
    ) -> FormattedParagraph:
        if not isinstance(preserve_physical_lines, bool):
            raise TypeError("preserve_physical_lines phải là bool.")
        if not 0.85 <= character_width_scale <= 1.15:
            raise ValueError("character_width_scale phải nằm trong [0.85, 1.15].")
        if not 0.80 <= word_gap_scale <= 1.20:
            raise ValueError("word_gap_scale phải nằm trong [0.80, 1.20].")
        if not 0.90 <= line_gap_scale <= 1.10:
            raise ValueError("line_gap_scale phải nằm trong [0.90, 1.10].")

        sections: list[list[FactorizedGrapheme]] = [[]]
        for grapheme in self.factorizer.factorize(text):
            if grapheme.class_name == "newline":
                sections.append([])
            else:
                sections[-1].append(grapheme)

        margin = 48
        if preserve_physical_lines:
            lines = sections
        else:
            usable_width = self.config.canvas_width - 2 * margin
            narrow, wide = set("fijlt"), set("mw")

            def advance(grapheme: FactorizedGrapheme) -> float:
                if grapheme.class_name == "digit":
                    width = 30.0
                elif grapheme.class_name == "punctuation":
                    width = 18.0
                elif grapheme.base.lower() in narrow:
                    width = 20.0
                elif grapheme.base.lower() in wide:
                    width = 42.0
                else:
                    width = 32.0
                return width * character_width_scale

            space = FactorizedGrapheme(
                " ",
                " ",
                "none",
                "none",
                "none",
                "space",
            )
            lines = []
            for section in sections:
                words: list[list[FactorizedGrapheme]] = []
                word: list[FactorizedGrapheme] = []
                for grapheme in section:
                    if grapheme.class_name == "space":
                        if word:
                            words.append(word)
                            word = []
                    else:
                        word.append(grapheme)
                if word:
                    words.append(word)
                if not words:
                    lines.append([])
                    continue

                line: list[FactorizedGrapheme] = []
                line_width = 0.0
                for word in words:
                    word_width = sum(advance(item) for item in word)
                    gap = 18.0 * word_gap_scale if line else 0.0
                    if (
                        line
                        and line_width + gap + word_width > usable_width
                    ):
                        lines.append(line)
                        line, line_width, gap = [], 0.0, 0.0
                    if word_width <= usable_width:
                        if line:
                            line.append(space)
                            line_width += gap
                        line.extend(word)
                        line_width += word_width
                    else:
                        for grapheme in word:
                            width = advance(grapheme)
                            if (
                                line
                                and line_width + width > usable_width
                            ):
                                lines.append(line)
                                line, line_width = [], 0.0
                            line.append(grapheme)
                            line_width += width
                if line:
                    lines.append(line)

        if len(lines) > self.config.max_lines:
            raise ValueError(
                f"Paragraph cần {len(lines)} dòng, vượt giới hạn {self.config.max_lines}."
            )
        line_height, line_gap = 112, round(24 * line_gap_scale)
        required_height = 96 + len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        if output_height is not None:
            if output_height not in self.config.height_buckets:
                raise ValueError(
                    "output_height phải thuộc height_buckets, nhận "
                    f"{output_height}."
                )
            if output_height < required_height:
                raise ValueError(
                    f"output_height={output_height} không đủ required_height="
                    f"{required_height}."
                )
            bucket_id = self.config.height_buckets.index(output_height)
        else:
            for bucket_id, selected_height in enumerate(
                self.config.height_buckets
            ):
                if selected_height >= required_height:
                    output_height = selected_height
                    break
            else:
                raise ValueError(
                    f"Paragraph cần {required_height}px, vượt bucket "
                    f"{self.config.height_buckets[-1]}px."
                )
        if output_height is None:
            raise RuntimeError("Không chọn được output height bucket.")

        flattened: list[FactorizedGrapheme] = []
        line_ids: list[int] = []
        positions: list[int] = []
        strings: list[str] = []
        for line_id, line in enumerate(lines):
            if len(line) > self.config.max_position_in_line:
                raise ValueError(
                    f"Dòng {line_id} có {len(line)} graphemes, vượt "
                    f"{self.config.max_position_in_line}."
                )
            strings.append("".join(item.surface for item in line))
            for position, grapheme in enumerate(line):
                flattened.append(grapheme)
                line_ids.append(line_id)
                positions.append(position)
            if line_id < len(lines) - 1:
                flattened.append(FactorizedGrapheme("\n", "\n", "none", "none", "none", "newline"))
                line_ids.append(line_id)
                positions.append(min(len(line), self.config.max_position_in_line - 1))
        if len(flattened) > self.config.max_graphemes:
            raise ValueError(
                f"Paragraph có {len(flattened)} tokens, vượt {self.config.max_graphemes}."
            )

        mask = torch.zeros(
            self.config.max_lines,
            output_height // 8,
            self.config.canvas_width // 8,
        )
        for line_id in range(len(lines)):
            if not lines[line_id]:
                continue
            y0 = (48 + line_id * (line_height + line_gap)) // 8
            y1 = math.ceil((48 + line_id * (line_height + line_gap) + line_height) / 8)
            mask[line_id, y0:y1, margin // 8 : (self.config.canvas_width - margin) // 8] = 1.0
        return FormattedParagraph(
            tuple(flattened),
            tuple(line_ids),
            tuple(positions),
            tuple(strings),
            bucket_id,
            output_height,
            mask,
        )


@dataclass(frozen=True, slots=True)
class GraphemeBatch:
    base_ids: Tensor
    shape_ids: Tensor
    tone_ids: Tensor
    case_ids: Tensor
    class_ids: Tensor
    line_ids: Tensor
    position_in_line_ids: Tensor
    height_bucket_ids: Tensor
    attention_mask: Tensor


@dataclass(frozen=True, slots=True)
class FormattedTextBatch:
    graphemes: GraphemeBatch
    canonical_line_slots: Tensor
    output_height: int


class GraphemeVocabulary:
    def __init__(
        self,
        base_to_id: dict[str, int],
        shape_to_id: dict[str, int],
        tone_to_id: dict[str, int],
        case_to_id: dict[str, int],
        class_to_id: dict[str, int],
    ) -> None:
        self.base_to_id = dict(base_to_id)
        self.shape_to_id = dict(shape_to_id)
        self.tone_to_id = dict(tone_to_id)
        self.case_to_id = dict(case_to_id)
        self.class_to_id = dict(class_to_id)
        for name, vocab in (
            ("base", self.base_to_id),
            ("shape", self.shape_to_id),
            ("tone", self.tone_to_id),
            ("case", self.case_to_id),
            ("class", self.class_to_id),
        ):
            if vocab.get("<pad>") != 0 or vocab.get("<unk>") != 1:
                raise ValueError(f"{name} vocabulary phải có <pad>=0 và <unk>=1.")

    @classmethod
    def default_vietnamese(cls) -> GraphemeVocabulary:
        symbols = list("abcdefghijklmnopqrstuvwxyz0123456789") + list(
            " \n.,;:!?\'\"()[]{}-–—…/\\%+=*@#&"
        )
        base = {"<pad>": 0, "<unk>": 1}
        for symbol in symbols:
            if symbol not in base:
                base[symbol] = len(base)
        return cls(
            base,
            {"<pad>": 0, "<unk>": 1, "none": 2, "breve": 3, "circumflex": 4, "horn": 5, "bar": 6},
            {"<pad>": 0, "<unk>": 1, "none": 2, "acute": 3, "grave": 4, "hook_above": 5, "tilde": 6, "dot_below": 7},
            {"<pad>": 0, "<unk>": 1, "none": 2, "lower": 3, "upper": 4},
            {"<pad>": 0, "<unk>": 1, "letter": 2, "digit": 3, "space": 4, "punctuation": 5, "newline": 6, "special": 7},
        )

    def encode_batch(
        self,
        paragraphs: list[FormattedParagraph],
        *,
        device: torch.device | str | None = None,
    ) -> FormattedTextBatch:
        if not paragraphs:
            raise ValueError("paragraphs không được rỗng.")
        output_height = paragraphs[0].output_height
        if any(item.output_height != output_height for item in paragraphs):
            raise ValueError("Mọi paragraph trong batch phải cùng height bucket.")
        batch, length = len(paragraphs), max(len(item.graphemes) for item in paragraphs)
        if length == 0:
            raise ValueError("Paragraph batch phải có ít nhất một grapheme token.")
        ids = [torch.zeros(batch, length, dtype=torch.long, device=device) for _ in range(7)]
        height_ids = torch.empty(batch, dtype=torch.long, device=device)
        attention = torch.zeros(batch, length, dtype=torch.bool, device=device)
        vocabs = (
            self.base_to_id,
            self.shape_to_id,
            self.tone_to_id,
            self.case_to_id,
            self.class_to_id,
        )
        for batch_id, paragraph in enumerate(paragraphs):
            for token_id, grapheme in enumerate(paragraph.graphemes):
                values = (
                    grapheme.base,
                    grapheme.shape,
                    grapheme.tone,
                    grapheme.case,
                    grapheme.class_name,
                )
                for tensor, vocab, value in zip(ids[:5], vocabs, values, strict=True):
                    tensor[batch_id, token_id] = vocab.get(value, 1)
                ids[5][batch_id, token_id] = paragraph.line_ids[token_id]
                ids[6][batch_id, token_id] = paragraph.positions_in_line[token_id]
            attention[batch_id, : len(paragraph.graphemes)] = True
            height_ids[batch_id] = paragraph.height_bucket_id
        return FormattedTextBatch(
            GraphemeBatch(*ids, height_ids, attention),
            torch.stack([item.line_slot_mask for item in paragraphs]).to(device),
            output_height,
        )


@dataclass(frozen=True, slots=True)
class GraphemeCondition:
    base_context: Tensor
    shape_context: Tensor
    tone_context: Tensor
    attention_mask: Tensor
    line_ids: Tensor


class FactorizedGraphemeEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.component_embedding_dim
        self.base_embedding = nn.Embedding(config.base_vocab_size, dim, padding_idx=0)
        self.shape_embedding = nn.Embedding(config.shape_vocab_size, dim, padding_idx=0)
        self.tone_embedding = nn.Embedding(config.tone_vocab_size, dim, padding_idx=0)
        self.case_embedding = nn.Embedding(config.case_vocab_size, dim, padding_idx=0)
        self.class_embedding = nn.Embedding(config.class_vocab_size, dim, padding_idx=0)
        self.input_projection = nn.Linear(3 * dim, config.model_dim)
        self.sequence_position = nn.Parameter(torch.randn(config.max_graphemes, config.model_dim) * 0.02)
        self.line_embedding = nn.Embedding(config.max_lines, config.model_dim)
        self.inline_embedding = nn.Embedding(config.max_position_in_line, config.model_dim)
        self.height_embedding = nn.Embedding(len(config.height_buckets), config.model_dim)
        layer = nn.TransformerEncoderLayer(
            config.model_dim,
            config.num_heads,
            config.ffn_dim,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.base_projection = nn.Linear(config.model_dim + dim, config.context_dim)
        self.shape_projection = nn.Linear(config.model_dim + dim, config.context_dim)
        self.tone_projection = nn.Linear(config.model_dim + dim, config.context_dim)

    def forward(self, batch: GraphemeBatch) -> GraphemeCondition:
        shape = batch.base_ids.shape
        tensors = (
            batch.shape_ids, batch.tone_ids, batch.case_ids, batch.class_ids,
            batch.line_ids, batch.position_in_line_ids, batch.attention_mask,
        )
        if batch.base_ids.ndim != 2 or any(item.shape != shape for item in tensors):
            raise ValueError("Mọi grapheme ID và attention_mask phải có shape [B,N].")
        if batch.height_bucket_ids.shape != (shape[0],):
            raise ValueError("height_bucket_ids phải có shape [B].")
        if shape[1] == 0 or shape[1] > self.config.max_graphemes:
            raise ValueError(
                "Sequence length phải nằm trong [1, "
                f"{self.config.max_graphemes}]."
            )
        if batch.attention_mask.dtype != torch.bool or not batch.attention_mask.any(dim=1).all():
            raise ValueError("Mỗi sample phải có attention_mask bool với ít nhất một token.")
        id_tensors = (
            batch.base_ids,
            batch.shape_ids,
            batch.tone_ids,
            batch.case_ids,
            batch.class_ids,
            batch.line_ids,
            batch.position_in_line_ids,
            batch.height_bucket_ids,
        )
        if any(item.dtype != torch.long for item in id_tensors):
            raise TypeError("Mọi grapheme ID phải có dtype torch.long.")
        base = self.base_embedding(batch.base_ids)
        shape_emb = self.shape_embedding(batch.shape_ids)
        tone = self.tone_embedding(batch.tone_ids)
        components = (
            base,
            self.case_embedding(batch.case_ids),
            self.class_embedding(batch.class_ids),
        )
        features = self.input_projection(torch.cat(components, dim=-1))
        features = (
            features
            + self.sequence_position[: shape[1]][None]
            + self.line_embedding(batch.line_ids)
            + self.inline_embedding(batch.position_in_line_ids)
            + self.height_embedding(batch.height_bucket_ids)[:, None]
        )
        features = self.transformer(features, src_key_padding_mask=~batch.attention_mask)
        padding = ~batch.attention_mask[:, :, None]
        return GraphemeCondition(
            self.base_projection(torch.cat((features, base), dim=-1)).masked_fill(padding, 0.0),
            self.shape_projection(torch.cat((features, shape_emb), dim=-1)).masked_fill(padding, 0.0),
            self.tone_projection(torch.cat((features, tone), dim=-1)).masked_fill(padding, 0.0),
            batch.attention_mask,
            batch.line_ids,
        )

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy text checkpoint: {path}")
        state: object = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and set(state) == {"model"}:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise ValueError("Text checkpoint không phải state_dict hợp lệ.")
        self.load_state_dict(dict(state), strict=True)
