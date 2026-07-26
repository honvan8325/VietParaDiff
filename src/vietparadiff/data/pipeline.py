"""Training-time image processors, datasets, samplers, and collators."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from typing import Literal, Protocol

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import pil_to_tensor

from vietparadiff.data.contracts import (
    eligible_reference,
    excluded_source_line_ids,
)
from vietparadiff.models.config import TextEncoderConfig
from vietparadiff.models.grapheme import (
    FormattedTextBatch,
    GraphemeVocabulary,
    ParagraphFormatter,
    VietnameseGraphemeFactorizer,
)

__all__ = [
    "AutoKLDataset",
    "HTRDataset",
    "HTRImageProcessor",
    "HTRVocabulary",
    "HeightBucketBatchSampler",
    "ParagraphImageProcessor",
    "ReferenceImageProcessor",
    "VietParaDiffCollator",
    "VietParaDiffDataset",
    "WidthBucketBatchSampler",
    "collate_autokl",
    "collate_htr",
]


ImageSource = Path | str | Image.Image
HEIGHT_BUCKETS = (384, 512, 640, 768, 896, 1024, 1280)


def _as_paths(paths: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(paths, Path):
        return (paths,)
    result = tuple(paths)
    if not result or not all(isinstance(path, Path) for path in result):
        raise TypeError("manifest paths phải là Path hoặc sequence Path không rỗng.")
    return result


def _read_jsonl(
    paths: Path | Sequence[Path],
    *,
    required_fields: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for path in _as_paths(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON lỗi tại {path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record tại {path}:{line_number} phải là object."
                    )
                missing = required_fields - record.keys()
                if missing:
                    raise ValueError(
                        f"Record tại {path}:{line_number} thiếu "
                        f"{sorted(missing)}."
                    )
                record_id = record.get("id", record.get("pair_id"))
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(
                        f"Record tại {path}:{line_number} thiếu ID hợp lệ."
                    )
                if record_id in seen_ids:
                    raise ValueError(f"Trùng record ID: {record_id}")
                seen_ids.add(record_id)
                records.append(dict(record))
    if not records:
        raise ValueError("Manifest không chứa record nào.")
    return records


def _resolve_image_path(path_value: object, image_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Image path không hợp lệ: {path_value!r}")
    path = Path(path_value)
    resolved = path if path.is_absolute() else image_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Không tìm thấy image: {resolved}")
    return resolved


def _load_grayscale(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        if source.width <= 0 or source.height <= 0:
            raise ValueError(f"Image dimensions không hợp lệ: {source.size}")
        return source.convert("L")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy image: {path}")
    with Image.open(path) as image:
        image.load()
        return image.convert("L")


def _crop_foreground(
    image: Image.Image,
    *,
    threshold: int,
    margin: int,
    background_delta: int = 12,
) -> Image.Image:
    if not 0 <= threshold <= 255:
        raise ValueError("foreground threshold phải nằm trong [0, 255].")
    if margin < 0:
        raise ValueError("crop margin không được âm.")
    if not 1 <= background_delta <= 255:
        raise ValueError("background_delta phải nằm trong [1, 255].")
    border_width = max(1, min(8, image.width // 2, image.height // 2))
    strips = (
        (0, 0, image.width, border_width),
        (0, image.height - border_width, image.width, image.height),
        (0, border_width, border_width, image.height - border_width),
        (
            image.width - border_width,
            border_width,
            image.width,
            image.height - border_width,
        ),
    )
    histogram = [0] * 256
    for box in strips:
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        strip = image.crop(box)
        try:
            for value, count in enumerate(strip.histogram()):
                histogram[value] += count
        finally:
            strip.close()
    total = sum(histogram)
    if total == 0:
        raise ValueError("Không đo được background từ image border.")
    midpoint = (total + 1) // 2
    cumulative = 0
    background = 255
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= midpoint:
            background = value
            break
    adaptive_threshold = min(
        threshold,
        max(1, background - background_delta),
    )
    foreground = image.point(
        lambda value: 255 if value < adaptive_threshold else 0,
        mode="1",
    )
    box = foreground.getbbox()
    foreground.close()
    if box is None:
        raise ValueError("Image không có foreground dưới ngưỡng trắng.")
    cropped = image.crop(box)
    if margin:
        bordered = ImageOps.expand(cropped, border=margin, fill=255)
        cropped.close()
        return bordered
    return cropped


def _normalized_tensor(image: Image.Image) -> Tensor:
    tensor = pil_to_tensor(image).to(torch.float32)
    return tensor.div(127.5).sub(1.0)


def _round_up(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("value và multiple phải dương.")
    return ((value + multiple - 1) // multiple) * multiple


class ParagraphImageProcessor:
    """Crop, aspect-preserving resize, and bucket-pad a paragraph image."""

    def __init__(
        self,
        *,
        canvas_width: int = 1024,
        height_buckets: tuple[int, ...] = HEIGHT_BUCKETS,
        foreground_threshold: int = 253,
        crop_margin: int = 16,
    ) -> None:
        if canvas_width != 1024:
            raise ValueError("Paragraph canvas width phải bằng 1024.")
        if height_buckets != HEIGHT_BUCKETS:
            raise ValueError(f"Height buckets phải bằng {HEIGHT_BUCKETS}.")
        self.canvas_width = canvas_width
        self.height_buckets = height_buckets
        self.foreground_threshold = foreground_threshold
        self.crop_margin = crop_margin

    def _prepare(
        self,
        source: ImageSource,
        *,
        output_height: int | None = None,
    ) -> tuple[Image.Image, tuple[int, int], int]:
        image = _load_grayscale(source)
        try:
            cropped = _crop_foreground(
                image,
                threshold=self.foreground_threshold,
                margin=self.crop_margin,
            )
        finally:
            image.close()
        if output_height is not None:
            if output_height not in self.height_buckets:
                cropped.close()
                raise ValueError(
                    f"output_height phải thuộc {self.height_buckets}, "
                    f"nhận {output_height}."
                )
            max_height = output_height
        else:
            max_height = self.height_buckets[-1]
        scale = min(
            self.canvas_width / cropped.width,
            max_height / cropped.height,
        )
        size = (
            max(1, min(self.canvas_width, round(cropped.width * scale))),
            max(1, min(max_height, round(cropped.height * scale))),
        )
        if output_height is not None:
            bucket = output_height
        else:
            bucket = next(
                (
                    height
                    for height in self.height_buckets
                    if height >= size[1]
                ),
                None,
            )
            if bucket is None:
                cropped.close()
                raise ValueError(
                    f"Paragraph height {size[1]} vượt bucket tối đa "
                    f"{self.height_buckets[-1]}."
                )
        return cropped, size, bucket

    def height_bucket(
        self,
        source: ImageSource,
        *,
        output_height: int | None = None,
    ) -> int:
        if output_height is not None:
            if output_height not in self.height_buckets:
                raise ValueError(
                    f"output_height phải thuộc {self.height_buckets}."
                )
            return output_height
        cropped, _, bucket = self._prepare(source)
        cropped.close()
        return bucket

    def __call__(
        self,
        source: ImageSource,
        *,
        output_height: int | None = None,
    ) -> dict[str, Tensor | int]:
        cropped, size, bucket = self._prepare(
            source,
            output_height=output_height,
        )
        try:
            resized = cropped.resize(size, Image.Resampling.LANCZOS)
        finally:
            cropped.close()
        canvas = Image.new("L", (self.canvas_width, bucket), 255)
        try:
            canvas.paste(
                resized,
                (
                    (self.canvas_width - resized.width) // 2,
                    (bucket - resized.height) // 2,
                ),
            )
            tensor = _normalized_tensor(canvas)
        finally:
            resized.close()
            canvas.close()
        return {"image": tensor, "height_bucket": bucket}


class HTRImageProcessor:
    """Normalize a line or word into a 64px-high HTR sequence image."""

    def __init__(
        self,
        *,
        content_height: int = 56,
        output_height: int = 64,
        foreground_threshold: int = 253,
        crop_margin: int = 4,
    ) -> None:
        if content_height != 56 or output_height != 64:
            raise ValueError("HTR phải dùng content height 56 và output height 64.")
        self.content_height = content_height
        self.output_height = output_height
        self.foreground_threshold = foreground_threshold
        self.crop_margin = crop_margin

    def __call__(self, source: ImageSource) -> dict[str, Tensor | int]:
        image = _load_grayscale(source)
        try:
            cropped = _crop_foreground(
                image,
                threshold=self.foreground_threshold,
                margin=self.crop_margin,
            )
        finally:
            image.close()
        width = max(
            1,
            round(cropped.width * self.content_height / cropped.height),
        )
        try:
            resized = cropped.resize(
                (width, self.content_height),
                Image.Resampling.LANCZOS,
            )
        finally:
            cropped.close()
        canvas = Image.new("L", (width, self.output_height), 255)
        try:
            canvas.paste(resized, (0, (self.output_height - self.content_height) // 2))
            tensor = _normalized_tensor(canvas)
        finally:
            resized.close()
            canvas.close()
        return {"image": tensor, "valid_width": width}


class ReferenceImageProcessor:
    """Prepare one real line reference for the style encoder contract."""

    def __init__(
        self,
        *,
        output_height: int = 256,
        max_width: int = 1536,
        width_multiple: int = 32,
        foreground_threshold: int = 253,
        crop_margin: int = 8,
    ) -> None:
        if (
            output_height != 256
            or max_width != 1536
            or width_multiple != 32
        ):
            raise ValueError(
                "Reference phải cao 256, rộng tối đa 1536 và pad bội số 32."
            )
        self.output_height = output_height
        self.max_width = max_width
        self.width_multiple = width_multiple
        self.foreground_threshold = foreground_threshold
        self.crop_margin = crop_margin

    def __call__(self, source: ImageSource) -> dict[str, Tensor]:
        image = _load_grayscale(source)
        try:
            cropped = _crop_foreground(
                image,
                threshold=self.foreground_threshold,
                margin=self.crop_margin,
            )
        finally:
            image.close()
        scale = min(
            self.output_height / cropped.height,
            self.max_width / cropped.width,
        )
        size = (
            max(1, min(self.max_width, round(cropped.width * scale))),
            max(1, min(self.output_height, round(cropped.height * scale))),
        )
        try:
            resized = cropped.resize(size, Image.Resampling.LANCZOS)
        finally:
            cropped.close()
        padded_width = _round_up(resized.width, self.width_multiple)
        if padded_width > self.max_width:
            resized.close()
            raise RuntimeError(
                f"Reference padded width {padded_width} vượt {self.max_width}."
            )
        canvas = Image.new("L", (padded_width, self.output_height), 255)
        y0 = (self.output_height - resized.height) // 2
        valid_mask = torch.zeros(
            1,
            self.output_height,
            padded_width,
            dtype=torch.bool,
        )
        try:
            canvas.paste(resized, (0, y0))
            valid_mask[:, y0 : y0 + resized.height, : resized.width] = True
            tensor = _normalized_tensor(canvas)
        finally:
            resized.close()
            canvas.close()
        return {"image": tensor, "valid_mask": valid_mask}


@dataclass(frozen=True, slots=True)
class HTRVocabulary:
    """Four deterministic CTC vocabularies built only from training text."""

    raw_to_id: dict[str, int]
    base_to_id: dict[str, int]
    shape_to_id: dict[str, int]
    tone_to_id: dict[str, int]

    def __post_init__(self) -> None:
        for name, vocabulary in (
            ("raw", self.raw_to_id),
            ("base", self.base_to_id),
            ("shape", self.shape_to_id),
            ("tone", self.tone_to_id),
        ):
            if (
                vocabulary.get("<blank>") != 0
                or vocabulary.get("<unk>") != 1
            ):
                raise ValueError(
                    f"{name} vocabulary phải có <blank>=0, <unk>=1."
                )
            if len(set(vocabulary.values())) != len(vocabulary):
                raise ValueError(f"{name} vocabulary chứa ID trùng.")

    @classmethod
    def build_from_manifests(
        cls,
        manifests: Path | Sequence[Path],
    ) -> HTRVocabulary:
        records = _read_jsonl(
            manifests,
            required_fields={"id", "text", "level"},
        )
        factorizer = VietnameseGraphemeFactorizer()
        values: dict[str, set[str]] = {
            "raw": set(),
            "base": set(),
            "shape": set(),
            "tone": set(),
        }
        for record in records:
            if record["level"] not in {"line", "word"}:
                raise ValueError(
                    f"HTR vocabulary chỉ nhận line/word, nhận "
                    f"{record['level']!r} tại {record['id']}."
                )
            text = unicodedata.normalize("NFC", str(record["text"]))
            graphemes = factorizer.factorize(text)
            if any(item.class_name == "newline" for item in graphemes):
                raise ValueError(f"HTR transcript chứa newline: {record['id']}")
            values["raw"].update(item.surface for item in graphemes)
            values["base"].update(item.base for item in graphemes)
            values["shape"].update(item.shape for item in graphemes)
            values["tone"].update(item.tone for item in graphemes)

        def vocabulary(tokens: set[str]) -> dict[str, int]:
            output = {"<blank>": 0, "<unk>": 1}
            for token in sorted(tokens):
                output[token] = len(output)
            return output

        return cls(
            vocabulary(values["raw"]),
            vocabulary(values["base"]),
            vocabulary(values["shape"]),
            vocabulary(values["tone"]),
        )

    def encode(self, text: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = unicodedata.normalize("NFC", text)
        graphemes = VietnameseGraphemeFactorizer().factorize(normalized)
        if any(item.class_name == "newline" for item in graphemes):
            raise ValueError("HTR transcript không được chứa newline.")

        def ids(vocabulary: Mapping[str, int], values: Sequence[str]) -> Tensor:
            return torch.tensor(
                [vocabulary.get(value, 1) for value in values],
                dtype=torch.long,
            )

        return (
            ids(self.raw_to_id, [item.surface for item in graphemes]),
            ids(self.base_to_id, [item.base for item in graphemes]),
            ids(self.shape_to_id, [item.shape for item in graphemes]),
            ids(self.tone_to_id, [item.tone for item in graphemes]),
        )

    def minimum_input_width(self, text: str) -> int:
        """Minimum pixel width whose x4-downsampled CTC path is feasible."""
        heads = self.encode(text)

        def required(target: Tensor) -> int:
            values = target.tolist()
            repeats = sum(
                first == second
                for first, second in zip(
                    values[:-1],
                    values[1:],
                    strict=True,
                )
            )
            return len(values) + repeats

        required_steps = max(required(target) for target in heads)
        return max(1, (required_steps - 1) * 4 + 1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "raw_to_id": self.raw_to_id,
                    "base_to_id": self.base_to_id,
                    "shape_to_id": self.shape_to_id,
                    "tone_to_id": self.tone_to_id,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> HTRVocabulary:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy HTR vocabulary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "raw_to_id",
            "base_to_id",
            "shape_to_id",
            "tone_to_id",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("HTR vocabulary JSON sai schema.")
        mappings = []
        for name in (
            "raw_to_id",
            "base_to_id",
            "shape_to_id",
            "tone_to_id",
        ):
            mapping = payload[name]
            if not isinstance(mapping, dict) or not all(
                isinstance(key, str) and isinstance(value, int)
                for key, value in mapping.items()
            ):
                raise ValueError(f"{name} phải là mapping str -> int.")
            mappings.append(dict(mapping))
        return cls(*mappings)


class _HeightBucketDataset(Protocol):
    def __len__(self) -> int: ...

    def height_bucket(self, index: int) -> int: ...


class _WidthBucketDataset(Protocol):
    def __len__(self) -> int: ...

    def valid_width(self, index: int) -> int: ...


class AutoKLDataset(Dataset[dict[str, object]]):
    """Real paragraph images for AutoKL reconstruction."""

    def __init__(
        self,
        manifest: Path,
        *,
        image_root: Path = Path("."),
        processor: ParagraphImageProcessor | None = None,
    ) -> None:
        self.records = _read_jsonl(
            manifest,
            required_fields={
                "id",
                "image",
                "level",
                "canonical_writer_id",
            },
        )
        if any(record["level"] != "paragraph" for record in self.records):
            raise ValueError("AutoKLDataset chỉ nhận paragraph records.")
        self.image_root = image_root
        self.processor = processor or ParagraphImageProcessor()
        self._bucket_cache: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, index: int) -> Path:
        return _resolve_image_path(
            self.records[index]["image"],
            self.image_root,
        )

    def height_bucket(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._bucket_cache:
            self._bucket_cache[index] = self.processor.height_bucket(
                self._image_path(index)
            )
        return self._bucket_cache[index]

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        processed = self.processor(self._image_path(index))
        bucket = int(processed["height_bucket"])
        cached = self._bucket_cache.get(index)
        if cached is not None and cached != bucket:
            raise RuntimeError(
                f"Height bucket thay đổi cho {record['id']}: {cached} -> {bucket}."
            )
        self._bucket_cache[index] = bucket
        return {
            "image": processed["image"],
            "sample_id": str(record["id"]),
            "writer_id": str(record["canonical_writer_id"]),
            "height_bucket": bucket,
        }


class HTRDataset(Dataset[dict[str, object]]):
    """Line/word images and four aligned factorized CTC targets."""

    def __init__(
        self,
        manifests: Path | Sequence[Path],
        vocabulary: HTRVocabulary,
        *,
        image_root: Path = Path("."),
        processor: HTRImageProcessor | None = None,
    ) -> None:
        self.records = _read_jsonl(
            manifests,
            required_fields={"id", "image", "text", "level"},
        )
        if any(
            record["level"] not in {"line", "word"}
            for record in self.records
        ):
            raise ValueError("HTRDataset chỉ nhận line/word records.")
        self.vocabulary = vocabulary
        self.image_root = image_root
        self.processor = processor or HTRImageProcessor()
        self._width_cache: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, index: int) -> Path:
        return _resolve_image_path(
            self.records[index]["image"],
            self.image_root,
        )

    def valid_width(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._width_cache:
            processed = self.processor(self._image_path(index))
            text = unicodedata.normalize(
                "NFC", str(self.records[index]["text"])
            )
            self._width_cache[index] = max(
                int(processed["valid_width"]),
                self.vocabulary.minimum_input_width(text),
            )
        return self._width_cache[index]

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        processed = self.processor(self._image_path(index))
        text = unicodedata.normalize("NFC", str(record["text"]))
        raw, base, shape, tone = self.vocabulary.encode(text)
        original_width = int(processed["valid_width"])
        valid_width = max(
            original_width,
            self.vocabulary.minimum_input_width(text),
        )
        image = processed["image"]
        if not isinstance(image, Tensor):
            raise TypeError("HTR processor image phải là Tensor.")
        if valid_width > original_width:
            padding = image.new_ones(
                image.shape[0],
                image.shape[1],
                valid_width - original_width,
            )
            image = torch.cat((image, padding), dim=2)
        self._width_cache[index] = valid_width
        return {
            "image": image,
            "valid_width": valid_width,
            "text": text,
            "raw_targets": raw,
            "base_targets": base,
            "shape_targets": shape,
            "tone_targets": tone,
            "sample_id": str(record["id"]),
            "sample_level": str(record["level"]),
        }


def _flat_text(text: str) -> str:
    return " ".join(text.split())


class VietParaDiffDataset(Dataset[dict[str, object]]):
    """Target/reference pairs for generator train or fixed test evaluation."""

    def __init__(
        self,
        manifest: Path | Sequence[Path],
        *,
        mode: Literal["train", "test"],
        reference_manifest: Path | None = None,
        image_root: Path = Path("."),
        paragraph_processor: ParagraphImageProcessor | None = None,
        reference_processor: ReferenceImageProcessor | None = None,
        formatter: ParagraphFormatter,
        seed: int = 0,
    ) -> None:
        if mode not in {"train", "test"}:
            raise ValueError("mode phải là 'train' hoặc 'test'.")
        self.mode = mode
        self.image_root = image_root
        self.paragraph_processor = (
            paragraph_processor or ParagraphImageProcessor()
        )
        self.reference_processor = (
            reference_processor or ReferenceImageProcessor()
        )
        self.formatter = formatter
        if not isinstance(seed, int):
            raise TypeError("seed phải là int.")
        self.seed = seed
        self._epoch = Value("q", 0, lock=False)
        self._bucket_cache: dict[int, int] = {}

        if mode == "train":
            if reference_manifest is None:
                raise ValueError("Train mode bắt buộc có reference_manifest.")
            self.records = _read_jsonl(
                manifest,
                required_fields={
                    "id",
                    "image",
                    "text",
                    "canonical_writer_id",
                    "formatter_mode",
                },
            )
            references = _read_jsonl(
                reference_manifest,
                required_fields={
                    "id",
                    "image",
                    "text",
                    "canonical_writer_id",
                    "level",
                },
            )
            if any(
                record["formatter_mode"] != "physical_lines"
                for record in self.records
            ):
                raise ValueError(
                    "Mọi generator target phải dùng formatter_mode="
                    "'physical_lines'."
                )
            if any(reference["level"] != "line" for reference in references):
                raise ValueError("Style reference pool chỉ được chứa line.")
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for reference in references:
                grouped[str(reference["canonical_writer_id"])].append(reference)
            self._eligible_references: list[list[dict[str, object]]] = []
            for target in self.records:
                excluded = set(excluded_source_line_ids(target))
                candidates = [
                    reference
                    for reference in grouped.get(
                        str(target["canonical_writer_id"]),
                        (),
                    )
                    if eligible_reference(
                        target,
                        reference,
                        excluded_reference_ids=excluded,
                    )
                ]
                if not candidates:
                    raise ValueError(
                        f"Target {target['id']} không có reference hợp lệ."
                    )
                self._eligible_references.append(
                    sorted(
                        candidates,
                        key=lambda item: str(item["id"]),
                    )
                )
        else:
            if reference_manifest is not None:
                raise ValueError(
                    "Test mode đọc fixed pair trực tiếp, không nhận "
                    "reference_manifest."
                )
            self.records = _read_jsonl(
                manifest,
                required_fields={
                    "pair_id",
                    "canonical_writer_id",
                    "target_id",
                    "target_image",
                    "target_text",
                    "reference_id",
                    "reference_image",
                },
            )
            self._eligible_references = []

    def __len__(self) -> int:
        return len(self.records)

    def _target_fields(
        self,
        index: int,
    ) -> tuple[Path, str, str, str]:
        record = self.records[index]
        if self.mode == "train":
            return (
                _resolve_image_path(record["image"], self.image_root),
                str(record["text"]),
                str(record["id"]),
                str(record["canonical_writer_id"]),
            )
        return (
            _resolve_image_path(record["target_image"], self.image_root),
            str(record["target_text"]),
            str(record["target_id"]),
            str(record["canonical_writer_id"]),
        )

    def height_bucket(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._bucket_cache:
            _, text, _, _ = self._target_fields(index)
            formatted = self.formatter.format(
                text,
                preserve_physical_lines=True,
            )
            self._bucket_cache[index] = formatted.output_height
        return self._bucket_cache[index]

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int):
            raise TypeError("epoch phải là int.")
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self._epoch.value = epoch

    def _select_train_reference(
        self,
        index: int,
        target_id: str,
    ) -> dict[str, object]:
        candidates = self._eligible_references[index]
        epoch = int(self._epoch.value)
        digest = hashlib.sha256(
            f"{self.seed}:{epoch}:{target_id}".encode("utf-8")
        ).digest()
        selected = int.from_bytes(digest[:8], "big") % len(candidates)
        return candidates[selected]

    def __getitem__(self, index: int) -> dict[str, object]:
        target_path, text, target_id, canonical_id = self._target_fields(index)
        formatted = self.formatter.format(
            text,
            preserve_physical_lines=True,
        )
        output_height = formatted.output_height
        target = self.paragraph_processor(
            target_path,
            output_height=output_height,
        )
        target_height = int(target["height_bucket"])
        if target_height != output_height:
            raise RuntimeError(
                f"Processor trả bucket {target_height}, formatter yêu cầu "
                f"{output_height}."
            )
        self._bucket_cache[index] = output_height

        record = self.records[index]
        if self.mode == "train":
            reference_record = self._select_train_reference(
                index,
                target_id,
            )
            reference_id = str(reference_record["id"])
            reference_path = _resolve_image_path(
                reference_record["image"],
                self.image_root,
            )
            reference_text = str(reference_record["text"])
            augmentation = record.get("augmentation")
            source_line_ids = (
                tuple(str(item) for item in augmentation["source_line_ids"])
                if isinstance(augmentation, dict)
                else ()
            )
        else:
            reference_id = str(record["reference_id"])
            reference_path = _resolve_image_path(
                record["reference_image"],
                self.image_root,
            )
            reference_text = ""
            source_line_ids = ()

        reference = self.reference_processor(reference_path)
        return {
            "target_image": target["image"],
            "height_bucket": target_height,
            "target_text": text,
            "formatter_mode": "physical_lines",
            "reference_image": reference["image"],
            "reference_valid_mask": reference["valid_mask"],
            "target_id": target_id,
            "reference_id": reference_id,
            "reference_text": reference_text,
            "canonical_writer_id": canonical_id,
            "source_line_ids": source_line_ids,
        }


class HeightBucketBatchSampler(Sampler[list[int]]):
    """Yield batches whose paragraph targets share one exact height bucket."""

    def __init__(
        self,
        dataset: _HeightBucketDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size phải dương.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            grouped[dataset.height_bucket(index)].append(index)
        if not grouped:
            raise ValueError("Không thể tạo sampler từ dataset rỗng.")
        self.groups = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket in sorted(self.groups):
            indices = list(self.groups[bucket])
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self.groups.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups.values()
        )


class WidthBucketBatchSampler(Sampler[list[int]]):
    """Group HTR samples by coarse width to reduce right-padding waste."""

    def __init__(
        self,
        dataset: _WidthBucketDataset,
        batch_size: int,
        *,
        bucket_width: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0 or bucket_width <= 0:
            raise ValueError("batch_size và bucket_width phải dương.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            width = dataset.valid_width(index)
            grouped[_round_up(width, bucket_width)].append(index)
        if not grouped:
            raise ValueError("Không thể tạo sampler từ dataset rỗng.")
        self.groups = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket in sorted(self.groups):
            indices = list(self.groups[bucket])
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self.groups.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.groups.values()
        )


def _image_tensor(sample: Mapping[str, object], key: str) -> Tensor:
    value = sample.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"{key} phải là Tensor.")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{key} phải là floating tensor hữu hạn.")
    if value.min() < -1.0 or value.max() > 1.0:
        raise ValueError(f"{key} phải nằm trong [-1, 1].")
    return value


def collate_autokl(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("AutoKL batch không được rỗng.")
    images = [_image_tensor(sample, "image") for sample in samples]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("AutoKL batch phải cùng height bucket.")
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 1024:
        raise ValueError(
            f"AutoKL image phải có shape [1,H,1024], nhận {tuple(shape)}."
        )
    buckets = [int(sample["height_bucket"]) for sample in samples]
    if any(bucket != shape[1] for bucket in buckets):
        raise ValueError("height_bucket không khớp image height.")
    return {
        "images": torch.stack(images),
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "writer_ids": [str(sample["writer_id"]) for sample in samples],
        "height_bucket": buckets[0],
    }


def collate_htr(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("HTR batch không được rỗng.")
    images = [_image_tensor(sample, "image") for sample in samples]
    if any(
        image.ndim != 3 or image.shape[:2] != (1, 64)
        for image in images
    ):
        raise ValueError("Mọi HTR image phải có shape [1,64,W].")
    valid_widths = torch.tensor(
        [int(sample["valid_width"]) for sample in samples],
        dtype=torch.long,
    )
    if any(
        width <= 0 or width > image.shape[-1]
        for width, image in zip(valid_widths.tolist(), images, strict=True)
    ):
        raise ValueError("HTR valid_width không khớp image.")
    padded_width = _round_up(max(image.shape[-1] for image in images), 4)
    batch_images = torch.ones(
        len(images),
        1,
        64,
        padded_width,
        dtype=images[0].dtype,
    )
    for index, image in enumerate(images):
        batch_images[index, :, :, : image.shape[-1]] = image

    target_names = (
        "raw_targets",
        "base_targets",
        "shape_targets",
        "tone_targets",
    )
    targets: dict[str, Tensor] = {}
    lengths: list[int] = []
    for sample in samples:
        raw = sample["raw_targets"]
        if not isinstance(raw, Tensor) or raw.ndim != 1:
            raise TypeError("raw_targets phải là Tensor [N].")
        lengths.append(raw.numel())
    max_length = max(lengths)
    for name in target_names:
        padded = torch.zeros(len(samples), max_length, dtype=torch.long)
        for index, sample in enumerate(samples):
            target = sample[name]
            if (
                not isinstance(target, Tensor)
                or target.dtype != torch.long
                or target.shape != (lengths[index],)
            ):
                raise ValueError(f"{name} phải là torch.long [target_length].")
            padded[index, : target.numel()] = target
        targets[name] = padded
    return {
        "images": batch_images,
        "valid_widths": valid_widths,
        "texts": [str(sample["text"]) for sample in samples],
        **targets,
        "target_lengths": torch.tensor(lengths, dtype=torch.long),
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "sample_levels": [
            str(sample["sample_level"]) for sample in samples
        ],
    }


class VietParaDiffCollator:
    """Collate one target-height bucket into direct model input tensors."""

    def __init__(
        self,
        formatter: ParagraphFormatter,
        vocabulary: GraphemeVocabulary,
    ) -> None:
        self.formatter = formatter
        self.vocabulary = vocabulary

    def __call__(
        self,
        samples: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        if not samples:
            raise ValueError("VietParaDiff batch không được rỗng.")
        targets = [
            _image_tensor(sample, "target_image")
            for sample in samples
        ]
        target_shape = targets[0].shape
        if any(target.shape != target_shape for target in targets):
            raise ValueError(
                "VietParaDiff target batch phải cùng height bucket."
            )
        if (
            len(target_shape) != 3
            or target_shape[0] != 1
            or target_shape[2] != 1024
        ):
            raise ValueError(
                "target_image phải có shape [1,H,1024], nhận "
                f"{tuple(target_shape)}."
            )
        output_height = target_shape[1]
        formatted = []
        for sample in samples:
            if sample.get("formatter_mode") != "physical_lines":
                raise ValueError(
                    "VietParaDiff collate yêu cầu formatter_mode="
                    "'physical_lines'."
                )
            formatted.append(
                self.formatter.format(
                    str(sample["target_text"]),
                    preserve_physical_lines=True,
                    output_height=output_height,
                )
            )
        text_batch: FormattedTextBatch = self.vocabulary.encode_batch(
            formatted
        )

        references = [
            _image_tensor(sample, "reference_image")
            for sample in samples
        ]
        masks = [sample.get("reference_valid_mask") for sample in samples]
        if any(
            reference.ndim != 3
            or reference.shape[0] != 1
            or reference.shape[1] != 256
            or reference.shape[-1] > 1536
            or reference.shape[-1] % 32
            for reference in references
        ):
            raise ValueError(
                "Reference phải có shape [1,256,W], W<=1536 và chia hết 32."
            )
        if any(
            not isinstance(mask, Tensor)
            or mask.dtype != torch.bool
            or mask.shape != reference.shape
            for mask, reference in zip(masks, references, strict=True)
        ):
            raise ValueError(
                "reference_valid_mask phải là bool Tensor cùng shape reference."
            )
        reference_width = max(reference.shape[-1] for reference in references)
        batch_references = torch.ones(
            len(references),
            1,
            256,
            reference_width,
            dtype=references[0].dtype,
        )
        batch_masks = torch.zeros(
            len(references),
            1,
            256,
            reference_width,
            dtype=torch.bool,
        )
        for index, (reference, mask) in enumerate(
            zip(references, masks, strict=True)
        ):
            width = reference.shape[-1]
            batch_references[index, :, :, :width] = reference
            batch_masks[index, :, :, :width] = mask

        return {
            "target_images": torch.stack(targets),
            "target_texts": [
                str(sample["target_text"]) for sample in samples
            ],
            "reference_images": batch_references,
            "reference_valid_mask": batch_masks,
            "graphemes": text_batch.graphemes,
            "canonical_line_slots": text_batch.canonical_line_slots,
            "output_height": text_batch.output_height,
            "target_ids": [
                str(sample["target_id"]) for sample in samples
            ],
            "reference_ids": [
                str(sample["reference_id"]) for sample in samples
            ],
            "canonical_writer_ids": [
                str(sample["canonical_writer_id"]) for sample in samples
            ],
        }
