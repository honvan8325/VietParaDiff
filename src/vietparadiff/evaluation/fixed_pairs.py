"""Resume-safe deterministic generation for fixed held-out test pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch import Tensor

from vietparadiff.artifacts import LatentStatistics, sha256_file
from vietparadiff.data.pipeline import ReferenceImageProcessor
from vietparadiff.inference.generator import (
    SamplingConfig,
    decode_scaled_latent,
    sample_scaled_latent,
)
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.models.grapheme import (
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.generator import VietParaDiff


@dataclass(frozen=True, slots=True)
class EvaluationModelConfig:
    checkpoint: Path
    contract: Path
    model_config: Path
    vocabulary: Path


@dataclass(frozen=True, slots=True)
class EvaluationAutoKLConfig:
    checkpoint: Path
    latent_statistics: Path


@dataclass(frozen=True, slots=True)
class EvaluationDataConfig:
    test_pairs: Path
    image_root: Path
    samples_per_pair: int

    def __post_init__(self) -> None:
        if self.samples_per_pair != 3:
            raise ValueError(
                "P0 evaluation khóa samples_per_pair=3."
            )


@dataclass(frozen=True, slots=True)
class EvaluationDiffusionConfig:
    num_inference_steps: int

    def __post_init__(self) -> None:
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps phải >= 2.")


@dataclass(frozen=True, slots=True)
class EvaluationInputConfig:
    reference_height: int
    maximum_reference_width: int

    def __post_init__(self) -> None:
        if (
            self.reference_height != 256
            or self.maximum_reference_width != 1536
        ):
            raise ValueError(
                "Evaluation reference contract phải là 256x<=1536."
            )


@dataclass(frozen=True, slots=True)
class EvaluationOutputConfig:
    directory: Path
    contact_sheet_pairs: int

    def __post_init__(self) -> None:
        if self.contact_sheet_pairs <= 0:
            raise ValueError("contact_sheet_pairs phải dương.")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    base_seed: int
    device: str
    precision: str
    model: EvaluationModelConfig
    autokl: EvaluationAutoKLConfig
    data: EvaluationDataConfig
    diffusion: EvaluationDiffusionConfig
    input: EvaluationInputConfig
    output: EvaluationOutputConfig

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed không được âm.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device phải là auto/cuda/mps/cpu.")
        if self.precision not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError("precision không hợp lệ.")


def _section(
    raw: Mapping[str, object],
    name: str,
    keys: set[str],
) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = (
            sorted(value)
            if isinstance(value, Mapping)
            else type(value).__name__
        )
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(keys)}, nhận {actual}."
        )
    return dict(value)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_root = {
        "base_seed",
        "device",
        "precision",
        "model",
        "autokl",
        "data",
        "diffusion",
        "input",
        "output",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_root:
        raise ValueError(
            "Evaluation config root sai schema."
        )
    model = _section(
        raw,
        "model",
        {"checkpoint", "contract", "model_config", "vocabulary"},
    )
    autokl = _section(
        raw,
        "autokl",
        {"checkpoint", "latent_statistics"},
    )
    data = _section(
        raw,
        "data",
        {"test_pairs", "image_root", "samples_per_pair"},
    )
    diffusion = _section(
        raw,
        "diffusion",
        {"num_inference_steps"},
    )
    input_config = _section(
        raw,
        "input",
        {"reference_height", "maximum_reference_width"},
    )
    output = _section(
        raw,
        "output",
        {"directory", "contact_sheet_pairs"},
    )
    return EvaluationConfig(
        base_seed=int(raw["base_seed"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        model=EvaluationModelConfig(
            checkpoint=Path(str(model["checkpoint"])),
            contract=Path(str(model["contract"])),
            model_config=Path(str(model["model_config"])),
            vocabulary=Path(str(model["vocabulary"])),
        ),
        autokl=EvaluationAutoKLConfig(
            checkpoint=Path(str(autokl["checkpoint"])),
            latent_statistics=Path(str(autokl["latent_statistics"])),
        ),
        data=EvaluationDataConfig(
            test_pairs=Path(str(data["test_pairs"])),
            image_root=Path(str(data["image_root"])),
            samples_per_pair=int(data["samples_per_pair"]),
        ),
        diffusion=EvaluationDiffusionConfig(
            num_inference_steps=int(
                diffusion["num_inference_steps"]
            ),
        ),
        input=EvaluationInputConfig(
            reference_height=int(input_config["reference_height"]),
            maximum_reference_width=int(
                input_config["maximum_reference_width"]
            ),
        ),
        output=EvaluationOutputConfig(
            directory=Path(str(output["directory"])),
            contact_sheet_pairs=int(output["contact_sheet_pairs"]),
        ),
    )


def stable_sample_seed(
    base_seed: int,
    pair_id: str,
    sample_index: int,
) -> int:
    if base_seed < 0 or sample_index < 0:
        raise ValueError("Seed/index không được âm.")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id phải là string không rỗng.")
    digest = hashlib.sha256(
        f"{base_seed}:{pair_id}:{sample_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


_PAIR_FIELDS = {
    "pair_id",
    "canonical_writer_id",
    "target_id",
    "target_image",
    "target_text",
    "reference_id",
    "reference_image",
}


def load_test_pairs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy test pairs manifest: {path}"
        )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping) or set(payload) != _PAIR_FIELDS:
            raise ValueError(
                f"test_pairs dòng {line_number} sai schema."
            )
        if not all(
            isinstance(payload[field], str) and payload[field]
            for field in _PAIR_FIELDS
        ):
            raise ValueError(
                f"test_pairs dòng {line_number} phải chứa string không rỗng."
            )
        record = {
            field: str(payload[field]) for field in _PAIR_FIELDS
        }
        if not record["target_text"].strip():
            raise ValueError(
                f"test_pairs dòng {line_number} có target_text rỗng."
            )
        if record["pair_id"] in seen:
            raise ValueError(
                f"pair_id bị trùng: {record['pair_id']}"
            )
        seen.add(record["pair_id"])
        records.append(record)
    if not records:
        raise ValueError("test_pairs manifest không được rỗng.")
    return records


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_png_atomic(image: Tensor, path: Path) -> None:
    if (
        image.shape[0:2] != (1, 1)
        or image.ndim != 4
        or not torch.isfinite(image).all()
    ):
        raise ValueError("Output image phải là finite [1,1,H,W].")
    pixels = (
        image[0, 0]
        .detach()
        .float()
        .cpu()
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    height, width = pixels.shape
    rendered = Image.frombytes(
        "L",
        (width, height),
        bytes(pixels.untyped_storage()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        rendered.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        rendered.close()
        if temporary.exists():
            temporary.unlink()


def _resolve_image(image_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else image_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {resolved}")
    return resolved


_RESULT_FIELDS = {
    "schema_version",
    "pair_id",
    "canonical_writer_id",
    "target_id",
    "target_image",
    "target_text",
    "reference_id",
    "reference_image",
    "sample_index",
    "seed",
    "output_height",
    "num_inference_steps",
    "generated_image",
    "generated_image_sha256",
    "artifact_sha256",
}


class FixedPairEvaluator:
    def __init__(
        self,
        model: VietParaDiff,
        autokl: HandwritingAutoKL,
        statistics: LatentStatistics,
        formatter: ParagraphFormatter,
        vocabulary: GraphemeVocabulary,
        processor: ReferenceImageProcessor,
        config: EvaluationConfig,
        *,
        num_train_timesteps: int,
        device: torch.device,
        artifact_sha256: Mapping[str, str],
    ) -> None:
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps phải >= 2.")
        if config.diffusion.num_inference_steps > num_train_timesteps:
            raise ValueError(
                "num_inference_steps vượt training schedule."
            )
        expected_artifacts = {
            "generator_checkpoint",
            "inference_contract",
            "model_config",
            "grapheme_vocabulary",
            "autokl_checkpoint",
            "latent_statistics",
            "test_pairs",
        }
        if set(artifact_sha256) != expected_artifacts:
            raise ValueError("Evaluation artifact hash schema sai.")
        self.model = model
        self.autokl = autokl
        self.statistics = statistics
        self.formatter = formatter
        self.vocabulary = vocabulary
        self.processor = processor
        self.config = config
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.artifact_sha256 = dict(artifact_sha256)
        self.output_dir = config.output.directory
        for module in (self.model, self.autokl):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _run_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "base_seed": self.config.base_seed,
            "samples_per_pair": self.config.data.samples_per_pair,
            "num_inference_steps": (
                self.config.diffusion.num_inference_steps
            ),
            "num_train_timesteps": self.num_train_timesteps,
            "artifact_sha256": self.artifact_sha256,
        }

    def _load_existing(
        self,
        *,
        resume: bool,
    ) -> list[dict[str, object]]:
        contract_path = self.output_dir / "evaluation_contract.json"
        results_path = self.output_dir / "results.jsonl"
        if not resume:
            if contract_path.exists() or results_path.exists():
                raise FileExistsError(
                    "Evaluation output đã tồn tại; dùng --resume."
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(contract_path, self._run_contract())
            return []
        if not contract_path.is_file():
            raise FileNotFoundError(
                "Resume yêu cầu evaluation_contract.json."
            )
        stored_contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        if stored_contract != self._run_contract():
            raise ValueError(
                "Evaluation resume contract không khớp artifacts/config."
            )
        if not results_path.exists():
            return []
        records: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()
        for line_number, line in enumerate(
            results_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            payload = json.loads(line)
            if not isinstance(payload, Mapping) or set(payload) != _RESULT_FIELDS:
                raise ValueError(
                    f"results.jsonl dòng {line_number} sai schema."
                )
            pair_id = payload["pair_id"]
            sample_index = payload["sample_index"]
            if not isinstance(pair_id, str) or not isinstance(
                sample_index, int
            ):
                raise TypeError("Result pair/index sai type.")
            key = (pair_id, sample_index)
            if key in seen:
                raise ValueError(f"Result bị trùng: {key}.")
            seen.add(key)
            generated_relative = Path(str(payload["generated_image"]))
            if (
                generated_relative.is_absolute()
                or ".." in generated_relative.parts
            ):
                raise ValueError(
                    "Result generated_image phải nằm trong output directory."
                )
            generated = self.output_dir / generated_relative
            if not generated.is_file():
                raise FileNotFoundError(
                    f"Result PNG bị thiếu: {generated}"
                )
            actual_hash = sha256_file(generated)
            if actual_hash != payload["generated_image_sha256"]:
                raise ValueError(
                    f"Result PNG hash không khớp: {generated}"
                )
            if payload["artifact_sha256"] != self.artifact_sha256:
                raise ValueError(
                    "Result artifact hashes không khớp run."
                )
            records.append(dict(payload))
        return records

    @torch.inference_mode()
    def _generate_pair(
        self,
        pair: Mapping[str, str],
        existing_keys: set[tuple[str, int]],
        records: list[dict[str, object]],
    ) -> None:
        pair_keys = {
            (pair["pair_id"], sample_index)
            for sample_index in range(
                self.config.data.samples_per_pair
            )
        }
        if pair_keys.issubset(existing_keys):
            return
        reference_path = _resolve_image(
            self.config.data.image_root,
            pair["reference_image"],
        )
        processed = self.processor(reference_path)
        reference_image = processed["image"][None].to(self.device)
        reference_mask = processed["valid_mask"][None].to(self.device)
        style = self.model.encode_reference(
            reference_image,
            reference_mask,
        )
        if not torch.equal(
            style.layout_scales,
            torch.ones_like(style.layout_scales),
        ):
            raise ValueError(
                "P0 evaluation yêu cầu neutral layout scales."
            )
        formatted = self.formatter.format(
            pair["target_text"],
            preserve_physical_lines=True,
        )
        text_batch = self.vocabulary.encode_batch(
            [formatted],
            device=self.device,
        )
        for sample_index in range(
            self.config.data.samples_per_pair
        ):
            key = (pair["pair_id"], sample_index)
            if key in existing_keys:
                continue
            seed = stable_sample_seed(
                self.config.base_seed,
                pair["pair_id"],
                sample_index,
            )
            scaled_latent = sample_scaled_latent(
                self.model,
                text_batch.graphemes,
                style,
                latent_height=formatted.output_height // 8,
                latent_width=128,
                config=SamplingConfig(
                    num_inference_steps=(
                        self.config.diffusion.num_inference_steps
                    ),
                    seed=seed,
                ),
                num_train_timesteps=self.num_train_timesteps,
                device=self.device,
            )
            _, image = decode_scaled_latent(
                self.autokl,
                self.statistics,
                scaled_latent,
            )
            relative_path = (
                Path("samples")
                / pair["pair_id"]
                / f"sample_{sample_index:02d}_seed_{seed}.png"
            )
            output_path = self.output_dir / relative_path
            _save_png_atomic(image, output_path)
            record: dict[str, object] = {
                "schema_version": 1,
                "pair_id": pair["pair_id"],
                "canonical_writer_id": pair[
                    "canonical_writer_id"
                ],
                "target_id": pair["target_id"],
                "target_image": pair["target_image"],
                "target_text": pair["target_text"],
                "reference_id": pair["reference_id"],
                "reference_image": pair["reference_image"],
                "sample_index": sample_index,
                "seed": seed,
                "output_height": formatted.output_height,
                "num_inference_steps": (
                    self.config.diffusion.num_inference_steps
                ),
                "generated_image": relative_path.as_posix(),
                "generated_image_sha256": sha256_file(output_path),
                "artifact_sha256": self.artifact_sha256,
            }
            records.append(record)
            records.sort(
                key=lambda item: (
                    str(item["pair_id"]),
                    int(item["sample_index"]),
                )
            )
            _write_jsonl_atomic(
                self.output_dir / "results.jsonl",
                records,
            )
            existing_keys.add(key)

    def _contact_sheets(
        self,
        pairs: Sequence[Mapping[str, str]],
        records: Sequence[Mapping[str, object]],
    ) -> None:
        records_by_pair: dict[str, list[Mapping[str, object]]] = {}
        for record in records:
            records_by_pair.setdefault(
                str(record["pair_id"]),
                [],
            ).append(record)
        sheet_dir = self.output_dir / "contact_sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = self.config.output.contact_sheet_pairs
        cell_width, cell_height, label_height = 300, 260, 24
        columns = 2 + self.config.data.samples_per_pair
        for chunk_index, start in enumerate(
            range(0, len(pairs), chunk_size)
        ):
            chunk = pairs[start : start + chunk_size]
            canvas = Image.new(
                "L",
                (
                    columns * cell_width,
                    len(chunk) * (cell_height + label_height),
                ),
                color=255,
            )
            draw = ImageDraw.Draw(canvas)
            try:
                for row, pair in enumerate(chunk):
                    pair_records = sorted(
                        records_by_pair[pair["pair_id"]],
                        key=lambda record: int(
                            record["sample_index"]
                        ),
                    )
                    paths = [
                        _resolve_image(
                            self.config.data.image_root,
                            pair["target_image"],
                        ),
                        _resolve_image(
                            self.config.data.image_root,
                            pair["reference_image"],
                        ),
                        *[
                            self.output_dir
                            / str(record["generated_image"])
                            for record in pair_records
                        ],
                    ]
                    labels = [
                        f"{pair['pair_id']} target",
                        "reference",
                        *[
                            f"sample {record['sample_index']}"
                            for record in pair_records
                        ],
                    ]
                    for column, (path, label) in enumerate(
                        zip(paths, labels, strict=True)
                    ):
                        with Image.open(path) as source:
                            image = source.convert("L")
                            try:
                                image.thumbnail(
                                    (cell_width, cell_height),
                                    Image.Resampling.LANCZOS,
                                )
                                x = column * cell_width + (
                                    cell_width - image.width
                                ) // 2
                                y = (
                                    row * (cell_height + label_height)
                                    + label_height
                                    + (cell_height - image.height) // 2
                                )
                                canvas.paste(image, (x, y))
                            finally:
                                image.close()
                        draw.text(
                            (
                                column * cell_width + 4,
                                row
                                * (cell_height + label_height)
                                + 4,
                            ),
                            label,
                            fill=0,
                        )
                path = sheet_dir / f"sheet_{chunk_index:04d}.png"
                temporary = path.with_suffix(".tmp")
                try:
                    canvas.save(temporary, format="PNG")
                    temporary.replace(path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            finally:
                canvas.close()

    def run(self, *, resume: bool = False) -> dict[str, object]:
        pairs = load_test_pairs(self.config.data.test_pairs)
        for pair in pairs:
            _resolve_image(
                self.config.data.image_root,
                pair["target_image"],
            )
            _resolve_image(
                self.config.data.image_root,
                pair["reference_image"],
            )
            self.formatter.format(
                pair["target_text"],
                preserve_physical_lines=True,
            )
        records = self._load_existing(resume=resume)
        valid_pair_ids = {pair["pair_id"] for pair in pairs}
        expected_keys = {
            (pair["pair_id"], sample_index)
            for pair in pairs
            for sample_index in range(
                self.config.data.samples_per_pair
            )
        }
        existing_keys = {
            (str(record["pair_id"]), int(record["sample_index"]))
            for record in records
        }
        if any(pair_id not in valid_pair_ids for pair_id, _ in existing_keys):
            raise ValueError(
                "results.jsonl chứa pair không thuộc manifest hiện tại."
            )
        if not existing_keys.issubset(expected_keys):
            raise ValueError("results.jsonl chứa sample_index ngoài contract.")
        for pair in pairs:
            self._generate_pair(
                pair,
                existing_keys,
                records,
            )
        if existing_keys != expected_keys:
            raise RuntimeError("Evaluation chưa sinh đủ fixed pairs.")
        self._contact_sheets(pairs, records)
        summary = {
            "schema_version": 1,
            "pair_count": len(pairs),
            "sample_count": len(records),
            "samples_per_pair": self.config.data.samples_per_pair,
            "base_seed": self.config.base_seed,
            "num_inference_steps": (
                self.config.diffusion.num_inference_steps
            ),
            "artifact_sha256": self.artifact_sha256,
        }
        _write_json_atomic(
            self.output_dir / "summary.json",
            summary,
        )
        return summary


__all__ = [
    "EvaluationConfig",
    "EvaluationDataConfig",
    "EvaluationModelConfig",
    "EvaluationOutputConfig",
    "FixedPairEvaluator",
    "load_evaluation_config",
    "load_test_pairs",
    "stable_sample_seed",
]
