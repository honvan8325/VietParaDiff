"""Score fixed-pair PNGs without rerunning generator sampling."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from vietparadiff.artifacts import sha256_file
from vietparadiff.data.pipeline import HTRVocabulary
from vietparadiff.inference.generator import load_model_config
from vietparadiff.metrics import (
    binary_auc_eer,
    edit_distance,
    mean_pairwise_cosine_distance,
    style_distribution_mmd,
)
from vietparadiff.models.config import WriterEncoderConfig
from vietparadiff.models.grapheme import ParagraphFormatter
from vietparadiff.models.writer import WriterStyleEncoder
from vietparadiff.training.htr import greedy_ctc_decode
from vietparadiff.training.htr_guidance import (
    FrozenHTRTeacher,
    HTRGuidanceConfig,
)
from vietparadiff.training.writer import (
    WriterImageProcessor,
    validate_writer_inference_contract,
)


@dataclass(frozen=True, slots=True)
class ScoreGenerationConfig:
    directory: Path
    results: Path
    evaluation_contract: Path


@dataclass(frozen=True, slots=True)
class ScoreHTRConfig:
    checkpoint: Path
    model_config: Path
    vocabulary: Path
    guidance_checkpoint: Path


@dataclass(frozen=True, slots=True)
class ScoreWriterConfig:
    checkpoint: Path
    model_config: Path
    vocabulary: Path
    contract: Path


@dataclass(frozen=True, slots=True)
class ScoreTextConfig:
    generator_model_config: Path
    image_root: Path


@dataclass(frozen=True, slots=True)
class ScoreMetricConfig:
    foreground_threshold: float
    blank_foreground_fraction: float
    mmd_subset_size: int
    mmd_subsets: int
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 < self.foreground_threshold < 1.0:
            raise ValueError("foreground_threshold phải trong (0,1).")
        if not 0.0 <= self.blank_foreground_fraction < 1.0:
            raise ValueError(
                "blank_foreground_fraction phải trong [0,1)."
            )
        if self.mmd_subset_size < 2 or self.mmd_subsets <= 0:
            raise ValueError("MMD subset settings không hợp lệ.")
        if self.seed < 0:
            raise ValueError("Metric seed không được âm.")


@dataclass(frozen=True, slots=True)
class ScoreOutputConfig:
    directory: Path


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    device: str
    generation: ScoreGenerationConfig
    htr: ScoreHTRConfig
    writer: ScoreWriterConfig
    text: ScoreTextConfig
    metrics: ScoreMetricConfig
    output: ScoreOutputConfig


def _section(
    raw: Mapping[str, object],
    name: str,
    keys: set[str],
) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(keys)}."
        )
    return dict(value)


def load_scoring_config(path: Path) -> ScoringConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy scoring config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = {"device", "generation", "htr", "writer", "text", "metrics", "output"}
    if not isinstance(raw, Mapping) or set(raw) != root:
        raise ValueError("Scoring config root sai schema.")
    generation = _section(
        raw,
        "generation",
        {"directory", "results", "evaluation_contract"},
    )
    htr = _section(
        raw,
        "htr",
        {
            "checkpoint",
            "model_config",
            "vocabulary",
            "guidance_checkpoint",
        },
    )
    writer = _section(
        raw,
        "writer",
        {"checkpoint", "model_config", "vocabulary", "contract"},
    )
    text = _section(
        raw,
        "text",
        {"generator_model_config", "image_root"},
    )
    metrics = _section(
        raw,
        "metrics",
        {
            "foreground_threshold",
            "blank_foreground_fraction",
            "mmd_subset_size",
            "mmd_subsets",
            "seed",
        },
    )
    output = _section(raw, "output", {"directory"})
    device = str(raw["device"])
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device phải là auto/cuda/mps/cpu.")
    return ScoringConfig(
        device=device,
        generation=ScoreGenerationConfig(
            directory=Path(str(generation["directory"])),
            results=Path(str(generation["results"])),
            evaluation_contract=Path(
                str(generation["evaluation_contract"])
            ),
        ),
        htr=ScoreHTRConfig(
            checkpoint=Path(str(htr["checkpoint"])),
            model_config=Path(str(htr["model_config"])),
            vocabulary=Path(str(htr["vocabulary"])),
            guidance_checkpoint=Path(
                str(htr["guidance_checkpoint"])
            ),
        ),
        writer=ScoreWriterConfig(
            checkpoint=Path(str(writer["checkpoint"])),
            model_config=Path(str(writer["model_config"])),
            vocabulary=Path(str(writer["vocabulary"])),
            contract=Path(str(writer["contract"])),
        ),
        text=ScoreTextConfig(
            generator_model_config=Path(
                str(text["generator_model_config"])
            ),
            image_root=Path(str(text["image_root"])),
        ),
        metrics=ScoreMetricConfig(
            foreground_threshold=float(
                metrics["foreground_threshold"]
            ),
            blank_foreground_fraction=float(
                metrics["blank_foreground_fraction"]
            ),
            mmd_subset_size=int(metrics["mmd_subset_size"]),
            mmd_subsets=int(metrics["mmd_subsets"]),
            seed=int(metrics["seed"]),
        ),
        output=ScoreOutputConfig(
            directory=Path(str(output["directory"]))
        ),
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS được yêu cầu nhưng không khả dụng.")
    return device


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy JSONL: {path}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} không phải object.")
        records.append(dict(payload))
    if not records:
        raise ValueError(f"{path} không được rỗng.")
    return records


def _write_json(
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


def _write_jsonl(
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


def _image_tensor(path: Path) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    with Image.open(path) as source:
        image = source.convert("L")
        tensor = torch.frombuffer(
            bytearray(image.tobytes()),
            dtype=torch.uint8,
        ).reshape(image.height, image.width)
    return tensor.float().div(127.5).sub(1.0)[None, None]


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_writer_model(
    config: ScoreWriterConfig,
    *,
    device: torch.device,
) -> WriterStyleEncoder:
    validate_writer_inference_contract(
        checkpoint=config.checkpoint,
        model_config=config.model_config,
        writer_vocabulary=config.vocabulary,
        contract_path=config.contract,
    )
    payload = json.loads(config.model_config.read_text(encoding="utf-8"))
    expected_config = {field.name for field in fields(WriterEncoderConfig)}
    if not isinstance(payload, Mapping) or set(payload) != expected_config:
        raise ValueError("Writer model config sai schema.")
    model = WriterStyleEncoder(
        WriterEncoderConfig(**dict(payload)),  # type: ignore[arg-type]
    )
    model.load_checkpoint(config.checkpoint)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def validate_independent_htr_checkpoints(
    guidance_checkpoint: Path,
    evaluation_checkpoint: Path,
) -> tuple[str, str]:
    """Return both hashes after proving guidance and scoring HTR differ."""
    guidance_hash = sha256_file(guidance_checkpoint)
    evaluation_hash = sha256_file(evaluation_checkpoint)
    if guidance_hash == evaluation_hash:
        raise ValueError(
            "Paper scoring HTR phải độc lập với HTR guidance teacher."
        )
    return guidance_hash, evaluation_hash


def _generated_writer_embedding(
    generated_path: Path,
    layout: Mapping[str, object],
    processor: WriterImageProcessor,
    model: WriterStyleEncoder,
    device: torch.device,
) -> Tensor | None:
    """Skip writer preprocessing deliberately for a truly blank sample."""
    if bool(layout["blank_output"]):
        return None
    writer_image = processor(generated_path)[None].to(device)
    return model(writer_image)[0].cpu()


def _load_real_writer_embeddings(
    results: Sequence[Mapping[str, object]],
    *,
    image_root: Path,
    processor: WriterImageProcessor,
    model: WriterStyleEncoder,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, str]]:
    """Encode the complete real gallery/distribution before generated data."""
    references: dict[str, Tensor] = {}
    targets: dict[str, Tensor] = {}
    reference_writers: dict[str, str] = {}
    reference_sources: dict[str, tuple[str, str]] = {}
    target_sources: dict[str, str] = {}
    for result in results:
        reference_id = str(result["reference_id"])
        target_id = str(result["target_id"])
        writer_id = str(result["canonical_writer_id"])
        reference_value = str(result["reference_image"])
        target_value = str(result["target_image"])
        reference_source = (reference_value, writer_id)
        previous_reference = reference_sources.get(reference_id)
        if (
            previous_reference is not None
            and previous_reference != reference_source
        ):
            raise ValueError(
                f"Reference ID {reference_id} map tới nhiều image/writer."
            )
        previous_target = target_sources.get(target_id)
        if previous_target is not None and previous_target != target_value:
            raise ValueError(
                f"Target ID {target_id} map tới nhiều image."
            )
        reference_sources[reference_id] = reference_source
        target_sources[target_id] = target_value
        if reference_id not in references:
            reference_path = _resolve(image_root, reference_value)
            references[reference_id] = model(
                processor(reference_path)[None].to(device)
            )[0].cpu()
            reference_writers[reference_id] = writer_id
        if target_id not in targets:
            target_path = _resolve(image_root, target_value)
            targets[target_id] = model(
                processor(target_path)[None].to(device)
            )[0].cpu()
    return references, targets, reference_writers


def _inverse(mapping: Mapping[str, int]) -> dict[int, str]:
    return {value: key for key, value in mapping.items()}


def _decode_lines(
    teacher: FrozenHTRTeacher,
    line_batch: Mapping[str, object],
) -> tuple[dict[str, list[list[int]]], dict[str, list[list[int]]]]:
    images = line_batch["images"]
    valid_widths = line_batch["valid_widths"]
    if not isinstance(images, Tensor) or not isinstance(valid_widths, Tensor):
        raise TypeError("HTR routed batch sai tensor contract.")
    output = teacher.model(images, valid_widths)
    hypotheses = {
        "raw": greedy_ctc_decode(
            output.raw_logits,
            output.input_lengths,
        ),
        "base": greedy_ctc_decode(
            output.base_logits,
            output.input_lengths,
        ),
        "shape": greedy_ctc_decode(
            output.shape_logits,
            output.input_lengths,
        ),
        "tone": greedy_ctc_decode(
            output.tone_logits,
            output.input_lengths,
        ),
    }
    target_lengths = line_batch["target_lengths"]
    if not isinstance(target_lengths, Tensor):
        raise TypeError("HTR target_lengths phải là Tensor.")
    targets: dict[str, list[list[int]]] = {}
    for name in ("raw", "base", "shape", "tone"):
        tensor = line_batch[f"{name}_targets"]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name}_targets phải là Tensor.")
        targets[name] = [
            row[: int(length)].detach().cpu().tolist()
            for row, length in zip(
                tensor,
                target_lengths.detach().cpu().tolist(),
                strict=True,
            )
        ]
    return targets, hypotheses


def _content_metrics(
    teacher: FrozenHTRTeacher,
    formatter: ParagraphFormatter,
    image: Tensor,
    text: str,
    output_height: int,
    sample_id: str,
) -> dict[str, object]:
    formatted = formatter.format(
        text,
        output_height=output_height,
        preserve_physical_lines=True,
    )
    formatted_text = "\n".join(formatted.lines)
    slots = formatted.line_slot_mask[None].to(image.device)
    routed = teacher.router(
        image,
        slots,
        [formatted_text],
        teacher.vocabulary,
        sample_ids=[sample_id],
    )
    targets, hypotheses = _decode_lines(teacher, routed)
    inverse = {
        "raw": _inverse(teacher.vocabulary.raw_to_id),
        "base": _inverse(teacher.vocabulary.base_to_id),
        "shape": _inverse(teacher.vocabulary.shape_to_id),
        "tone": _inverse(teacher.vocabulary.tone_to_id),
    }
    output: dict[str, object] = {}
    for name in ("raw", "base", "shape", "tone"):
        edits = sum(
            edit_distance(target, hypothesis)
            for target, hypothesis in zip(
                targets[name],
                hypotheses[name],
                strict=True,
            )
        )
        tokens = sum(len(target) for target in targets[name])
        output[f"{name}_error_rate"] = edits / max(tokens, 1)
    target_lines = [
        "".join(inverse["raw"].get(token, "<unk>") for token in line)
        for line in targets["raw"]
    ]
    predicted_lines = [
        "".join(inverse["raw"].get(token, "<unk>") for token in line)
        for line in hypotheses["raw"]
    ]
    target_paragraph = "\n".join(target_lines)
    predicted_paragraph = "\n".join(predicted_lines)
    output["paragraph_cer"] = (
        edit_distance(target_paragraph, predicted_paragraph)
        / max(len(target_paragraph), 1)
    )
    target_words = target_paragraph.split()
    predicted_words = predicted_paragraph.split()
    output["paragraph_wer"] = (
        edit_distance(target_words, predicted_words)
        / max(len(target_words), 1)
    )
    output["exact_line_accuracy"] = sum(
        target == predicted
        for target, predicted in zip(
            target_lines,
            predicted_lines,
            strict=True,
        )
    ) / len(target_lines)
    output["htr_prediction"] = predicted_paragraph
    return output


def _layout_metrics(
    image: Tensor,
    slots: Tensor,
    config: ScoreMetricConfig,
) -> dict[str, object]:
    ink = ((1.0 - image) / 2.0).clamp(0.0, 1.0)
    foreground = ink >= config.foreground_threshold
    foreground_fraction = float(foreground.float().mean())
    upsampled = F.interpolate(
        slots[None],
        size=image.shape[-2:],
        mode="nearest",
    ).sum(dim=1, keepdim=True) > 0
    foreground_count = foreground.sum().clamp_min(1)
    overflow = (foreground & ~upsampled).sum() / foreground_count
    active_rows = torch.nonzero(
        upsampled[0, 0].any(dim=1),
        as_tuple=False,
    ).flatten()
    if active_rows.numel() > 0:
        span = torch.zeros_like(upsampled)
        span[
            :,
            :,
            int(active_rows[0]) : int(active_rows[-1]) + 1,
            :,
        ] = True
        gap = span & ~upsampled
        interline = (foreground & gap).sum() / foreground_count
    else:
        interline = overflow.new_zeros(())
    return {
        "ink_mean": float(ink.mean()),
        "foreground_fraction": foreground_fraction,
        "blank_output": (
            foreground_fraction
            < config.blank_foreground_fraction
        ),
        "overflow_ink_ratio": float(overflow),
        "interline_bleed_ratio": float(interline),
    }


class EvaluationScorer:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self.device = _device(config.device)
        validate_independent_htr_checkpoints(
            config.htr.guidance_checkpoint,
            config.htr.checkpoint,
        )
        htr_config = HTRGuidanceConfig(
            checkpoint=config.htr.checkpoint,
            model_config=config.htr.model_config,
            vocabulary=config.htr.vocabulary,
            maximum_weight=0.05,
            warmup_steps=5000,
            maximum_timestep=250,
            every_n_optimizer_steps=4,
            raw_weight=1.0,
            base_weight=0.5,
            shape_weight=0.25,
            tone_weight=0.25,
        )
        self.teacher = FrozenHTRTeacher.load(
            htr_config,
            device=self.device,
        )
        self.writer = _load_writer_model(
            config.writer,
            device=self.device,
        )
        self.writer_processor = WriterImageProcessor(
            self.writer.config
        )
        model_config = load_model_config(
            config.text.generator_model_config
        )
        self.formatter = ParagraphFormatter(model_config.text)

    def _contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generation_results_sha256": sha256_file(
                self.config.generation.results
            ),
            "evaluation_contract_sha256": sha256_file(
                self.config.generation.evaluation_contract
            ),
            "htr_checkpoint_sha256": sha256_file(
                self.config.htr.checkpoint
            ),
            "guidance_htr_checkpoint_sha256": sha256_file(
                self.config.htr.guidance_checkpoint
            ),
            "htr_model_config_sha256": sha256_file(
                self.config.htr.model_config
            ),
            "htr_vocabulary_sha256": sha256_file(
                self.config.htr.vocabulary
            ),
            "writer_checkpoint_sha256": sha256_file(
                self.config.writer.checkpoint
            ),
            "writer_model_config_sha256": sha256_file(
                self.config.writer.model_config
            ),
            "writer_vocabulary_sha256": sha256_file(
                self.config.writer.vocabulary
            ),
            "writer_contract_sha256": sha256_file(
                self.config.writer.contract
            ),
            "generator_model_config_sha256": sha256_file(
                self.config.text.generator_model_config
            ),
            "metric_config": {
                field.name: getattr(self.config.metrics, field.name)
                for field in fields(ScoreMetricConfig)
            },
        }

    @torch.inference_mode()
    def run(self, *, resume: bool) -> dict[str, object]:
        output_dir = self.config.output.directory
        contract_path = output_dir / "metrics_contract.json"
        metrics_path = output_dir / "metrics.jsonl"
        expected_contract = self._contract()
        if resume:
            if not contract_path.is_file():
                raise FileNotFoundError(
                    "Scoring resume yêu cầu metrics_contract.json."
                )
            stored = json.loads(contract_path.read_text(encoding="utf-8"))
            if stored != expected_contract:
                raise ValueError(
                    "Scoring resume contract không khớp artifacts."
                )
            if metrics_path.exists():
                existing = _read_jsonl(metrics_path)
                keys = [
                    (record.get("pair_id"), record.get("sample_index"))
                    for record in existing
                ]
                if len(keys) != len(set(keys)):
                    raise ValueError("metrics.jsonl có record trùng.")
        elif contract_path.exists() or metrics_path.exists():
            raise FileExistsError(
                "Scoring output đã tồn tại; dùng --resume."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(contract_path, expected_contract)
        results = _read_jsonl(self.config.generation.results)
        sample_records: list[dict[str, object]] = []
        valid_style_entries: list[tuple[dict[str, object], Tensor]] = []
        generated_embeddings: list[Tensor] = []
        (
            reference_embeddings,
            target_embeddings,
            reference_writers,
        ) = _load_real_writer_embeddings(
            results,
            image_root=self.config.text.image_root,
            processor=self.writer_processor,
            model=self.writer,
            device=self.device,
        )
        embeddings_by_pair: dict[str, list[Tensor]] = defaultdict(list)
        for result in results:
            generated_path = (
                self.config.generation.directory
                / Path(str(result["generated_image"]))
            )
            if sha256_file(generated_path) != result["generated_image_sha256"]:
                raise ValueError(
                    f"Generated PNG hash không khớp: {generated_path}"
                )
            image = _image_tensor(generated_path).to(self.device)
            formatted = self.formatter.format(
                str(result["target_text"]),
                output_height=int(result["output_height"]),
                preserve_physical_lines=True,
            )
            content = _content_metrics(
                self.teacher,
                self.formatter,
                image,
                str(result["target_text"]),
                int(result["output_height"]),
                f"{result['pair_id']}:{result['sample_index']}",
            )
            layout = _layout_metrics(
                image,
                formatted.line_slot_mask.to(self.device),
                self.config.metrics,
            )
            target_id = str(result["target_id"])
            reference_id = str(result["reference_id"])
            record: dict[str, object] = {
                "schema_version": 1,
                "pair_id": result["pair_id"],
                "sample_index": result["sample_index"],
                "canonical_writer_id": result[
                    "canonical_writer_id"
                ],
                "target_id": target_id,
                "reference_id": reference_id,
                "generated_image": result["generated_image"],
                "generated_image_sha256": result[
                    "generated_image_sha256"
                ],
                **content,
                **layout,
                "artifact_sha256": expected_contract,
            }
            generated_embedding = _generated_writer_embedding(
                generated_path,
                layout,
                self.writer_processor,
                self.writer,
                self.device,
            )
            if generated_embedding is None:
                record.update(
                    {
                        "style_metric_valid": False,
                        "style_cosine": None,
                        "retrieved_writer_id": None,
                        "writer_retrieval_correct": None,
                    }
                )
            else:
                generated_embeddings.append(generated_embedding)
                embeddings_by_pair[str(result["pair_id"])].append(
                    generated_embedding
                )
                record.update(
                    {
                        "style_metric_valid": True,
                        "style_cosine": float(
                            generated_embedding
                            @ reference_embeddings[reference_id]
                        ),
                    }
                )
                valid_style_entries.append(
                    (record, generated_embedding)
                )
            sample_records.append(record)

        writer_centroids: dict[str, Tensor] = {}
        grouped_references: dict[str, list[Tensor]] = defaultdict(list)
        for reference_id, embedding in reference_embeddings.items():
            grouped_references[reference_writers[reference_id]].append(
                embedding
            )
        for writer_id, embeddings in grouped_references.items():
            writer_centroids[writer_id] = F.normalize(
                torch.stack(embeddings).mean(dim=0),
                dim=0,
            )
        verification_labels: list[bool] = []
        verification_scores: list[float] = []
        retrieval_correct = 0
        for record, embedding in valid_style_entries:
            similarities = {
                writer_id: float(embedding @ centroid)
                for writer_id, centroid in writer_centroids.items()
            }
            predicted_writer = max(similarities, key=similarities.__getitem__)
            target_writer = str(record["canonical_writer_id"])
            retrieval_correct += int(predicted_writer == target_writer)
            record["retrieved_writer_id"] = predicted_writer
            record["writer_retrieval_correct"] = (
                predicted_writer == target_writer
            )
            for writer_id, score in similarities.items():
                verification_labels.append(writer_id == target_writer)
                verification_scores.append(score)
        if (
            verification_labels
            and any(verification_labels)
            and not all(verification_labels)
        ):
            auc, eer = binary_auc_eer(
                verification_labels,
                verification_scores,
            )
        else:
            auc, eer = None, None
        if len(target_embeddings) >= 2 and len(
            generated_embeddings
        ) >= 2:
            mmd_mean, mmd_subset_std = style_distribution_mmd(
                torch.stack(list(target_embeddings.values())),
                torch.stack(generated_embeddings),
                subset_size=self.config.metrics.mmd_subset_size,
                subsets=self.config.metrics.mmd_subsets,
                seed=self.config.metrics.seed,
            )
        else:
            mmd_mean, mmd_subset_std = None, None
        all_pair_ids = {
            str(record["pair_id"]) for record in sample_records
        }
        diversity = {
            pair_id: (
                mean_pairwise_cosine_distance(
                    torch.stack(embeddings)
                )
                if len(embeddings) >= 2
                else None
            )
            for pair_id in all_pair_ids
            for embeddings in [embeddings_by_pair.get(pair_id, [])]
        }
        for record in sample_records:
            record["pair_diversity"] = diversity.get(
                str(record["pair_id"])
            )
        _write_jsonl(metrics_path, sample_records)

        scalar_names = (
            "paragraph_cer",
            "paragraph_wer",
            "exact_line_accuracy",
            "base_error_rate",
            "shape_error_rate",
            "tone_error_rate",
            "ink_mean",
            "foreground_fraction",
            "overflow_ink_ratio",
            "interline_bleed_ratio",
        )
        valid_style_count = len(valid_style_entries)
        summary: dict[str, object] = {
            "schema_version": 1,
            "sample_count": len(sample_records),
            "pair_count": len(all_pair_ids),
            "writer_retrieval_accuracy": (
                retrieval_correct / valid_style_count
                if valid_style_count > 0
                else None
            ),
            "writer_verification_auc": auc,
            "writer_verification_eer": eer,
            "style_metric_coverage": (
                valid_style_count / len(sample_records)
            ),
            "style_cosine": (
                sum(
                    float(record["style_cosine"])
                    for record, _ in valid_style_entries
                )
                / valid_style_count
                if valid_style_count > 0
                else None
            ),
            "style_distribution_mmd_mean": mmd_mean,
            "style_distribution_mmd_subset_std": mmd_subset_std,
            "pair_diversity": (
                sum(
                    float(value)
                    for value in diversity.values()
                    if value is not None
                )
                / sum(
                    value is not None
                    for value in diversity.values()
                )
                if any(
                    value is not None
                    for value in diversity.values()
                )
                else None
            ),
            "blank_output_rate": sum(
                bool(record["blank_output"])
                for record in sample_records
            )
            / len(sample_records),
            "artifact_sha256": expected_contract,
        }
        for name in scalar_names:
            summary[name] = sum(
                float(record[name]) for record in sample_records
            ) / len(sample_records)
        per_writer: dict[str, dict[str, float]] = {}
        grouped_records: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in sample_records:
            grouped_records[str(record["canonical_writer_id"])].append(
                record
            )
        for writer_id, writer_records in grouped_records.items():
            per_writer[writer_id] = {
                name: sum(
                    float(record[name]) for record in writer_records
                )
                / len(writer_records)
                for name in scalar_names
            }
            valid_writer_styles = [
                record
                for record in writer_records
                if bool(record["style_metric_valid"])
            ]
            per_writer[writer_id]["style_metric_coverage"] = (
                len(valid_writer_styles) / len(writer_records)
            )
            if valid_writer_styles:
                per_writer[writer_id]["style_cosine"] = sum(
                    float(record["style_cosine"])
                    for record in valid_writer_styles
                ) / len(valid_writer_styles)
        summary["per_writer"] = per_writer
        _write_json(output_dir / "metrics_summary.json", summary)
        return summary


__all__ = [
    "EvaluationScorer",
    "ScoringConfig",
    "load_scoring_config",
    "validate_independent_htr_checkpoints",
]
