"""Sequential multi-seed experiment DAG and statistical aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    use_shape_condition: bool
    use_tone_condition: bool
    use_high_frequency_style: bool
    use_local_style_tokens: bool
    use_harmonizer: bool
    use_synthetic_data: bool
    use_htr_guidance: bool


@dataclass(frozen=True, slots=True)
class ExternalBaselineExperiment:
    name: str
    config_pattern: str
    output_pattern: str

    def __post_init__(self) -> None:
        if self.name not in {"one_dm", "paragraph_ldm"}:
            raise ValueError(
                "External experiment phải là one_dm/paragraph_ldm."
            )
        for value, label in (
            (self.config_pattern, "config_pattern"),
            (self.output_pattern, "output_pattern"),
        ):
            if "{seed}" not in value:
                raise ValueError(
                    f"External {label} phải chứa {{seed}}."
                )

    def config_path(self, seed: int) -> Path:
        return Path(self.config_pattern.format(seed=seed))

    def output_dir(self, seed: int) -> Path:
        return Path(self.output_pattern.format(seed=seed))


@dataclass(frozen=True, slots=True)
class PaperPreflightConfig:
    data_audit: Path
    data_split_root: Path
    data_image_root: Path
    vision_contract: Path
    convnext_checkpoint: Path
    resnet_checkpoint: Path
    autokl_checkpoint: Path
    latent_statistics: Path
    guidance_htr_checkpoint: Path
    guidance_htr_model_config: Path
    guidance_htr_vocabulary: Path
    evaluation_htr_checkpoint: Path
    evaluation_htr_model_config: Path
    evaluation_htr_vocabulary: Path
    writer_checkpoint: Path
    writer_model_config: Path
    writer_vocabulary: Path
    writer_contract: Path


@dataclass(frozen=True, slots=True)
class PaperExperimentConfig:
    seeds: tuple[int, ...]
    output_root: Path
    require_clean_git: bool
    pretrain_config: Path
    finetune_config: Path
    htr_guided_config: Path
    evaluate_config: Path
    metrics_config: Path
    variants: tuple[ExperimentVariant, ...]
    external_baselines: tuple[ExternalBaselineExperiment, ...]
    preflight: PaperPreflightConfig

    def __post_init__(self) -> None:
        if self.seeds != (42, 43, 44):
            raise ValueError("Paper training seeds phải là (42,43,44).")
        expected_names = ("a0", "a1", "a2", "a3", "a4", "full")
        if tuple(variant.name for variant in self.variants) != expected_names:
            raise ValueError(
                f"Experiment variants phải là {expected_names}."
            )
        expected_flags = {
            "a0": (False, False, False, False, False, False, False),
            "a1": (True, True, False, False, False, False, False),
            "a2": (True, True, True, True, False, False, False),
            "a3": (True, True, True, True, True, False, False),
            "a4": (True, True, True, True, True, True, False),
            "full": (True, True, True, True, True, True, True),
        }
        for variant in self.variants:
            actual = (
                variant.use_shape_condition,
                variant.use_tone_condition,
                variant.use_high_frequency_style,
                variant.use_local_style_tokens,
                variant.use_harmonizer,
                variant.use_synthetic_data,
                variant.use_htr_guidance,
            )
            if actual != expected_flags[variant.name]:
                raise ValueError(
                    f"Variant {variant.name} không đúng cumulative matrix."
                )
        if tuple(item.name for item in self.external_baselines) != (
            "one_dm",
            "paragraph_ldm",
        ):
            raise ValueError(
                "External baselines phải gồm one_dm và paragraph_ldm."
            )


def load_experiment_config(path: Path) -> PaperExperimentConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy experiment config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "seeds",
        "output_root",
        "require_clean_git",
        "base_configs",
        "variants",
        "external_baselines",
        "preflight",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Experiment config root sai schema.")
    base = raw["base_configs"]
    if not isinstance(base, Mapping) or set(base) != {
        "pretrain",
        "finetune",
        "htr_guided",
        "evaluate",
        "metrics",
    }:
        raise ValueError("Experiment base_configs sai schema.")
    seeds = raw["seeds"]
    variants = raw["variants"]
    external = raw["external_baselines"]
    preflight_raw = raw["preflight"]
    preflight_keys = {
        field.name for field in fields(PaperPreflightConfig)
    }
    if (
        not isinstance(preflight_raw, Mapping)
        or set(preflight_raw) != preflight_keys
    ):
        raise ValueError("Experiment preflight sai schema.")
    if not isinstance(raw["require_clean_git"], bool):
        raise TypeError("require_clean_git phải là bool.")
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or not isinstance(variants, Sequence)
        or isinstance(variants, (str, bytes))
        or not isinstance(external, Sequence)
        or isinstance(external, (str, bytes))
    ):
        raise TypeError("seeds/variants phải là sequences.")
    parsed_variants: list[ExperimentVariant] = []
    variant_keys = {
        "name",
        "use_shape_condition",
        "use_tone_condition",
        "use_high_frequency_style",
        "use_local_style_tokens",
        "use_harmonizer",
        "use_synthetic_data",
        "use_htr_guidance",
    }
    for item in variants:
        if not isinstance(item, Mapping) or set(item) != variant_keys:
            raise ValueError("Experiment variant sai schema.")
        for key in variant_keys - {"name"}:
            if not isinstance(item[key], bool):
                raise TypeError(f"Variant {key} phải là bool.")
        parsed_variants.append(
            ExperimentVariant(
                name=str(item["name"]),
                use_shape_condition=item["use_shape_condition"],
                use_tone_condition=item["use_tone_condition"],
                use_high_frequency_style=item[
                    "use_high_frequency_style"
                ],
                use_local_style_tokens=item[
                    "use_local_style_tokens"
                ],
                use_harmonizer=item["use_harmonizer"],
                use_synthetic_data=item["use_synthetic_data"],
                use_htr_guidance=item["use_htr_guidance"],
            )
        )
    parsed_external: list[ExternalBaselineExperiment] = []
    for item in external:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "config_pattern",
            "output_pattern",
        }:
            raise ValueError("External baseline experiment sai schema.")
        parsed_external.append(
            ExternalBaselineExperiment(
                name=str(item["name"]),
                config_pattern=str(item["config_pattern"]),
                output_pattern=str(item["output_pattern"]),
            )
        )
    return PaperExperimentConfig(
        seeds=tuple(int(seed) for seed in seeds),
        output_root=Path(str(raw["output_root"])),
        require_clean_git=raw["require_clean_git"],
        pretrain_config=Path(str(base["pretrain"])),
        finetune_config=Path(str(base["finetune"])),
        htr_guided_config=Path(str(base["htr_guided"])),
        evaluate_config=Path(str(base["evaluate"])),
        metrics_config=Path(str(base["metrics"])),
        variants=tuple(parsed_variants),
        external_baselines=tuple(parsed_external),
        preflight=PaperPreflightConfig(
            **{
                key: Path(str(preflight_raw[key]))
                for key in preflight_keys
            }
        ),
    )


def _git(command: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_provenance(*, allow_dirty: bool) -> dict[str, object]:
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"])
    if status and not allow_dirty:
        raise RuntimeError(
            "Paper experiment yêu cầu clean Git worktree; "
            "dùng --allow-dirty để record patch hash."
        )
    patch_hash = None
    if status:
        digest = hashlib.sha256()
        digest.update(
            subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
        )
        digest.update(status.encode("utf-8"))
        for line in status.splitlines():
            if line.startswith("?? "):
                path = Path(line[3:])
                if path.is_file():
                    digest.update(path.as_posix().encode("utf-8"))
                    digest.update(path.read_bytes())
        patch_hash = digest.hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_patch_sha256": patch_hash,
    }


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config {path} phải là mapping.")
    return dict(payload)


def _save_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(
            dict(payload),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_path(
    payload: dict[str, object],
    path: Sequence[str],
    value: object,
) -> None:
    current: dict[str, object] = payload
    for key in path[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            if not isinstance(nested, Mapping):
                raise ValueError(
                    f"Không thể override config path {'.'.join(path)}."
                )
            nested = dict(nested)
            current[key] = nested
        current = nested
    current[path[-1]] = value


def resolved_pipeline_configs(
    config: PaperExperimentConfig,
    variant: ExperimentVariant,
    seed: int,
) -> list[tuple[str, dict[str, object], list[str]]]:
    run_dir = config.output_root / variant.name / f"seed_{seed}"
    pretrain_dir = run_dir / "pretrain"
    finetune_dir = run_dir / "finetune"
    guided_dir = run_dir / "htr_guided"
    evaluation_dir = run_dir / "evaluation"

    pretrain = _load_yaml(config.pretrain_config)
    _set_path(pretrain, ("seed",), seed)
    _set_path(
        pretrain,
        ("checkpoint", "output_dir"),
        str(pretrain_dir),
    )
    for name in (
        "use_shape_condition",
        "use_tone_condition",
        "use_high_frequency_style",
        "use_local_style_tokens",
        "use_harmonizer",
    ):
        _set_path(
            pretrain,
            ("behavior", name),
            getattr(variant, name),
        )

    finetune = _load_yaml(config.finetune_config)
    _set_path(finetune, ("seed",), seed)
    _set_path(
        finetune,
        ("checkpoint", "output_dir"),
        str(finetune_dir),
    )
    _set_path(
        finetune,
        ("data", "use_synthetic_data"),
        variant.use_synthetic_data,
    )
    if not variant.use_synthetic_data:
        _set_path(finetune, ("data", "synthetic_targets"), None)
    for name, filename in (
        ("checkpoint", "best.pt"),
        ("contract", "inference_contract.json"),
        ("model_config", "model_config.json"),
        ("vocabulary", "grapheme_vocabulary.json"),
    ):
        _set_path(
            finetune,
            ("initialization", name),
            str(pretrain_dir / filename),
        )

    stages: list[tuple[str, dict[str, object], list[str]]] = [
        (
            "pretrain",
            pretrain,
            ["uv", "run", "python", "scripts/train_generator.py"],
        ),
        (
            "finetune",
            finetune,
            ["uv", "run", "python", "scripts/train_generator.py"],
        ),
    ]
    selected_dir = finetune_dir
    if variant.use_htr_guidance:
        guided = _load_yaml(config.htr_guided_config)
        _set_path(guided, ("seed",), seed)
        _set_path(
            guided,
            ("checkpoint", "output_dir"),
            str(guided_dir),
        )
        for name, filename in (
            ("checkpoint", "best.pt"),
            ("contract", "inference_contract.json"),
            ("model_config", "model_config.json"),
            ("vocabulary", "grapheme_vocabulary.json"),
        ):
            _set_path(
                guided,
                ("initialization", name),
                str(finetune_dir / filename),
            )
        stages.append(
            (
                "htr_guided",
                guided,
                ["uv", "run", "python", "scripts/train_generator.py"],
            )
        )
        selected_dir = guided_dir

    evaluate = _load_yaml(config.evaluate_config)
    _set_path(evaluate, ("base_seed",), 42)
    _set_path(
        evaluate,
        ("output", "directory"),
        str(evaluation_dir),
    )
    for name, filename in (
        ("checkpoint", "best.pt"),
        ("contract", "inference_contract.json"),
        ("model_config", "model_config.json"),
        ("vocabulary", "grapheme_vocabulary.json"),
    ):
        _set_path(
            evaluate,
            ("model", name),
            str(selected_dir / filename),
        )
    stages.append(
        (
            "evaluate",
            evaluate,
            ["uv", "run", "python", "scripts/evaluate.py"],
        )
    )

    metrics = _load_yaml(config.metrics_config)
    _set_path(
        metrics,
        ("generation", "directory"),
        str(evaluation_dir),
    )
    _set_path(
        metrics,
        ("generation", "results"),
        str(evaluation_dir / "results.jsonl"),
    )
    _set_path(
        metrics,
        ("generation", "evaluation_contract"),
        str(evaluation_dir / "evaluation_contract.json"),
    )
    _set_path(
        metrics,
        ("text", "generator_model_config"),
        str(selected_dir / "model_config.json"),
    )
    _set_path(
        metrics,
        ("metrics", "seed"),
        42,
    )
    _set_path(
        metrics,
        ("output", "directory"),
        str(evaluation_dir),
    )
    stages.append(
        (
            "score",
            metrics,
            ["uv", "run", "python", "scripts/score_evaluation.py"],
        )
    )
    return stages


def _runtime_manifest(
    provenance: Mapping[str, object],
    *,
    seed: int,
    variant: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "variant": variant,
        "seed": seed,
        **dict(provenance),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "started_at_unix": time.time(),
        "finished_at_unix": None,
        "stages": {},
    }


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
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


def validate_data_audit_report(
    report_path: Path,
    *,
    split_root: Path,
    image_root: Path,
) -> dict[str, object]:
    from vietparadiff.data.audit import compute_dataset_snapshot

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Thiếu full-data audit report: {report_path}"
        )
    audit = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "split_root",
        "image_root",
        "manifest_sha256",
        "image_inventory_sha256",
        "dataset_snapshot_sha256",
        "snapshot_image_count",
        "error_count",
    }
    if (
        not isinstance(audit, Mapping)
        or not required.issubset(audit)
        or audit.get("schema_version") != 2
        or not isinstance(audit.get("error_count"), int)
    ):
        raise ValueError("Full-data audit report schema v2 không hợp lệ.")
    if audit["error_count"] != 0:
        raise ValueError(
            "Paper preflight từ chối data audit có "
            f"error_count={audit['error_count']}."
        )
    if (
        audit["split_root"] != str(split_root)
        or audit["image_root"] != str(image_root)
    ):
        raise ValueError(
            "Data audit split_root/image_root không khớp paper config."
        )
    current = compute_dataset_snapshot(
        split_root,
        image_root=image_root,
    )
    expected_fields = current.report_fields()
    for name, actual in expected_fields.items():
        if audit.get(name) != actual:
            raise ValueError(
                "Data audit đã stale; phải chạy lại audit_dataset.py "
                f"(mismatch {name})."
            )
    return dict(audit)


def run_paper_preflight(
    config: PaperExperimentConfig,
) -> dict[str, object]:
    from vietparadiff.artifacts import (
        LatentStatistics,
        sha256_file,
        verify_visual_backbone,
    )
    from vietparadiff.baselines import (
        load_external_baseline_config,
        preflight_external_baseline,
    )
    from vietparadiff.evaluation.fixed_pairs import (
        load_evaluation_config,
    )
    from vietparadiff.evaluation.scoring import load_scoring_config
    from vietparadiff.models import (
        AutoKLConfig,
        HTRConfig,
        HandwritingAutoKL,
        VietnameseHTR,
        WriterEncoderConfig,
        WriterStyleEncoder,
    )
    from vietparadiff.training.generator import (
        load_vietparadiff_training_config,
    )
    from vietparadiff.training.htr import (
        load_htr_training_config,
        validate_htr_training_contract,
    )
    from vietparadiff.training.htr_guidance import (
        HTRGuidanceConfig,
        load_htr_model_config,
        validate_htr_inference_contract,
    )
    from vietparadiff.training.writer import (
        load_writer_metric_config,
        validate_writer_inference_contract,
    )

    preflight = config.preflight
    validate_data_audit_report(
        preflight.data_audit,
        split_root=preflight.data_split_root,
        image_root=preflight.data_image_root,
    )
    convnext_hash = verify_visual_backbone(
        preflight.vision_contract,
        name="convnext_tiny_imagenet1k_v1",
        checkpoint=preflight.convnext_checkpoint,
    )
    resnet_hash = verify_visual_backbone(
        preflight.vision_contract,
        name="resnet18_imagenet1k_v1",
        checkpoint=preflight.resnet_checkpoint,
    )
    statistics_payload = json.loads(
        preflight.latent_statistics.read_text(encoding="utf-8")
    )
    if not isinstance(statistics_payload, Mapping):
        raise ValueError("Latent statistics phải là mapping.")
    statistics = LatentStatistics(**dict(statistics_payload))
    autokl_hash = sha256_file(preflight.autokl_checkpoint)
    if statistics.autokl_checkpoint_sha256 != autokl_hash:
        raise ValueError(
            "Latent statistics không khớp AutoKL checkpoint."
        )
    autokl = HandwritingAutoKL(AutoKLConfig())
    autokl.load_checkpoint(preflight.autokl_checkpoint)
    del autokl

    def validate_htr(
        checkpoint: Path,
        model_config: Path,
        vocabulary: Path,
    ) -> str:
        guidance = HTRGuidanceConfig(
            checkpoint=checkpoint,
            model_config=model_config,
            vocabulary=vocabulary,
            maximum_weight=0.05,
            warmup_steps=5000,
            maximum_timestep=250,
            every_n_optimizer_steps=4,
            raw_weight=1.0,
            base_weight=0.5,
            shape_weight=0.25,
            tone_weight=0.25,
        )
        validate_htr_inference_contract(guidance)
        htr_config = load_htr_model_config(model_config)
        if not isinstance(htr_config, HTRConfig):
            raise RuntimeError("HTR model config loader sai type.")
        model = VietnameseHTR(htr_config)
        model.load_checkpoint(checkpoint)
        del model
        return sha256_file(checkpoint)

    guidance_htr_hash = validate_htr(
        preflight.guidance_htr_checkpoint,
        preflight.guidance_htr_model_config,
        preflight.guidance_htr_vocabulary,
    )
    evaluation_htr_hash = validate_htr(
        preflight.evaluation_htr_checkpoint,
        preflight.evaluation_htr_model_config,
        preflight.evaluation_htr_vocabulary,
    )
    if guidance_htr_hash == evaluation_htr_hash:
        raise ValueError(
            "Paper scoring HTR phải độc lập với HTR guidance teacher."
        )
    guidance_htr_training = load_htr_training_config(
        Path("configs/htr/train.yaml")
    )
    evaluation_htr_training = load_htr_training_config(
        Path("configs/htr/eval.yaml")
    )
    htr_artifact_bindings = (
        (
            "guidance",
            guidance_htr_training,
            preflight.guidance_htr_checkpoint,
            preflight.guidance_htr_model_config,
            preflight.guidance_htr_vocabulary,
        ),
        (
            "evaluation",
            evaluation_htr_training,
            preflight.evaluation_htr_checkpoint,
            preflight.evaluation_htr_model_config,
            preflight.evaluation_htr_vocabulary,
        ),
    )
    for (
        name,
        training_config,
        checkpoint,
        model_config,
        vocabulary,
    ) in htr_artifact_bindings:
        output_dir = training_config.checkpoint.output_dir
        expected_paths = (
            output_dir / "best.pt",
            output_dir / "model_config.json",
            output_dir / "vocabulary.json",
        )
        actual_paths = (checkpoint, model_config, vocabulary)
        if any(
            expected.resolve() != actual.resolve()
            for expected, actual in zip(
                expected_paths,
                actual_paths,
                strict=True,
            )
        ):
            raise ValueError(
                f"Paper preflight {name} HTR artifact paths "
                "không khớp training config."
            )
    guidance_training_contract = validate_htr_training_contract(
        guidance_htr_training
    )
    evaluation_training_contract = validate_htr_training_contract(
        evaluation_htr_training
    )
    if (
        guidance_training_contract["seed"]
        == evaluation_training_contract["seed"]
    ):
        raise ValueError(
            "Guidance và evaluation HTR phải dùng seed khác nhau."
        )
    if (
        guidance_training_contract["augmentation"]
        == evaluation_training_contract["augmentation"]
    ):
        raise ValueError(
            "Guidance và evaluation HTR phải dùng augmentation khác nhau."
        )
    validate_writer_inference_contract(
        checkpoint=preflight.writer_checkpoint,
        model_config=preflight.writer_model_config,
        writer_vocabulary=preflight.writer_vocabulary,
        contract_path=preflight.writer_contract,
    )
    writer_payload = json.loads(
        preflight.writer_model_config.read_text(encoding="utf-8")
    )
    if not isinstance(writer_payload, Mapping):
        raise ValueError("Writer model config phải là mapping.")
    writer = WriterStyleEncoder(
        WriterEncoderConfig(**dict(writer_payload))
    )
    writer.load_checkpoint(preflight.writer_checkpoint)
    del writer

    for path in (
        config.pretrain_config,
        config.finetune_config,
        config.htr_guided_config,
    ):
        load_vietparadiff_training_config(path)
    load_writer_metric_config(Path("configs/writer_metric/train.yaml"))
    load_evaluation_config(config.evaluate_config)
    scoring = load_scoring_config(config.metrics_config)
    scoring_htr_paths = (
        scoring.htr.guidance_checkpoint,
        scoring.htr.checkpoint,
        scoring.htr.model_config,
        scoring.htr.vocabulary,
    )
    preflight_htr_paths = (
        preflight.guidance_htr_checkpoint,
        preflight.evaluation_htr_checkpoint,
        preflight.evaluation_htr_model_config,
        preflight.evaluation_htr_vocabulary,
    )
    if any(
        scoring_path.resolve() != preflight_path.resolve()
        for scoring_path, preflight_path in zip(
            scoring_htr_paths,
            preflight_htr_paths,
            strict=True,
        )
    ):
        raise ValueError(
            "Scoring HTR artifacts không khớp paper preflight."
        )
    if (
        sha256_file(scoring.htr.guidance_checkpoint)
        == sha256_file(scoring.htr.checkpoint)
    ):
        raise ValueError("Scoring config tái sử dụng guidance HTR.")
    with tempfile.TemporaryDirectory(
        prefix="vietparadiff_preflight_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for variant in config.variants:
            for seed in config.seeds:
                for stage, payload, _ in resolved_pipeline_configs(
                    config,
                    variant,
                    seed,
                ):
                    path = (
                        temporary_root
                        / variant.name
                        / f"{seed}_{stage}.yaml"
                    )
                    _save_yaml(path, payload)
                    if stage in {
                        "pretrain",
                        "finetune",
                        "htr_guided",
                    }:
                        load_vietparadiff_training_config(path)
                    elif stage == "evaluate":
                        load_evaluation_config(path)
                    elif stage == "score":
                        load_scoring_config(path)
                    else:
                        raise ValueError(
                            f"Resolved stage không hợp lệ: {stage}."
                        )

    baseline_hashes: dict[str, dict[int, str]] = {}
    for baseline in config.external_baselines:
        baseline_hashes[baseline.name] = {}
        for seed in config.seeds:
            baseline_config = load_external_baseline_config(
                baseline.config_path(seed)
            )
            preflight_external_baseline(
                baseline_config,
                require_generator_model_config=False,
            )
            baseline_hashes[baseline.name][seed] = (
                baseline_config.checkpoint_sha256
            )
    return {
        "data_audit_sha256": sha256_file(preflight.data_audit),
        "convnext_checkpoint_sha256": convnext_hash,
        "resnet_checkpoint_sha256": resnet_hash,
        "autokl_checkpoint_sha256": autokl_hash,
        "guidance_htr_checkpoint_sha256": guidance_htr_hash,
        "evaluation_htr_checkpoint_sha256": evaluation_htr_hash,
        "guidance_htr_training_contract_sha256": sha256_file(
            guidance_htr_training.checkpoint.output_dir
            / "training_contract.json"
        ),
        "evaluation_htr_training_contract_sha256": sha256_file(
            evaluation_htr_training.checkpoint.output_dir
            / "training_contract.json"
        ),
        "writer_checkpoint_sha256": sha256_file(
            preflight.writer_checkpoint
        ),
        "external_baselines": baseline_hashes,
    }


def _stage_artifacts(
    stage: str,
    payload: Mapping[str, object],
) -> dict[str, str]:
    if stage in {"pretrain", "finetune", "htr_guided"}:
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{stage} checkpoint config sai schema.")
        root = Path(str(checkpoint["output_dir"]))
        names = (
            "best.pt",
            "last.pt",
            "inference_contract.json",
            "model_config.json",
            "grapheme_vocabulary.json",
            "training_lineage.json",
        )
    elif stage == "baseline":
        root = Path(str(payload["output_dir"]))
        names = (
            "evaluation_contract.json",
            "results.jsonl",
            "summary.json",
        )
    elif stage == "evaluate":
        output = payload.get("output")
        if not isinstance(output, Mapping):
            raise ValueError("evaluate output config sai schema.")
        root = Path(str(output["directory"]))
        names = (
            "evaluation_contract.json",
            "results.jsonl",
            "summary.json",
        )
    elif stage == "score":
        output = payload.get("output")
        if not isinstance(output, Mapping):
            raise ValueError("score output config sai schema.")
        root = Path(str(output["directory"]))
        names = (
            "metrics_contract.json",
            "metrics.jsonl",
            "metrics_summary.json",
        )
    else:
        raise ValueError(f"Experiment stage không hợp lệ: {stage}.")
    snapshot: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage {stage} thiếu expected artifact: {path}"
            )
        snapshot[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _validate_completed_stage(
    previous: Mapping[str, object],
    *,
    current_config_hash: str,
    command: Sequence[str],
    actual_artifacts: Mapping[str, str],
    label: str,
) -> None:
    if previous.get("resolved_config_sha256") != current_config_hash:
        raise ValueError(f"Resume config mismatch tại {label}.")
    if previous.get("command") != list(command):
        raise ValueError(f"Resume command mismatch tại {label}.")
    if previous.get("artifact_sha256") != dict(actual_artifacts):
        raise ValueError(f"Completed {label} artifact mismatch.")


class PaperExperimentRunner:
    def __init__(
        self,
        config: PaperExperimentConfig,
        *,
        allow_dirty: bool,
    ) -> None:
        self.config = config
        self.provenance = git_provenance(
            allow_dirty=(
                allow_dirty or not config.require_clean_git
            )
        )
        self.preflight_artifacts: dict[str, object] | None = None

    def run(self, *, dry_run: bool, resume: bool) -> None:
        if not dry_run:
            self.preflight_artifacts = run_paper_preflight(
                self.config
            )
        for variant in self.config.variants:
            for seed in self.config.seeds:
                run_dir = (
                    self.config.output_root
                    / variant.name
                    / f"seed_{seed}"
                )
                manifest_path = run_dir / "run_manifest.json"
                if manifest_path.exists():
                    if not resume:
                        raise FileExistsError(
                            f"Run đã tồn tại: {run_dir}; dùng --resume."
                        )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(manifest, dict):
                        raise ValueError("run_manifest.json sai schema.")
                    for key, value in self.provenance.items():
                        if manifest.get(key) != value:
                            raise ValueError(
                                f"Run provenance mismatch tại {key}."
                            )
                else:
                    manifest = _runtime_manifest(
                        self.provenance,
                        seed=seed,
                        variant=variant.name,
                    )
                    manifest["preflight_artifacts"] = (
                        self.preflight_artifacts
                    )
                if manifest.get("preflight_artifacts") != (
                    self.preflight_artifacts
                ):
                    raise ValueError(
                        "Run preflight artifacts mismatch."
                    )
                for stage, payload, base_command in resolved_pipeline_configs(
                    self.config,
                    variant,
                    seed,
                ):
                    config_path = run_dir / "configs" / f"{stage}.yaml"
                    command = [
                        *base_command,
                        "--config",
                        str(config_path),
                    ]
                    if dry_run:
                        print(" ".join(command))
                        continue
                    _save_yaml(config_path, payload)
                    current_config_hash = hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest()
                    stage_state = manifest["stages"]  # type: ignore[index]
                    if not isinstance(stage_state, dict):
                        raise ValueError("run manifest stages sai schema.")
                    previous = stage_state.get(stage)
                    if isinstance(previous, Mapping) and previous.get(
                        "status"
                    ) == "complete":
                        _validate_completed_stage(
                            previous,
                            current_config_hash=current_config_hash,
                            command=command,
                            actual_artifacts=_stage_artifacts(
                                stage,
                                payload,
                            ),
                            label=f"stage {stage}",
                        )
                        continue
                    executed_command = list(command)
                    if resume:
                        if stage in {"pretrain", "finetune", "htr_guided"}:
                            last = (
                                Path(str(payload["checkpoint"]["output_dir"]))  # type: ignore[index]
                                / "last.pt"
                            )
                            if last.is_file():
                                executed_command.extend(
                                    ["--resume", str(last)]
                                )
                        elif stage in {"evaluate", "score"}:
                            executed_command.append("--resume")
                    started = time.time()
                    subprocess.run(executed_command, check=True)
                    stage_state[stage] = {
                        "status": "complete",
                        "started_at_unix": started,
                        "finished_at_unix": time.time(),
                        "resolved_config_sha256": current_config_hash,
                        "command": command,
                        "executed_command": executed_command,
                        "artifact_sha256": _stage_artifacts(
                            stage,
                            payload,
                        ),
                    }
                    _save_json(manifest_path, manifest)
                if not dry_run:
                    manifest["finished_at_unix"] = time.time()
                    _save_json(manifest_path, manifest)
        for baseline in self.config.external_baselines:
            for seed in self.config.seeds:
                self._run_external(
                    baseline,
                    seed,
                    dry_run=dry_run,
                    resume=resume,
                )

    def _run_external(
        self,
        baseline: ExternalBaselineExperiment,
        seed: int,
        *,
        dry_run: bool,
        resume: bool,
    ) -> None:
        output_dir = baseline.output_dir(seed)
        run_dir = output_dir.parent
        resolved_baseline_path = run_dir / "configs" / "baseline.yaml"
        resolved_score_path = run_dir / "configs" / "score.yaml"
        if dry_run:
            print(
                " ".join(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/run_baseline.py",
                        "--config",
                        str(baseline.config_path(seed)),
                    ]
                )
            )
            print(
                " ".join(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/score_evaluation.py",
                        "--config",
                        str(resolved_score_path),
                    ]
                )
            )
            return
        source_path = baseline.config_path(seed)
        source = _load_yaml(source_path)
        if source.get("name") != baseline.name:
            raise ValueError(
                f"External config {source_path} sai baseline name."
            )
        source["base_seed"] = 42
        source["output_dir"] = str(output_dir)
        generator_model_config = source.get("generator_model_config")
        if not isinstance(generator_model_config, str):
            raise TypeError(
                "External generator_model_config phải là string path."
            )
        score = _load_yaml(self.config.metrics_config)
        _set_path(
            score,
            ("generation", "directory"),
            str(output_dir),
        )
        _set_path(
            score,
            ("generation", "results"),
            str(output_dir / "results.jsonl"),
        )
        _set_path(
            score,
            ("generation", "evaluation_contract"),
            str(output_dir / "evaluation_contract.json"),
        )
        _set_path(
            score,
            ("text", "generator_model_config"),
            generator_model_config,
        )
        _set_path(score, ("metrics", "seed"), 42)
        _set_path(score, ("output", "directory"), str(output_dir))
        _save_yaml(resolved_baseline_path, source)
        _save_yaml(resolved_score_path, score)
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"External run đã tồn tại: {run_dir}; dùng --resume."
                )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest, dict):
                raise ValueError("External run manifest sai schema.")
            for key, value in self.provenance.items():
                if manifest.get(key) != value:
                    raise ValueError(
                        f"External provenance mismatch tại {key}."
                    )
        else:
            manifest = _runtime_manifest(
                self.provenance,
                seed=seed,
                variant=baseline.name,
            )
            manifest["preflight_artifacts"] = (
                self.preflight_artifacts
            )
        if manifest.get("preflight_artifacts") != (
            self.preflight_artifacts
        ):
            raise ValueError("External preflight artifacts mismatch.")
        stages = (
            (
                "baseline",
                source,
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/run_baseline.py",
                    "--config",
                    str(resolved_baseline_path),
                ],
                resolved_baseline_path,
            ),
            (
                "score",
                score,
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/score_evaluation.py",
                    "--config",
                    str(resolved_score_path),
                ],
                resolved_score_path,
            ),
        )
        stage_state = manifest.get("stages")
        if not isinstance(stage_state, dict):
            raise ValueError("External run manifest stages sai schema.")
        for stage, payload, command, config_path in stages:
            current_config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()
            previous = stage_state.get(stage)
            if isinstance(previous, Mapping) and previous.get(
                "status"
            ) == "complete":
                _validate_completed_stage(
                    previous,
                    current_config_hash=current_config_hash,
                    command=command,
                    actual_artifacts=_stage_artifacts(stage, payload),
                    label=f"external stage {stage}",
                )
                continue
            executed_command = list(command)
            if resume and stage == "score":
                executed_command.append("--resume")
            started = time.time()
            subprocess.run(executed_command, check=True)
            stage_state[stage] = {
                "status": "complete",
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "resolved_config_sha256": current_config_hash,
                "command": command,
                "executed_command": executed_command,
                "artifact_sha256": _stage_artifacts(stage, payload),
            }
            _save_json(manifest_path, manifest)
        manifest["finished_at_unix"] = time.time()
        _save_json(manifest_path, manifest)


def aggregate_experiments(
    config: PaperExperimentConfig,
    *,
    output_dir: Path,
) -> dict[str, object]:
    t_critical = 4.302652729911275
    summaries: dict[str, list[dict[str, object]]] = defaultdict(list)
    for variant in config.variants:
        for seed in config.seeds:
            path = (
                config.output_root
                / variant.name
                / f"seed_{seed}"
                / "evaluation"
                / "metrics_summary.json"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Thiếu experiment summary: {path}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"Summary sai schema: {path}")
            summaries[variant.name].append(dict(payload))
    for baseline in config.external_baselines:
        for seed in config.seeds:
            path = baseline.output_dir(seed) / "metrics_summary.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Thiếu external baseline summary: {path}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"Summary sai schema: {path}")
            summaries[baseline.name].append(dict(payload))

    aggregate: dict[str, object] = {
        "schema_version": 1,
        "seeds": list(config.seeds),
        "variants": {},
        "style_mmd_subset_statistics": {},
    }
    csv_rows: list[dict[str, object]] = []
    per_writer_rows: list[dict[str, object]] = []
    for variant, runs in summaries.items():
        scalar_keys = sorted(
            key
            for key, value in runs[0].items()
            if isinstance(value, (int, float))
            and all(
                isinstance(run.get(key), (int, float))
                for run in runs
            )
            and key not in {
                "schema_version",
                "style_distribution_mmd_subset_std",
                "sample_count",
                "pair_count",
            }
        )
        metrics: dict[str, dict[str, float]] = {}
        for key in scalar_keys:
            values = [float(run[key]) for run in runs]
            mean = statistics.fmean(values)
            std = statistics.stdev(values)
            ci = t_critical * std / math.sqrt(len(values))
            metrics[key] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci,
                "ci95_high": mean + ci,
            }
            csv_rows.append(
                {"variant": variant, "metric": key, **metrics[key]}
            )
        aggregate["variants"][variant] = metrics  # type: ignore[index]
        aggregate["style_mmd_subset_statistics"][variant] = [  # type: ignore[index]
            {
                "seed": seed,
                "style_distribution_mmd_mean": run.get(
                    "style_distribution_mmd_mean"
                ),
                "style_distribution_mmd_subset_std": run.get(
                    "style_distribution_mmd_subset_std"
                ),
            }
            for seed, run in zip(config.seeds, runs, strict=True)
        ]
        writer_ids = sorted(
            set().union(
                *[
                    set(run.get("per_writer", {}))
                    for run in runs
                    if isinstance(run.get("per_writer"), Mapping)
                ]
            )
        )
        for writer_id in writer_ids:
            first_writer = runs[0]["per_writer"][writer_id]  # type: ignore[index]
            if not isinstance(first_writer, Mapping):
                raise ValueError("per_writer summary sai schema.")
            for key in sorted(first_writer):
                values = [
                    float(run["per_writer"][writer_id][key])  # type: ignore[index]
                    for run in runs
                ]
                per_writer_rows.append(
                    {
                        "variant": variant,
                        "writer_id": writer_id,
                        "metric": key,
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "aggregate.json", aggregate)
    _write_csv(output_dir / "aggregate.csv", csv_rows)
    _write_csv(output_dir / "per_writer.csv", per_writer_rows)
    markdown = [
        "| Variant | Metric | Mean | Std | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in csv_rows:
        markdown.append(
            f"| {row['variant']} | {row['metric']} | "
            f"{float(row['mean']):.6f} | {float(row['std']):.6f} | "
            f"[{float(row['ci95_low']):.6f}, "
            f"{float(row['ci95_high']):.6f}] |"
        )
    (output_dir / "aggregate.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _write_csv(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    if not records:
        raise ValueError(f"Không có records để ghi {path}.")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


__all__ = [
    "ExternalBaselineExperiment",
    "ExperimentVariant",
    "PaperExperimentConfig",
    "PaperExperimentRunner",
    "aggregate_experiments",
    "git_provenance",
    "load_experiment_config",
    "resolved_pipeline_configs",
]
