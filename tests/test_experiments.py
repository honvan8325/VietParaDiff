"""Experiment DAG and three-seed aggregation tests."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from vietparadiff.experiments import (
    ExternalBaselineExperiment,
    _validate_completed_stage,
    aggregate_experiments,
    load_experiment_config,
    resolved_pipeline_configs,
    validate_data_audit_report,
)
from vietparadiff.data.audit import compute_dataset_snapshot


def test_experiment_matrix_and_dag_are_locked() -> None:
    config = load_experiment_config(
        Path("configs/experiments/paper.yaml")
    )
    assert config.seeds == (42, 43, 44)
    assert tuple(variant.name for variant in config.variants) == (
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "full",
    )
    a0_stages = resolved_pipeline_configs(
        config,
        config.variants[0],
        42,
    )
    full_stages = resolved_pipeline_configs(
        config,
        config.variants[-1],
        42,
    )
    assert [stage for stage, _, _ in a0_stages] == [
        "pretrain",
        "finetune",
        "evaluate",
        "score",
    ]
    assert [stage for stage, _, _ in full_stages] == [
        "pretrain",
        "finetune",
        "htr_guided",
        "evaluate",
        "score",
    ]
    for baseline in config.external_baselines:
        for seed in config.seeds:
            assert baseline.config_path(seed).is_file()


def test_resume_rejects_config_and_command_changes() -> None:
    previous = {
        "resolved_config_sha256": "config-a",
        "command": ["python", "train.py"],
        "artifact_sha256": {"best.pt": "artifact-a"},
    }
    with pytest.raises(ValueError, match="Resume config mismatch"):
        _validate_completed_stage(
            previous,
            current_config_hash="config-b",
            command=["python", "train.py"],
            actual_artifacts={"best.pt": "artifact-a"},
            label="stage train",
        )
    with pytest.raises(ValueError, match="Resume command mismatch"):
        _validate_completed_stage(
            previous,
            current_config_hash="config-a",
            command=["python", "changed.py"],
            actual_artifacts={"best.pt": "artifact-a"},
            label="stage train",
        )


def test_preflight_rejects_stale_dataset_audit(tmp_path: Path) -> None:
    split_root = tmp_path / "splits"
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image version one")
    manifest = split_root / "stage" / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = compute_dataset_snapshot(
        split_root,
        image_root=tmp_path,
    )
    report = {
        "schema_version": 2,
        "split_root": str(split_root),
        "image_root": str(tmp_path),
        "error_count": 0,
        **snapshot.report_fields(),
    }
    report_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validate_data_audit_report(
        report_path,
        split_root=split_root,
        image_root=tmp_path,
    )

    image_path.write_bytes(b"image version two")
    with pytest.raises(ValueError, match="Data audit đã stale"):
        validate_data_audit_report(
            report_path,
            split_root=split_root,
            image_root=tmp_path,
        )


def test_aggregator_uses_sample_std_and_student_t_ci(
    tmp_path: Path,
) -> None:
    base = load_experiment_config(
        Path("configs/experiments/paper.yaml")
    )
    config = replace(
        base,
        output_root=tmp_path / "runs",
        external_baselines=tuple(
            ExternalBaselineExperiment(
                name=item.name,
                config_pattern=item.config_pattern,
                output_pattern=str(
                    tmp_path
                    / "runs"
                    / item.name
                    / "seed_{seed}"
                    / "evaluation"
                ),
            )
            for item in base.external_baselines
        ),
    )
    for variant in config.variants:
        for offset, seed in enumerate(config.seeds):
            directory = (
                config.output_root
                / variant.name
                / f"seed_{seed}"
                / "evaluation"
            )
            directory.mkdir(parents=True)
            payload = {
                "schema_version": 1,
                "sample_count": 3,
                "pair_count": 1,
                "paragraph_cer": float(offset + 1),
                "style_distribution_mmd_mean": float(offset + 2),
                "style_distribution_mmd_subset_std": (
                    float(offset) / 10.0
                ),
                "per_writer": {
                    "writer": {
                        "paragraph_cer": float(offset + 1)
                    }
                },
            }
            (directory / "metrics_summary.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
    for baseline in config.external_baselines:
        for offset, seed in enumerate(config.seeds):
            directory = baseline.output_dir(seed)
            directory.mkdir(parents=True)
            payload = {
                "schema_version": 1,
                "sample_count": 3,
                "pair_count": 1,
                "paragraph_cer": float(offset + 1),
                "style_distribution_mmd_mean": float(offset + 2),
                "style_distribution_mmd_subset_std": (
                    float(offset) / 10.0
                ),
                "per_writer": {
                    "writer": {
                        "paragraph_cer": float(offset + 1)
                    }
                },
            }
            (directory / "metrics_summary.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
    result = aggregate_experiments(
        config,
        output_dir=tmp_path / "aggregate",
    )
    metric = result["variants"]["a0"]["paragraph_cer"]
    assert metric["mean"] == pytest.approx(2.0)
    assert metric["std"] == pytest.approx(1.0)
    half_width = 4.302652729911275 / math.sqrt(3)
    assert metric["ci95_low"] == pytest.approx(2.0 - half_width)
    assert metric["ci95_high"] == pytest.approx(2.0 + half_width)
    mmd = result["style_mmd_subset_statistics"]["a0"]
    assert [
        item["style_distribution_mmd_subset_std"]
        for item in mmd
    ] == [0.0, 0.1, 0.2]
    assert "one_dm" in result["variants"]
    assert "paragraph_ldm" in result["variants"]
