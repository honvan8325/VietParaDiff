#!/usr/bin/env python3
"""Generate every fixed held-out pair with resume-safe deterministic seeds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vietparadiff.artifacts import (
    load_latent_statistics,
    sha256_file,
)
from vietparadiff.data.pipeline import ReferenceImageProcessor
from vietparadiff.evaluation.fixed_pairs import (
    FixedPairEvaluator,
    load_evaluation_config,
)
from vietparadiff.inference.generator import (
    checkpoint_loading_config,
    load_inference_contract,
    load_model_config,
)
from vietparadiff.models.autokl import HandwritingAutoKL
from vietparadiff.models.grapheme import (
    GraphemeVocabulary,
    ParagraphFormatter,
)
from vietparadiff.models.generator import VietParaDiff
from vietparadiff.runtime import (
    autocast_context,
    resolve_runtime,
    seed_everything,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all fixed test pairs with three stable seeds each."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietparadiff/evaluate.yaml"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when evaluation contract and PNG hashes match.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evaluation_config(args.config)
    seed_everything(config.base_seed)
    runtime = resolve_runtime(config.device, config.precision)
    contract = load_inference_contract(
        config.model.contract,
        generator_checkpoint=config.model.checkpoint,
        model_config=config.model.model_config,
        vocabulary=config.model.vocabulary,
        autokl_checkpoint=config.autokl.checkpoint,
        latent_statistics=config.autokl.latent_statistics,
    )
    stored_model_config = load_model_config(config.model.model_config)
    loading_model_config = checkpoint_loading_config(
        stored_model_config
    )
    vocabulary = GraphemeVocabulary.load(config.model.vocabulary)
    expected_sizes = (
        len(vocabulary.base_to_id),
        len(vocabulary.shape_to_id),
        len(vocabulary.tone_to_id),
        len(vocabulary.case_to_id),
        len(vocabulary.class_to_id),
    )
    actual_sizes = (
        stored_model_config.text.base_vocab_size,
        stored_model_config.text.shape_vocab_size,
        stored_model_config.text.tone_vocab_size,
        stored_model_config.text.case_vocab_size,
        stored_model_config.text.class_vocab_size,
    )
    if expected_sizes != actual_sizes:
        raise ValueError(
            "Evaluation vocabulary không khớp model config."
        )
    statistics = load_latent_statistics(
        config.autokl.latent_statistics,
        expected_autokl_checkpoint=config.autokl.checkpoint,
    )
    autokl = HandwritingAutoKL(stored_model_config.autokl)
    autokl.load_checkpoint(config.autokl.checkpoint)
    autokl.to(runtime.device).eval()
    model = VietParaDiff(loading_model_config)
    model.load_checkpoint(config.model.checkpoint)
    model.to(runtime.device).eval()
    formatter = ParagraphFormatter(stored_model_config.text)
    processor = ReferenceImageProcessor(
        output_height=config.input.reference_height,
        max_width=config.input.maximum_reference_width,
    )
    artifact_sha256 = {
        "generator_checkpoint": sha256_file(
            config.model.checkpoint
        ),
        "inference_contract": sha256_file(config.model.contract),
        "model_config": sha256_file(config.model.model_config),
        "grapheme_vocabulary": sha256_file(
            config.model.vocabulary
        ),
        "autokl_checkpoint": sha256_file(
            config.autokl.checkpoint
        ),
        "latent_statistics": sha256_file(
            config.autokl.latent_statistics
        ),
        "test_pairs": sha256_file(config.data.test_pairs),
    }
    evaluator = FixedPairEvaluator(
        model,
        autokl,
        statistics,
        formatter,
        vocabulary,
        processor,
        config,
        num_train_timesteps=contract.num_train_timesteps,
        device=runtime.device,
        artifact_sha256=artifact_sha256,
    )
    with autocast_context(runtime):
        summary = evaluator.run(resume=args.resume)
    print(
        "Evaluation complete: "
        f"pairs={summary['pair_count']} "
        f"samples={summary['sample_count']} "
        f"output={config.output.directory}"
    )


if __name__ == "__main__":
    main()
