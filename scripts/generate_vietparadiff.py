#!/usr/bin/env python3
"""Generate one grayscale paragraph with deterministic VietParaDiff sampling."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autokl_training import autocast_context
from src.data.training import ReferenceImageProcessor
from src.models.autokl import HandwritingAutoKL
from src.models.text import (
    GraphemeVocabulary,
    ParagraphFormatter,
)
from src.models.vietparadiff import VietParaDiff
from src.vietparadiff_sampling import (
    SamplingConfig,
    checkpoint_loading_config,
    generate_paragraph,
    load_generation_config,
    load_inference_contract,
    load_model_config,
)
from src.vietparadiff_training import (
    load_latent_statistics,
    resolve_runtime,
    seed_everything,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one paragraph; no CFG, stochastic eta or reranking."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietparadiff_generate.yaml"),
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="UTF-8 text file; hard newlines are preserved.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Real one-line handwriting reference image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path; defaults to config output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override deterministic sampling seed.",
    )
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed không được âm.")
    return args


def _save_png(image: torch.Tensor, path: Path) -> None:
    if (
        image.shape[0:2] != (1, 1)
        or image.ndim != 4
        or not torch.isfinite(image).all()
    ):
        raise ValueError("Output image phải là finite [1,1,H,W].")
    if path.suffix.lower() != ".png":
        raise ValueError("Output path phải có extension .png.")
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(path, format="PNG")
    finally:
        rendered.close()


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy text file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Text file không phải UTF-8 hợp lệ: {path}"
        ) from error
    if not text.strip():
        raise ValueError("Text file UTF-8 không được rỗng.")
    return text


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_generation_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    seed_everything(seed)
    runtime = resolve_runtime(config.device, config.precision)
    text = _read_text(args.text_file)
    if not args.reference.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy reference image: {args.reference}"
        )

    contract = load_inference_contract(
        config.model.contract,
        generator_checkpoint=config.model.checkpoint,
        model_config=config.model.model_config,
        vocabulary=config.model.vocabulary,
        autokl_checkpoint=config.autokl.checkpoint,
        latent_statistics=config.autokl.latent_statistics,
    )
    if (
        config.diffusion.num_inference_steps
        > contract.num_train_timesteps
    ):
        raise ValueError(
            "num_inference_steps vượt num_train_timesteps trong "
            "inference contract."
        )
    stored_model_config = load_model_config(config.model.model_config)
    model_config = checkpoint_loading_config(stored_model_config)
    vocabulary = GraphemeVocabulary.load(config.model.vocabulary)
    expected_sizes = (
        len(vocabulary.base_to_id),
        len(vocabulary.shape_to_id),
        len(vocabulary.tone_to_id),
        len(vocabulary.case_to_id),
        len(vocabulary.class_to_id),
    )
    actual_sizes = (
        model_config.text.base_vocab_size,
        model_config.text.shape_vocab_size,
        model_config.text.tone_vocab_size,
        model_config.text.case_vocab_size,
        model_config.text.class_vocab_size,
    )
    if actual_sizes != expected_sizes:
        raise ValueError(
            "Model text vocabulary sizes không khớp deterministic "
            f"vocabulary: expected={expected_sizes}, actual={actual_sizes}."
        )
    formatter = ParagraphFormatter(model_config.text)
    processor = ReferenceImageProcessor(
        output_height=config.input.reference_height,
        max_width=config.input.maximum_reference_width,
    )
    processed = processor(args.reference)
    reference_image = processed["image"][None]
    reference_valid_mask = processed["valid_mask"][None]

    statistics = load_latent_statistics(
        config.autokl.latent_statistics,
        expected_autokl_checkpoint=config.autokl.checkpoint,
    )
    autokl = HandwritingAutoKL(model_config.autokl)
    autokl.load_checkpoint(config.autokl.checkpoint)
    autokl.to(runtime.device).eval()
    model = VietParaDiff(model_config)
    model.load_checkpoint(config.model.checkpoint)
    model.to(runtime.device).eval()
    for module in (model, autokl):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    sampling = SamplingConfig(
        num_inference_steps=config.diffusion.num_inference_steps,
        seed=seed,
    )
    with autocast_context(runtime):
        result = generate_paragraph(
            model,
            autokl,
            statistics,
            formatter,
            vocabulary,
            text=text,
            reference_image=reference_image,
            reference_valid_mask=reference_valid_mask,
            sampling_config=sampling,
            num_train_timesteps=(
                contract.num_train_timesteps
            ),
            device=runtime.device,
        )
    output_path = (
        config.output.directory
        / f"{args.text_file.stem}_seed{seed}.png"
        if args.output is None
        else args.output
    )
    _save_png(result.image, output_path)
    print(
        f"Saved {output_path} shape={tuple(result.image.shape)} "
        f"height_bucket={result.formatted_text.output_height} "
        f"steps={sampling.num_inference_steps} seed={seed}"
    )


if __name__ == "__main__":
    main()
