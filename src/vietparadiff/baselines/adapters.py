"""Pinned-checkout adapters for One-DM and Paragraph LDM."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image

from vietparadiff.artifacts import sha256_file
from vietparadiff.evaluation.fixed_pairs import (
    load_test_pairs,
    stable_sample_seed,
)
from vietparadiff.inference.generator import load_model_config
from vietparadiff.models.grapheme import ParagraphFormatter


BaselineName = Literal["one_dm", "paragraph_ldm"]
PINNED_COMMITS = {
    "one_dm": "dde2205a70a2c70d1786503d198a795358c80ee4",
    "paragraph_ldm": "8a53e91b99c868614f7e615f41bc49c3f73c75b9",
}


def _validate_environment_executable(
    config: ExternalBaselineConfig,
) -> None:
    executable = Path(config.command[0])
    if executable.parent != Path("."):
        executable_path = (
            executable
            if executable.is_absolute()
            else config.checkout / executable
        )
        if not executable_path.is_file():
            raise FileNotFoundError(
                "Baseline environment command không tồn tại: "
                f"{executable_path}"
            )
    elif shutil.which(config.command[0]) is None:
        raise FileNotFoundError(
            "Baseline environment command không có trên PATH: "
            f"{config.command[0]}"
        )


@dataclass(frozen=True, slots=True)
class ExternalBaselineConfig:
    name: BaselineName
    checkout: Path
    expected_commit: str
    checkpoint: Path
    checkpoint_sha256: str
    command: tuple[str, ...]
    test_pairs: Path
    image_root: Path
    generator_model_config: Path
    output_dir: Path
    base_seed: int
    samples_per_pair: int

    def __post_init__(self) -> None:
        if self.name not in PINNED_COMMITS:
            raise ValueError("Baseline phải là one_dm/paragraph_ldm.")
        if self.expected_commit != PINNED_COMMITS[self.name]:
            raise ValueError(
                f"{self.name} phải pin commit {PINNED_COMMITS[self.name]}."
            )
        if (
            len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_sha256
            )
        ):
            raise ValueError("Baseline checkpoint SHA-256 không hợp lệ.")
        if not self.command or any(not item for item in self.command):
            raise ValueError("Baseline command không được rỗng.")
        required = {"{requests}", "{output_dir}"}
        joined = "\n".join(self.command)
        if not all(token in joined for token in required):
            raise ValueError(
                "Baseline command phải chứa {requests} và {output_dir}."
            )
        if self.base_seed < 0 or self.samples_per_pair != 3:
            raise ValueError(
                "Baseline khóa base_seed không âm và 3 samples/pair."
            )


def load_external_baseline_config(
    path: Path,
) -> ExternalBaselineConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy baseline config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "name",
        "checkout",
        "expected_commit",
        "checkpoint",
        "checkpoint_sha256",
        "command",
        "test_pairs",
        "image_root",
        "generator_model_config",
        "output_dir",
        "base_seed",
        "samples_per_pair",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("External baseline config sai schema.")
    command = raw["command"]
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not all(isinstance(item, str) for item in command)
    ):
        raise TypeError("Baseline command phải là list string.")
    return ExternalBaselineConfig(
        name=str(raw["name"]),  # type: ignore[arg-type]
        checkout=Path(str(raw["checkout"])),
        expected_commit=str(raw["expected_commit"]),
        checkpoint=Path(str(raw["checkpoint"])),
        checkpoint_sha256=str(raw["checkpoint_sha256"]),
        command=tuple(command),
        test_pairs=Path(str(raw["test_pairs"])),
        image_root=Path(str(raw["image_root"])),
        generator_model_config=Path(
            str(raw["generator_model_config"])
        ),
        output_dir=Path(str(raw["output_dir"])),
        base_seed=int(raw["base_seed"]),
        samples_per_pair=int(raw["samples_per_pair"]),
    )


def preflight_external_baseline(
    config: ExternalBaselineConfig,
    *,
    require_generator_model_config: bool = True,
) -> dict[str, str]:
    if not config.checkout.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy baseline checkout: {config.checkout}"
        )
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(config.checkout),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != config.expected_commit:
        raise ValueError(
            f"Baseline checkout commit mismatch: "
            f"expected={config.expected_commit}, actual={commit}."
        )
    checkpoint_hash = sha256_file(config.checkpoint)
    if checkpoint_hash != config.checkpoint_sha256:
        raise ValueError("External baseline checkpoint SHA-256 mismatch.")
    _validate_environment_executable(config)
    artifacts = {
        "checkout_commit": commit,
        "checkpoint_sha256": checkpoint_hash,
        "test_pairs_sha256": sha256_file(config.test_pairs),
    }
    if require_generator_model_config:
        artifacts["generator_model_config_sha256"] = sha256_file(
            config.generator_model_config
        )
    return artifacts


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


def _foreground_crop(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    mask = grayscale.point(lambda value: 255 if value < 253 else 0)
    bbox = mask.getbbox()
    mask.close()
    if bbox is None:
        grayscale.close()
        raise ValueError("Baseline output không có foreground.")
    cropped = grayscale.crop(bbox)
    grayscale.close()
    return cropped


def stitch_word_images(
    lines: Sequence[Sequence[Path]],
    *,
    line_ranges: Sequence[tuple[int, int]],
    output_height: int,
    canvas_width: int = 1024,
    margin: int = 48,
    word_gap: int = 18,
) -> Image.Image:
    if (
        len(lines) != len(line_ranges)
        or output_height <= 0
        or canvas_width != 1024
    ):
        raise ValueError("One-DM stitch contract không hợp lệ.")
    canvas = Image.new("L", (canvas_width, output_height), 255)
    for paths, (y0, y1) in zip(lines, line_ranges, strict=True):
        if not paths or y0 < 0 or y1 <= y0 or y1 > output_height:
            canvas.close()
            raise ValueError("One-DM line inputs/y-range không hợp lệ.")
        crops: list[Image.Image] = []
        try:
            content_height = max(1, round((y1 - y0) * 0.8))
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Thiếu One-DM word output: {path}"
                    )
                with Image.open(path) as source:
                    crop = _foreground_crop(source)
                width = max(
                    1,
                    round(crop.width * content_height / crop.height),
                )
                resized = crop.resize(
                    (width, content_height),
                    Image.Resampling.LANCZOS,
                )
                crop.close()
                crops.append(resized)
            available = canvas_width - 2 * margin
            raw_width = sum(image.width for image in crops)
            raw_width += word_gap * max(0, len(crops) - 1)
            if raw_width > available:
                scale = available / raw_width
                resized_crops: list[Image.Image] = []
                for image in crops:
                    resized_crops.append(
                        image.resize(
                            (
                                max(1, round(image.width * scale)),
                                max(1, round(image.height * scale)),
                            ),
                            Image.Resampling.LANCZOS,
                        )
                    )
                    image.close()
                crops = resized_crops
                actual_gap = max(1, round(word_gap * scale))
            else:
                actual_gap = word_gap
            x = margin
            for image in crops:
                y = y0 + (y1 - y0 - image.height) // 2
                canvas.paste(image, (x, y))
                x += image.width + actual_gap
        finally:
            for image in crops:
                image.close()
    return canvas


def normalize_paragraph_output(
    source_path: Path,
    *,
    output_height: int,
) -> Image.Image:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Thiếu Paragraph LDM output: {source_path}"
        )
    with Image.open(source_path) as source:
        cropped = _foreground_crop(source)
    scale = min(1024 / cropped.width, output_height / cropped.height)
    size = (
        max(1, min(1024, round(cropped.width * scale))),
        max(1, min(output_height, round(cropped.height * scale))),
    )
    resized = cropped.resize(size, Image.Resampling.LANCZOS)
    cropped.close()
    canvas = Image.new("L", (1024, output_height), 255)
    canvas.paste(
        resized,
        ((1024 - resized.width) // 2, (output_height - resized.height) // 2),
    )
    resized.close()
    return canvas


class ExternalBaselineRunner:
    def __init__(self, config: ExternalBaselineConfig) -> None:
        self.config = config
        model_config = load_model_config(
            config.generator_model_config
        )
        self.formatter = ParagraphFormatter(model_config.text)

    def _verify(self) -> dict[str, str]:
        return preflight_external_baseline(self.config)

    def preflight(self) -> dict[str, str]:
        return self._verify()

    def _requests(
        self,
        pairs: Sequence[Mapping[str, str]],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        requests: list[dict[str, object]] = []
        plans: dict[str, object] = {}
        for pair in pairs:
            formatted = self.formatter.format(
                pair["target_text"],
                preserve_physical_lines=True,
            )
            for sample_index in range(self.config.samples_per_pair):
                seed = stable_sample_seed(
                    self.config.base_seed,
                    pair["pair_id"],
                    sample_index,
                )
                sample_id = f"{pair['pair_id']}:{sample_index}"
                if self.config.name == "one_dm":
                    word_ids_by_line: list[dict[str, object]] = []
                    for line_index, line in enumerate(formatted.lines):
                        word_ids: list[str] = []
                        for word_index, word in enumerate(line.split()):
                            request_id = (
                                f"{pair['pair_id']}__{sample_index:02d}"
                                f"__l{line_index:02d}__w{word_index:03d}"
                            )
                            word_seed = stable_sample_seed(
                                seed,
                                request_id,
                                word_index,
                            )
                            requests.append(
                                {
                                    "request_id": request_id,
                                    "text": word,
                                    "reference_image": pair[
                                        "reference_image"
                                    ],
                                    "seed": word_seed,
                                    "granularity": "word",
                                }
                            )
                            word_ids.append(request_id)
                        if word_ids:
                            word_ids_by_line.append(
                                {
                                    "line_index": line_index,
                                    "request_ids": word_ids,
                                }
                            )
                    plans[sample_id] = {
                        "pair": dict(pair),
                        "sample_index": sample_index,
                        "seed": seed,
                        "output_height": formatted.output_height,
                        "lines": word_ids_by_line,
                        "line_slot_mask": formatted.line_slot_mask.tolist(),
                    }
                else:
                    request_id = (
                        f"{pair['pair_id']}__{sample_index:02d}"
                    )
                    requests.append(
                        {
                            "request_id": request_id,
                            "text": "\n".join(formatted.lines),
                            "reference_image": pair["reference_image"],
                            "seed": seed,
                            "granularity": "paragraph",
                        }
                    )
                    plans[sample_id] = {
                        "pair": dict(pair),
                        "sample_index": sample_index,
                        "seed": seed,
                        "output_height": formatted.output_height,
                        "request_id": request_id,
                    }
        return requests, plans

    def run(self) -> dict[str, object]:
        artifacts = self.preflight()
        pairs = load_test_pairs(self.config.test_pairs)
        requests, plans = self._requests(pairs)
        work = self.config.output_dir / "external"
        requests_path = work / "requests.jsonl"
        external_output = work / "outputs"
        _write_jsonl(requests_path, requests)
        external_output.mkdir(parents=True, exist_ok=True)
        replacements = {
            "{requests}": str(requests_path.resolve()),
            "{output_dir}": str(external_output.resolve()),
            "{checkpoint}": str(self.config.checkpoint.resolve()),
            "{checkout}": str(self.config.checkout.resolve()),
        }
        command: list[str] = []
        for item in self.config.command:
            resolved = item
            for token, value in replacements.items():
                resolved = resolved.replace(token, value)
            command.append(resolved)
        subprocess.run(
            command,
            cwd=self.config.checkout,
            check=True,
        )
        output_manifest = external_output / "outputs.jsonl"
        outputs = _read_external_outputs(output_manifest)
        expected_ids = {
            str(request["request_id"]) for request in requests
        }
        if set(outputs) != expected_ids:
            raise ValueError(
                "External baseline outputs không khớp request IDs."
            )
        records: list[dict[str, object]] = []
        for sample_id, plan_value in sorted(plans.items()):
            if not isinstance(plan_value, Mapping):
                raise RuntimeError("Baseline plan sai type.")
            plan = dict(plan_value)
            pair = plan["pair"]
            if not isinstance(pair, Mapping):
                raise RuntimeError("Baseline pair plan sai type.")
            pair = dict(pair)
            relative = (
                Path("samples")
                / str(pair["pair_id"])
                / (
                    f"sample_{int(plan['sample_index']):02d}"
                    f"_seed_{int(plan['seed'])}.png"
                )
            )
            destination = self.config.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if self.config.name == "one_dm":
                line_ids = plan["lines"]
                if not isinstance(line_ids, Sequence):
                    raise TypeError("One-DM line plan phải là sequence.")
                lines: list[list[Path]] = []
                line_indices: list[int] = []
                for line in line_ids:
                    if not isinstance(line, Mapping):
                        raise TypeError("One-DM line entry phải là mapping.")
                    request_ids = line.get("request_ids")
                    line_index = line.get("line_index")
                    if (
                        not isinstance(request_ids, Sequence)
                        or isinstance(request_ids, (str, bytes))
                        or not isinstance(line_index, int)
                    ):
                        raise TypeError("One-DM line entry sai schema.")
                    lines.append(
                        [
                            outputs[str(request_id)]
                            for request_id in request_ids
                        ]
                    )
                    line_indices.append(line_index)
                slot_mask = plan["line_slot_mask"]
                if not isinstance(slot_mask, Sequence):
                    raise TypeError("One-DM slot mask phải là sequence.")
                ranges: list[tuple[int, int]] = []
                for line_index in line_indices:
                    if not 0 <= line_index < len(slot_mask):
                        raise ValueError("One-DM line index vượt slot mask.")
                    slot = slot_mask[line_index]
                    active = [
                        row_index
                        for row_index, row in enumerate(slot)
                        if any(float(value) > 0.0 for value in row)
                    ]
                    if not active:
                        raise ValueError("One-DM active line thiếu slot.")
                    ranges.append((active[0] * 8, (active[-1] + 1) * 8))
                rendered = stitch_word_images(
                    lines,
                    line_ranges=ranges,
                    output_height=int(plan["output_height"]),
                )
            else:
                rendered = normalize_paragraph_output(
                    outputs[str(plan["request_id"])],
                    output_height=int(plan["output_height"]),
                )
            temporary = destination.with_suffix(".tmp")
            rendered.save(temporary, format="PNG")
            rendered.close()
            temporary.replace(destination)
            records.append(
                {
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
                    "sample_index": int(plan["sample_index"]),
                    "seed": int(plan["seed"]),
                    "output_height": int(plan["output_height"]),
                    "num_inference_steps": -1,
                    "generated_image": str(relative),
                    "generated_image_sha256": sha256_file(destination),
                    "artifact_sha256": artifacts,
                }
            )
        _write_jsonl(self.config.output_dir / "results.jsonl", records)
        contract = {
            "schema_version": 1,
            "baseline": self.config.name,
            "stitched_word_level": self.config.name == "one_dm",
            "samples_per_pair": self.config.samples_per_pair,
            "artifact_sha256": artifacts,
        }
        _write_json(
            self.config.output_dir / "evaluation_contract.json",
            contract,
        )
        summary = {
            "schema_version": 1,
            "baseline": self.config.name,
            "pair_count": len(pairs),
            "sample_count": len(records),
            "artifact_sha256": artifacts,
        }
        _write_json(self.config.output_dir / "summary.json", summary)
        return summary


def _read_external_outputs(path: Path) -> dict[str, Path]:
    if not path.is_file():
        raise FileNotFoundError(
            f"External worker không tạo {path}."
        )
    outputs: dict[str, Path] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"request_id", "image"}
            or not isinstance(payload["request_id"], str)
            or not isinstance(payload["image"], str)
        ):
            raise ValueError(
                f"External outputs dòng {line_number} sai schema."
            )
        request_id = payload["request_id"]
        if request_id in outputs:
            raise ValueError(
                f"External output request ID trùng: {request_id}"
            )
        image = Path(payload["image"])
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise FileNotFoundError(
                f"External output image bị thiếu: {image}"
            )
        outputs[request_id] = image
    return outputs


__all__ = [
    "ExternalBaselineConfig",
    "ExternalBaselineRunner",
    "load_external_baseline_config",
    "normalize_paragraph_output",
    "stitch_word_images",
]
