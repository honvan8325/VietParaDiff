"""Train an independent ArcFace writer-verification encoder."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

from vietparadiff.artifacts import (
    sha256_file,
    verify_visual_backbone,
)
from vietparadiff.metrics import binary_auc_eer
from vietparadiff.models.config import WriterEncoderConfig
from vietparadiff.models.writer import ArcFaceHead, WriterStyleEncoder


@dataclass(frozen=True, slots=True)
class WriterMetricDataConfig:
    line_manifest: Path
    paragraph_manifest: Path
    image_root: Path
    num_workers: int
    writers_per_batch: int
    samples_per_writer: int
    validation_writer_fraction: float

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("data.num_workers không được âm.")
        if self.writers_per_batch < 2 or self.samples_per_writer < 2:
            raise ValueError(
                "Writer batches cần >=2 writers và >=2 samples/writer."
            )
        if not 0.0 < self.validation_writer_fraction < 0.5:
            raise ValueError(
                "validation_writer_fraction phải nằm trong (0,0.5)."
            )


@dataclass(frozen=True, slots=True)
class WriterMetricBackboneConfig:
    checkpoint: Path
    contract: Path


@dataclass(frozen=True, slots=True)
class WriterMetricOptimizerConfig:
    learning_rate: float
    weight_decay: float
    epochs: int
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Writer optimizer values không hợp lệ.")
        if self.epochs <= 0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("Writer epochs/gradient clip phải dương.")


@dataclass(frozen=True, slots=True)
class WriterMetricCheckpointConfig:
    output_dir: Path


@dataclass(frozen=True, slots=True)
class WriterMetricTrainingConfig:
    seed: int
    device: str
    selection_protocol: str
    data: WriterMetricDataConfig
    backbone: WriterMetricBackboneConfig
    optimizer: WriterMetricOptimizerConfig
    checkpoint: WriterMetricCheckpointConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed không được âm.")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device phải là auto/cuda/mps/cpu.")
        if self.selection_protocol != (
            "writer_disjoint_internal_validation"
        ):
            raise ValueError(
                "Writer metric phải công khai dùng internal "
                "writer-disjoint validation."
            )

    def resolved_dict(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {
                    str(key): convert(item)
                    for key, item in value.items()
                }
            return value

        payload = convert(asdict(self))
        if not isinstance(payload, dict):
            raise RuntimeError("Writer config phải serialize thành mapping.")
        return payload


def _section(
    raw: Mapping[str, object],
    name: str,
    keys: set[str],
) -> dict[str, object]:
    section = raw.get(name)
    if not isinstance(section, Mapping) or set(section) != keys:
        raise ValueError(
            f"config.{name} keys phải bằng {sorted(keys)}."
        )
    return dict(section)


def load_writer_metric_config(
    path: Path,
) -> WriterMetricTrainingConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = {
        "seed",
        "device",
        "selection_protocol",
        "data",
        "backbone",
        "optimizer",
        "checkpoint",
    }
    if not isinstance(raw, Mapping) or set(raw) != root:
        raise ValueError("Writer metric config root sai schema.")
    data = _section(
        raw,
        "data",
        {
            "line_manifest",
            "paragraph_manifest",
            "image_root",
            "num_workers",
            "writers_per_batch",
            "samples_per_writer",
            "validation_writer_fraction",
        },
    )
    backbone = _section(raw, "backbone", {"checkpoint", "contract"})
    optimizer = _section(
        raw,
        "optimizer",
        {
            "learning_rate",
            "weight_decay",
            "epochs",
            "gradient_clip_norm",
        },
    )
    checkpoint = _section(raw, "checkpoint", {"output_dir"})
    return WriterMetricTrainingConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        selection_protocol=str(raw["selection_protocol"]),
        data=WriterMetricDataConfig(
            line_manifest=Path(str(data["line_manifest"])),
            paragraph_manifest=Path(str(data["paragraph_manifest"])),
            image_root=Path(str(data["image_root"])),
            num_workers=int(data["num_workers"]),
            writers_per_batch=int(data["writers_per_batch"]),
            samples_per_writer=int(data["samples_per_writer"]),
            validation_writer_fraction=float(
                data["validation_writer_fraction"]
            ),
        ),
        backbone=WriterMetricBackboneConfig(
            checkpoint=Path(str(backbone["checkpoint"])),
            contract=Path(str(backbone["contract"])),
        ),
        optimizer=WriterMetricOptimizerConfig(
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            epochs=int(optimizer["epochs"]),
            gradient_clip_norm=float(
                optimizer["gradient_clip_norm"]
            ),
        ),
        checkpoint=WriterMetricCheckpointConfig(
            output_dir=Path(str(checkpoint["output_dir"]))
        ),
    )


def _read_records(paths: Sequence[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            required = {"id", "image", "canonical_writer_id", "level"}
            if (
                not isinstance(payload, Mapping)
                or not required.issubset(payload)
            ):
                raise ValueError(
                    f"{path}:{line_number} thiếu writer metric fields."
                )
            record = {
                key: str(payload[key]) for key in required
            }
            if record["id"] in seen_ids:
                raise ValueError(
                    f"Writer metric sample ID trùng: {record['id']}"
                )
            if record["level"] not in {"line", "paragraph"}:
                raise ValueError(
                    f"Writer metric không nhận level {record['level']}."
                )
            seen_ids.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError("Writer metric manifests không được rỗng.")
    return records


def split_writer_ids(
    records: Sequence[Mapping[str, str]],
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["canonical_writer_id"])] += 1
    eligible = [writer for writer, count in counts.items() if count >= 2]
    if len(eligible) < 4:
        raise ValueError("Writer metric cần ít nhất bốn writers có >=2 ảnh.")
    ordered = sorted(
        eligible,
        key=lambda writer: hashlib.sha256(
            f"{seed}:{writer}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = max(
        2,
        min(
            len(ordered) - 2,
            round(len(ordered) * validation_fraction),
        ),
    )
    validation = tuple(sorted(ordered[:validation_count]))
    train = tuple(sorted(ordered[validation_count:]))
    return train, validation


class WriterImageProcessor:
    def __init__(
        self,
        config: WriterEncoderConfig,
        *,
        threshold: int = 253,
        margin: int = 4,
    ) -> None:
        self.config = config
        self.threshold = threshold
        self.margin = margin

    def __call__(self, path: Path) -> Tensor:
        with Image.open(path) as source:
            image = source.convert("L")
        mask = image.point(
            lambda value: 255 if value < self.threshold else 0
        )
        bbox = mask.getbbox()
        mask.close()
        if bbox is None:
            image.close()
            raise ValueError(f"Writer metric image không có foreground: {path}")
        left, top, right, bottom = bbox
        cropped = image.crop(
            (
                max(0, left - self.margin),
                max(0, top - self.margin),
                min(image.width, right + self.margin),
                min(image.height, bottom + self.margin),
            )
        )
        image.close()
        scale = min(
            self.config.input_height / cropped.height,
            self.config.max_width / cropped.width,
        )
        width = max(1, min(self.config.max_width, round(cropped.width * scale)))
        height = max(1, min(self.config.input_height, round(cropped.height * scale)))
        resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
        cropped.close()
        canvas = Image.new(
            "L",
            (width, self.config.input_height),
            255,
        )
        canvas.paste(
            resized,
            (0, (self.config.input_height - height) // 2),
        )
        resized.close()
        tensor = torch.frombuffer(
            bytearray(canvas.tobytes()),
            dtype=torch.uint8,
        ).reshape(self.config.input_height, width)
        canvas.close()
        return tensor.float().div(127.5).sub(1.0).unsqueeze(0)


class WriterMetricDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        *,
        writer_ids: Sequence[str],
        image_root: Path,
        processor: WriterImageProcessor,
        writer_to_id: Mapping[str, int],
    ) -> None:
        allowed = set(writer_ids)
        self.records = [
            dict(record)
            for record in records
            if str(record["canonical_writer_id"]) in allowed
        ]
        if not self.records:
            raise ValueError("Writer metric dataset không được rỗng.")
        self.image_root = image_root
        self.processor = processor
        self.writer_to_id = dict(writer_to_id)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        writer = record["canonical_writer_id"]
        return {
            "image": self.processor(image_path),
            "writer_id": writer,
            "writer_label": self.writer_to_id.get(writer, -1),
            "sample_id": record["id"],
        }


class WriterBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: WriterMetricDataset,
        *,
        writers_per_batch: int,
        samples_per_writer: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.writers_per_batch = writers_per_batch
        self.samples_per_writer = samples_per_writer
        self.seed = seed
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            grouped[record["canonical_writer_id"]].append(index)
        if len(grouped) < writers_per_batch:
            raise ValueError("Không đủ writers cho một balanced batch.")
        self.grouped = dict(grouped)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch không được âm.")
        self.epoch = epoch

    def __len__(self) -> int:
        batch_size = self.writers_per_batch * self.samples_per_writer
        return max(1, math.ceil(len(self.dataset) / batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        writers = sorted(self.grouped)
        pools = {
            writer: generator.sample(indices, len(indices))
            for writer, indices in self.grouped.items()
        }
        positions = {writer: 0 for writer in writers}
        for _ in range(len(self)):
            selected_writers = generator.sample(
                writers,
                self.writers_per_batch,
            )
            batch: list[int] = []
            for writer in selected_writers:
                pool = pools[writer]
                for _ in range(self.samples_per_writer):
                    position = positions[writer]
                    if position >= len(pool):
                        pool = generator.sample(
                            self.grouped[writer],
                            len(self.grouped[writer]),
                        )
                        pools[writer] = pool
                        position = 0
                    batch.append(pool[position])
                    positions[writer] = position + 1
            yield batch


def collate_writer_metric(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("Writer metric batch không được rỗng.")
    images = [sample["image"] for sample in samples]
    if not all(isinstance(image, Tensor) for image in images):
        raise TypeError("Writer batch image phải là Tensor.")
    width = max(image.shape[-1] for image in images)  # type: ignore[union-attr]
    batch = torch.ones(
        len(images),
        1,
        128,
        width,
        dtype=torch.float32,
    )
    for index, image in enumerate(images):
        batch[index, :, :, : image.shape[-1]] = image  # type: ignore[index,union-attr]
    return {
        "images": batch,
        "labels": torch.tensor(
            [int(sample["writer_label"]) for sample in samples],
            dtype=torch.long,
        ),
        "writer_ids": tuple(
            str(sample["writer_id"]) for sample in samples
        ),
        "sample_ids": tuple(
            str(sample["sample_id"]) for sample in samples
        ),
    }


def validation_verification(
    embeddings: Tensor,
    writer_ids: Sequence[str],
) -> tuple[float, float]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(writer_ids):
        raise ValueError("Validation embeddings/writer_ids không khớp.")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, writer in enumerate(writer_ids):
        grouped[writer].append(index)
    if any(len(indices) < 2 for indices in grouped.values()):
        raise ValueError("Mỗi validation writer cần ít nhất hai ảnh.")
    labels: list[bool] = []
    scores: list[float] = []
    normalized = F.normalize(embeddings.float(), dim=1)
    writers = sorted(grouped)
    for writer in writers:
        indices = grouped[writer]
        for first, second in zip(indices[:-1], indices[1:], strict=True):
            labels.append(True)
            scores.append(float(normalized[first] @ normalized[second]))
    for first_writer, second_writer in zip(
        writers,
        writers[1:] + writers[:1],
        strict=True,
    ):
        first = grouped[first_writer][0]
        second = grouped[second_writer][0]
        labels.append(False)
        scores.append(float(normalized[first] @ normalized[second]))
    return binary_auc_eer(labels, scores)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_writer_artifacts(
    output_dir: Path,
    *,
    model: WriterStyleEncoder,
    model_config: WriterEncoderConfig,
    writer_to_id: Mapping[str, int],
    config: WriterMetricTrainingConfig,
    validation_auc: float,
    validation_eer: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = output_dir / "best.pt"
    temporary = output_dir / "best.pt.tmp"
    torch.save({"model": model.state_dict()}, temporary)
    temporary.replace(best)
    model_config_path = output_dir / "model_config.json"
    writers_path = output_dir / "writer_vocabulary.json"
    _write_json(model_config_path, asdict(model_config))
    _write_json(writers_path, {"writer_to_id": dict(writer_to_id)})
    contract = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(best),
        "model_config_sha256": sha256_file(model_config_path),
        "writer_vocabulary_sha256": sha256_file(writers_path),
        "resnet18_checkpoint_sha256": verify_visual_backbone(
            config.backbone.contract,
            name="resnet18_imagenet1k_v1",
            checkpoint=config.backbone.checkpoint,
        ),
        "visual_backbone_contract_sha256": sha256_file(
            config.backbone.contract
        ),
        "line_manifest_sha256": sha256_file(
            config.data.line_manifest
        ),
        "paragraph_manifest_sha256": sha256_file(
            config.data.paragraph_manifest
        ),
        "validation_auc": validation_auc,
        "validation_eer": validation_eer,
        "selection_protocol": config.selection_protocol,
    }
    _write_json(output_dir / "inference_contract.json", contract)


def validate_writer_inference_contract(
    *,
    checkpoint: Path,
    model_config: Path,
    writer_vocabulary: Path,
    contract_path: Path,
) -> Mapping[str, object]:
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy writer contract: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "checkpoint_sha256",
        "model_config_sha256",
        "writer_vocabulary_sha256",
        "resnet18_checkpoint_sha256",
        "visual_backbone_contract_sha256",
        "line_manifest_sha256",
        "paragraph_manifest_sha256",
        "validation_auc",
        "validation_eer",
        "selection_protocol",
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != expected
        or contract["schema_version"] != 1
    ):
        raise ValueError("Writer inference contract sai schema.")
    for path, key in (
        (checkpoint, "checkpoint_sha256"),
        (model_config, "model_config_sha256"),
        (writer_vocabulary, "writer_vocabulary_sha256"),
    ):
        if sha256_file(path) != contract[key]:
            raise ValueError(f"Writer artifact sai hash: {path}")
    if contract["selection_protocol"] != (
        "writer_disjoint_internal_validation"
    ):
        raise ValueError("Writer selection protocol không hợp lệ.")
    return contract


def train_writer_metric(
    config: WriterMetricTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    backbone_hash = verify_visual_backbone(
        config.backbone.contract,
        name="resnet18_imagenet1k_v1",
        checkpoint=config.backbone.checkpoint,
    )
    del backbone_hash
    records = _read_records(
        (config.data.line_manifest, config.data.paragraph_manifest)
    )
    train_writers, validation_writers = split_writer_ids(
        records,
        seed=config.seed,
        validation_fraction=config.data.validation_writer_fraction,
    )
    writer_to_id = {
        writer: index for index, writer in enumerate(train_writers)
    }
    model_config = WriterEncoderConfig()
    processor = WriterImageProcessor(model_config)
    train_dataset = WriterMetricDataset(
        records,
        writer_ids=train_writers,
        image_root=config.data.image_root,
        processor=processor,
        writer_to_id=writer_to_id,
    )
    validation_dataset = WriterMetricDataset(
        records,
        writer_ids=validation_writers,
        image_root=config.data.image_root,
        processor=processor,
        writer_to_id=writer_to_id,
    )
    sampler = WriterBalancedBatchSampler(
        train_dataset,
        writers_per_batch=config.data.writers_per_batch,
        samples_per_writer=config.data.samples_per_writer,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_writer_metric,
        num_workers=config.data.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_writer_metric,
        num_workers=config.data.num_workers,
    )
    model = WriterStyleEncoder(
        model_config,
        imagenet_checkpoint=config.backbone.checkpoint,
    ).to(device)
    head = ArcFaceHead(256, len(writer_to_id)).to(device)
    optimizer = AdamW(
        [*model.parameters(), *head.parameters()],
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    best_eer = math.inf
    best_auc = 0.0
    for epoch in range(config.optimizer.epochs):
        sampler.set_epoch(epoch)
        model.train()
        head.train()
        for batch in train_loader:
            images = batch["images"]
            labels = batch["labels"]
            if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
                raise TypeError("Writer train batch sai tensor contract.")
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(model(images), labels)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Writer ArcFace loss không hữu hạn.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*model.parameters(), *head.parameters()],
                config.optimizer.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
        model.eval()
        validation_embeddings: list[Tensor] = []
        validation_ids: list[str] = []
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["images"]
                if not isinstance(images, Tensor):
                    raise TypeError("Writer validation images phải là Tensor.")
                validation_embeddings.append(
                    model(images.to(device)).cpu()
                )
                validation_ids.extend(batch["writer_ids"])  # type: ignore[arg-type]
        auc, eer = validation_verification(
            torch.cat(validation_embeddings),
            validation_ids,
        )
        if eer < best_eer:
            best_eer = eer
            best_auc = auc
            save_writer_artifacts(
                config.checkpoint.output_dir,
                model=model,
                model_config=model_config,
                writer_to_id=writer_to_id,
                config=config,
                validation_auc=auc,
                validation_eer=eer,
            )
        print(
            f"writer epoch={epoch + 1} validation_auc={auc:.6f} "
            f"validation_eer={eer:.6f}"
        )
    return {"validation_auc": best_auc, "validation_eer": best_eer}


__all__ = [
    "WriterBalancedBatchSampler",
    "WriterImageProcessor",
    "WriterMetricDataset",
    "WriterMetricTrainingConfig",
    "collate_writer_metric",
    "load_writer_metric_config",
    "split_writer_ids",
    "train_writer_metric",
    "validation_verification",
    "validate_writer_inference_contract",
]
