"""CLI for staged paper-level VietParaDiff training."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from vietparadiff.data.archetypes import ArchetypeRenderer
from vietparadiff.data.collate import paragraph_collate
from vietparadiff.data.dataset import ParagraphDataset
from vietparadiff.data.graphemes import VietnameseTokenizer
from vietparadiff.models.pipeline import VietParaDiff
from vietparadiff.training.checkpoint import load_modules_from_checkpoint
from vietparadiff.training.trainer import Trainer
from vietparadiff.utils.config import apply_overrides, load_config
from vietparadiff.utils.device import choose_device
from vietparadiff.utils.seed import seed_everything


DEPENDENCY_MODULES: dict[str, list[str]] = {
    "vae": ["vae"],
    "htr": ["htr"],
    "style_layout": ["content", "style", "layout"],
    "topology": ["topology"],
}

STAGE_DEPENDENCIES: dict[str, list[str]] = {
    "vae": [],
    "htr": [],
    "style_layout": [],
    # The topology detector uses frozen style-layout predictions to build exact/weak maps.
    "topology": ["style_layout"],
    # Diffusion consumes frozen low/high VAE, style/layout conditioning, topology, and optional HTR.
    "diffusion": ["vae", "htr", "style_layout", "topology"],
}


def load_stage_dependencies(model: VietParaDiff, cfg: dict, stage: str) -> None:
    """Restore dependency modules declared under training.dependency_checkpoints."""

    deps = cfg.get("training", {}).get("dependency_checkpoints", {}) or {}
    for dep_name in STAGE_DEPENDENCIES[stage]:
        path = deps.get(dep_name)
        if not path:
            raise RuntimeError(
                f"Stage '{stage}' requires training.dependency_checkpoints.{dep_name}. "
                f"Pass it with --set training.dependency_checkpoints.{dep_name}=runs/vietparadiff/{dep_name}/latest.pt"
            )
        info = load_modules_from_checkpoint(path, model, DEPENDENCY_MODULES[dep_name])
        print(f"loaded dependency {dep_name}: {info['modules']} from {info['path']} ({info['loaded_keys']} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one VietParaDiff stage with dependency checkpoint loading")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["vae", "htr", "style_layout", "topology", "diffusion"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--font-dir", default="fonts")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None, help="Full same-stage checkpoint to resume model, optimizer, scaler, and step")
    parser.add_argument("--save-every", type=int, default=None, help="Override training.checkpoint_every_steps")
    parser.add_argument("--log-every", type=int, default=None, help="Override training.log_every_steps")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.save_every is not None:
        overrides.append(f"training.checkpoint_every_steps={args.save_every}")
    if args.log_every is not None:
        overrides.append(f"training.log_every_steps={args.log_every}")
    cfg = apply_overrides(load_config(args.config), overrides)
    seed_everything(cfg.get("seed", 2026))
    device = choose_device(args.device)

    tokenizer = VietnameseTokenizer(max_tokens=cfg["text"]["max_tokens"])
    renderer = ArchetypeRenderer(size=cfg["text"]["archetype_size"], font_dir=args.font_dir)
    dataset = ParagraphDataset(args.manifest, args.root, tokenizer, renderer, cfg["image"]["height"], cfg["image"]["width"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=paragraph_collate,
    )

    model = VietParaDiff(cfg, tokenizer)
    if args.resume:
        # A full same-stage checkpoint already contains dependency module weights,
        # optimizer state, AMP scaler state, and the completed optimizer step.
        # Loading is done inside Trainer after the optimizer/scaler are created.
        print(f"resume requested for stage {args.stage}: {args.resume}")
    else:
        load_stage_dependencies(model, cfg, args.stage)

    Trainer(model, cfg, device).fit(loader, args.stage, args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
