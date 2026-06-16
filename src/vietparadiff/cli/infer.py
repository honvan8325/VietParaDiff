"""CLI for inference from a prepared manifest row."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from torch.utils.data import DataLoader

from vietparadiff.data.archetypes import ArchetypeRenderer
from vietparadiff.data.collate import paragraph_collate
from vietparadiff.data.dataset import ParagraphDataset
from vietparadiff.data.graphemes import VietnameseTokenizer
from vietparadiff.models.pipeline import VietParaDiff
from vietparadiff.utils.config import load_config
from vietparadiff.utils.device import choose_device
from vietparadiff.utils.image import tensor_to_pil


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate handwriting images for samples in a manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-dir", default="fonts")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = choose_device(args.device)
    tokenizer = VietnameseTokenizer(max_tokens=cfg["text"]["max_tokens"])
    renderer = ArchetypeRenderer(size=cfg["text"]["archetype_size"], font_dir=args.font_dir)
    dataset = ParagraphDataset(args.manifest, args.root, tokenizer, renderer, cfg["image"]["height"], cfg["image"]["width"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=paragraph_collate)
    model = VietParaDiff(cfg, tokenizer).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                batch[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        image = model.generate(batch, steps=args.steps, guidance=args.guidance)
        tensor_to_pil(image).save(out_dir / f"{batch['id'][0]}.png")


if __name__ == "__main__":
    main()
