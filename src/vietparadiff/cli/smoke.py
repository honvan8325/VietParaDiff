"""Fast CPU smoke test for tensor shapes without reading fonts or datasets."""

from __future__ import annotations

import torch

from vietparadiff.data.graphemes import VietnameseTokenizer
from vietparadiff.models.pipeline import VietParaDiff


def tiny_cfg() -> dict:
    return {
        "seed": 1,
        "image": {"channels": 1, "height": 64, "width": 128, "downsample": 8},
        "text": {"dim": 64, "layers": 1, "heads": 4, "max_tokens": 32, "archetype_size": 16, "visual_base": 16},
        "style": {"tokens_per_group": 2, "codebook_size": 16, "writer_classes": 8, "base_channels": 16, "transformer_layers": 1, "heads": 4, "pooled_grid": [2, 4]},
        "layout": {"hidden_dim": 64, "fields": 5, "teacher_forcing_start": 1.0, "teacher_forcing_end": 0.1, "teacher_forcing_decay_steps": 10},
        "vae": {"latent_channels": 2, "base_channels": 16},
        "htr": {"hidden_dim": 64},
        "topology": {"base_channels": 16},
        "unet": {"base_channels": 16, "channel_mults": [1, 2], "num_res_blocks": 1, "heads": 4, "time_dim": 64},
        "diffusion": {"steps": 16, "min_snr_gamma": 5.0, "cond_drop_prob": 0.0},
        "loss": {"kl": 1e-6, "frequency": 0.5, "orthogonality": 0.01, "style": 0.1, "refiner": 1.0, "topology": 0.5, "break": 0.1},
    }


def main() -> None:
    cfg = tiny_cfg()
    tokenizer = VietnameseTokenizer(max_tokens=cfg["text"]["max_tokens"])
    model = VietParaDiff(cfg, tokenizer)
    b, n, s = 1, 8, cfg["text"]["archetype_size"]
    text = {k: torch.randint(0, tokenizer.vocab_sizes[k], (b, n)) for k in ["base", "modifier", "tone", "case", "type", "full"]}
    mask = torch.ones(b, n, dtype=torch.bool)
    batch = {
        "image": torch.randn(b, 1, cfg["image"]["height"], cfg["image"]["width"]),
        "reference": torch.randn(b, 1, cfg["image"]["height"], cfg["image"]["width"]),
        "text": text,
        "text_mask": mask,
        "archetypes": torch.randn(b, n, 4, s, s),
        "boxes": torch.rand(b, n, 4).sort(dim=-1).values,
        "line_ids": torch.zeros(b, n, dtype=torch.long),
        "line_boxes": torch.tensor([[[0.05, 0.1, 0.95, 0.3]]]),
        "writer": torch.zeros(b, dtype=torch.long),
        "parts": [tokenizer.parts("Tiếng Việt")[:n]],
        "line_texts": [["Tiếng Việt"]],
        "transcript": ["Tiếng Việt"],
    }
    out = model.forward_style_layout(batch)
    print("style_layout loss", float(out["loss"]))
    vae = model.forward_vae(batch)
    print("vae loss", float(vae["loss"]))


if __name__ == "__main__":
    main()
