from __future__ import annotations

import torch

from vietparadiff.data.graphemes import VietnameseTokenizer, decompose_grapheme
from vietparadiff.models.diffusion import LatentDiffusion
from vietparadiff.models.layout import AutoregressiveLayoutPlanner, rasterize_layout
from vietparadiff.cli.smoke import tiny_cfg
from vietparadiff.models.pipeline import VietParaDiff


def test_vietnamese_decomposition():
    p = decompose_grapheme("ắ")
    assert p.base == "a"
    assert p.modifier == "breve"
    assert p.tone == "acute"
    p2 = decompose_grapheme("Đ")
    assert p2.base == "D"
    assert p2.modifier == "stroke"


def test_layout_shapes():
    planner = AutoregressiveLayoutPlanner(dim=32, hidden_dim=32)
    content = torch.randn(2, 7, 32)
    mask = torch.tensor([[1,1,1,1,1,1,1],[1,1,1,0,0,0,0]], dtype=torch.bool)
    style = torch.randn(2, 32)
    out = planner(content, mask, style)
    assert out["boxes"].shape == (2, 7, 4)
    fields = rasterize_layout(out["boxes"], out["anchors"], mask, 16, 32)
    assert fields.shape == (2, 5, 16, 32)


def test_diffusion_shapes():
    diffusion = LatentDiffusion(steps=16)
    clean = torch.randn(2, 4, 8, 16)
    noise = torch.randn_like(clean)
    t = torch.tensor([0, 15])
    noisy = diffusion.q_sample(clean, t, noise)
    assert noisy.shape == clean.shape
    loss = diffusion.noise_loss(noise, noise, t)
    assert torch.isfinite(loss)


def test_tiny_pipeline_stage():
    cfg = tiny_cfg()
    tok = VietnameseTokenizer(max_tokens=32)
    model = VietParaDiff(cfg, tok)
    b, n, s = 1, 6, 16
    text = {k: torch.randint(0, tok.vocab_sizes[k], (b, n)) for k in ["base", "modifier", "tone", "case", "type", "full"]}
    batch = {
        "image": torch.randn(b, 1, 64, 128),
        "reference": torch.randn(b, 1, 64, 128),
        "text": text,
        "text_mask": torch.ones(b, n, dtype=torch.bool),
        "archetypes": torch.randn(b, n, 4, s, s),
        "boxes": torch.tensor([[[0.05,0.1,0.10,0.2],[0.12,0.1,0.17,0.2],[0.19,0.1,0.23,0.2],[0.25,0.1,0.30,0.2],[0.32,0.1,0.36,0.2],[0.38,0.1,0.42,0.2]]]),
        "line_ids": torch.zeros(b, n, dtype=torch.long),
        "line_boxes": torch.tensor([[[0.05, 0.1, 0.95, 0.3]]]),
        "writer": torch.zeros(b, dtype=torch.long),
        "parts": [tok.parts("Tiếng")[:n]],
        "line_texts": [["Tiếng"]],
        "transcript": ["Tiếng"],
    }
    out = model.forward_style_layout(batch)
    assert torch.isfinite(out["loss"])
