"""Vietnamese visual-grapheme content encoder."""

from __future__ import annotations

import torch
from torch import nn

from .common import ResidualBlock, group_norm


class VisualGraphemeContentEncoder(nn.Module):
    """Encode four-channel grapheme archetypes plus decomposed symbolic streams."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        dim: int = 384,
        layers: int = 8,
        heads: int = 8,
        max_tokens: int = 1024,
        visual_base: int = 96,
    ) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(4, visual_base, 3, padding=1), nn.SiLU(),
            ResidualBlock(visual_base),
            nn.Conv2d(visual_base, visual_base * 2, 4, stride=2, padding=1), nn.SiLU(),
            ResidualBlock(visual_base * 2),
            nn.Conv2d(visual_base * 2, visual_base * 4, 4, stride=2, padding=1), nn.SiLU(),
            ResidualBlock(visual_base * 4),
            group_norm(visual_base * 4), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.visual_proj = nn.Linear(visual_base * 4, dim)
        self.embedding_keys = ["base", "modifier", "tone", "case", "type"]
        self.embeddings = nn.ModuleDict({("kind" if k == "type" else k): nn.Embedding(vocab_sizes[k], dim) for k in self.embedding_keys})
        self.position = nn.Embedding(max_tokens, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text: dict[str, torch.Tensor], archetypes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n = archetypes.shape[:2]
        vis = self.visual(archetypes.reshape(b * n, *archetypes.shape[2:])).flatten(1)
        x = self.visual_proj(vis).view(b, n, -1)
        for key in self.embedding_keys:
            module_key = "kind" if key == "type" else key
            x = x + self.embeddings[module_key](text[key])
        pos = torch.arange(n, device=archetypes.device).clamp_max(self.position.num_embeddings - 1)
        x = x + self.position(pos)[None]
        x = self.transformer(x, src_key_padding_mask=~mask)
        return self.norm(x)
