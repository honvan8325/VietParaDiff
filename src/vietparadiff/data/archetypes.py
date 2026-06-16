"""Visual grapheme archetype rendering.

Each grapheme is represented by four rendered channels: base glyph, structural
modifier, tone mark, and full composed glyph.  The renderer uses only project
fonts, never system discovery.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from .fonts import list_project_fonts, validate_font
from .graphemes import GraphemeParts, NONE, NEWLINE, SPACE

MODIFIER_GLYPHS = {"breve": "˘", "circumflex": "ˆ", "horn": "̛", "stroke": "đ", NONE: ""}
TONE_GLYPHS = {"acute": "´", "grave": "`", "hook": "̉", "tilde": "˜", "dot": ".", NONE: ""}


class ArchetypeRenderer:
    """Render grapheme archetypes as normalized tensors."""

    def __init__(self, size: int = 48, font_path: str | Path | None = None, font_dir: str | Path | None = None) -> None:
        self.size = int(size)
        if font_path:
            self.font_path = validate_font(font_path)
        else:
            fonts = list_project_fonts(font_dir)
            if not fonts:
                raise RuntimeError(
                    "No Vietnamese-capable fonts found. Put .ttf/.otf files in fonts/ or pass --font-dir. "
                    "System font discovery is intentionally disabled."
                )
            self.font_path = fonts[0]
        self.font = ImageFont.truetype(str(self.font_path), max(12, int(size * 0.72)))
        self.small = ImageFont.truetype(str(self.font_path), max(10, int(size * 0.48)))

    def _render_text(self, text: str, *, small: bool = False) -> np.ndarray:
        image = Image.new("L", (self.size, self.size), 0)
        if not text:
            return np.zeros((self.size, self.size), dtype=np.float32)
        draw = ImageDraw.Draw(image)
        font = self.small if small else self.font
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (self.size - tw) / 2 - bbox[0]
        y = (self.size - th) / 2 - bbox[1]
        draw.text((x, y), text, fill=255, font=font)
        return np.asarray(image, dtype=np.float32) / 255.0

    def render_part(self, part: GraphemeParts) -> torch.Tensor:
        """Render one decomposed grapheme into `[4, S, S]`."""

        if part.base in {SPACE, NEWLINE} or part.kind in {"space", "newline"}:
            return torch.zeros(4, self.size, self.size, dtype=torch.float32)
        base = self._render_text(part.base)
        modifier = self._render_text(MODIFIER_GLYPHS.get(part.modifier, ""), small=True)
        tone = self._render_text(TONE_GLYPHS.get(part.tone, ""), small=True)
        full = self._render_text(part.surface)
        return torch.from_numpy(np.stack([base, modifier, tone, full], axis=0)).float()

    def render_batch(self, parts: list[GraphemeParts]) -> torch.Tensor:
        if not parts:
            return torch.zeros(0, 4, self.size, self.size, dtype=torch.float32)
        return torch.stack([self.render_part(p) for p in parts], dim=0)
