"""Project-local font loading and Vietnamese glyph validation.

Synthetic generation is reproducible only when fonts are explicit.  Therefore
this module intentionally refuses to scan system directories.  All `.ttf`/`.otf`
files must live in the repository `fonts/` folder or in a user supplied
`--font-dir`.
"""

from __future__ import annotations

from pathlib import Path
from fontTools.ttLib import TTFont

REQUIRED_TEXT = (
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "eèéẻẽẹêềếểễệ"
    "iìíỉĩị"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữự"
    "yỳýỷỹỵđ"
    "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐ"
)


class FontValidationError(RuntimeError):
    """Raised when a font does not cover the Vietnamese glyph set."""


def project_font_dir() -> Path:
    """Return the repository-local `fonts` directory for an installed checkout."""

    # src/vietparadiff/data/fonts.py -> repo root is parents[4] in editable tree.
    here = Path(__file__).resolve()
    for parent in [*here.parents]:
        candidate = parent / "fonts"
        if candidate.exists():
            return candidate
    return Path.cwd() / "fonts"


def font_coverage(path: str | Path, required_text: str = REQUIRED_TEXT) -> tuple[float, list[str]]:
    """Return coverage ratio and missing characters using the Unicode cmap table."""

    path = Path(path)
    tt = TTFont(str(path), lazy=True)
    cmap: set[int] = set()
    for table in tt["cmap"].tables:
        cmap.update(table.cmap.keys())
    required = sorted(set(required_text))
    missing = [ch for ch in required if ord(ch) not in cmap]
    return 1.0 - len(missing) / max(1, len(required)), missing


def validate_font(path: str | Path, required_text: str = REQUIRED_TEXT) -> Path:
    """Validate that a font covers Vietnamese glyphs and return the path."""

    path = Path(path)
    if not path.exists():
        raise FontValidationError(f"Font does not exist: {path}")
    try:
        coverage, missing = font_coverage(path, required_text)
    except Exception as exc:  # pragma: no cover - depends on font parsing internals
        raise FontValidationError(f"Cannot read font {path}: {exc}") from exc
    if missing:
        preview = "".join(missing[:32])
        raise FontValidationError(f"Font {path.name} covers {coverage:.1%}; missing Vietnamese glyphs: {preview}")
    return path


def list_project_fonts(font_dir: str | Path | None = None, required_text: str = REQUIRED_TEXT) -> list[Path]:
    """List valid fonts from a project-local directory only.

    No system font discovery is performed.  This is intentional to keep
    synthetic datasets reproducible across machines.
    """

    root = Path(font_dir) if font_dir else project_font_dir()
    if not root.exists():
        return []
    candidates = sorted([*root.glob("*.ttf"), *root.glob("*.otf"), *root.glob("*.ttc")])
    valid: list[Path] = []
    errors: list[str] = []
    for font in candidates:
        try:
            valid.append(validate_font(font, required_text))
        except FontValidationError as exc:
            errors.append(str(exc))
    if not valid and candidates:
        raise FontValidationError("No valid Vietnamese-capable fonts found in " + str(root) + "\n" + "\n".join(errors[:5]))
    return valid
