"""Synthetic Vietnamese paragraph generation with exact line and token annotation.

The renderer is intentionally conservative about geometry.  It renders each line
on an expanded scratch layer, applies writer slant on that layer only, computes
the actual ink bounding box after deformation, and pastes the line back into the
page with safe margins.  This prevents clipped first/last glyphs, invalid boxes,
and repeated paragraphs that would poison layout/diffusion training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from .corpus import VietnameseCorpusGenerator
from .fonts import list_project_fonts, validate_font
from .graphemes import split_graphemes
from .manifest import LineAnnotation, ManifestRow, TokenAnnotation, read_jsonl, write_jsonl


@dataclass
class SyntheticStyle:
    """Writer-like rendering parameters sampled per synthetic writer."""

    writer_id: str
    font_path: Path
    font_size: int
    line_gap: int
    ink: int
    blur: float
    slant: float
    jitter: int
    word_space_scale: float


@dataclass
class RenderedLine:
    """A rendered line layer and geometry needed for annotation."""

    text: str
    image: Image.Image
    ink_bbox: tuple[int, int, int, int]
    advances: list[float]
    total_advance: float


def _ink_bbox(image: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-white ink pixels."""

    arr = np.asarray(image.convert("L"))
    ys, xs = np.where(arr < threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _shift_rows_for_slant(image: Image.Image, slant: float) -> Image.Image:
    """Apply row-wise shear to a scratch line image without clipping ink."""

    if abs(slant) < 1e-4:
        return image
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape
    max_shift = int(abs(slant) * h / 2) + 8
    out = np.full((h, w + 2 * max_shift), 255, dtype=np.uint8)
    center = (h - 1) / 2.0
    for y in range(h):
        shift = int(round(slant * (y - center))) + max_shift
        out[y, shift : shift + w] = np.minimum(out[y, shift : shift + w], arr[y])
    return Image.fromarray(out, mode="L")


def _glyph_advance(font: ImageFont.FreeTypeFont, surface: str, word_space_scale: float) -> float:
    """Measure one grapheme advance, including a controllable space width."""

    if surface == " ":
        return max(float(font.getlength("n")) * word_space_scale, 1.0)
    return max(float(font.getlength(surface)), 1.0)


def _measure_line_width(text: str, font: ImageFont.FreeTypeFont, style: SyntheticStyle) -> float:
    """Measure a line using the same grapheme advances used for annotation."""

    return sum(_glyph_advance(font, g, style.word_space_scale) for g in split_graphemes(text))


def wrap_words(text: str, font: ImageFont.FreeTypeFont, max_width: int, style: SyntheticStyle) -> list[str]:
    """Wrap text by font metrics and reserve room for slant/jitter.

    The function never emits a line whose measured width exceeds the safe line
    width unless a single word is itself too long; in that case it falls back to
    grapheme-level breaking.
    """

    safe_width = max(48, max_width - int(abs(style.slant) * style.font_size * 3) - 2 * style.jitter - 16)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            probe = word if not current else f"{current} {word}"
            if _measure_line_width(probe, font, style) <= safe_width or not current:
                current = probe
                continue
            lines.append(current)
            current = word
            if _measure_line_width(current, font, style) > safe_width:
                chunk = ""
                for g in split_graphemes(current):
                    probe_chunk = chunk + g
                    if _measure_line_width(probe_chunk, font, style) <= safe_width or not chunk:
                        chunk = probe_chunk
                    else:
                        lines.append(chunk)
                        chunk = g
                current = chunk
        if current:
            lines.append(current)
    return lines or [""]


def _render_line(line: str, font: ImageFont.FreeTypeFont, style: SyntheticStyle) -> RenderedLine:
    """Render one line on a padded layer, then apply safe slant deformation."""

    graphemes = split_graphemes(line)
    advances = [_glyph_advance(font, g, style.word_space_scale) for g in graphemes]
    total_advance = max(sum(advances), 1.0)
    ascent, descent = font.getmetrics()
    raw_bbox = font.getbbox(line or " ")
    raw_w = max(int(np.ceil(total_advance)), raw_bbox[2] - raw_bbox[0], 1)
    raw_h = max(ascent + descent, raw_bbox[3] - raw_bbox[1], style.font_size)
    pad_x = max(16, int(style.font_size * (1.2 + abs(style.slant) * 2.5)))
    pad_y = max(14, int(style.font_size * 0.9))
    layer = Image.new("L", (raw_w + 2 * pad_x, raw_h + 2 * pad_y), 255)
    draw = ImageDraw.Draw(layer)
    draw.text((pad_x - raw_bbox[0], pad_y - raw_bbox[1]), line, fill=style.ink, font=font)
    layer = _shift_rows_for_slant(layer, style.slant)
    bbox = _ink_bbox(layer)
    if bbox is None:
        bbox = (0, 0, 1, 1)
    return RenderedLine(line, layer, bbox, advances, total_advance)


def _build_font(style: SyntheticStyle, size: int | None = None) -> ImageFont.FreeTypeFont:
    """Load the writer font at a selected size."""

    return ImageFont.truetype(str(style.font_path), int(size or style.font_size))


def _stacked_ink_height(rendered: list[RenderedLine], line_gap: int, margin_y: int) -> int:
    """Compute paragraph height from actual ink boxes, not scratch-layer padding."""

    ink_heights = [max(1, rl.ink_bbox[3] - rl.ink_bbox[1]) for rl in rendered]
    return margin_y * 2 + sum(ink_heights) + line_gap * max(0, len(rendered) - 1)


def _fit_lines_to_box(
    style: SyntheticStyle,
    text: str,
    width: int,
    margin_x: int,
    margin_y: int,
    target_height: int | None,
) -> tuple[ImageFont.FreeTypeFont, list[str], list[RenderedLine], int]:
    """Choose the largest font size that fits the requested canvas.

    A binary search is used instead of a linear decrement so large synthetic
    datasets are not slowed down by repeated full-line rendering.
    """

    min_size = max(17, int(style.font_size * 0.50))
    max_size = style.font_size
    max_line_width = width - 2 * margin_x
    best: tuple[ImageFont.FreeTypeFont, list[str], list[RenderedLine], int] | None = None

    lo, hi = min_size, max_size
    while lo <= hi:
        size = (lo + hi) // 2
        trial_style = SyntheticStyle(**{**style.__dict__, "font_size": size})
        font = _build_font(trial_style, size)
        lines = wrap_words(text, font, max_line_width, trial_style)
        rendered = [_render_line(line, font, trial_style) for line in lines]
        ascent, descent = font.getmetrics()
        gap = max(int(style.line_gap * size / max(style.font_size, 1)), int((ascent + descent) * 0.14), 4)
        width_ok = all((rl.ink_bbox[2] - rl.ink_bbox[0]) <= max_line_width for rl in rendered)
        height_ok = target_height is None or _stacked_ink_height(rendered, gap, margin_y) <= target_height
        if width_ok and height_ok:
            best = (font, lines, rendered, gap)
            lo = size + 1
        else:
            hi = size - 1

    if best is not None:
        return best

    final_style = SyntheticStyle(**{**style.__dict__, "font_size": min_size})
    font = _build_font(final_style, min_size)
    lines = wrap_words(text, font, max_line_width, final_style)
    rendered = [_render_line(line, font, final_style) for line in lines]
    ascent, descent = font.getmetrics()
    gap = max(int(style.line_gap * min_size / max(style.font_size, 1)), int((ascent + descent) * 0.14), 4)
    return font, lines, rendered, gap



def choose_canvas_height(natural_height: int, height_buckets: list[int] | None, multiple: int = 64) -> int:
    """Choose the training canvas height for a natural paragraph height.

    The renderer never rescales text vertically to fit a fixed height.  It first
    computes the natural ink-driven height, then pads to the smallest configured
    bucket.  If a paragraph is taller than every bucket, the height is rounded up
    to the next multiple so the sample remains valid instead of being clipped.
    """

    natural_height = max(int(natural_height), multiple)
    if height_buckets:
        buckets = sorted({int(b) for b in height_buckets if int(b) > 0})
        for bucket in buckets:
            if natural_height <= bucket:
                return bucket
    return int(np.ceil(natural_height / multiple) * multiple)

def render_sample(
    style: SyntheticStyle,
    text: str,
    width: int,
    sample_id: str,
    rng: random.Random,
    height: int | None = None,
    height_buckets: list[int] | None = None,
) -> tuple[Image.Image, ManifestRow]:
    """Render one paragraph image and exact metric annotations."""

    margin_x = max(64, width // 14)
    target_height = height
    margin_y = max(30, min(width // 28, (target_height or width) // 9))
    font, lines, rendered_lines, line_gap = _fit_lines_to_box(style, text, width, margin_x, margin_y, target_height)
    natural_h = _stacked_ink_height(rendered_lines, line_gap, margin_y)
    if target_height is not None:
        if natural_h > target_height:
            raise RuntimeError(f"Synthetic render height overflow in {sample_id}: natural={natural_h}, target={target_height}")
        canvas_h = int(target_height)
    else:
        canvas_h = choose_canvas_height(natural_h, height_buckets)
    image = Image.new("L", (width, canvas_h), 255)

    y = margin_y
    line_annotations: list[LineAnnotation] = []
    token_annotations: list[TokenAnnotation] = []
    transcript_lines: list[str] = []

    for line_id, rl in enumerate(rendered_lines):
        bbox_w = rl.ink_bbox[2] - rl.ink_bbox[0]
        # Left margin is never allowed to go negative; long decorative lines are centered inside safe area.
        jitter = rng.randint(-style.jitter, style.jitter)
        ink_x = margin_x + jitter
        if ink_x + bbox_w > width - margin_x:
            ink_x = max(margin_x, (width - bbox_w) // 2)
        ink_x = max(margin_x // 2, min(ink_x, width - margin_x - bbox_w))
        paste_x = int(round(ink_x - rl.ink_bbox[0]))
        paste_y = int(round(y - rl.ink_bbox[1]))
        paste_x = max(0, min(paste_x, max(0, width - rl.image.size[0])))
        paste_y = max(0, min(paste_y, max(0, canvas_h - rl.image.size[1])))

        mask = Image.eval(rl.image, lambda p: 255 - p)
        image.paste(rl.image, (paste_x, paste_y), mask)

        line_box = [
            float(paste_x + rl.ink_bbox[0]),
            float(paste_y + rl.ink_bbox[1]),
            float(paste_x + rl.ink_bbox[2]),
            float(paste_y + rl.ink_bbox[3]),
        ]
        line_annotations.append(LineAnnotation(rl.text, line_box))

        line_w = max(line_box[2] - line_box[0], 1.0)
        cursor = 0.0
        graphemes = split_graphemes(rl.text)
        for surface, advance in zip(graphemes, rl.advances, strict=False):
            tx1 = line_box[0] + (cursor / rl.total_advance) * line_w
            tx2 = line_box[0] + ((cursor + advance) / rl.total_advance) * line_w
            token_annotations.append(TokenAnnotation(surface, [tx1, line_box[1], tx2, line_box[3]], line_id))
            cursor += advance
        if line_id < len(rendered_lines) - 1:
            # Keep token count aligned with transcripts that join lines using "\n".
            token_annotations.append(TokenAnnotation("\n", [line_box[2], line_box[1], line_box[2] + 1.0, line_box[3]], line_id))

        transcript_lines.append(rl.text)
        y = int(round(line_box[3] + line_gap))

    if style.blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(style.blur))

    page_bbox = _ink_bbox(image)
    if page_bbox is None:
        page_bbox = (0, 0, 1, 1)
    # Hard guard: a generated training image with edge-touching ink is invalid.
    if page_bbox[0] <= 1 or page_bbox[2] >= width - 1 or page_bbox[3] >= canvas_h - 1:
        raise RuntimeError(f"Synthetic render overflow in {sample_id}: ink bbox={page_bbox}, width={width}")

    row = ManifestRow(
        id=sample_id,
        writer_id=style.writer_id,
        document_id=f"{style.writer_id}_D0",
        image=f"images/{sample_id}.png",
        transcript="\n".join(transcript_lines),
        lines=line_annotations,
        tokens=token_annotations,
        meta={
            "font": str(style.font_path),
            "font_size": font.size,
            "requested_font_size": style.font_size,
            "slant": style.slant,
            "line_gap": line_gap,
            "word_space_scale": style.word_space_scale,
            "page_ink_bbox": list(map(int, page_bbox)),
            "natural_height": int(natural_h),
            "canvas_width": int(width),
            "canvas_height": int(canvas_h),
            "height_buckets": height_buckets or [],
        },
    )
    return image, row


def _sample_seed(seed: int, writer_index: int, sample_index: int) -> int:
    """Return a stable per-sample seed independent of resume/skip order."""

    # Large odd constants avoid collisions for realistic dataset sizes and keep
    # each sample reproducible even if earlier samples were skipped or retried.
    return int(seed + 1_000_003 * writer_index + 9_176_489 * sample_index)


def _writer_seed(seed: int, writer_index: int) -> int:
    """Return a stable per-writer seed for style parameters."""

    return int(seed + 65_537 * writer_index)


def _append_row_jsonl(path: Path, row: ManifestRow) -> None:
    """Append one manifest row immediately so generation can resume after a crash."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _load_resumable_rows(manifest: Path, root: Path) -> dict[str, ManifestRow]:
    """Load existing rows whose image files are present and non-empty.

    Duplicate IDs can happen if a previous process was killed while rewriting a
    canonical manifest.  The last valid row wins, and the final manifest is
    rewritten in deterministic ID order by ``generate_dataset``.
    """

    if not manifest.exists():
        return {}
    rows: dict[str, ManifestRow] = {}
    for row in read_jsonl(manifest):
        image_path = root / row.image
        if image_path.exists() and image_path.stat().st_size > 0:
            rows[row.id] = row
    return rows


def _validate_resume_config(info_path: Path, expected: dict) -> None:
    """Guard against accidentally resuming a dataset with incompatible settings."""

    if not info_path.exists():
        return
    old = json.loads(info_path.read_text(encoding="utf-8"))
    checked = ["num_writers", "samples_per_writer", "seed", "width", "height", "height_buckets", "font_dir"]
    mismatches = []
    for key in checked:
        if old.get(key) != expected.get(key):
            mismatches.append((key, old.get(key), expected.get(key)))
    if mismatches:
        details = "; ".join(f"{k}: old={a!r}, new={b!r}" for k, a, b in mismatches)
        raise RuntimeError(f"Cannot resume synthetic dataset with different generation settings: {details}")


def generate_dataset(
    output: str | Path,
    num_writers: int,
    samples_per_writer: int,
    width: int,
    seed: int,
    height: int | None = None,
    height_buckets: list[int] | None = None,
    min_sentences: int = 2,
    max_sentences: int = 6,
    corpus_path: str | None = None,
    lexicon_path: str | None = None,
    font_dir: str | None = None,
    fonts: list[str] | None = None,
    overwrite: bool = False,
    resume: bool = False,
    show_progress: bool = True,
    render_retries: int = 12,
) -> Path:
    """Generate a resumable synthetic Vietnamese paragraph dataset.

    The generator is deterministic at the **writer** and **sample** level.  This
    is important for paper-scale runs: if the process is killed at sample 300k,
    rerunning with ``--resume`` skips completed samples, regenerates missing
    samples with the same IDs and random seeds, appends new manifest rows as it
    goes, and rewrites a canonical manifest at the end.

    Fonts are loaded exclusively from ``font_dir`` or explicitly passed font
    files.  System font discovery is intentionally disabled for reproducibility.
    """

    out = Path(output)
    manifest = out / "manifest.jsonl"
    info_path = out / "dataset_info.json"

    if overwrite and resume:
        raise ValueError("Use only one of overwrite=True or resume=True.")
    if out.exists() and overwrite:
        shutil.rmtree(out)
    if out.exists() and not resume and not overwrite and any(out.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty: {out}. Use --overwrite or --resume.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    valid_fonts = [validate_font(f) for f in fonts] if fonts else list_project_fonts(font_dir)
    if not valid_fonts:
        raise RuntimeError("No valid fonts found. Put Vietnamese-capable .ttf/.otf files in fonts/ or pass --font-dir.")

    expected_info = {
        "num_writers": num_writers,
        "samples_per_writer": samples_per_writer,
        "seed": seed,
        "width": width,
        "height": height,
        "height_buckets": height_buckets or [],
        "height_mode": "fixed" if height is not None else "dynamic_bucketed",
        "font_dir": str(font_dir or "fonts"),
        "corpus_path": corpus_path,
        "lexicon_path": lexicon_path,
    }
    if resume:
        _validate_resume_config(info_path, expected_info)
    existing = _load_resumable_rows(manifest, out) if resume else {}

    corpus = VietnameseCorpusGenerator.from_files(seed=seed, corpus_path=corpus_path, lexicon_path=lexicon_path)
    total = num_writers * samples_per_writer
    completed = len(existing)
    rows_by_id: dict[str, ManifestRow] = dict(existing)
    generated_count = 0
    skipped_count = 0
    retry_count = 0

    pbar = tqdm(total=total, initial=completed, disable=not show_progress, desc="synthetic", unit="img")
    try:
        for wi in range(num_writers):
            writer_rng = random.Random(_writer_seed(seed, wi))
            font_path = valid_fonts[wi % len(valid_fonts)]
            style = SyntheticStyle(
                writer_id=f"W{wi:05d}",
                font_path=font_path,
                font_size=writer_rng.randint(34, 52),
                line_gap=writer_rng.randint(12, 30),
                ink=writer_rng.randint(12, 78),
                blur=writer_rng.choice([0.0, 0.05, 0.10, 0.16]),
                slant=writer_rng.uniform(-0.10, 0.10),
                jitter=writer_rng.randint(0, 3),
                word_space_scale=writer_rng.uniform(0.75, 1.10),
            )
            for si in range(samples_per_writer):
                sample_id = f"{style.writer_id}_P{si:05d}"
                if sample_id in rows_by_id:
                    # The progress bar already starts at len(existing); do not
                    # update it again for skipped samples or resume progress will
                    # exceed the true total.
                    skipped_count += 1
                    if (skipped_count + generated_count) % 100 == 0:
                        pbar.set_postfix(generated=generated_count, skipped=skipped_count, retries=retry_count)
                    continue

                sample_rng = random.Random(_sample_seed(seed, wi, si))
                last_error: Exception | None = None
                for attempt in range(max(1, render_retries)):
                    text = corpus.paragraph(sample_rng, min_sentences=min_sentences, max_sentences=max_sentences)
                    try:
                        image, row = render_sample(style, text, width, sample_id, sample_rng, height=height, height_buckets=height_buckets)
                        row.meta["sample_seed"] = _sample_seed(seed, wi, si)
                        row.meta["render_attempt"] = attempt
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        retry_count += 1
                else:
                    raise RuntimeError(f"Failed to render {sample_id} after {render_retries} attempts. Last error: {last_error}")

                image_path = out / row.image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write: keep a real image extension so Pillow can infer
                # the encoder, and still replace the final PNG in one step.
                # Using image_path.with_suffix(image_path.suffix + ".tmp")
                # creates files ending in .tmp, which Pillow cannot save unless
                # format is passed explicitly. The .tmp.png suffix is both
                # crash-safe and Pillow-compatible.
                tmp_path = image_path.with_name(f".{image_path.name}.tmp.png")
                image.save(tmp_path, format="PNG")
                tmp_path.replace(image_path)
                _append_row_jsonl(manifest, row)
                rows_by_id[row.id] = row
                generated_count += 1
                pbar.update(1)
                if (skipped_count + generated_count) % 100 == 0:
                    pbar.set_postfix(generated=generated_count, skipped=skipped_count, retries=retry_count)
    finally:
        pbar.close()

    # Rewrite a canonical, duplicate-free manifest in deterministic sample order.
    ordered_rows: list[ManifestRow] = []
    missing: list[str] = []
    for wi in range(num_writers):
        for si in range(samples_per_writer):
            sample_id = f"W{wi:05d}_P{si:05d}"
            row = rows_by_id.get(sample_id)
            if row is None:
                missing.append(sample_id)
            else:
                ordered_rows.append(row)
    if missing:
        raise RuntimeError(f"Synthetic generation incomplete; first missing sample IDs: {missing[:10]}")
    write_jsonl(manifest, ordered_rows)

    info = {
        **expected_info,
        "fonts_used": sorted({str(r.meta["font"]) for r in ordered_rows}),
        "renderer": "safe_line_layer_slant_dynamic_resume_v4",
        "resume_supported": True,
        "completed_samples": len(ordered_rows),
        "generated_this_run": generated_count,
        "skipped_this_run": skipped_count,
        "render_retries_this_run": retry_count,
        "guarantees": [
            "project-local fonts only",
            "paragraph-level duplicate sentence rejection",
            "line-level slant without page-edge clipping",
            "actual post-render ink bounding boxes in manifest metadata",
            "dynamic natural paragraph height padded to buckets, never vertical-resized",
            "token annotations aligned with newline-inclusive transcripts",
            "manifest rows appended during generation for crash-safe resume",
            "deterministic writer/sample seeds independent of resume order",
        ],
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
