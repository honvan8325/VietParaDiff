"""CLI for resumable synthetic Vietnamese paragraph dataset generation."""

from __future__ import annotations

import argparse
from vietparadiff.data.synthetic import generate_dataset


def _parse_height(value: str) -> int | None:
    """Parse ``--height``.  ``auto`` means natural height + bucket padding."""

    if value.lower() in {"auto", "dynamic", "none"}:
        return None
    return int(value)


def _parse_buckets(value: str) -> list[int]:
    """Parse a comma-separated height bucket list."""

    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resumable synthetic Vietnamese paragraph handwriting data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-writers", type=int, default=64)
    parser.add_argument("--samples-per-writer", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--height",
        default="auto",
        help="Use 'auto' for natural paragraph height padded to buckets, or an integer for fixed-height ablation only.",
    )
    parser.add_argument(
        "--height-buckets",
        default="256,384,512,768,1024,1536",
        help="Comma-separated canvas height buckets used when --height auto.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=6)
    parser.add_argument("--corpus", default=None, help="Optional UTF-8 text file, one sentence per line")
    parser.add_argument("--lexicon", default=None, help="Optional JSON lexicon overriding the built-in corpus generator")
    parser.add_argument("--font-dir", "--synthetic-font-dir", default="fonts/synthetic", help="Directory containing many synthetic rendering fonts; system fonts are not searched")
    parser.add_argument("--font", action="append", default=None, help="Explicit font file; can be passed multiple times")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory and regenerate from scratch")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted generation by skipping manifest rows with existing images")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--render-retries", type=int, default=12, help="Retry with a new paragraph if a sample overflows geometry guards")
    args = parser.parse_args()

    height = _parse_height(args.height)
    manifest = generate_dataset(
        output=args.output,
        num_writers=args.num_writers,
        samples_per_writer=args.samples_per_writer,
        width=args.width,
        seed=args.seed,
        height=height,
        height_buckets=None if height is not None else _parse_buckets(args.height_buckets),
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        corpus_path=args.corpus,
        lexicon_path=args.lexicon,
        font_dir=args.font_dir,
        fonts=args.font,
        overwrite=args.overwrite,
        resume=args.resume,
        show_progress=not args.no_progress,
        render_retries=args.render_retries,
    )
    print(manifest)


if __name__ == "__main__":
    main()
