# VietParaDiff SourceFinal

Paper-level source tree for **VietParaDiff: Diacritic-aware Layout-Factorized Latent Diffusion for One-shot Vietnamese Paragraph Handwriting Generation**.

This version implements the final training pipeline:

- project-local `fonts/` only; no system font discovery;
- synthetic paragraph rendering with dynamic natural height and bucket padding;
- resumable synthetic generation with tqdm progress;
- staged training with dependency checkpoints;
- true same-stage training resume with model + optimizer + AMP scaler + completed optimizer step;
- VAE, HTR, style-layout, topology, diffusion/refiner, inference CLI;
- Python 3.12 and `uv`.

## Install

```bash
uv python install 3.12
uv sync --extra dev
uv run pytest -q
```

## Fonts

```bash
uv run vpd-download-fonts --output fonts
```

You can also put licensed `.ttf` / `.otf` files directly under `fonts/`.

## Synthetic generation

Fresh generation:

```bash
uv run vpd-synthetic \
  --output data/synthetic_paper_v1 \
  --num-writers 4096 \
  --samples-per-writer 128 \
  --width 1024 \
  --height auto \
  --height-buckets 256,384,512,768,1024,1536 \
  --seed 2026 \
  --min-sentences 2 \
  --max-sentences 8 \
  --font-dir fonts \
  --overwrite
```

Resume interrupted generation:

```bash
uv run vpd-synthetic \
  --output data/synthetic_paper_v1 \
  --num-writers 4096 \
  --samples-per-writer 128 \
  --width 1024 \
  --height auto \
  --height-buckets 256,384,512,768,1024,1536 \
  --seed 2026 \
  --min-sentences 2 \
  --max-sentences 8 \
  --font-dir fonts \
  --resume
```

Rows are appended to `manifest.jsonl` during generation, then rewritten in canonical order at the end. The renderer uses deterministic writer/sample seeds so resume order does not change missing samples.

## Split

```bash
uv run vpd-prepare \
  --input data/synthetic_paper_v1/manifest.jsonl \
  --output-dir data/synthetic_paper_v1/splits \
  --seed 2026 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## Train stages

`training.max_steps` means completed optimizer updates. With `grad_accum_steps=16`, `max_steps=400000` means 400k optimizer updates.

VAE:

```bash
uv run vpd-train --config configs/base.yaml --stage vae \
  --manifest data/synthetic_paper_v1/splits/train.jsonl \
  --root data/synthetic_paper_v1 --font-dir fonts --device cuda \
  --set training.batch_size=4 --set training.grad_accum_steps=8 \
  --set training.max_steps=250000 --set training.lr=2e-4
```

Resume VAE:

```bash
uv run vpd-train --config configs/base.yaml --stage vae \
  --manifest data/synthetic_paper_v1/splits/train.jsonl \
  --root data/synthetic_paper_v1 --font-dir fonts --device cuda \
  --resume runs/vietparadiff/vae/latest.pt
```

Diffusion with dependencies:

```bash
uv run vpd-train --config configs/base.yaml --stage diffusion \
  --manifest data/synthetic_paper_v1/splits/train.jsonl \
  --root data/synthetic_paper_v1 --font-dir fonts --device cuda \
  --set training.dependency_checkpoints.vae=runs/vietparadiff/vae/latest.pt \
  --set training.dependency_checkpoints.htr=runs/vietparadiff/htr/latest.pt \
  --set training.dependency_checkpoints.style_layout=runs/vietparadiff/style_layout/latest.pt \
  --set training.dependency_checkpoints.topology=runs/vietparadiff/topology/latest.pt \
  --set training.batch_size=2 --set training.grad_accum_steps=16 \
  --set training.max_steps=400000 --set training.lr=1e-4
```

Resume diffusion:

```bash
uv run vpd-train --config configs/base.yaml --stage diffusion \
  --manifest data/synthetic_paper_v1/splits/train.jsonl \
  --root data/synthetic_paper_v1 --font-dir fonts --device cuda \
  --resume runs/vietparadiff/diffusion/latest.pt
```


## Font layout

VietParaDiff now separates the two font roles explicitly:

```text
fonts/synthetic/   # many fonts used only by vpd-synthetic
fonts/gnu/         # exactly one GNU/Unicode font used by train/infer archetypes
```

Download both folders:

```bash
uv run vpd-download-fonts --output fonts
```

Generate synthetic data with many style fonts:

```bash
uv run vpd-synthetic --font-dir fonts/synthetic ...
```

Train/infer with one stable archetype font:

```bash
uv run vpd-train --archetype-font fonts/gnu/unifont-15.1.05.otf ...
uv run vpd-infer --archetype-font fonts/gnu/unifont-15.1.05.otf ...
```
