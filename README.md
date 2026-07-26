# VietParaDiff

**One-shot Vietnamese paragraph handwriting generation with latent diffusion,
scale-separated grapheme conditioning, and inter-line style harmonization.**

VietParaDiff generates an entire grayscale paragraph in one latent canvas from:

1. target Vietnamese text; and
2. one real handwriting reference line.

The generator is trained from scratch. It does not use a pretrained
Paragraph LDM, Stable Diffusion VAE, GAN, VQ/SAQ codebook, OCR correction,
best-of-N reranking, line-by-line generation, or stitched output at inference.
ImageNet initialization is used only for the visual style and writer-metric
backbones.

> [!IMPORTANT]
> This repository implements the complete training and evaluation
> infrastructure, but it does not ship trained research checkpoints or paper
> results. External baseline templates intentionally contain invalid
> placeholder hashes until real baseline artifacts are supplied.

## Highlights

- Direct paragraph generation at fixed width `1024` and dynamic height.
- Writer-disjoint Vietnamese adaptation and evaluation using UIT-HWDB only.
- Four-channel handwriting AutoKL trained from scratch.
- Vietnamese base–shape–tone grapheme factorization.
- Dual-frequency one-shot style encoding with continuous local and global
  conditions.
- Scale-separated shape and tone residual adapters.
- Attention-based inter-line harmonization without line boxes, pseudo masks,
  or a vertical spatial prior.
- Velocity-prediction latent diffusion.
- Differentiable HTR-guided fine-tuning through the frozen AutoKL decoder.
- Strict, hash-bound checkpoints, vocabularies, latent statistics, inference
  contracts, evaluation outputs, and experiment manifests.
- Deterministic three-seed evaluation and aggregation infrastructure.

## Method overview

```text
REFERENCE LINE
    ├── raw grayscale stem
    └── foreground-masked Laplacian stem
                 │
                 ▼
       shared ConvNeXt-Tiny trunk
                 │
        ┌────────┴────────┐
        ▼                 ▼
16 local style tokens   global style vector

TARGET TEXT
    │
    ├── NFC normalization
    ├── grapheme segmentation
    ├── base / shape / tone factorization
    └── deterministic paragraph formatting
                 │
                 ▼
       Factorized Grapheme Transformer
        ├── base context
        ├── shape context
        └── tone context
                 │
                 ▼
       Paragraph Latent Diffusion U-Net
        ├── base: every scale
        ├── shape: high + medium scales
        ├── tone: highest scale only
        ├── local style: cross-attention
        ├── global style: FiLM / AdaGN
        └── text-guided inter-line harmonizer
                 │
                 ▼
          clean paragraph latent
                 │
                 ▼
       frozen Handwriting AutoKL decoder
                 │
                 ▼
       grayscale paragraph image
```

### Locked model contracts

| Component | Contract |
| --- | --- |
| Output canvas | Grayscale, width `1024`, height in `384…1280`, at most 8 lines |
| AutoKL | Base 32, multipliers `[1,2,4,8]`, 4 latent channels, downsample ×8 |
| AutoKL bottleneck | Row + column axial attention |
| Text encoder | 6 layers, dim 512, 8 heads, FFN 2048 |
| Grapheme contexts | Base, shape, and tone contexts of dimension 768 |
| Style encoder | Dual grayscale stems, shared ConvNeXt-Tiny, 16 local tokens |
| U-Net | Channels `[128,256,512,768]`, 2 ResBlocks per level |
| Spatial attention | Row at high; axial at medium/low; global at deepest/middle |
| Harmonizer | Dim 512, 2 layers, 8 heads; no spatial mask or vertical prior |
| Diffusion target | Velocity prediction with a cosine training schedule |
| HTR teacher | Line-level CNN-Conformer with raw/base/shape/tone CTC heads |

The formatter is neutral and deterministic. It preserves hard newlines, uses
fixed wrapping priors, and selects a supported height bucket. The current
implementation does **not** claim learned reference-calibrated character width,
word gap, or line gap.

## Data protocol

VietParaDiff uses three normalized real handwriting corpora:

- **CVL**
- **IAM**
- **UIT-HWDB**

Vietnamese training and evaluation use UIT-HWDB only. This avoids ambiguous
cross-corpus writer identities.

| Stage | Targets | References |
| --- | --- | --- |
| AutoKL | Real CVL, IAM, and UIT-HWDB paragraphs | — |
| HTR teacher | Real UIT-HWDB lines and words | — |
| HTR evaluator | Real UIT-HWDB lines and words; independently trained | — |
| Generator pretrain | IAM and supported CVL paragraphs | Same-corpus real lines |
| Vietnamese fine-tune | Real UIT-HWDB paragraphs + train-only synthetic paragraphs | Same-writer real UIT-HWDB lines |
| HTR-guided fine-tune | Same pool as Vietnamese fine-tune | Same-writer real UIT-HWDB lines |
| Final evaluation | Real paragraphs from held-out UIT-HWDB writers | Different-content real line from the same writer |

Synthetic paragraphs:

- are constructed only from lines belonging to the same training writer;
- preserve `augmentation.source_line_ids`;
- never provide a synthetic style reference;
- never enter test manifests; and
- increase layout diversity, not writer diversity.

For all normalized records:

```text
canonical_writer_id == writer_id
```

The split is deterministic and writer-disjoint within CVL, IAM, and UIT-HWDB.

## Installation

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PyTorch-compatible CPU, Apple MPS, or CUDA runtime
- Node.js/npm only when regenerating `llms.txt` through the pre-commit hook

Install the locked environment:

```bash
uv sync
uv run python -c \
  "import torch, torchvision, vietparadiff; print(torch.__version__)"
```

All commands below assume the repository root as the working directory.

## Processed data

A processed dataset archive is available here:

**[Download VietParaDiff processed data](https://drive.google.com/file/d/1uoHTAEH5JmWVdcyB4B0AMkGtyDdAwdOh/view?usp=sharing)**

Extract it into `data/` while preserving paths. Always regenerate
`data/splits/` with the current code before training:

```bash
uv run python scripts/create_splits.py \
  --data-root data \
  --output-root data/splits \
  --test-fraction 0.2 \
  --seed 42 \
  --overwrite
```

The processed archive does not replace the original dataset licenses. For
paper-grade provenance, normalized datasets must also include build reports
created from the original raw inventories by the current committed builder
source.

## Building data from raw corpora

Expected layout:

```text
data/raw/
├── CVL/
│   ├── cvl-database-1-1/
│   └── cvl-database-cropped-1-1/
├── IAM/
│   └── archive/
│       ├── ascii/ascii/forms.txt
│       ├── forms/forms/
│       └── xml/
└── UIT_HWDB/
    ├── UIT_HWDB_word/
    ├── UIT_HWDB_line/
    └── UIT_HWDB_paragraph/
```

Build each real dataset:

```bash
uv run python scripts/build_dataset.py cvl
uv run python scripts/build_dataset.py iam
uv run python scripts/build_dataset.py uithwdb
```

Each builder writes atomically:

```text
data/<dataset>/
├── images/
├── manifest.jsonl
└── build_report.json
```

`build_report.json` binds the accepted records and expected rejections to:

- raw inventory SHA-256;
- resolved builder configuration;
- builder-source Git commit and dirty-patch provenance; and
- output manifest SHA-256.

Build reports created from uncommitted builder code intentionally become stale
after that code is committed. For an official run, commit the builder source
first, then rebuild from raw data.

### Normalized manifest schema

```json
{
  "id": "uithwdb_line_100_1",
  "image": "data/uithwdb/images/uithwdb_line_100_1.png",
  "text": "Sự tổ chức phối đủ bộ tọa.",
  "writer_id": "uithwdb_100",
  "level": "line",
  "width": 1024,
  "height": 91
}
```

`level` is one of `paragraph`, `line`, or `word`. Paragraph transcripts retain
their physical line breaks.

Images are 8-bit grayscale and aspect-preserving:

| Level | Maximum width | Upscaling |
| --- | ---: | --- |
| Paragraph | 1024 | Never |
| Line | 1024 | Never |
| Word | 512 | Never |

UIT-HWDB paragraphs without valid native line alignment are recorded as
expected rejections. No annotation from another dataset is substituted.

### Synthetic UIT-HWDB paragraphs

```bash
uv run python scripts/augment_paragraphs.py uithwdb \
  --data-root data \
  --output data/uithwdb_augmented \
  --samples 10000 \
  --seed 42 \
  --min-lines 2 \
  --max-lines 8 \
  --overwrite
```

### Dataset statistics

```bash
uv run python scripts/summarize_datasets.py \
  cvl iam uithwdb uithwdb_augmented
```

Current real normalized manifests:

| Dataset | Writers | Paragraphs | Lines | Words | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVL | 310 | 1,598 | 13,440 | 99,899 | 114,937 |
| IAM | 657 | 1,539 | 13,353 | 96,422 | 111,314 |
| UIT-HWDB | 255 | 1,144 | 7,229 | 110,488 | 118,861 |
| **Total** | **1,222** | **4,281** | **34,022** | **306,809** | **345,112** |

## Creating stage manifests

```bash
uv run python scripts/create_splits.py \
  --data-root data \
  --output-root data/splits \
  --test-fraction 0.2 \
  --seed 42 \
  --overwrite
```

Outputs:

```text
data/splits/
├── writers/
│   ├── train.json
│   └── test.json
├── autokl/
│   ├── train_paragraphs.jsonl
│   └── test_paragraphs.jsonl
├── htr/
│   ├── train_lines.jsonl
│   ├── train_words.jsonl
│   ├── test_lines.jsonl
│   └── test_words.jsonl
└── vietparadiff/
    ├── pretrain_targets.jsonl
    ├── pretrain_references.jsonl
    ├── finetune_targets_real.jsonl
    ├── finetune_targets_synthetic.jsonl
    ├── finetune_references.jsonl
    ├── test_pairs.jsonl
    └── rejected_targets.jsonl
```

The split builder:

- retains all eligible records rather than enforcing batch ratios;
- validates formatter limits before accepting generator targets;
- requires real, same-writer, different-content line references;
- excludes every synthetic source line from that target's reference pool;
- keeps rejected generator targets available to AutoKL; and
- writes deterministic fixed test pairs.

Generator formatter limits:

```text
maximum physical lines:        8
maximum graphemes per line:  128
maximum paragraph tokens:    768
```

## Data validation

Render all six training-data inspection grids:

```bash
uv run python scripts/inspect_training_data.py
```

```text
outputs/data_check/
├── autokl_batch.png
├── htr_lines.png
├── htr_words.png
├── style_references.png
├── vietparadiff_pairs.png
└── vietparadiff_canonical_slots.png
```

Run the full schema, image, split, duplicate, reference, CTC, and provenance
audit:

```bash
uv run python scripts/audit_dataset.py \
  --split-root data/splits \
  --image-root . \
  --output outputs/data_audit.json
```

The audit reports separate counts:

```text
hard_error_count
expected_rejection_count
warning_count
```

Paper preflight requires `hard_error_count == 0`. Expected rejections are
allowed only when their record ID, reason, source, and build provenance are
complete. A conflict, corrupt accepted image, writer leakage, invalid
reference, or label inconsistency cannot be downgraded to an expected
rejection.

## Training

### 1. Prepare local visual backbones

This is the only command allowed to download torchvision ImageNet weights:

```bash
uv run python scripts/prepare_visual_backbones.py
```

It writes hash-bound ConvNeXt-Tiny and ResNet-18 artifacts under
`checkpoints/vision/`. Training and evaluation fail if a required local
checkpoint is missing or its hash differs.

### 2. Train AutoKL

```bash
uv run python scripts/train_autokl.py \
  --config configs/autokl/train.yaml
```

Resume at an epoch boundary:

```bash
uv run python scripts/train_autokl.py \
  --config configs/autokl/train.yaml \
  --resume outputs/autokl/last.pt
```

AutoKL uses weighted reconstruction, fixed Laplacian edge loss, and a
normalized KL term with warm-up. Training samples the posterior; deterministic
evaluation uses its mode.

Compute training-set posterior-mode latent statistics:

```bash
uv run python scripts/compute_autokl_latent_stats.py \
  --checkpoint outputs/autokl/best.pt \
  --output outputs/autokl/latent_statistics.json
```

The generator uses:

```text
z_scaled = (z - latent_mean) * scaling_factor
```

No fixed Stable Diffusion scaling constant is used.

### 3. Train two independent HTR models

Guidance teacher:

```bash
uv run python scripts/train_htr.py \
  --config configs/htr/train.yaml
```

Paper-scoring evaluator:

```bash
uv run python scripts/train_htr.py \
  --config configs/htr/eval.yaml
```

The two runs use distinct seeds, augmentation policies, output directories,
vocabularies, and artifact contracts. Paper scoring rejects identical teacher
and evaluator checkpoint hashes.

HTR is line-level. Paragraph images are never collapsed directly across their
height. Generated paragraphs are routed into differentiable line crops only
for the HTR-guidance or content-scoring paths.

### 4. Pretrain VietParaDiff

```bash
uv run python scripts/train_generator.py \
  --config configs/vietparadiff/pretrain.yaml
```

This stage uses IAM and factorizer-supported CVL paragraphs with real
same-corpus references. Its objective is velocity MSE.

### 5. Fine-tune on Vietnamese handwriting

```bash
uv run python scripts/train_generator.py \
  --config configs/vietparadiff/finetune.yaml
```

The trainer exhausts real batches once per epoch and inserts one synthetic
batch after every three real batches. A fresh optimizer and scheduler are
created after strict-loading the pretrained generator weights.

### 6. HTR-guided fine-tuning

```bash
uv run python scripts/train_generator.py \
  --config configs/vietparadiff/htr_guided.yaml
```

The differentiable auxiliary path is:

```text
predicted velocity
→ predicted clean latent
→ latent denormalization
→ frozen AutoKL decoder
→ generated paragraph
→ differentiable generated-line routing
→ frozen four-head HTR
→ CTC loss
→ generator gradient
```

Canonical slots are used only to route the generated image. They are not sent
to the diffusion U-Net and do not supervise regions of real target images.

The pre-training structural probe checks tensor shapes, finite values, CTC
feasibility, slot/ink coverage, gradient flow to the generator, and absence of
AutoKL/HTR parameter gradients. It does not impose a quality or CER threshold
on an untrained generator.

For `htr_guided`, `best.pt` is the last completed epoch rather than the minimum
of a non-stationary warm-up objective. `last.pt` remains the strict resume
artifact.

## Checkpoints, logging, and resume

AutoKL, HTR, generator, and writer-metric trainers support:

- automatic CUDA/MPS/CPU selection;
- BF16 or FP16 autocast on supported CUDA devices;
- gradient clipping;
- deterministic data-worker seeding;
- TensorBoard;
- W&B, configured offline by default;
- atomic `last.pt` checkpoints; and
- strict epoch-boundary resume.

Typical logs:

```text
outputs/<stage>/
├── last.pt
├── best.pt
├── tensorboard/
└── wandb/
```

`last.pt` contains optimizer, scheduler, scaler, counters, RNG state, resolved
configuration, and artifact hashes. `best.pt` is model-only for strict
downstream loading. Resume is exact at an epoch boundary on the same runtime
class; mid-epoch and arbitrary cross-device resume are not claimed.

## Inference

Create a UTF-8 target file and preserve any hard newlines:

```bash
uv run python scripts/generate.py \
  --config configs/vietparadiff/generate.yaml \
  --text-file target.txt \
  --reference reference.png \
  --output outputs/generated.png
```

Sampling is deterministic DDIM-style velocity inversion. The main inference
path does not use classifier-free guidance, stochastic eta, reranking, OCR
correction, or post-generation line stitching.

The inference contract binds:

- generator checkpoint;
- model configuration;
- exact grapheme ID mapping;
- diffusion schedule;
- AutoKL checkpoint;
- latent statistics; and
- prediction type.

Changing any bound artifact causes loading to fail.

## Fixed-pair evaluation and scoring

Generate three deterministic samples for every held-out pair:

```bash
uv run python scripts/evaluate.py \
  --config configs/vietparadiff/evaluate.yaml
```

Resume only when the test manifest, generation contract, artifacts, records,
and existing PNG hashes still match:

```bash
uv run python scripts/evaluate.py \
  --config configs/vietparadiff/evaluate.yaml \
  --resume
```

Train the independent writer verifier:

```bash
uv run python scripts/train_writer_metric.py \
  --config configs/writer_metric/train.yaml
```

Score existing PNGs without running diffusion again:

```bash
uv run python scripts/score_evaluation.py \
  --config configs/vietparadiff/metrics.yaml
```

The scorer writes per-sample `metrics.jsonl` before
`metrics_summary.json`. Metrics include:

- paragraph CER/WER and exact-line accuracy;
- base, shape, and tone error rates;
- generated/reference writer-feature cosine similarity;
- writer retrieval, verification AUC, and EER;
- writer-feature distribution MMD;
- multi-seed feature diversity;
- foreground density and blank rate; and
- ink outside canonical slots and inter-line bleed.

Writer-feature MMD is intentionally not called standard Inception KID.
Completely blank generations remain content failures and contribute to
`blank_output_rate`; they are excluded only from style embeddings, with
explicit style-metric coverage.

## Ablations and paper experiments

Behavior flags can disable shape, tone, local style, high-frequency style, or
harmonization without changing state-dict keys. The experiment configuration
defines cumulative A0–A4 and Full variants across seeds 42, 43, and 44.

Dry-run the complete DAG:

```bash
uv run python scripts/run_experiments.py --dry-run --allow-dirty
```

Run or resume from a clean worktree:

```bash
uv run python scripts/run_experiments.py
uv run python scripts/run_experiments.py --resume
uv run python scripts/aggregate_results.py
```

The runner records the Git commit, dirty-patch hash when explicitly allowed,
resolved configurations, environment, GPU, commands, selected checkpoints,
seeds, timestamps, and artifact hashes. Aggregation reports mean, sample
standard deviation, and 95% Student-t confidence intervals over the three
training seeds.

### External baselines

External source is not vendored. The adapters require:

- One-DM commit `dde2205a70a2c70d1786503d198a795358c80ee4`; or
- Paragraph LDM commit `8a53e91b99c868614f7e615f41bc49c3f73c75b9`.

Run one strict adapter:

```bash
uv run python scripts/run_baseline.py \
  --config configs/baselines/one_dm/seed_42.yaml
```

The in-repository wrapper, external checkout commit, backend script, runtime
command, checkpoint, request schema, and output schema are hash-validated
before subprocess execution. Blank outputs are retained as white failures.

The six baseline YAML files contain deliberate all-zero checkpoint hashes.
Replace them with real seed-specific artifacts before starting a non-dry paper
DAG.

## Repository structure

```text
.
├── configs/
│   ├── autokl/
│   ├── htr/
│   ├── vietparadiff/
│   ├── writer_metric/
│   ├── baselines/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── cvl/
│   ├── iam/
│   ├── uithwdb/
│   ├── uithwdb_augmented/
│   └── splits/
├── scripts/                    # Thin CLI entry points
├── src/vietparadiff/
│   ├── artifacts.py           # Hash-bound artifact schemas
│   ├── diffusion.py           # Shared diffusion equations
│   ├── runtime.py             # Device, precision, AMP, and RNG
│   ├── data/                  # Builders, splits, datasets, audit
│   ├── models/                # AutoKL, grapheme, style, HTR, U-Net
│   ├── training/              # Stage-specific trainers
│   ├── inference/             # Deterministic paragraph sampling
│   ├── evaluation/            # Fixed-pair generation and scoring
│   └── baselines/             # External adapter contracts
├── tests/
├── tools/
└── uv.lock
```

Public imports use the installed `src`-layout package:

```python
from vietparadiff.models import HandwritingAutoKL, VietParaDiff
from vietparadiff.data import HTRVocabulary, VietParaDiffDataset
```

## Development

Run static import compilation and the full test suite:

```bash
uv lock --check
uv run python -m compileall -q src/vietparadiff scripts tests
uv run pytest -q
```

Install repository hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

The hooks:

1. stage a deterministic sample of processed data images; and
2. regenerate and stage `llms.txt` with Repomix `1.17.0`.

Inspect any CLI without changing state:

```bash
uv run python scripts/build_dataset.py --help
uv run python scripts/create_splits.py --help
uv run python scripts/train_autokl.py --help
uv run python scripts/train_htr.py --help
uv run python scripts/train_generator.py --help
uv run python scripts/generate.py --help
uv run python scripts/evaluate.py --help
uv run python scripts/score_evaluation.py --help
uv run python scripts/run_experiments.py --help
```

## Reproducibility boundary

The repository provides strict contracts and deterministic orchestration; it
does not by itself establish paper results.

Do not claim:

- paper metrics before training the frozen models;
- external-baseline superiority before replacing placeholder artifacts;
- complete reproducibility before all three-seed runs finish;
- A100/H100 throughput or memory performance without measurements on that
  hardware; or
- a clean paper dataset while the full audit reports any hard error.

The minimum gate before an official experiment is:

```text
committed source
→ normalized datasets with current build reports
→ deterministic stage manifests
→ full audit with hard_error_count == 0
→ local vision backbone contracts
→ frozen AutoKL, HTR teacher/evaluator, and writer verifier
→ three-seed generator and baseline runs
→ fixed-pair scoring and aggregation
```

Dataset licenses and terms remain those of CVL, IAM, and UIT-HWDB.
