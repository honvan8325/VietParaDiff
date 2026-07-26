# VietParaDiff

VietParaDiff provides a common data-building pipeline for three handwriting
corpora:

- CVL
- IAM
- UIT-HWDB

Vietnamese domain adaptation and held-out-writer evaluation use UIT-HWDB
only. Synthetic Vietnamese paragraphs are assembled from UIT-HWDB training
lines and are never used for evaluation.

The builders convert the original dataset layouts into one shared format:
width-limited grayscale PNG images plus a UTF-8 JSON Lines manifest. Each
manifest can contain paragraph-, line-, and word-level samples.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- The original datasets, downloaded separately and used according to their
  respective licenses

Install the project dependencies:

```bash
uv sync
```

All commands below assume the current working directory is the repository
root. Dataset paths are intentionally repository-relative.

## Processed data

A ready-to-use copy of the processed datasets is available here:

**[Download VietParaDiff processed data](https://drive.google.com/file/d/1uoHTAEH5JmWVdcyB4B0AMkGtyDdAwdOh/view?usp=sharing)**

Use this download to skip rebuilding the normalized images and manifests from
the original dataset layouts. Extract the contents into the repository's
`data/` directory while preserving the included directory structure. The
original dataset licenses and usage terms still apply.

## Repository layout

```text
.
├── configs/
│   ├── autokl/train.yaml
│   ├── htr/
│   │   ├── train.yaml
│   │   └── eval.yaml
│   ├── baselines/              # Three seeds for each external baseline
│   ├── writer_metric/train.yaml
│   ├── experiments/paper.yaml
│   └── vietparadiff/
│       ├── pretrain.yaml
│       ├── finetune.yaml
│       ├── htr_guided.yaml
│       ├── generate.yaml
│       ├── evaluate.yaml
│       └── metrics.yaml
├── data/                       # Raw, processed, and split manifests
├── scripts/
│   ├── build_dataset.py
│   ├── summarize_datasets.py
│   ├── inspect_training_data.py
│   ├── train_autokl.py
│   ├── train_htr.py
│   ├── train_generator.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── score_evaluation.py
│   ├── train_writer_metric.py
│   ├── audit_dataset.py
│   ├── run_baseline.py
│   ├── run_experiments.py
│   └── aggregate_results.py
└── src/
    └── vietparadiff/
        ├── artifacts.py        # Checkpoint-bound artifact contracts
        ├── diffusion.py        # Shared velocity-diffusion equations
        ├── runtime.py          # Device, precision, AMP, and RNG setup
        ├── data/               # Builders and training data pipeline
        ├── models/             # AutoKL, HTR, style, grapheme, generator
        ├── training/           # Stage-specific trainers
        ├── inference/          # Sampling and generation pipeline
        ├── evaluation/         # Fixed-pair generation and scoring
        └── baselines/          # Strict external-checkout adapters
```

## Expected raw-data layout

The builders expect the following source directories:

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
│   ├── UIT_HWDB_word/
│   ├── UIT_HWDB_line/
│   └── UIT_HWDB_paragraph/
```

UIT-HWDB level directories must contain the original `train_data` and
`test_data` writer directories. Each writer directory is expected to contain
a `label.json` file and its referenced images.

## Building a dataset

Run the dispatcher with exactly one dataset name:

```bash
uv run python scripts/build_dataset.py cvl
uv run python scripts/build_dataset.py iam
uv run python scripts/build_dataset.py uithwdb
```

The dispatcher builds in a same-filesystem staging directory and replaces
`data/<dataset>` only after the manifest and build report are complete. Do not
call a builder function directly when an atomic replacement is required.

Each builder:

1. Reads the original annotations and source images.
2. Reconstructs paragraph transcripts from ordered physical-line annotations,
   preserving each real line break as `\n`.
3. Validates the metadata needed for each sample.
4. Converts accepted images to width-limited 8-bit grayscale PNG files.
5. Creates dataset-prefixed sample and writer identifiers.
6. Records each final image's width and height.
7. Writes `data/<dataset>/manifest.jsonl` and a hash-bound
   `build_report.json`.

Individual builders are also available as Python functions:

```python
from vietparadiff.data import build_iam_dataset

build_iam_dataset()
```

## Manifest format

Every non-blank manifest line is an independent JSON object:

```json
{
  "id": "iam_a01_000u_00",
  "image": "data/iam/images/iam_a01_000u_00.png",
  "text": "A MOVE to stop Mr. Gaitskell from",
  "writer_id": "iam_000",
  "level": "line",
  "width": 1024,
  "height": 143
}
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Globally namespaced sample identifier. |
| `image` | string | Repository-relative path to the normalized PNG image. |
| `text` | string | Ground-truth transcription. Paragraphs preserve line breaks. |
| `writer_id` | string | Dataset-prefixed writer identifier. |
| `level` | string | One of `paragraph`, `line`, or `word`. |
| `width` | integer | Final normalized image width in pixels. |
| `height` | integer | Final normalized image height in pixels. |

Dataset prefixes prevent identifier collisions when manifests are combined.

### Image sizing policy

Resizing is based only on image width:

| Sample level | Maximum width | Height |
| --- | ---: | --- |
| `paragraph` | 1024 px | Unconstrained |
| `line` | 1024 px | Unconstrained |
| `word` | 512 px | Unconstrained |

An image is downscaled only when its width exceeds the applicable limit.
Smaller images are never upscaled. Every resize preserves the original aspect
ratio and uses Pillow's Lanczos resampling filter.

## Dataset statistics

Calculate statistics for every directory under `data/` that contains a
`manifest.jsonl` file:

```bash
uv run python scripts/summarize_datasets.py
```

Select one or more datasets:

```bash
uv run python scripts/summarize_datasets.py iam cvl
```

Produce machine-readable JSON:

```bash
uv run python scripts/summarize_datasets.py iam --json
```

Read manifests from a different root:

```bash
uv run python scripts/summarize_datasets.py \
  --data-root /path/to/data \
  iam
```

The statistics script streams each manifest instead of loading it entirely
into memory. It reports unique writers, paragraph samples, line samples, word
samples, and the total number of records. The final row is the aggregate of
all selected datasets.

Current generated manifests contain:

| Dataset | Writers | Paragraphs | Lines | Words | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVL | 310 | 1,598 | 13,440 | 99,899 | 114,937 |
| IAM | 657 | 1,539 | 13,353 | 96,422 | 111,314 |
| UIT-HWDB | 255 | 1,144 | 7,229 | 110,488 | 118,861 |
| **Total** | **1,222** | **4,281** | **34,022** | **306,809** | **345,112** |

## Creating training splits

Build all stage-specific manifests after the three normalized real manifests
(`cvl`, `iam`, and `uithwdb`) plus `uithwdb_augmented` are available:

```bash
uv run python scripts/create_splits.py
```

The default split is deterministic, writer-disjoint within each corpus, and
reserves 20% of writers for test with seed 42. Change
these values explicitly when needed:

```bash
uv run python scripts/create_splits.py \
  --test-fraction 0.2 \
  --seed 42 \
  --overwrite
```

CVL, IAM, and UIT-HWDB writers are split independently within their corpus.
For every record, `canonical_writer_id` equals the dataset-prefixed
`writer_id`. Synthetic UIT-HWDB paragraphs inherit their source writer's
training split and never enter test manifests.

The command writes:

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

AutoKL receives only real paragraphs. HTR receives real Vietnamese line and
word samples. Generator pretraining uses IAM and factorizer-supported CVL
paragraphs with same-corpus line references. Finetuning uses real Vietnamese
paragraphs plus train-only stitched paragraphs, preserving each synthetic
record's `augmentation.source_line_ids`. Fixed test pairs contain a real
Vietnamese paragraph and a different-content real line from the same unseen
canonical writer. Targets without any valid same-writer reference are omitted
from the generator manifests and recorded in `rejected_targets.jsonl` with
their stage, reason code, and concrete reason. They remain available to
AutoKL because that stage does not require text/style conditioning.

Generator targets use `formatter_mode="physical_lines"`. Their annotated
newlines are preserved rather than wrapped a second time, and every accepted
target is validated against the model contract:

```text
maximum physical lines:          8
maximum graphemes per line:    128
maximum paragraph tokens:      768
```

Every accepted target is also guaranteed to have at least one real line
reference from the same canonical writer whose content is outside the target.
For stitched targets, all `augmentation.source_line_ids` are additionally
excluded from the eligible reference set.

The manifests retain all eligible records. The HTR line/word ratio and the
generator real/synthetic ratio belong to the later training sampler; the split
builder does not downsample records to enforce those ratios.

## Training data layer

`src/vietparadiff/data/pipeline.py` converts split records into the exact
tensors consumed by the three training stages:

- `ParagraphImageProcessor` produces aspect-preserved grayscale paragraph
  canvases `[1, H_bucket, 1024]`.
- `HTRImageProcessor` produces one-line images `[1, 64, W]` and their valid
  widths.
- `ReferenceImageProcessor` produces style references
  `[1, 256, W_pad]` with boolean valid masks, where `W_pad <= 1536` and is a
  multiple of 32.
- `AutoKLDataset`, `HTRDataset`, and `VietParaDiffDataset` load the respective
  stage manifests without performing training.
- `HeightBucketBatchSampler` prevents paragraph heights from being mixed in a
  batch. `WidthBucketBatchSampler` reduces HTR right-padding.
- `collate_autokl`, `collate_htr`, and `VietParaDiffCollator` construct padded
  batches and factorized text tensors.

For generator data, the formatter alone selects the output height used at
both train and inference. The target image is isotropically fit into that
exact canvas; image geometry never increases the formatter bucket.
`VietParaDiffDataset.set_epoch(epoch)` and the dataset seed select each train
reference through a stable hash of `seed:epoch:target_id`, so worker count,
resume, and debugging reads do not change the selected reference.

Build the four HTR CTC vocabularies only from the training manifests:

```python
from pathlib import Path
from vietparadiff.data import HTRVocabulary

vocabulary = HTRVocabulary.build_from_manifests(
    (
        Path("data/splits/htr/train_lines.jsonl"),
        Path("data/splits/htr/train_words.jsonl"),
    )
)
vocabulary.save(Path("outputs/htr_vocabulary.json"))
```

Test transcripts may map unseen symbols to `<unk>`, but never expand these
training vocabularies.

Render processed real samples for manual inspection before training:

```bash
uv run python scripts/inspect_training_data.py
```

The command validates real collated batches and writes:

```text
outputs/data_check/
├── autokl_batch.png
├── htr_lines.png
├── htr_words.png
├── style_references.png
├── vietparadiff_pairs.png
└── vietparadiff_canonical_slots.png
```

Canonical slots describe the layout requested for generated images and are
reserved for the later generated-image HTR auxiliary path. They are not
treated as regions of real target images and are never passed to the
diffusion U-Net. Inter-line alignment inside the U-Net is learned from
physical line IDs using text-guided spatial cross-attention without a
vertical spatial prior; it does not use line boxes, line detection, or pseudo
masks.

The formatter is a neutral deterministic layout component. It preserves hard
newlines, wraps by fixed priors, and chooses a supported height bucket. The
current implementation does not claim learned or reference-calibrated
character width, word gap, or line gap.

## Dataset-specific behavior

### CVL

CVL PAGE XML files may declare UTF-16 even when their actual bytes use another
encoding. The builder decodes each payload defensively and normalizes the XML
declaration before parsing. Existing line and word TIFF crops are converted
directly, while the cropped page image is used for paragraph-level output.

### IAM

IAM paragraph, line, and word bounding boxes are computed from the union of
their XML `cmp` components. Configurable padding is added and clamped to the
form boundary. Lines with unreliable IAM word segmentation are retained as
line and paragraph content, but their word crops are excluded.

### UIT-HWDB

UIT-HWDB provides separate paragraph, line, and word exports. The builder
normalizes all three layouts through the same `label.json`-based path and
orders numeric filenames deterministically.

## Training and generation

Prepare both ImageNet backbones explicitly before training. This is the only
command allowed to use torchvision's download mechanism:

```bash
uv run python scripts/prepare_visual_backbones.py
```

It writes ConvNeXt-Tiny and ResNet-18 state dictionaries under
`checkpoints/vision/` plus `manifest.json` containing their SHA-256 hashes.
Generator pretraining and writer-verifier training fail if a local checkpoint
is missing or does not match that contract; they never download implicitly.

Every stage has a distinct configuration path:

```bash
uv run python scripts/train_autokl.py \
  --config configs/autokl/train.yaml

uv run python scripts/train_htr.py \
  --config configs/htr/train.yaml

uv run python scripts/train_htr.py \
  --config configs/htr/eval.yaml

uv run python scripts/train_generator.py \
  --config configs/vietparadiff/pretrain.yaml

uv run python scripts/train_generator.py \
  --config configs/vietparadiff/finetune.yaml

uv run python scripts/train_generator.py \
  --config configs/vietparadiff/htr_guided.yaml
```

Fine-tune stages strict-load only generator model weights from the previous
stage, then create a new optimizer and scheduler. Vietnamese fine-tuning
exhausts real batches once per epoch and inserts one synthetic batch after
every three real batches. HTR guidance uses canonical slots only to crop
generated images for the frozen line-level teacher; slots never enter the
diffusion U-Net. The frozen teacher is bound to `best.pt`,
`model_config.json`, and the exact HTR vocabulary through
`outputs/htr/inference_contract.json`.

The first HTR run is the guidance teacher. The second is the paper-scoring
evaluator, trained separately with a different seed, augmentation policy,
vocabulary artifact path, output directory, and W&B project. Scoring rejects
the artifacts if their checkpoint SHA-256 values are equal.

Each HTR run also writes `training_contract.json`, binding its selected
checkpoint to the canonical resolved-config SHA-256, seed, augmentation,
training line/word manifest hashes, and the
`minimum_train_loss` selection policy. Paper preflight validates both
contracts and requires the guidance/evaluation seeds and augmentations to
differ. Test manifests are never part of checkpoint selection.

For the `htr_guided` generator stage, `best.pt` deliberately means the model
from the final completed epoch. It is overwritten after every guided epoch
instead of comparing non-stationary objectives while the HTR weight is
warming up. `last.pt` remains the strict resume artifact.

Compute AutoKL latent normalization statistics before generator training:

```bash
uv run python scripts/compute_autokl_latent_stats.py --help
```

Generate a paragraph with the checkpoint-bound inference contract:

```bash
uv run python scripts/generate.py \
  --config configs/vietparadiff/generate.yaml \
  --text-file target.txt \
  --reference reference.png
```

Generate all fixed held-out pairs with three stable seeds per pair:

```bash
uv run python scripts/evaluate.py \
  --config configs/vietparadiff/evaluate.yaml
```

Use `--resume` only with the same manifest, checkpoints, vocabulary, latent
statistics, inference settings, and existing PNG hashes.

## Paper scoring

Train the independent grayscale ResNet-18 writer verifier on real training
lines and paragraphs:

```bash
uv run python scripts/train_writer_metric.py \
  --config configs/writer_metric/train.yaml
```

Resume at an epoch boundary with:

```bash
uv run python scripts/train_writer_metric.py \
  --config configs/writer_metric/train.yaml \
  --resume outputs/writer_metric/last.pt
```

The verifier uses 256-dimensional L2-normalized embeddings and ArcFace
(`scale=30`, `margin=0.5`). Its train/validation writers are deterministically
split 90/10 and disjoint. This is an explicitly allowed internal validation
protocol for the auxiliary evaluator only; held-out paper test writers remain
untouched. The selection protocol is serialized in the artifact contract.
`best.pt` is selected by validation EER and is bound to the exact model
config, writer mapping, input manifests, and local ResNet-18 artifact.

Score already generated fixed-pair PNGs without running diffusion again:

```bash
uv run python scripts/score_evaluation.py \
  --config configs/vietparadiff/metrics.yaml
```

The scorer writes per-sample `metrics.jsonl` before aggregating
`metrics_summary.json`. It reports HTR content errors, independent writer
verification, writer-feature style-distribution MMD, multi-seed diversity,
foreground density, blank outputs,
ink outside canonical slots, and inter-line bleed. Every record is bound to
the PNG and all model/vocabulary/manifest hashes.

Style embeddings, retrieval, verification, and style-distribution MMD
intentionally exclude completely blank generated images. Their coverage is
reported as `style_metric_coverage`; content errors and `blank_output_rate`
still include every sample. No zero embedding or replacement image is used.
This MMD is not called standard Inception KID because it is computed in the
independent writer-feature space.

The reference retrieval gallery always contains every unique reference in
`test_pairs.jsonl`, and the real MMD distribution always contains every
unique target paragraph. These two real sets are encoded before inspecting
generated outputs. Only the generated side is filtered for blank images, so a
difficult or blank generation cannot remove its writer or target from the
evaluation population.

## Ablations, baselines, and multi-seed experiments

Model behavior flags disable shape, tone, local style, high-frequency style,
or harmonization without deleting modules or changing any state-dict key.
`configs/experiments/paper.yaml` defines the cumulative A0, A1, A2, A3, A4,
and Full variants for seeds 42, 43, and 44:

```bash
uv run python scripts/run_experiments.py --dry-run --allow-dirty
uv run python scripts/run_experiments.py --resume
uv run python scripts/aggregate_results.py
```

Paper runs require a clean Git worktree by default. `--allow-dirty` records a
SHA-256 of the binary patch and untracked files in every run manifest.
Aggregation produces JSON, CSV, and Markdown with the mean, sample standard
deviation, and 95% Student-t confidence interval across the three training
seeds. MMD subset variation remains separate from training-seed variation.

Before the first subprocess, a non-dry paper run performs a complete
preflight: clean full-data audit, local vision backbones, AutoKL and latent
statistics, distinct guidance/evaluation HTRs, writer verifier, every
resolved internal config, and every external checkout/command/checkpoint.
Resume compares the current resolved-config SHA-256 and canonical command
against the completed-stage manifest before reusing an artifact.

External One-DM and Paragraph LDM code is not vendored. Create a baseline YAML
using the strict schema accepted by `scripts/run_baseline.py`, point it at a
separate checkout, and run:

```bash
uv run python scripts/run_baseline.py --config /path/to/baseline.yaml
```

The adapters require the exact pinned commits
`dde2205a70a2c70d1786503d198a795358c80ee4` for One-DM and
`8a53e91b99c868614f7e615f41bc49c3f73c75b9` for Paragraph LDM, verify the
external checkpoint and adapter-script hashes, and enforce a common JSONL
request/output schema. Blank baseline outputs remain white samples and are
scored as failures rather than aborting evaluation.
One-DM outputs are explicitly reported as a deterministically stitched
word-level baseline. Paragraph LDM output is aspect-preserved and white-padded
to the target bucket.

For the paper DAG, provide seed-specific external configs at
`configs/baselines/one_dm/seed_{42,43,44}.yaml` and
`configs/baselines/paragraph_ldm/seed_{42,43,44}.yaml`. Each must reference
the checkpoint retrained for that training seed. The common runner then runs
generation and scoring for all six external artifacts, records the same Git
and artifact provenance, and includes `one_dm` and `paragraph_ldm` in the
three-seed aggregate. Missing external configs, checkout commits, environment
commands, or checkpoint hashes fail explicitly.

Six template configs are included at those exact paths. Their all-zero
checkpoint hashes are deliberate invalid placeholders. The provenance bridge
is the hash-bound in-repository
`tools/baselines/run_pinned_adapter.py`; each `backend_script` must resolve to
a file tracked by the exact pinned external commit. Replace checkpoint paths,
hashes, and backend paths with real trained artifacts before starting the
paper DAG. Preflight rejects incomplete templates before any subprocess.

Dependency resolution is pinned by the tracked `uv.lock`; this work does not
modify or regenerate it.

No paper metric, baseline superiority, or full reproducibility claim is valid
until the frozen models are trained and all three-seed runs complete.

## Full-data audit

Audit every split record and referenced image before paper training:

```bash
uv run python scripts/audit_dataset.py \
  --output outputs/data_audit.json
```

The audit checks schema, IDs, paths, dimensions, image decode, duplicate
content, writer/image leakage, formatter acceptance, HTR CTC feasibility,
target/reference eligibility, and excluded source-line rules. It exits
nonzero when `hard_error_count` is nonzero and retains concrete issue records
in the JSON report. Expected build rejections and warnings are reported
separately and do not hide blocking failures. Audit schema v3 also stores
every manifest SHA-256, an image
inventory SHA-256, and a combined dataset snapshot SHA-256. Paper preflight
recomputes this snapshot without decoding images and rejects a stale report
when any split manifest, referenced image, normalized source manifest, build
report, writer split, or augmented-source manifest has changed. Audit also
recomputes each raw inventory and verifies builder
config, Git commit/dirty-patch provenance, issue-list counts, accepted record
count, and output-manifest hash.

Before an HTR-guided training run begins, the trainer executes one structural
probe through generator velocity, differentiable AutoKL decode, canonical
generated-line routing, and frozen HTR CTC. The probe checks finite values,
CTC feasibility, slot/ink coverage, and gradient ownership only. It does not
apply an untrained-model CER or image-quality threshold.

## Validation

Compile the source and scripts without rebuilding data:

```bash
uv run python -m compileall -q src/vietparadiff scripts
uv run pytest
```

Inspect command-line options:

```bash
uv run python scripts/build_dataset.py --help
uv run python scripts/summarize_datasets.py --help
uv run python scripts/train_autokl.py --help
uv run python scripts/train_htr.py --help
uv run python scripts/train_generator.py --help
uv run python scripts/generate.py --help
uv run python scripts/evaluate.py --help
uv run python scripts/prepare_visual_backbones.py --help
uv run python scripts/audit_dataset.py --help
uv run python scripts/train_writer_metric.py --help
uv run python scripts/score_evaluation.py --help
uv run python scripts/run_baseline.py --help
uv run python scripts/run_experiments.py --help
uv run python scripts/aggregate_results.py --help
```
