# VietParaDiff

VietParaDiff provides a common data-building pipeline for four handwriting
corpora:

- CVL
- IAM
- UIT-HWDB
- VNOnDB

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
│   ├── htr/train.yaml
│   └── vietparadiff/
│       ├── pretrain.yaml
│       └── generate.yaml
├── data/                       # Raw, processed, and split manifests
├── scripts/
│   ├── build_dataset.py
│   ├── summarize_datasets.py
│   ├── inspect_training_data.py
│   ├── train_autokl.py
│   ├── train_htr.py
│   ├── train_generator.py
│   └── generate.py
└── src/
    └── vietparadiff/
        ├── artifacts.py        # Checkpoint-bound artifact contracts
        ├── diffusion.py        # Shared velocity-diffusion equations
        ├── runtime.py          # Device, precision, AMP, and RNG setup
        ├── data/               # Builders and training data pipeline
        ├── models/             # AutoKL, HTR, style, grapheme, generator
        ├── training/           # Stage-specific trainers
        └── inference/          # Sampling and generation pipeline
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
├── UIT_HWDB/
│   ├── UIT_HWDB_word/
│   ├── UIT_HWDB_line/
│   └── UIT_HWDB_paragraph/
└── VNOnDB/
    └── Data_processed/
        ├── InkData_word_processed/
        ├── InkData_line_processed/
        └── InkData_paragraph_processed/
```

UIT-HWDB level directories must contain the original `train_data` and
`test_data` writer directories. Each writer directory is expected to contain
a `label.json` file and its referenced images.

VNOnDB expects every source PNG to have a same-stem `.txt` transcript file.

## Building a dataset

Run the dispatcher with exactly one dataset name:

```bash
uv run python scripts/build_dataset.py cvl
uv run python scripts/build_dataset.py iam
uv run python scripts/build_dataset.py uithwdb
uv run python scripts/build_dataset.py vnondb
```

> [!WARNING]
> A builder deletes its existing `data/<dataset>` directory before writing the
> replacement. Do not keep manually created or irreplaceable files inside a
> generated dataset directory.

Each builder:

1. Reads the original annotations and source images.
2. Reconstructs paragraph transcripts from ordered physical-line annotations,
   preserving each real line break as `\n`.
3. Validates the metadata needed for each sample.
4. Converts accepted images to width-limited 8-bit grayscale PNG files.
5. Creates dataset-prefixed sample and writer identifiers.
6. Records each final image's width and height.
7. Writes `data/<dataset>/manifest.jsonl`.

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
| VNOnDB | 224 | 1,144 | 7,296 | 110,746 | 119,186 |
| **Total** | **1,446** | **5,425** | **41,318** | **417,555** | **464,298** |

## Creating training splits

Build all stage-specific manifests after the five normalized manifests
(`cvl`, `iam`, `uithwdb`, `vnondb`, and `uithwdb_augmented`) are available:

```bash
uv run python scripts/create_splits.py
```

The default split is deterministic, writer-disjoint, stratified by corpus
family, and reserves 20% of canonical writers for test with seed 42. Change
these values explicitly when needed:

```bash
uv run python scripts/create_splits.py \
  --test-fraction 0.2 \
  --seed 42 \
  --overwrite
```

UIT-HWDB and VNOnDB writers are canonicalized as a single Vietnamese writer
family using matching paragraph transcripts and handwriting-image signatures.
Consequently, the same physical writer cannot appear under one dataset in
train and the other dataset in test. Synthetic UIT-HWDB paragraphs inherit
their source writer's split and are train-only.

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
physical line IDs using text-guided spatial cross-attention with a weak
monotonic vertical prior; it does not use line boxes, line detection, or
pseudo masks.

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

### VNOnDB

VNOnDB writer identifiers are derived from the first two underscore-separated
fields of each source filename. Image/transcript pairs that are missing,
empty, undecodable, or invalid are logged and skipped.

## Training and generation

Every stage has a distinct configuration path:

```bash
uv run python scripts/train_autokl.py \
  --config configs/autokl/train.yaml

uv run python scripts/train_htr.py \
  --config configs/htr/train.yaml

uv run python scripts/train_generator.py \
  --config configs/vietparadiff/pretrain.yaml
```

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
```
