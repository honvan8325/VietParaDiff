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

## Repository layout

```text
.
├── data/
│   ├── raw/                    # Original datasets; not tracked by Git
│   ├── cvl/
│   │   ├── images/             # Normalized grayscale PNG files
│   │   └── manifest.jsonl
│   ├── iam/
│   ├── uithwdb/
│   └── vnondb/
├── scripts/
│   ├── dataset_statistics.py   # Manifest-based dataset statistics
│   └── reprocess_data.py       # Dataset builder dispatcher
└── src/
    ├── data/
    │   ├── cvl.py
    │   ├── iam.py
    │   ├── image_utils.py       # Shared image normalization policy
    │   ├── uithwdb.py
    │   └── vnondb.py
    └── logger.py
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
uv run python scripts/reprocess_data.py cvl
uv run python scripts/reprocess_data.py iam
uv run python scripts/reprocess_data.py uithwdb
uv run python scripts/reprocess_data.py vnondb
```

> [!WARNING]
> A builder deletes its existing `data/<dataset>` directory before writing the
> replacement. Do not keep manually created or irreplaceable files inside a
> generated dataset directory.

Each builder:

1. Reads the original annotations and source images.
2. Validates the metadata needed for each sample.
3. Converts accepted images to width-limited 8-bit grayscale PNG files.
4. Creates dataset-prefixed sample and writer identifiers.
5. Records each final image's width and height.
6. Writes `data/<dataset>/manifest.jsonl`.

Individual builders are also available as Python functions:

```python
from src.data import build_iam_dataset

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
uv run python scripts/dataset_statistics.py
```

Select one or more datasets:

```bash
uv run python scripts/dataset_statistics.py iam cvl
```

Produce machine-readable JSON:

```bash
uv run python scripts/dataset_statistics.py iam --json
```

Read manifests from a different root:

```bash
uv run python scripts/dataset_statistics.py \
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

## Validation

Compile the source and scripts without rebuilding data:

```bash
uv run python -m compileall -q src scripts
```

Inspect command-line options:

```bash
uv run python scripts/reprocess_data.py --help
uv run python scripts/dataset_statistics.py --help
```
