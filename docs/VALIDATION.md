# Validation checklist

Run these before experiments:

```bash
python -m compileall -q src tests
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
uv run vpd-smoke
```

Recommended experiment metrics:

- Grapheme CER and WER.
- Base-character CER.
- Structural modifier error rate.
- Tone-mark error rate.
- Diacritic placement error.
- Baseline deviation.
- Line-spacing and word-gap distribution error.
- Overlap/overflow rate.
- Writer retrieval mAP / HWD when a writer embedding model is available.
- Patch-level FID/KID for paragraph realism.
