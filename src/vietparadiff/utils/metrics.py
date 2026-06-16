"""Evaluation metrics for Vietnamese handwriting generation."""

from __future__ import annotations

from vietparadiff.data.graphemes import decompose_grapheme, split_graphemes


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, target: str) -> float:
    tgt = split_graphemes(target)
    return edit_distance(split_graphemes(pred), tgt) / max(1, len(tgt))


def decomposed_error_rates(pred: str, target: str) -> dict[str, float]:
    p = [decompose_grapheme(x) for x in split_graphemes(pred)]
    t = [decompose_grapheme(x) for x in split_graphemes(target)]
    n = max(1, len(t))
    m = min(len(p), len(t))
    base = sum(p[i].base != t[i].base for i in range(m)) + abs(len(p) - len(t))
    modifier = sum(p[i].modifier != t[i].modifier for i in range(m)) + abs(len(p) - len(t))
    tone = sum(p[i].tone != t[i].tone for i in range(m)) + abs(len(p) - len(t))
    return {"base_cer": base / n, "modifier_error": modifier / n, "tone_error": tone / n}
