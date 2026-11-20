"""Answer scoring and statistics shared by all angles."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


# ------------------------------------------------------------- answer scoring
def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation/articles/extra space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        common = Counter(p_toks) & Counter(g_toks)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p_toks), n / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def first_line_answer(generation: str) -> str:
    """Readers are told to answer on one line; take the first non-empty line and
    strip a leading 'Answer:' if the model echoes the label."""
    for line in generation.strip().splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^(answer\s*[:\-]\s*)", "", line, flags=re.I).strip()
    return ""


# ------------------------------------------------------------- set metrics
def set_recall(kept: list[int], gold: list[int]) -> float:
    if not gold:
        return 1.0
    return len(set(kept) & set(gold)) / len(set(gold))


def set_precision(kept: list[int], gold: list[int]) -> float:
    if not kept:
        return 0.0
    return len(set(kept) & set(gold)) / len(set(kept))


# ------------------------------------------------------------- statistics
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> dict:
    """Mean of (a - b) with a 95 % percentile bootstrap CI over paired items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(n), "significant": bool(lo > 0 or hi < 0)}


def mean_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(x))}
