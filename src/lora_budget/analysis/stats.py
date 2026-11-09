"""Statistics for the final BANKING77 analysis (proposal Sections 9.1 and 9.3).

- macro_f1: fast macro-F1 with numpy (same result as sklearn, average="macro").
- hierarchical_bootstrap: 95% interval for "method A minus method B", resampling
  BOTH the matched seeds and the shared test examples, 10,000 times.
- decide: the one-point practical limit we set in advance.
    lower end of the interval above -1  -> "within the 1-point limit"
    upper end below -1                  -> "worse than the 1-point limit"
    interval crosses -1                 -> "inconclusive"
"""

from __future__ import annotations

import numpy as np


def macro_f1(gold: np.ndarray, pred: np.ndarray, num_labels: int) -> float:
    """Macro-F1 in percent over all labels (a label with no gold and no predictions scores 0)."""
    cm = np.bincount(gold * num_labels + pred, minlength=num_labels * num_labels).reshape(num_labels, num_labels)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)                      # predicted + gold counts
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return 100.0 * f1.mean()


def hierarchical_bootstrap(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                           metric: str = "accuracy", num_labels: int = 77,
                           n_boot: int = 10_000, seed: int = 2026) -> dict:
    """Paired A-minus-B difference with a 95% hierarchical bootstrap interval.

    gold:   (n_examples,) gold labels of the shared evaluation set
    pred_a: (n_seeds, n_examples) predictions of method A, one row per seed
    pred_b: (n_seeds, n_examples) predictions of method B, rows matched by seed
    Each resample draws seeds with replacement, then examples with replacement,
    and takes the mean over the drawn seeds of (score A - score B).
    """
    gold, pred_a, pred_b = np.asarray(gold), np.asarray(pred_a), np.asarray(pred_b)
    assert pred_a.shape == pred_b.shape and pred_a.shape[1] == gold.shape[0]
    n_seeds, n = pred_a.shape

    def score(g, p):
        return 100.0 * (g == p).mean() if metric == "accuracy" else macro_f1(g, p, num_labels)

    point = float(np.mean([score(gold, pred_a[s]) - score(gold, pred_b[s]) for s in range(n_seeds)]))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        seeds = rng.integers(0, n_seeds, n_seeds)
        idx = rng.integers(0, n, n)
        g = gold[idx]
        diffs[i] = np.mean([score(g, pred_a[s, idx]) - score(g, pred_b[s, idx]) for s in seeds])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric, "diff": round(point, 3), "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3), "n_boot": n_boot, "n_seeds": n_seeds,
            "n_examples": n, "bootstrap_seed": seed}


def decide(ci_low: float, ci_high: float, margin: float = -1.0) -> str:
    if ci_low > margin:
        return "within the 1-point limit"
    if ci_high < margin:
        return "worse than the 1-point limit"
    return "inconclusive"
