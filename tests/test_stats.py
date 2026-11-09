"""Tests for the bootstrap, macro-F1 and the one-point decision rule."""

import numpy as np
from sklearn.metrics import f1_score

from lora_budget.analysis.stats import decide, hierarchical_bootstrap, macro_f1


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    g, p = rng.integers(0, 77, 3000), rng.integers(0, 77, 3000)
    want = 100 * f1_score(g, p, average="macro", labels=list(range(77)), zero_division=0)
    assert abs(macro_f1(g, p, 77) - want) < 1e-9


def test_identical_methods_give_zero_difference():
    rng = np.random.default_rng(1)
    gold = rng.integers(0, 77, 500)
    preds = np.where(rng.random((3, 500)) < 0.9, gold, rng.integers(0, 77, (3, 500)))
    b = hierarchical_bootstrap(gold, preds, preds, n_boot=200)
    assert b["diff"] == 0 and b["ci_low"] == 0 and b["ci_high"] == 0


def test_known_gap_is_inside_the_interval():
    rng = np.random.default_rng(2)
    n = 3000
    gold = rng.integers(0, 77, n)
    better = np.where(rng.random((3, n)) < 0.92, gold, (gold + 1) % 77)
    worse = better.copy()
    flip = rng.random((3, n)) < 0.05          # B loses about 0.05 * 0.92 = 4.6 points
    worse[flip] = (gold[np.where(flip)[1]] + 1) % 77
    b = hierarchical_bootstrap(gold, worse, better, n_boot=500)
    assert b["ci_low"] < b["diff"] < b["ci_high"] < 0
    assert -6 < b["diff"] < -3


def test_decision_rule():
    assert decide(-0.8, 0.3) == "within the 1-point limit"
    assert decide(-1.6, -0.4) == "inconclusive"
    assert decide(-2.5, -1.2) == "worse than the 1-point limit"
