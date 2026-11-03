"""Loader validation test for BANKING77 (proposal Section 7.3).

Checks schemas, row counts, empty fields, label count, duplicate hashes,
and that the splits do not overlap. Needs data/raw/ and manifests/banking77/,
which `python scripts/make_banking77_manifest.py` creates.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from lora_budget.data.banking77 import load_split
from lora_budget.data.hashing import text_hash

MAN = Path("manifests/banking77")
RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(
    not (MAN / "meta.json").exists() or not (RAW / "banking77_train.csv").exists(),
    reason="run scripts/make_banking77_manifest.py first",
)


@pytest.fixture(scope="module")
def splits():
    return {s: load_split(s, RAW, MAN) for s in ("train", "tune", "test")}


@pytest.fixture(scope="module")
def meta():
    return json.loads((MAN / "meta.json").read_text())


def test_schema_and_no_empty_fields(splits):
    for name, df in splits.items():
        assert list(df.columns) == ["id", "text", "label"], name
        assert df.notna().all().all(), f"{name} has empty fields"
        assert (df["text"].str.strip().str.len() > 0).all(), f"{name} has blank text"


def test_row_counts_match_meta(splits, meta):
    for name, df in splits.items():
        assert len(df) == meta["final_counts"][name], name
    assert meta["raw_counts"] == {"train": 10003, "test": 3080}
    removed = sum(meta["duplicates_removed"].values())
    assert len(splits["train"]) + len(splits["tune"]) + removed == 10003


def test_77_labels_in_every_split(splits):
    for name, df in splits.items():
        assert df["label"].nunique() == 77, name
        assert df["label"].between(0, 76).all(), name


def test_ids_unique_and_splits_isolated(splits):
    ids = {n: set(df["id"]) for n, df in splits.items()}
    for n, df in splits.items():
        assert df["id"].is_unique, n
    assert not ids["train"] & ids["tune"]
    assert not ids["train"] & ids["test"]
    assert not ids["tune"] & ids["test"]


def test_no_exact_duplicates_across_splits(splits):
    h = {n: set(df["text"].map(text_hash)) for n, df in splits.items()}
    assert not h["train"] & h["test"], "train text found in test"
    assert not h["tune"] & h["test"], "tuning text found in test"
    assert not h["train"] & h["tune"], "train text found in tuning set"


def test_tuning_split_is_stratified(splits):
    tr = splits["train"]["label"].value_counts(normalize=True)
    tu = splits["tune"]["label"].value_counts(normalize=True)
    diff = (tr - tu).abs().max()
    assert diff < 0.005, f"label proportions differ by {diff:.4f}"


def test_manifest_checksums_match(meta):
    from lora_budget.data.hashing import sha256_file
    for n in ("train", "tune", "test"):
        assert sha256_file(MAN / f"{n}.csv") == meta["manifest_sha256"][n], n
