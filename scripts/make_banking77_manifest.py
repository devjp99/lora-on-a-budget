"""Build the fixed BANKING77 split manifests (proposal Section 7.1).

What this does:
  1. Downloads the official BANKING77 train/test CSVs from the authors' repo
     (PolyAI-LDN/task-specific-datasets, CC BY 4.0) into data/raw/ (not committed).
  2. Records a SHA-256 checksum of each downloaded file.
  3. Makes a stratified train/tuning split of the released training set
     (8,002 train / 2,001 tuning) with the fixed subset seed 2026.
  4. Keeps the official 3,080-example test set as the final evaluation set.
  5. Checks for exact duplicates across splits using hashes of normalized text.
     If a training or tuning example duplicates a test example (or a training
     example duplicates a tuning example), it is removed from the training side
     and the removal is recorded. Test membership never changes. Repeats inside
     the training split are counted and reported but kept.
  6. Writes manifests/banking77/{train,tune,test}.csv, label_map.json, meta.json.

Run from the repo root:
    python scripts/make_banking77_manifest.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lora_budget.data.hashing import sha256_file, text_hash

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
FILES = {"train": "train.csv", "test": "test.csv"}
SUBSET_SEED = 2026
TUNE_SIZE = 2001          # 10,003 released train -> 8,002 train + 2,001 tuning
EXPECTED = {"train_raw": 10003, "test": 3080, "labels": 77}


def download(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, name in FILES.items():
        dest = raw_dir / f"banking77_{name}"
        if not dest.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        paths[split] = dest
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="manifests/banking77")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = download(raw_dir)

    train_raw = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    for name, df in [("train", train_raw), ("test", test)]:
        assert list(df.columns) == ["text", "category"], f"{name}: unexpected columns {list(df.columns)}"

    # Stable IDs from the original file order (the file checksum pins that order).
    train_raw = train_raw.reset_index().rename(columns={"index": "row"})
    test = test.reset_index().rename(columns={"index": "row"})
    train_raw["id"] = "train-" + train_raw["row"].astype(str).str.zfill(5)
    test["id"] = "test-" + test["row"].astype(str).str.zfill(5)

    # Deterministic label dictionary: sorted category names -> 0..76.
    labels = sorted(set(train_raw["category"]) | set(test["category"]))
    label_map = {name: i for i, name in enumerate(labels)}
    for df in (train_raw, test):
        df["label"] = df["category"].map(label_map)
        df["text_sha256"] = df["text"].map(text_hash)

    # Stratified train / tuning split from the released training set only.
    train, tune = train_test_split(
        train_raw, test_size=TUNE_SIZE, random_state=SUBSET_SEED,
        stratify=train_raw["label"], shuffle=True,
    )

    # Exact-duplicate screening. Evaluation membership stays fixed.
    removed = {"train_in_test": 0, "tune_in_test": 0, "train_in_tune": 0}
    test_hashes = set(test["text_sha256"])
    m = train["text_sha256"].isin(test_hashes); removed["train_in_test"] = int(m.sum()); train = train[~m]
    m = tune["text_sha256"].isin(test_hashes); removed["tune_in_test"] = int(m.sum()); tune = tune[~m]
    m = train["text_sha256"].isin(set(tune["text_sha256"])); removed["train_in_tune"] = int(m.sum()); train = train[~m]
    # Repeats inside the training split are not leakage, so they are only reported.
    within_train = int(train["text_sha256"].duplicated().sum())

    cols = ["id", "label", "category", "text_sha256"]
    for name, df in [("train", train), ("tune", tune), ("test", test)]:
        df.sort_values("id")[cols].to_csv(out_dir / f"{name}.csv", index=False)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    meta = {
        "dataset": "BANKING77",
        "source": BASE_URL,
        "license": "CC BY 4.0",
        "subset_seed": SUBSET_SEED,
        "raw_sha256": {k: sha256_file(p) for k, p in paths.items()},
        "raw_counts": {"train": len(train_raw), "test": len(test)},
        "planned_counts": {"train": EXPECTED["train_raw"] - TUNE_SIZE, "tune": TUNE_SIZE, "test": EXPECTED["test"]},
        "final_counts": {"train": len(train), "tune": len(tune), "test": len(test)},
        "duplicates_removed": removed,
        "repeats_within_train_kept": within_train,
        "num_labels": len(label_map),
        "manifest_sha256": {n: sha256_file(out_dir / f"{n}.csv") for n in ["train", "tune", "test"]},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps({k: meta[k] for k in ["raw_counts", "final_counts", "duplicates_removed", "repeats_within_train_kept", "num_labels"]}, indent=2))
    if meta["raw_counts"]["train"] != EXPECTED["train_raw"] or meta["raw_counts"]["test"] != EXPECTED["test"]:
        print("WARNING: raw counts differ from the published split sizes in the proposal.")


if __name__ == "__main__":
    main()
