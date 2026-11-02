"""BANKING77 loader that reads the raw CSVs through the fixed manifests.

Proposal Sections 7.1-7.3: every method reads the exact same manifest files,
labels come from the saved label dictionary, inputs are capped at 128 tokens,
and padding is dynamic within each minibatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from lora_budget.data.hashing import sha256_file

MAX_LEN = 128
SPLITS = ("train", "tune", "test")


def load_split(split: str, raw_dir: str | Path = "data/raw",
               manifest_dir: str | Path = "manifests/banking77",
               verify: bool = True) -> pd.DataFrame:
    """Return the rows of one split as a DataFrame with columns id, text, label."""
    assert split in SPLITS, f"split must be one of {SPLITS}"
    raw_dir, manifest_dir = Path(raw_dir), Path(manifest_dir)
    meta = json.loads((manifest_dir / "meta.json").read_text())

    raw_name = "test" if split == "test" else "train"
    raw_path = raw_dir / f"banking77_{raw_name}.csv"
    if verify:
        got = sha256_file(raw_path)
        want = meta["raw_sha256"][raw_name]
        assert got == want, f"{raw_path} checksum {got[:12]} does not match manifest {want[:12]}"

    raw = pd.read_csv(raw_path).reset_index().rename(columns={"index": "row"})
    raw["id"] = f"{raw_name}-" + raw["row"].astype(str).str.zfill(5)
    manifest = pd.read_csv(manifest_dir / f"{split}.csv")
    df = manifest.merge(raw[["id", "text"]], on="id", how="left", validate="one_to_one")
    assert df["text"].notna().all(), f"{split}: some manifest IDs are missing from the raw file"
    return df[["id", "text", "label"]].reset_index(drop=True)


def label_names(manifest_dir: str | Path = "manifests/banking77") -> list[str]:
    label_map = json.loads((Path(manifest_dir) / "label_map.json").read_text())
    return [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]


class Banking77Dataset(Dataset):
    """Tokenized split. Padding is left to the collator (dynamic padding)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        enc = tokenizer(list(df["text"]), truncation=True, max_length=max_len)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = list(df["label"].astype(int))
        self.ids = list(df["id"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def make_collator(pad_token_id: int):
    """Dynamic padding within each minibatch (pad to the longest example in the batch)."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
            mask[i, :n] = torch.tensor(x["attention_mask"], dtype=torch.long)
        labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate
