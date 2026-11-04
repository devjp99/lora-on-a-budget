"""Pick the learning rate for each BANKING77 method from the seed-42 search (proposal Section 8.2).

Rule: highest tuning accuracy wins. If two learning rates tie, the one whose best
epoch came EARLIER wins; if that also ties, the SMALLER learning rate wins.
Only completed, real runs from the lr_search phase count (test runs are ignored).

Writes results/selected_lr_banking77.json, which later phases read.
Run from the repo root:
    python scripts/select_lr_banking77.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def select(ledger: pd.DataFrame) -> dict:
    df = ledger[(ledger["phase"] == "lr_search") & (ledger["status"] == "completed")].copy()
    df = df[~df["notes"].fillna("").str.contains("TEST RUN")]
    df = df.drop_duplicates(subset="run_id", keep="last")  # if a run was repeated, use the latest
    chosen = {}
    for cfg, g in df.groupby("config"):
        g = g.sort_values(["tune_accuracy", "best_epoch", "lr"], ascending=[False, True, True])
        top = g.iloc[0]
        chosen[Path(cfg).stem] = {
            "config": cfg,
            "method": top["method"],
            "lr": float(top["lr"]),
            "seed42_run_id": top["run_id"],
            "tune_accuracy": float(top["tune_accuracy"]),
            "tune_macro_f1": float(top["tune_macro_f1"]),
            "best_epoch": int(top["best_epoch"]),
            "candidates": g[["lr", "tune_accuracy", "tune_macro_f1", "best_epoch"]].to_dict("records"),
        }
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results/train_ledger.csv")
    ap.add_argument("--out", default="results/selected_lr_banking77.json")
    args = ap.parse_args()
    chosen = select(pd.read_csv(args.ledger))
    Path(args.out).write_text(json.dumps(chosen, indent=2))
    for name, c in chosen.items():
        print(f"{name:10s} -> lr {c['lr']:.0e}  tune acc {c['tune_accuracy']:.2f}  "
              f"macro-F1 {c['tune_macro_f1']:.2f}  best epoch {c['best_epoch']}  "
              f"({len(c['candidates'])} candidates)")


if __name__ == "__main__":
    main()
