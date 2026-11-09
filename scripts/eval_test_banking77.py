"""Final TEST-set evaluation for BANKING77 (proposal Section 8.2: run once, after code is locked).

For every locked run it:
  1. Finds the run's best checkpoint (saved by train_banking77.py) under --ckpt-roots.
  2. Rebuilds the model the same way training did and loads the checkpoint.
  3. Re-scores the TUNING set and checks it matches the training log (within
     --tol points). This proves we loaded the right weights. If it does not
     match, the run is marked failed and the test set is NOT scored for it.
  4. Scores the TEST set once and saves every prediction.

Safety rules:
  - Refuses to run if tracked files have uncommitted changes (code must be locked),
    unless --allow-dirty (for local testing only; marked TEST RUN).
  - Refuses to overwrite existing test predictions unless --force.

Outputs:
  results/test/preds/<run_id>.json   ids, gold, pred for the test set
  results/test/test_metrics.csv      one row per run

Run from the repo root:
    python scripts/eval_test_banking77.py --ckpt-roots /kaggle/input/week2/checkpoints /kaggle/input/week3/checkpoints
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from lora_budget.data.banking77 import Banking77Dataset, load_split, make_collator
from lora_budget.models.roberta import build_model

NUM_LABELS = 77
SELECTED = Path("results/selected_lr_banking77.json")
LEDGER = Path("results/train_ledger.csv")


def locked_runs(ledger: pd.DataFrame, selected: dict) -> list[dict]:
    """The runs that go to the test set: E1 (3 methods x 3 seeds), E3 ranks 4/16, supplementary."""
    led = ledger[ledger["status"] == "completed"].drop_duplicates("run_id", keep="last")
    led = led[~led["notes"].fillna("").str.contains("TEST RUN")]
    runs = []
    for name in ("full_ft", "lora_r8", "head_only"):
        runs.append({"run_id": selected[name]["seed42_run_id"], "group": "E1", "setup": name})
        for rid in led[(led["phase"] == "confirm") & (led["config"].str.endswith(f"/{name}.yaml"))]["run_id"]:
            runs.append({"run_id": rid, "group": "E1", "setup": name})
    for rid, cfg in led[led["phase"] == "rank_sweep"][["run_id", "config"]].values:
        runs.append({"run_id": rid, "group": "E3", "setup": Path(cfg).stem})
    for rid in led[led["phase"] == "supplementary"]["run_id"]:
        runs.append({"run_id": rid, "group": "supplementary", "setup": "head_only"})
    return runs


def find_best_dir(run_id: str, roots: list[str]) -> Path | None:
    for root in roots:
        d = Path(root) / run_id / "best"
        if d.is_dir():
            return d
    return None


def load_run_model(method: str, cfg: dict, seed: int, best_dir: Path, offline: bool = False):
    """Rebuild the model exactly as training did, then load the saved trainable weights."""
    kw = {}
    if method == "lora":
        kw = {"rank": cfg["lora"]["rank"], "alpha": cfg["lora"]["alpha"], "dropout": cfg["lora"]["dropout"],
              "target_modules": tuple(cfg["lora"]["target_modules"])}
    model = build_model(method, seed=seed, offline_random_init=offline,
                        base_model=cfg["model"]["name"], revision=cfg["model"]["revision"], **kw)
    if method == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(str(best_dir / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, state)
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora_" in k or "classifier" in k]
        assert not bad, f"checkpoint keys not loaded: {bad[:5]}"
    else:
        state = torch.load(best_dir / "trainable_state.pt", map_location="cpu")
        res = model.load_state_dict(state, strict=(method == "full_ft"))
        assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
        if method == "head_only":
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            assert trainable <= set(state), f"head weights missing: {sorted(trainable - set(state))[:5]}"
    return model


@torch.no_grad()
def predict(model, loader, device: str, use_amp: bool) -> list[int]:
    """Same settings as training-time evaluation: fp16 autocast, batch 64, argmax."""
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).tolist()


def scores(gold, pred) -> tuple[float, float]:
    acc = round(100 * accuracy_score(gold, pred), 3)
    f1 = round(100 * f1_score(gold, pred, average="macro", labels=list(range(NUM_LABELS)), zero_division=0), 3)
    return acc, f1


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return commit or "unknown", bool(dirty)
    except FileNotFoundError:
        return "unknown", True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-roots", nargs="+", required=True, help="folders that hold <run_id>/best")
    ap.add_argument("--out", default="results/test")
    ap.add_argument("--tol", type=float, default=0.15, help="allowed tuning-accuracy mismatch, in points")
    ap.add_argument("--force", action="store_true", help="overwrite existing test predictions")
    ap.add_argument("--allow-dirty", action="store_true", help="local testing only")
    ap.add_argument("--offline", action="store_true", help="random-init model + fake tokenizer, testing only")
    args = ap.parse_args()

    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        sys.exit("Uncommitted changes in tracked files. Commit (lock) the code before touching the test set.")
    out, pred_dir = Path(args.out), Path(args.out) / "preds"
    if pred_dir.exists() and any(pred_dir.iterdir()) and not args.force:
        sys.exit(f"{pred_dir} already has test predictions. The test set is scored once; use --force only if "
                 "the earlier run was broken, and say so in the docs.")
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ledger = pd.read_csv(LEDGER)
    runs = locked_runs(ledger, json.loads(SELECTED.read_text()))
    print(f"{len(runs)} runs to evaluate, code at commit {commit[:10]}{' (DIRTY - TEST RUN)' if dirty else ''}")

    tok_cache, data_cache, rows = {}, {}, []
    for r in runs:
        rep = json.loads(Path(f"results/runs/{r['run_id']}.json").read_text())
        cfg, a = rep["config"], rep["args"]
        method, seed = cfg["method"], int(a["seed"])
        row = {**r, "method": method, "seed": seed, "lr": a["lr"], "rank": cfg.get("lora", {}).get("rank", ""),
               "logged_tune_accuracy": rep["ledger_row"]["tune_accuracy"], "commit": commit,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "notes": "TEST RUN - not evidence." if (dirty or args.offline) else ""}
        print(f"\n== {r['run_id']} ({r['group']})")
        try:
            best_dir = find_best_dir(r["run_id"], args.ckpt_roots)
            if best_dir is None:
                raise FileNotFoundError("best checkpoint not found under --ckpt-roots")
            key = (cfg["model"]["name"], cfg["model"]["revision"])
            if key not in tok_cache:
                if args.offline:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from train_banking77 import OfflineTokenizer
                    tok_cache[key] = OfflineTokenizer()
                else:
                    from transformers import AutoTokenizer
                    tok_cache[key] = AutoTokenizer.from_pretrained(key[0], revision=key[1])
                tok = tok_cache[key]
                loaders = {}
                for split in ("tune", "test"):
                    ds = Banking77Dataset(load_split(split, cfg["data"]["raw_dir"], cfg["data"]["manifest_dir"]),
                                          tok, cfg["data"]["max_len"])
                    loaders[split] = (ds, DataLoader(ds, batch_size=64, shuffle=False,
                                                     collate_fn=make_collator(tok.pad_token_id)))
                data_cache[key] = loaders
            loaders = data_cache[key]
            use_amp = device == "cuda" and cfg["train"]["precision"] == "fp16"
            model = load_run_model(method, cfg, seed, best_dir, offline=args.offline).to(device)

            tune_ds, tune_loader = loaders["tune"]
            tune_acc, _ = scores(tune_ds.labels, predict(model, tune_loader, device, use_amp))
            row["reloaded_tune_accuracy"] = tune_acc
            gap = abs(tune_acc - float(row["logged_tune_accuracy"]))
            if gap > args.tol:
                raise ValueError(f"reloaded tuning accuracy {tune_acc} != logged {row['logged_tune_accuracy']}")

            test_ds, test_loader = loaders["test"]
            pred = predict(model, test_loader, device, use_amp)
            row["test_accuracy"], row["test_macro_f1"] = scores(test_ds.labels, pred)
            row["status"] = "completed"
            (pred_dir / f"{r['run_id']}.json").write_text(json.dumps(
                {"run_id": r["run_id"], "ids": list(test_ds.ids), "gold": test_ds.labels, "pred": pred}))
            print(f"tune {tune_acc} (logged {row['logged_tune_accuracy']})  "
                  f"TEST acc {row['test_accuracy']}  macro-F1 {row['test_macro_f1']}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # recorded, never silently dropped
            row["status"] = "failed"
            row["notes"] = (row["notes"] + f" {type(e).__name__}: {e}"[:300]).strip()
            print("FAILED:", row["notes"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "test_metrics.csv", index=False)
    print("\n" + df[["run_id", "status", "reloaded_tune_accuracy", "test_accuracy", "test_macro_f1"]]
          .to_string(index=False))
    n_bad = int((df["status"] != "completed").sum())
    if n_bad:
        sys.exit(f"{n_bad} run(s) failed; see {out / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()
