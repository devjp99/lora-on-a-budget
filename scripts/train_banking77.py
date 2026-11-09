"""Full training run for one BANKING77 config and one learning rate (proposal Sections 6.1, 8.1-8.2).

What one run does:
  1. Trains for the configured number of epochs (5) with the Table 5 settings:
     AdamW, weight decay 0.01 (not on bias/LayerNorm), linear decay with 6% warmup,
     gradient clipping 1.0, fp16 mixed precision, effective batch 32 (8 x 4).
  2. After every epoch, scores the TUNING set: accuracy and macro-F1.
  3. Keeps the epoch with the best tuning accuracy. Ties go to the earlier epoch.
  4. Saves that best checkpoint (trainable part only) and its tuning predictions.
  5. Saves a "last" checkpoint every epoch so an interrupted session can resume
     (model, optimizer, scheduler, loss scaler, and RNG state). It is deleted when
     the run finishes, unless --keep-last is given.
  6. Measures peak GPU memory, loading/training/eval time, and throughput.
  7. Appends one row to results/train_ledger.csv and writes results/runs/<run_id>.json.

The TEST set is never touched here. Final test evaluation happens later, only
after configs and analysis code are locked.

Examples (from the repo root):
    python scripts/train_banking77.py --config configs/banking77/lora_r8.yaml --lr 1e-4 --seed 42 --phase lr_search
    python scripts/train_banking77.py --config configs/banking77/lora_r8.yaml --lr 1e-4 --seed 42 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from lora_budget.data.banking77 import Banking77Dataset, load_split, make_collator
from lora_budget.models.roberta import build_model, count_params

NUM_LABELS = 77
NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight")
LEDGER = Path("results/train_ledger.csv")
LEDGER_FIELDS = [
    "run_id", "timestamp_utc", "phase", "experiment", "config", "method", "rank", "alpha",
    "seed", "lr", "epochs", "status", "best_epoch", "tune_accuracy", "tune_macro_f1",
    "trainable_params", "peak_alloc_mb", "peak_reserved_mb", "load_s", "train_s", "eval_s",
    "train_examples_per_s", "checkpoint_mb", "resumed", "device", "torch", "transformers",
    "peft", "notes",
]


# ----------------------------------------------------------------------------- helpers
def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mb(x: int) -> float:
    return round(x / 2**20, 1)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")  # 1e-4, 3e-5


class OfflineTokenizer:
    """Only for testing this script without internet. Never use for real runs."""
    pad_token_id = 1

    def __call__(self, texts, truncation=True, max_length=128):
        ids = [[0] + [3 + (zlib.crc32(w.encode()) % 50000) for w in t.split()][: max_length - 2] + [2] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def trainable_state(model, method: str) -> dict:
    """State dict of only the parts that train (what we need to save and resume)."""
    if method == "full_ft":
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@torch.no_grad()
def evaluate(model, loader, device: str, use_amp: bool) -> dict:
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu())
        golds.append(labels.cpu())
    p, g = torch.cat(preds).numpy(), torch.cat(golds).numpy()
    model.train()
    return {
        "accuracy": round(100 * accuracy_score(g, p), 3),
        "macro_f1": round(100 * f1_score(g, p, average="macro", labels=list(range(NUM_LABELS)),
                                         zero_division=0), 3),
        "preds": p.tolist(),
    }


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", default="lr_search", choices=["lr_search", "confirm", "rank_sweep", "supplementary", "debug"])
    ap.add_argument("--ckpt-root", default="checkpoints", help="NOT committed (.gitignore)")
    ap.add_argument("--resume", action="store_true", help="continue from the last saved epoch")
    ap.add_argument("--epochs", type=int, default=None, help="override; for quick tests only")
    ap.add_argument("--offline", action="store_true", help="random-init model + fake tokenizer, for testing only")
    ap.add_argument("--max-train-examples", type=int, default=None, help="for quick tests only")
    ap.add_argument("--max-tune-examples", type=int, default=None, help="for quick tests only")
    ap.add_argument("--keep-last", action="store_true", help="keep the resume file after the run finishes")
    ap.add_argument("--stop-after-epoch", type=int, default=None, help="testing only: simulate an interrupted session")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    method, tr = cfg["method"], cfg["train"]
    epochs = args.epochs or tr["epochs"]
    rank = cfg.get("lora", {}).get("rank")
    alpha = cfg.get("lora", {}).get("alpha")
    run_id = f"{args.phase}-{Path(args.config).stem}-lr{lr_tag(args.lr)}-s{args.seed}"
    run_dir = Path(args.ckpt_root) / run_id
    last_path, best_dir = run_dir / "last.pt", run_dir / "best"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and tr["precision"] == "fp16"

    import peft
    import transformers
    row = {k: "" for k in LEDGER_FIELDS}
    row.update(run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               phase=args.phase, experiment=cfg["experiment"], config=args.config, method=method,
               rank=rank or "", alpha=alpha or "", seed=args.seed, lr=args.lr, epochs=epochs,
               resumed=args.resume, torch=torch.__version__, transformers=transformers.__version__,
               peft=peft.__version__,
               device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    report: dict = {"run_id": run_id, "config": cfg, "args": vars(args), "epochs": []}
    if args.offline or args.epochs or args.max_train_examples or args.max_tune_examples or args.stop_after_epoch:
        row["notes"] = "TEST RUN (offline/subset/epochs override) - not evidence."

    try:
        # ---------------- load data + model (timed)
        sync(); t_load = time.perf_counter()
        if args.offline:
            tok = OfflineTokenizer()
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        train_df = load_split("train", cfg["data"]["raw_dir"], cfg["data"]["manifest_dir"])
        tune_df = load_split("tune", cfg["data"]["raw_dir"], cfg["data"]["manifest_dir"])
        if args.max_train_examples:
            train_df = train_df.head(args.max_train_examples)
        if args.max_tune_examples:
            tune_df = tune_df.head(args.max_tune_examples)
        train_ds = Banking77Dataset(train_df, tok, cfg["data"]["max_len"])
        tune_ds = Banking77Dataset(tune_df, tok, cfg["data"]["max_len"])
        collate = make_collator(tok.pad_token_id)
        micro, accum = tr["micro_batch"], tr["grad_accum"]
        tune_loader = DataLoader(tune_ds, batch_size=64, shuffle=False, collate_fn=collate)

        if device == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        kw = {"rank": rank, "alpha": alpha, "dropout": cfg["lora"]["dropout"],
              "target_modules": tuple(cfg["lora"]["target_modules"])} if method == "lora" else {}
        model = build_model(method, seed=args.seed, offline_random_init=args.offline,
                            base_model=cfg["model"]["name"], revision=cfg["model"]["revision"], **kw)
        model.to(device).train()
        counts = count_params(model)
        row["trainable_params"] = counts["trainable"]
        report["param_counts"] = counts

        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (no_decay if any(k in n for k in NO_DECAY) else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": tr["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
        steps_per_epoch = len(train_ds) // micro // accum
        total_steps = steps_per_epoch * epochs
        warm = max(1, math.ceil(tr["warmup_ratio"] * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        sync(); load_s = time.perf_counter() - t_load

        # ---------------- resume (epoch granularity)
        start_epoch, best = 0, {"epoch": None, "accuracy": -1.0}
        train_s = eval_s = 0.0
        if args.resume and last_path.exists():
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"], strict=False)
            opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"]); set_rng_state(ck["rng"])
            start_epoch, best = ck["epoch"] + 1, ck["best"]
            report["epochs"], train_s, eval_s = ck["epoch_log"], ck["train_s"], ck["eval_s"]
            print(f"Resumed from end of epoch {ck['epoch'] + 1}")
        elif args.resume:
            print("No last.pt found; starting from scratch")

        # ---------------- train
        for epoch in range(start_epoch, epochs):
            # Same example order for every method at a given seed and epoch.
            g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
            loader = DataLoader(train_ds, batch_size=micro, shuffle=True, generator=g,
                                collate_fn=collate, drop_last=True)
            it = iter(loader)
            sync(); t0 = time.perf_counter()
            running, seen = 0.0, 0
            for step in range(steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                for _ in range(accum):
                    batch = {k: v.to(device) for k, v in next(it).items()}
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        loss = model(**batch).loss / accum
                    scaler.scale(loss).backward()
                    running += loss.item(); seen += batch["labels"].shape[0]
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               tr["max_grad_norm"])
                scaler.step(opt); scaler.update(); sched.step()
                if (step + 1) % 50 == 0:
                    print(f"epoch {epoch + 1} step {step + 1}/{steps_per_epoch} loss {running / (step + 1):.4f}")
            sync(); ep_train_s = time.perf_counter() - t0
            train_s += ep_train_s
            train_loss = running / max(1, steps_per_epoch)
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"non-finite training loss in epoch {epoch + 1}")

            t1 = time.perf_counter()
            m = evaluate(model, tune_loader, device, use_amp)
            sync(); ep_eval_s = time.perf_counter() - t1
            eval_s += ep_eval_s
            log = {"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                   "tune_accuracy": m["accuracy"], "tune_macro_f1": m["macro_f1"],
                   "train_s": round(ep_train_s, 1), "eval_s": round(ep_eval_s, 1),
                   "examples_per_s": round(seen / ep_train_s, 1)}
            report["epochs"].append(log)
            print(json.dumps(log))

            # Best epoch by tuning accuracy; strict ">" keeps the EARLIER epoch on ties.
            if m["accuracy"] > best["accuracy"]:
                best = {"epoch": epoch + 1, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                best_dir.mkdir(parents=True, exist_ok=True)
                if method == "lora":
                    model.save_pretrained(best_dir)
                else:
                    torch.save(trainable_state(model, method), best_dir / "trainable_state.pt")
                (best_dir / "tune_preds.json").write_text(json.dumps(
                    {"ids": list(tune_ds.ids), "pred": m["preds"], "gold": tune_ds.labels}))

            # "last" checkpoint for resuming
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": trainable_state(model, method),
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng_state(), "best": best,
                        "epoch_log": report["epochs"], "train_s": train_s, "eval_s": eval_s},
                       last_path)
            if args.stop_after_epoch and epoch + 1 == args.stop_after_epoch:
                raise KeyboardInterrupt(f"simulated interruption after epoch {epoch + 1}")

        if not args.keep_last and last_path.exists():
            last_path.unlink()  # only needed to resume an unfinished run

        # ---------------- summary
        total_seen = sum(e["examples_per_s"] * e["train_s"] for e in report["epochs"])
        row.update(status="completed", best_epoch=best["epoch"],
                   tune_accuracy=best["accuracy"], tune_macro_f1=best.get("macro_f1", ""),
                   load_s=round(load_s, 1), train_s=round(train_s, 1), eval_s=round(eval_s, 1),
                   train_examples_per_s=round(total_seen / max(train_s, 1e-9), 1),
                   checkpoint_mb=mb(dir_size(best_dir)) if best_dir.exists() else "")
        if device == "cuda":
            row.update(peak_alloc_mb=mb(torch.cuda.max_memory_allocated()),
                       peak_reserved_mb=mb(torch.cuda.max_memory_reserved()))
        report["best"] = best

    except KeyboardInterrupt as e:  # stopped by hand or by a session limit; rerun with --resume
        row.update(status="interrupted", notes=(row["notes"] + f" {e}").strip())
        report["error"] = repr(e)
        raise
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="infeasible_oom", notes=(row["notes"] + " " + str(e)[:250]).strip())
        report["error"] = str(e)
    except Exception as e:  # recorded, never silently dropped
        row.update(status="failed_code_or_data",
                   notes=(row["notes"] + f" {type(e).__name__}: {e}"[:250]).strip())
        report["error"] = repr(e)
        raise
    finally:
        out = Path("results/runs"); out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_id}.json").write_text(json.dumps({"ledger_row": row, **report}, indent=2, default=str))
        append_ledger(row)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
