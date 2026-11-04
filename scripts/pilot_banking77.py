"""20-step compatibility pilot for one BANKING77 config (proposal Sections 8.3-8.4).

A config passes the pilot only if it:
  - finishes the requested optimizer steps with a finite loss,
  - puts gradients only on the tensors that are supposed to train,
  - saves and reloads its checkpoint, and
  - reproduces one fixed evaluation batch within a recorded tolerance.

It also measures peak allocated/reserved GPU memory (after the first optimizer
step and in steady state), the mean step time after warmup, and the saved
checkpoint size, then projects GPU hours for a full run. The result goes to
results/pilots/<run_id>.json and one row is appended to results/run_ledger.csv.

Run on Colab/Kaggle from the repo root:
    python scripts/pilot_banking77.py --config configs/banking77/lora_r8.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lora_budget.data.banking77 import Banking77Dataset, load_split, make_collator
from lora_budget.models.roberta import build_model, count_params

LEDGER_FIELDS = [
    "run_id", "timestamp_utc", "phase", "experiment", "config", "method", "rank", "alpha",
    "seed", "lr", "steps", "status", "final_loss", "peak_alloc_mb_first_step",
    "peak_alloc_mb_steady", "peak_reserved_mb_steady", "mean_step_s", "trainable_params",
    "checkpoint_mb", "reload_max_abs_diff", "projected_run_gpu_h", "device", "torch",
    "transformers", "peft", "notes",
]
NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mb(x: int) -> float:
    return round(x / 2**20, 1)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class OfflineTokenizer:
    """Only for testing this script without internet. Never use for real runs."""
    pad_token_id = 1

    def __call__(self, texts, truncation=True, max_length=128):
        ids = [[0] + [3 + (hash(w) % 50000) for w in t.split()][: max_length - 2] + [2] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def save_and_reload(model, method: str, cfg: dict, seed: int, offline: bool, out_dir: Path, probe: dict):
    """Save the trainable part, reload into a fresh model, and compare logits."""
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        before = model(**probe).logits.float()
    out_dir.mkdir(parents=True, exist_ok=True)
    if method == "lora":
        from peft import PeftModel
        model.save_pretrained(out_dir)
        base = build_model("full_ft", seed=seed, offline_random_init=offline,
                           base_model=cfg["model"]["name"], revision=cfg["model"]["revision"])
        fresh = PeftModel.from_pretrained(base, out_dir)
    elif method == "head_only":
        torch.save(model.classifier.state_dict(), out_dir / "head.pt")
        fresh = build_model(method, seed=seed, offline_random_init=offline,
                            base_model=cfg["model"]["name"], revision=cfg["model"]["revision"])
        for p in fresh.classifier.parameters():
            torch.nn.init.normal_(p)
        fresh.classifier.load_state_dict(torch.load(out_dir / "head.pt"))
    else:
        model.save_pretrained(out_dir)
        fresh = type(model).from_pretrained(out_dir)
    fresh.to(before.device).eval()
    with torch.no_grad():
        after = fresh(**probe).logits.float()
    return (before - after).abs().max().item(), dir_size(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=None, help="default: first value in the config grid")
    ap.add_argument("--warmup-steps-for-timing", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default="checkpoints/pilots", help="NOT committed (.gitignore)")
    ap.add_argument("--offline", action="store_true", help="random-init model + fake tokenizer, for testing only")
    ap.add_argument("--max-train-examples", type=int, default=None, help="for quick local tests only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    method = cfg["method"]
    lr = args.lr if args.lr is not None else cfg["train"]["learning_rates"][0]
    rank = cfg.get("lora", {}).get("rank")
    alpha = cfg.get("lora", {}).get("alpha")
    run_id = f"pilot-{Path(args.config).stem}-s{args.seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and cfg["train"]["precision"] == "fp16"
    row = {k: "" for k in LEDGER_FIELDS}
    row.update(run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               phase="pilot", experiment=cfg["experiment"], config=args.config, method=method,
               rank=rank or "", alpha=alpha or "", seed=args.seed, lr=lr, steps=args.steps,
               device=torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
               torch=torch.__version__)
    import peft
    import transformers
    row.update(transformers=transformers.__version__, peft=peft.__version__)
    report: dict = {"run_id": run_id, "config": cfg, "checks": {}}

    try:
        # ---- data (same shuffled order for every method at a given seed)
        if args.offline:
            tok = OfflineTokenizer()
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        train_df = load_split("train", cfg["data"]["raw_dir"], cfg["data"]["manifest_dir"])
        tune_df = load_split("tune", cfg["data"]["raw_dir"], cfg["data"]["manifest_dir"])
        if args.max_train_examples:
            train_df = train_df.head(args.max_train_examples)
        train_ds = Banking77Dataset(train_df, tok, cfg["data"]["max_len"])
        collate = make_collator(tok.pad_token_id)
        g = torch.Generator().manual_seed(args.seed)
        micro, accum = cfg["train"]["micro_batch"], cfg["train"]["grad_accum"]
        loader = DataLoader(train_ds, batch_size=micro, shuffle=True, generator=g,
                            collate_fn=collate, drop_last=True)
        probe_ds = Banking77Dataset(tune_df.head(8), tok, cfg["data"]["max_len"])
        probe = collate([probe_ds[i] for i in range(len(probe_ds))])
        probe.pop("labels")
        probe = {k: v.to(device) for k, v in probe.items()}

        # ---- model
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
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
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)
        warm = max(1, math.ceil(cfg["train"]["warmup_ratio"] * args.steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (args.steps - s) / max(1, args.steps - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # ---- train for N optimizer steps
        losses, step_times, it = [], [], iter(loader)
        grads_ok, peak_first = True, None
        for step in range(args.steps):
            sync(); t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader); batch = next(it)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss / accum
                scaler.scale(loss).backward()
                step_loss += loss.item()
            if step == 0:  # gradients only where intended
                for n, p in model.named_parameters():
                    if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                        grads_ok = False; report["checks"].setdefault("bad_grad", []).append(n)
                    if not p.requires_grad and p.grad is not None:
                        grads_ok = False; report["checks"].setdefault("frozen_got_grad", []).append(n)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           cfg["train"]["max_grad_norm"])
            scaler.step(opt); scaler.update(); sched.step()
            sync(); step_times.append(time.perf_counter() - t0)
            losses.append(step_loss)
            if step == 0 and device == "cuda":
                peak_first = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()  # steady-state peak measured from here on
            print(f"step {step + 1:>3}/{args.steps}  loss {step_loss:.4f}  {step_times[-1]:.2f}s")

        finite = all(math.isfinite(x) for x in losses)
        timed = step_times[args.warmup_steps_for_timing:] or step_times
        mean_step = sum(timed) / len(timed)
        if device == "cuda":
            row["peak_alloc_mb_first_step"] = mb(peak_first)
            row["peak_alloc_mb_steady"] = mb(torch.cuda.max_memory_allocated())
            row["peak_reserved_mb_steady"] = mb(torch.cuda.max_memory_reserved())

        # ---- save / reload / reproduce
        diff, size = save_and_reload(model, method, cfg, args.seed, args.offline,
                                     Path(args.ckpt_dir) / run_id, probe)
        steps_per_epoch = math.ceil(len(train_ds) / (micro * accum))
        projected_h = mean_step * steps_per_epoch * cfg["train"]["epochs"] / 3600

        report["checks"].update(finite_loss=finite, grads_only_where_intended=grads_ok,
                                reload_max_abs_diff=diff, reload_within_tol=diff < args.tol)
        passed = finite and grads_ok and diff < args.tol
        row.update(status="completed" if passed else "failed_code_or_data",
                   final_loss=round(losses[-1], 4), mean_step_s=round(mean_step, 3),
                   checkpoint_mb=mb(size), reload_max_abs_diff=f"{diff:.2e}",
                   projected_run_gpu_h=round(projected_h, 2) if device == "cuda" else "",
                   notes="" if passed else json.dumps(report["checks"])[:300])
        report.update(losses=losses, step_times=step_times, steps_per_epoch=steps_per_epoch,
                      passed=passed)

    except torch.cuda.OutOfMemoryError as e:
        row.update(status="infeasible_oom", notes=str(e)[:300])
        report["error"] = str(e)
    except Exception as e:  # recorded, never silently dropped
        row.update(status="failed_code_or_data", notes=f"{type(e).__name__}: {e}"[:300])
        report["error"] = repr(e)
        raise
    finally:
        if args.offline:
            row["notes"] = ("OFFLINE TEST RUN - not evidence. " + row["notes"]).strip()
        out = Path("results/pilots"); out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_id}.json").write_text(json.dumps({"ledger_row": row, **report}, indent=2, default=str))
        ledger = Path("results/run_ledger.csv")
        new = not ledger.exists()
        with open(ledger, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
