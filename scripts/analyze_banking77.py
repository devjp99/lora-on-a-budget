"""Final BANKING77 analysis from the saved test predictions (proposal Sections 9.1 and 9.3).

Reads results/test/test_metrics.csv, results/test/preds/*.json and results/train_ledger.csv.
Never trains or re-scores anything. Writes to results/analysis/:

  e1_per_seed.csv        test accuracy / macro-F1 for every E1 run
  e1_summary.csv         mean and sample std over 3 seeds, plus cost columns from training
  e1_bootstrap.json      LoRA - full FT (and head-only - full FT): paired difference,
                         95% hierarchical bootstrap interval (10,000 resamples), and the
                         one-point decision
  e3_ranks.csv           LoRA rank 4 / 8 / 16 (seed 42): quality and cost
  supplementary.csv      extra head-only learning rates (reported separately)
  per_class_f1.csv       F1 per intent for full FT and LoRA (mean over seeds) and the gap
  confused_pairs.csv     most common (true intent -> predicted intent) mistakes per method
  agreement.json         how often LoRA and full FT give the same answer (same seed)
  fig_quality_cost.png   test accuracy vs peak training memory and training time
  fig_per_class_f1.png   per-intent F1, LoRA vs full FT
  summary.md             everything above in readable form

Run from the repo root:
    python scripts/analyze_banking77.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lora_budget.analysis.stats import decide, hierarchical_bootstrap
from lora_budget.data.banking77 import label_names

NUM_LABELS = 77
COST_COLS = ["trainable_params", "peak_alloc_mb", "peak_reserved_mb", "train_s",
             "train_examples_per_s", "checkpoint_mb"]


def load_preds(pred_dir: Path, run_ids) -> tuple[np.ndarray, np.ndarray, list[str]]:
    gold, preds, ids = None, [], None
    for rid in run_ids:
        d = json.loads((pred_dir / f"{rid}.json").read_text())
        if ids is None:
            ids, gold = d["ids"], np.array(d["gold"])
        assert d["ids"] == ids, f"{rid}: test examples are in a different order"
        preds.append(np.array(d["pred"]))
    return gold, np.stack(preds), ids


def per_class_f1(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.bincount(gold * NUM_LABELS + pred, minlength=NUM_LABELS ** 2).reshape(NUM_LABELS, NUM_LABELS)
    tp, denom = np.diag(cm).astype(float), cm.sum(0) + cm.sum(1)
    return 100 * np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="results/test")
    ap.add_argument("--ledger", default="results/train_ledger.csv")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--manifest-dir", default="manifests/banking77")
    args = ap.parse_args()

    test_dir, out = Path(args.test_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tm = pd.read_csv(test_dir / "test_metrics.csv")
    assert (tm["status"] == "completed").all(), "some test evaluations failed; fix those first"
    ledger = pd.read_csv(args.ledger).drop_duplicates("run_id", keep="last").set_index("run_id")
    tm = tm.join(ledger[COST_COLS], on="run_id")
    names = label_names(args.manifest_dir)
    md = ["# BANKING77 final results (test set, 3,080 examples)", ""]
    if tm["notes"].fillna("").str.contains("TEST RUN").any():
        md += ["**TEST RUN - these numbers are not evidence.**", ""]

    # ------------------------------------------------------------------ E1 per seed + summary
    e1 = tm[tm["group"] == "E1"].sort_values(["setup", "seed"])
    e1[["setup", "seed", "lr", "test_accuracy", "test_macro_f1", "run_id"]].to_csv(out / "e1_per_seed.csv", index=False)
    summ = e1.groupby("setup")[["test_accuracy", "test_macro_f1"] + COST_COLS].agg(["mean", "std"]).round(2)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reindex(["full_ft", "lora_r8", "head_only"])
    summ.to_csv(out / "e1_summary.csv")
    md += ["## E1: full fine-tuning vs LoRA r=8 vs head-only (3 seeds)", "",
           "| Setup | Test acc (mean ± sd) | Test macro-F1 | Per-seed acc | Peak alloc MB | Train s | Examples/s | Checkpoint MB | Trainable params |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in summ.index:
        r, per = summ.loc[s], e1[e1["setup"] == s].sort_values("seed")
        md.append(f"| {s} | {r.test_accuracy_mean:.2f} ± {r.test_accuracy_std:.2f} | "
                  f"{r.test_macro_f1_mean:.2f} ± {r.test_macro_f1_std:.2f} | "
                  f"{', '.join(f'{v:.2f}' for v in per['test_accuracy'])} | {r.peak_alloc_mb_mean:,.0f} | "
                  f"{r.train_s_mean:.0f} | {r.train_examples_per_s_mean:.0f} | {r.checkpoint_mb_mean:.1f} | "
                  f"{r.trainable_params_mean:,.0f} |")

    # ------------------------------------------------------------------ paired bootstrap
    def runs_of(setup):
        return e1[e1["setup"] == setup].sort_values("seed")
    full, lora, head = runs_of("full_ft"), runs_of("lora_r8"), runs_of("head_only")
    assert list(full["seed"]) == list(lora["seed"]) == list(head["seed"]), "seeds are not matched"
    gold, p_full, test_ids = load_preds(test_dir / "preds", full["run_id"])
    _, p_lora, _ = load_preds(test_dir / "preds", lora["run_id"])
    _, p_head, _ = load_preds(test_dir / "preds", head["run_id"])
    boot = {}
    for label, pa in (("lora_r8 - full_ft", p_lora), ("head_only - full_ft", p_head)):
        for metric in ("accuracy", "macro_f1"):
            b = hierarchical_bootstrap(gold, pa, p_full, metric=metric, num_labels=NUM_LABELS, n_boot=args.n_boot)
            b["decision"] = decide(b["ci_low"], b["ci_high"])
            boot[f"{label} | {metric}"] = b
    (out / "e1_bootstrap.json").write_text(json.dumps(boot, indent=2))
    md += ["", "## Paired difference vs full fine-tuning (95% hierarchical bootstrap, "
           f"{args.n_boot:,} resamples over seeds and test examples)", "",
           "Practical limit set in advance: -1 point. Within the limit only if the lower end is above -1.", "",
           "| Comparison | Metric | Mean difference | 95% interval | Decision |", "|---|---|---|---|---|"]
    for k, b in boot.items():
        comp, metric = k.split(" | ")
        md.append(f"| {comp} | {metric} | {b['diff']:+.2f} | [{b['ci_low']:+.2f}, {b['ci_high']:+.2f}] | {b['decision']} |")

    # ------------------------------------------------------------------ E3
    r8 = e1[(e1["setup"] == "lora_r8") & (e1["seed"] == 42)]
    e3 = pd.concat([tm[tm["group"] == "E3"], r8]).sort_values("rank")
    e3[["rank", "test_accuracy", "test_macro_f1"] + COST_COLS + ["run_id"]].to_csv(out / "e3_ranks.csv", index=False)
    md += ["", "## E3: LoRA rank (seed 42, same learning rate, alpha = 2r)", "",
           "| Rank | Test acc | Test macro-F1 | Trainable params | Peak alloc MB | Train s | Checkpoint MB |",
           "|---|---|---|---|---|---|---|"]
    for _, r in e3.iterrows():
        md.append(f"| {int(r['rank'])} | {r.test_accuracy:.2f} | {r.test_macro_f1:.2f} | {int(r.trainable_params):,} | "
                  f"{r.peak_alloc_mb:,.0f} | {r.train_s:.0f} | {r.checkpoint_mb:.1f} |")
    md += ["", "One seed per rank, so small differences may be seed noise (E1 seed sd is a useful yardstick)."]

    # ------------------------------------------------------------------ supplementary
    sup = tm[tm["group"] == "supplementary"].sort_values("lr")
    if len(sup):
        sup[["lr", "test_accuracy", "test_macro_f1", "run_id"]].to_csv(out / "supplementary.csv", index=False)
        md += ["", "## Supplementary: head-only at larger learning rates (seed 42; outside the pre-registered grid)", "",
               "| LR | Test acc | Test macro-F1 |", "|---|---|---|"]
        md += [f"| {r.lr:g} | {r.test_accuracy:.2f} | {r.test_macro_f1:.2f} |" for r in sup.itertuples()]

    # ------------------------------------------------------------------ per-class F1 and confusions
    f1_full = np.mean([per_class_f1(gold, p) for p in p_full], axis=0)
    f1_lora = np.mean([per_class_f1(gold, p) for p in p_lora], axis=0)
    pc = pd.DataFrame({"label": range(NUM_LABELS), "intent": names, "f1_full_ft": f1_full.round(2),
                       "f1_lora_r8": f1_lora.round(2), "lora_minus_full": (f1_lora - f1_full).round(2),
                       "support": np.bincount(gold, minlength=NUM_LABELS)}).sort_values("lora_minus_full")
    pc.to_csv(out / "per_class_f1.csv", index=False)

    conf_rows = []
    for setup, P in (("full_ft", p_full), ("lora_r8", p_lora)):
        wrong = [(g, p) for row in P for g, p in zip(gold, row) if g != p]
        counts = pd.Series(wrong).value_counts()
        for (g, p), c in counts.head(15).items():
            conf_rows.append({"setup": setup, "true_intent": names[g], "predicted_intent": names[p],
                              "count_over_3_seeds": int(c)})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(out / "confused_pairs.csv", index=False)

    same = float(np.mean(p_full == p_lora))
    c_full, c_lora = p_full == gold, p_lora == gold
    agree = {"same_prediction": round(100 * same, 2),
             "both_correct": round(100 * float(np.mean(c_full & c_lora)), 2),
             "only_full_ft_correct": round(100 * float(np.mean(c_full & ~c_lora)), 2),
             "only_lora_correct": round(100 * float(np.mean(~c_full & c_lora)), 2),
             "both_wrong": round(100 * float(np.mean(~c_full & ~c_lora)), 2),
             "both_wrong_same_answer": round(100 * float(np.mean(~c_full & ~c_lora & (p_full == p_lora))), 2),
             "note": "percent of (test example, seed) pairs; seeds matched"}
    (out / "agreement.json").write_text(json.dumps(agree, indent=2))

    md += ["", "## Where LoRA and full fine-tuning differ", "",
           f"- Same prediction on {agree['same_prediction']}% of test examples (matched seeds).",
           f"- Only full FT correct: {agree['only_full_ft_correct']}%. Only LoRA correct: {agree['only_lora_correct']}%. "
           f"Both wrong: {agree['both_wrong']}% (same wrong answer: {agree['both_wrong_same_answer']}%).", "",
           "Intents where LoRA loses the most F1 (mean over 3 seeds):", "",
           "| Intent | Full FT F1 | LoRA F1 | Gap |", "|---|---|---|---|"]
    md += [f"| {r.intent} | {r.f1_full_ft:.1f} | {r.f1_lora_r8:.1f} | {r.lora_minus_full:+.1f} |"
           for r in pc.head(8).itertuples()]
    md += ["", "Most common mistakes (summed over 3 seeds):", "",
           "| Setup | True intent | Predicted as | Count |", "|---|---|---|---|"]
    for setup in ("full_ft", "lora_r8"):
        md += [f"| {r.setup} | {r.true_intent} | {r.predicted_intent} | {r.count_over_3_seeds} |"
               for r in conf[conf["setup"] == setup].head(5).itertuples()]

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = pd.concat([e1, e3[e3["group"] == "E3"]])
    pts = pts.assign(label=np.where(pts["group"] == "E3", "lora_r" + pts["rank"].astype(str).str.replace(".0", ""),
                                    pts["setup"]))
    agg = pts.groupby("label")[["test_accuracy", "peak_alloc_mb", "train_s"]].mean()
    order = ["full_ft", "lora_r16", "lora_r8", "lora_r4", "head_only"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, xl in ((axes[0], "peak_alloc_mb", "Peak training memory (MB)"),
                        (axes[1], "train_s", "Training time, 5 epochs (s)")):
        for (lab, r), mk in zip(agg.reindex(order).dropna().iterrows(), "osD^v"):
            ax.scatter(r[col], r["test_accuracy"], marker=mk, s=45, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel("Test accuracy (%)"); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="center right")
    fig.suptitle("BANKING77 / RoBERTa-base on one T4: quality vs cost")
    fig.tight_layout(); fig.savefig(out / "fig_quality_cost.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pc["f1_full_ft"], pc["f1_lora_r8"], s=12)
    lo = min(pc["f1_full_ft"].min(), pc["f1_lora_r8"].min()) - 2
    ax.plot([lo, 100], [lo, 100], "--", color="gray", lw=1)
    for r in pc.head(5).itertuples():
        ax.annotate(r.intent, (r.f1_full_ft, r.f1_lora_r8), fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("Full FT F1 per intent"); ax.set_ylabel("LoRA r=8 F1 per intent")
    ax.set_title("Per-intent F1, mean of 3 seeds\n(below the line = LoRA worse)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_per_class_f1.png", dpi=200); plt.close(fig)

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
