# BANKING77 / RoBERTa-base (owner: Kunj Sevak)

Covers experiments **E1** (full fine-tuning vs. head-only vs. LoRA r=8) and **E3** (LoRA ranks 4, 8, 16).

## Data

| Split | Source | Count |
|---|---|---|
| train | released train, stratified, seed 2026 | 7,995 |
| tune | released train, stratified, seed 2026 | 2,001 |
| test | official test | 3,080 |

The planned split was 8,002 / 2,001 / 3,080. The duplicate check found **7 training
examples that exactly match test examples** (after lowercasing and collapsing spaces),
so they were removed from training. Test membership did not change. Four repeated
texts inside the training split were kept, because they are not leakage.
All of this is recorded in `manifests/banking77/meta.json`.

Source: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets), CC BY 4.0.
The raw CSVs are downloaded into `data/raw/` and are not committed.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch && pip install -r requirements.txt && pip install -e .
python scripts/make_banking77_manifest.py          # rebuilds identical manifests
python -m pytest -q                                # offline smoke + loader tests (17)
LORA_BUDGET_ONLINE=1 python -m pytest -q           # same tests with real roberta-base
python scripts/param_audit_roberta.py              # checks counts vs proposal Table 1
python scripts/pilot_banking77.py --config configs/banking77/lora_r8.yaml
```

## Parameter counts (verified against proposal Table 1 with real weights)

| Setup | LoRA factors | Classification head | Trainable total |
|---|---|---|---|
| Full fine-tuning | 0 | 649,805 | 124,704,845 |
| Head-only | 0 | 649,805 | 649,805 |
| LoRA r=4 (alpha 8) | 147,456 | 649,805 | 797,261 |
| LoRA r=8 (alpha 16) | 294,912 | 649,805 | 944,717 |
| LoRA r=16 (alpha 32) | 589,824 | 649,805 | 1,239,629 |

## Files

- `scripts/make_banking77_manifest.py`: downloads, splits, deduplicates, writes manifests
- `src/lora_budget/data/banking77.py`: loads splits through the manifests, tokenizes, pads per batch
- `src/lora_budget/models/roberta.py`: builds full_ft / head_only / lora models, counts parameters
- `scripts/param_audit_roberta.py`: parameter audit
- `scripts/pilot_banking77.py`: 20-step pilot gate; writes `results/pilots/*.json` and `results/run_ledger.csv`
- `tests/`: loader validation and smoke tests
- `configs/banking77/*.yaml`: settings from proposal Table 5
- `notebooks/week1_banking77_pilot.ipynb`: runs all of the above on a Kaggle T4

## Week 2: learning-rate search (E1, seed 42)

Each run trains for 5 epochs, scores the tuning set (accuracy, macro-F1) after every epoch,
and keeps the best epoch (ties go to the earlier epoch). The test set is not used.

```bash
python scripts/train_banking77.py --config configs/banking77/lora_r8.yaml --lr 1e-4 --seed 42 --phase lr_search
python scripts/select_lr_banking77.py      # writes results/selected_lr_banking77.json
```

- `results/train_ledger.csv`: one row per run (status, best epoch, tuning scores, memory, time)
- `results/runs/<run_id>.json`: per-epoch log for each run
- If a session stops, rerun the same command with `--resume`
- Learning-rate rule: highest tuning accuracy; ties go to the earlier best epoch, then the smaller learning rate

### Week 2 results (seed 42, tuning set, one T4)

| Method | Picked LR | Tune accuracy | Tune macro-F1 | Best epoch |
|---|---|---|---|---|
| Full fine-tuning | 3e-5 | 91.85 | 91.78 | 5 |
| LoRA r=8 | 3e-4 | 90.45 | 90.56 | 5 |
| Head-only | 3e-4 | 22.14 | 15.86 | 5 |

Head-only is under-trained, not broken. Its training loss only went from 4.31 to 3.88
in 5 epochs (random guessing over 77 classes is ln 77 = 4.34), and tuning accuracy was
still going up at epoch 5.

## Week 3: seed confirmation (E1) and rank sweep (E3)

- Confirmation: each method reruns its picked LR with seeds 43 and 44 (`--phase confirm`).
  The seed-42 pick counts as the first seed, so every method has 3 seeds.
- E3: LoRA r=4 and r=16 at seed 42 with the LoRA r=8 learning rate (`--phase rank_sweep`).
  Rank 8 reuses the E1 seed-42 run.
- Notebook: `notebooks/week3_banking77_confirm_e3.ipynb`

**Deviation (supplementary check, added after the LR search):** we also trained head-only
at 1e-3 and 3e-3 (seed 42, `--phase supplementary`), because the pre-registered grid
(1e-4, 3e-4) left it under-trained. These runs are never used by the LR picker and are
reported in a separate row. The pre-registered head-only pick (3e-4) stays the main E1
result. The test set has still not been used.

### Week 3 results (tuning set, one T4)

E1, 3 seeds (seed-42 pick + seeds 43, 44), mean ± sample std:

| Method | Tune accuracy | Tune macro-F1 | Peak alloc MB | Train s (5 epochs) |
|---|---|---|---|---|
| Full fine-tuning | 91.89 ± 0.25 | 91.71 ± 0.22 | 2,451 | 265 |
| LoRA r=8 | 90.34 ± 0.25 | 90.38 ± 0.25 | 930 | 216 |
| Head-only | 22.99 ± 2.30 | 16.31 ± 2.07 | 637 | 79 |

E3 (seed 42, LR 3e-4): rank 4 / 8 / 16 gave 90.01 / 90.46 / 90.66 tuning accuracy,
with nearly the same memory (927 / 929 / 935 MB) and time (217 / 210 / 214 s).
Supplementary head-only (seed 42): 1e-3 gave 56.17 and 3e-3 gave 71.91, still rising at epoch 5.

## Week 4: final test evaluation (run once) and analysis

The analysis code below was committed **before** the test set was used. The eval script
refuses to run on uncommitted code and refuses to overwrite earlier test predictions.

```bash
python scripts/eval_test_banking77.py --ckpt-roots <week2 checkpoints> <week3 checkpoints>
python scripts/analyze_banking77.py
```

- `scripts/eval_test_banking77.py`: rebuilds each of the 13 locked runs (E1 9, E3 2,
  supplementary 2), loads its best checkpoint, re-scores the tuning set to prove the right
  weights loaded (must match the training log within 0.15 points), then scores the test set
  once and saves every prediction to `results/test/preds/`.
- `scripts/analyze_banking77.py`: per-seed scores, mean and sample std, cost columns,
  LoRA minus full FT with a 95% hierarchical bootstrap (10,000 resamples over matched seeds
  and test examples), the one-point decision rule (Section 9.1), the rank table,
  per-intent F1, the most confused intent pairs, and two figures. Output: `results/analysis/`.
- `src/lora_budget/analysis/stats.py`: macro-F1, the bootstrap, and the decision rule.
- Notebook: `notebooks/week4_banking77_test_eval.ipynb` (attach the Week 2 and Week 3
  Kaggle notebook outputs as inputs; that is where the checkpoints are).
