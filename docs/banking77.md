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
