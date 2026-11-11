# BillSum / Qwen2.5-0.5B — Jainish Patel

Scope: proposal Weeks 1–2, E2 preprocessing and decoder compatibility pilots.
BillSum learning-rate selection belongs to Week 3 and remains pending.

## Week 1 verification — November 11, 2026

The BillSum preparation checks and CPU parameter audit were rerun in Kaggle.
All 14 BillSum tests passed.

- [Week 1 report](billsum_week1.md)
- [Week 1 notebook with saved outputs](../notebooks/jainish-week-1-billsum.ipynb)
- [Week 1 evidence and logs](../results/billsum/week1/)

The report records the environment used, the torchao compatibility fix,
parameter counts, and truncation statistics.

The first audit failed because torchao 0.10.0 was incompatible with PEFT.
After removing torchao, the CPU audit passed. Both attempts and the removal
logs are preserved.

T4 access was checked separately, with one GPU visible to PyTorch.
The parameter audit itself ran on the CPU. No model training or final
test-set scoring was performed during this verification.

## What is verified

The U.S. Parquet release was downloaded, its schema and row counts checked,
and the existing SHA-256-pinned manifests verified. The subset seed is 2026.
California is excluded.

The selected groups contain:

- 3,000 training examples.
- 500 tuning examples.
- 500 final U.S. test examples.

No shared IDs or normalized-text fingerprints were found across the groups.
One repeated text fingerprint remains within the training group.
Membership records and input hashes are committed; raw bill texts and model
weights are not.

The model and dataset revisions are pinned. Training labels mask the
instruction, source, separator, and padding. EOS remains supervised even
when EOS is used as padding.

The prompt budget is 1,024 tokens including the instruction and separator.
The summary budget is 256 tokens including EOS. Full reference summaries
remain stored separately. The preprocessing keeps the beginning of each
bill, and the pilot code uses a consistent seed-driven example order.

The November 11 audit recorded:

| Group | Examples checked | Bill inputs shortened | Targets shortened |
|---|---:|---:|---:|
| Training | 3,000 | 2,953 | 1,099 |
| Tuning | 500 | 488 | 164 |

Shortening may remove important information. These counts show how many
examples exceeded the limits, not how much text was removed.

The CPU parameter audit recorded:

| Method | Trainable parameters |
|---|---:|
| Full fine-tuning | 494,032,768 |
| LoRA | 4,399,104 |
| DoRA | 4,703,232 |

These are trainable parameter counts, not measurements of GPU memory
usage or summary quality.

## Pending GPU checks

Run all four CUDA pilots on the same single T4 GPU. QLoRA NF4 kernels,
actual GPU precision, memory fit including full fine-tuning, and GPU reload
tolerance remain unverified by these CPU checks.

Only after the GPU checks pass should both teammates agree on a shared
environment lock and freeze the resolved configurations.

`requirements-billsum.txt` is an earlier pinned candidate environment.
The November 11 Kaggle environment is recorded in the Week 1 report.
Neither should be described as validated for all four GPU training methods
based on the CPU audit alone.

Kunj's requirements remain unchanged.

The existing `notebooks/week1_2_billsum_pilots.ipynb` is the earlier combined
runner. Use the dedicated Week 1 notebook linked above to review the
completed preparation checks. A separate Week 2 notebook will be prepared
for the GPU checks.

## Earlier candidate reproduction instructions

The commands below describe the earlier Python 3.12 candidate setup.
They are not the installation procedure used for the November 11 Kaggle
verification, which used Python 3.13.16 and the package versions recorded
in the Week 1 report.

For the candidate setup, use a clean Python 3.12 environment. On Kaggle or
Colab, select one GPU and enable internet. Run from the repository root.

Preparation and CPU checks:

```bash
python -m pip install -r requirements-billsum.txt
python -m pip install -e . --no-deps
python scripts/make_billsum_manifest.py --verify-existing
python -m pytest -q tests/test_billsum.py
python scripts/audit_billsum.py
```

GPU checks, still pending:

```bash
python scripts/pilot_billsum.py --config configs/billsum/full_ft.yaml
python scripts/pilot_billsum.py --config configs/billsum/lora.yaml
python scripts/pilot_billsum.py --config configs/billsum/dora.yaml
python scripts/pilot_billsum.py --config configs/billsum/qlora.yaml
```

Do not regenerate manifests over existing ones. `--verify-existing`
restores the raw files from the locked release and validates their checksums.

To independently reproduce membership, use the revision recorded in
`manifests/billsum/meta.json` with `--revision` and a new `--out-dir`,
then compare the CSV checksums with the committed ones.

## GPU pilot checks and outputs

Each pilot requires at least 20 successful optimizer steps.
AMP overflow skips do not count.

The pilot checks:

- Finite loss.
- Finite, nonzero trainable gradients.
- Frozen parameter integrity.
- Expected adapter parameter counts.
- Saved/reloaded logits on a fixed tuning batch, with absolute tolerance 0.001.

The original model and optimizer are released before reload to avoid
measuring two models' memory as the single-model requirement.
The pilot never loads the final test subset for scoring.

Runs write unique JSON reports and an append-only ledger under
`results/billsum/`. Reports include the resolved configuration, manifest
hash, package freeze, GPU/CUDA details, parameter and optimizer dtypes,
load time, memory peaks, per-step times, token throughput, and reload error.

Memory fields record PyTorch allocated and reserved bytes, not total
process or device memory. Broader inference and NVML profiling belongs
to later E4 work.

Failures and out-of-memory errors raise a nonzero exit code and remain
in the ledger. Preserve failed reports. Checkpoints stay ignored in
`checkpoints/billsum/`.

These short pilots save inference checkpoints. They are not resumable
main training runs and must not be counted as confirmation results.

## Training configuration implemented for the pilots

Full fine-tuning uses FP32 master weights and AdamW states under FP16
autocast with dynamic loss scaling.

LoRA and DoRA use FP16 frozen backbones and FP32 trainable adapter weights.

QLoRA uses NF4, double quantization, FP16 computation, and ordinary AdamW.
PEFT may retain some nonquantized tensors in FP32; the pilot records
the resulting dtypes.

All methods enable gradient checkpointing and disable the training cache.
The configuration does not use CPU offloading. FP32 trainable weights
avoid GradScaler's unsupported FP16-gradient unscale path.

These describe the implemented configuration, not completed GPU validation.

If full fine-tuning runs out of memory, preserve that report. A shorter
source limit requires a documented common change for all four methods,
followed by rerunning all pilots. Do not shorten only one method.

Archive successful reports and package freezes, review the four
configurations jointly, then begin Week 3 learning-rate selection.

## Historical teammate-work review

This section records the earlier review at base commit
`3ebdd001b36ebe8ad4b670534f89d62465e25182`.
It does not describe the current merge status of teammate branches.

At that review, Kunj's Week 1 work included BANKING77 manifests, RoBERTa
model builders, a parameter audit, tests, and T4 pilot reports, including
a retained failed LoRA attempt and its successful retry.

His `kunj/week2-lr-selection` branch contained six seed-42 learning-rate
trials, selection code, and selected learning rates:

- Full fine-tuning: 3e-5.
- Head-only: 3e-4.
- LoRA: 3e-4.

That branch had not been merged into main at the time of the earlier review.
His Week 3 branch was outside that review's scope.

The November 11 BillSum verification update does not change BANKING77
code or results.

## Earlier CPU audit and local validation

`results/billsum/cpu_audit.json` contains the earlier CPU model audit.
Its trainable parameter counts match those observed in the November 11 run.

The new run is stored separately under
`results/billsum/week1/20261111_201331/`.
The `previous_cpu_audit.json` files are backups of the earlier report,
not results produced by the failed attempt.

The earlier validation record states that on November 6, 2026,
the full local suite passed 31 tests with no skips: 17 BANKING77 tests
and 14 BillSum tests. After CSV line-ending normalization, all 14 BillSum
tests passed again, and the raw files and manifest checksums were reverified.

The November 11 verification reran the 14 BillSum tests only.
These checks do not exercise CUDA training or four-bit kernels.
