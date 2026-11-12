# BillSum / Qwen2.5-0.5B — Jainish Patel

Scope: Weeks 1–2 preparation and GPU compatibility checks for BillSum.
Week 3 learning-rate selection remains pending.

## Reports and notebooks

- [Week 1 report](billsum_week1.md)
- [Week 1 notebook](../notebooks/jainish-week-1-billsum.ipynb)
- [Week 1 evidence](../results/billsum/week1/)
- [Week 2 report](billsum_week2.md)
- [Week 2 notebook](../notebooks/week2_billsum_pilots.ipynb)
- [Week 2 logs and setup records](../results/billsum/week2/)

## Week 1 — preparation and CPU verification

The BillSum preparation checks and CPU parameter audit were rerun
in Kaggle on November 11, 2026. All 14 BillSum tests passed.

The first CPU audit failed because torchao 0.10.0 was incompatible
with PEFT. After removing torchao, the audit passed. Both attempts
and their available logs are preserved in the Week 1 evidence.

T4 access was checked separately, with one GPU visible to PyTorch.
The parameter audit itself ran on the CPU. No model training or
final test-set scoring was performed during Week 1 verification.

### Dataset checks

The U.S. Parquet release was downloaded, its schema and row counts
checked, and the existing SHA-256-pinned manifests verified.

The subset seed is 2026. California is excluded.

- Training: 3,000 examples.
- Tuning: 500 examples.
- Final U.S. test: 500 examples.

No shared IDs or normalized-text fingerprints were found across
the splits. One repeated text fingerprint remains within training.

Membership records and input hashes are committed. Raw bill texts
and model weights are not committed.

### Preprocessing

The model and dataset revisions are pinned.

Each prompt contains an instruction, the beginning of the bill,
and a "Summary:" separator.

The prompt limit is 1,024 tokens, including the instruction and
separator. The target limit is 256 tokens, including EOS.

Training labels exclude the instruction, bill text, separator,
and padding from the loss. Actual summary and EOS tokens remain
supervised. Full reference summaries remain stored separately.

The pilot code uses a consistent seed-driven example order.
Skipped optimizer attempts can nevertheless affect which examples
correspond to each successful update.

| Split | Examples checked | Bill inputs shortened | Targets shortened |
|---|---:|---:|---:|
| Training | 3,000 | 2,953 | 1,099 |
| Tuning | 500 | 488 | 164 |

Shortening may remove important information. These counts show
how many examples exceeded the limits, not how much text was removed.

### CPU parameter audit

| Method | Trainable parameters |
|---|---:|
| Full fine-tuning | 494,032,768 |
| LoRA | 4,399,104 |
| DoRA | 4,703,232 |

These are parameter counts, not GPU memory or summary-quality
measurements.

## Week 2 — GPU checks completed

Full fine-tuning, LoRA, DoRA, and 4-bit QLoRA each passed a short
training check on one Tesla T4 GPU in the recorded environment.

Each run completed 20 successful optimizer updates and passed
its checkpoint save/reload check.

| Method | Updates | Steady allocated memory (GiB) | Steady reserved memory (GiB) | Mean steady step time (seconds) | Reload maximum absolute difference |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | 20 | 10.27 | 11.62 | 14.39 | 0.0 |
| LoRA | 20 | 3.24 | 3.87 | 14.10 | 0.0 |
| DoRA | 20 | 3.25 | 3.92 | 24.33 | 0.0 |
| QLoRA | 20 | 3.30 | 3.92 | 16.15 | 0.0 |

The reload tolerance was 0.001.

Memory values record PyTorch allocated and reserved memory,
not total device or process memory. Timings are from single short
runs and are not repeated benchmark estimates.

QLoRA used slightly more recorded allocated memory than LoRA
in this setup. Four-bit weight storage does not guarantee the
lowest total training memory.

These runs establish compatibility for the tested configuration.
They do not establish summary quality, select a learning rate,
or replace longer training experiments.

### Latest reports

- [Full fine-tuning](../results/billsum/pilots/billsum-full_ft-4114c72ac3a7.json)
- [LoRA](../results/billsum/pilots/billsum-lora-1564994f1226.json)
- [DoRA](../results/billsum/pilots/billsum-dora-5ccc88dbe43a.json)
- [QLoRA](../results/billsum/pilots/billsum-qlora-b6bcc21e3505.json)

### QLoRA failure and fix

The first session's QLoRA attempt completed 20 updates but failed
the frozen-parameter integrity check.

A diagnostic forward pass showed that bitsandbytes converted an
inspected frozen bias from FP32 to FP16 without an optimizer update.
Its numerical values were unchanged, but its byte representation changed.

The model builder now converts frozen Linear4bit biases to their
compute dtype before integrity fingerprints are taken. It rejects
any conversion that changes numerical values.

Frozen-parameter checks remain enabled. The corrected QLoRA run
passed training and checkpoint reload checks.

### Session history

The original session ended before its report files were downloaded.
Available earlier outputs and debugging history remain in the notebook.

The four checks were rerun to regenerate the report files.
The latest reports and logs were downloaded together.
Earlier outputs are not counted as additional experimental repetitions.

The Week 2 notebook contains earlier diagnostics and later reruns.
It is an execution record, not yet a clean uninterrupted Run All workflow.

## Tested environment

The latest four GPU checks used:

- Python 3.13.16.
- PyTorch 2.11.0+cu130.
- Transformers 5.20.2.
- PEFT 0.21.2.
- Accelerate 1.15.0.
- bitsandbytes 0.50.2.

The incompatible optional torchao 0.10.0 package was removed.
All 14 BillSum tests passed in the restored environment, and the
existing manifests were verified again.

The runs used base commit 483f0fe with the local QLoRA bias fix.

- [Exact base commit](../results/billsum/week2/setup/git_commit.txt)
- [Code diff](../results/billsum/week2/setup/git_diff.patch)
- [Package freeze](../results/billsum/week2/setup/pip_freeze.txt)
- [BillSum test output](../results/billsum/week2/setup/billsum_tests.txt)

`requirements-billsum.txt` still describes the earlier Python 3.12
candidate environment. It is not the environment used for these GPU runs.

The package freeze records the observed environment; it is not yet
an agreed shared installation lock. Review the environment and resolved
configurations with Kunj before longer experiments.

## Commands used for verification

Run from the repository root after preparing the tested environment
and including the QLoRA fix. Select one GPU before importing PyTorch.

Preparation and tests:

```bash
python -m pip install -e . --no-deps
python scripts/make_billsum_manifest.py --verify-existing
python -m pytest -q tests/test_billsum.py
```

CPU audit, when needed:

```bash
python scripts/audit_billsum.py
```

The four GPU checks:

```bash
python scripts/pilot_billsum.py --config configs/billsum/full_ft.yaml --steps 20 --seed 42
python scripts/pilot_billsum.py --config configs/billsum/lora.yaml --steps 20 --seed 42
python scripts/pilot_billsum.py --config configs/billsum/dora.yaml --steps 20 --seed 42
python scripts/pilot_billsum.py --config configs/billsum/qlora.yaml --steps 20 --seed 42
```

These commands are not a complete fresh-environment installation recipe.
Use the environment records and resolve dependencies before execution.

Do not regenerate manifests over existing ones. `--verify-existing`
restores the locked raw release and verifies the existing manifests.

To independently reproduce membership, use the revision recorded in
`manifests/billsum/meta.json` with `--revision` and a new `--out-dir`,
then compare the CSV checksums with the committed ones.

## What the GPU checks do

Each pilot requires at least 20 successful optimizer steps.
AMP overflow skips do not count.

The checks cover:

- Finite loss.
- Finite, nonzero trainable gradients.
- Frozen-parameter integrity.
- Expected adapter parameter counts.
- Saved/reloaded logits on a fixed tuning batch.

The original model and optimizer are released before reload.
The pilot does not score the final test subset.

Runs write unique JSON reports and an append-only ledger under
`results/billsum/`. Reports record configuration, data provenance,
environment details, dtypes, memory measurements, timing, and reload error.

Preserve reports and logs from failures as well as successes.
Download evidence before ending a session.

Checkpoints stay ignored in `checkpoints/billsum/`. The uploaded
evidence does not include model checkpoints or raw datasets.

These short pilots save inference checkpoints, not complete
resumable training states. They are not confirmation experiments.
Broader inference and NVML profiling belongs to later E4 work.

## Training configuration

The pilots use a micro-batch size of 1, gradient accumulation of 16,
and seed 42.

They use the first candidate learning rate in each configuration:
1e-5 for full fine-tuning and 1e-4 for the adapter methods.
They do not run all configured epochs, seeds, or learning rates.

Full fine-tuning uses FP32 trainable weights and AdamW states,
with FP16 autocast and dynamic loss scaling.

LoRA and DoRA use FP16 frozen backbones and FP32 trainable adapters.

QLoRA uses NF4, double quantization, FP16 computation, and ordinary
AdamW. Some nonquantized tensors remain FP32; actual dtypes are
recorded by the pilot.

All methods enable gradient checkpointing and disable the training
cache. CPU offloading is not used.

Full fine-tuning fit within the tested GPU configuration.
If future settings cause an out-of-memory error, preserve that report.
Any shorter source limit used for comparison must be documented and
applied consistently across all four methods.

## Remaining work

- Agree on the shared environment and resolved configurations with Kunj.
- Prepare a clean execution sequence for future training.
- Begin Week 3 learning-rate selection.
- Run longer experiments and evaluate summary quality separately.

## Historical teammate-work review

The earlier review used base commit
`3ebdd001b36ebe8ad4b670534f89d62465e25182`.
This section does not describe the current status of teammate branches.

At that review, Kunj's Week 1 work included BANKING77 manifests,
RoBERTa builders, a parameter audit, tests, and T4 pilot reports,
including a failed LoRA attempt and successful retry.

His `kunj/week2-lr-selection` branch contained six seed-42 trials,
selection code, and these selected learning rates:

- Full fine-tuning: 3e-5.
- Head-only: 3e-4.
- LoRA: 3e-4.

That branch had not been merged at the time of the earlier review.
His Week 3 branch was outside that review's scope.

This BillSum update does not change BANKING77 code or results.

## Earlier CPU audit and local validation

`results/billsum/cpu_audit.json` contains the earlier CPU audit.
Its parameter counts match the November 11 Week 1 audit.

The Week 1 rerun is stored under
`results/billsum/week1/20261111_201331/`.

The `previous_cpu_audit.json` files are backups of an earlier report,
not results produced by the failed attempt.

The earlier validation record states that on November 6, 2026,
31 local tests passed with no skips: 17 BANKING77 and 14 BillSum.
After CSV line-ending normalization, the 14 BillSum tests passed again,
and raw-file and manifest checksums were reverified.

The subsequent Week 1 and Week 2 checks reran the 14 BillSum tests.
Those tests alone do not exercise CUDA training or four-bit kernels;
the separate Week 2 GPU reports provide that execution evidence.
