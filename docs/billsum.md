# BillSum / Qwen2.5-0.5B — Jainish Patel

Scope: proposal Weeks 1–2, E2 preprocessing and decoder compatibility pilots.
BillSum LR selection belongs to Week 3 and is not implemented or claimed here.

## What is verified

The real U.S. Parquet release was downloaded, schema/row counts and all text and
summary fields checked, and SHA-256-pinned manifests generated with subset seed
2026. California is excluded. Counts are 3,000 training, 500 tuning, 500 final
U.S. test examples. No cross-split normalized-text duplicates occurred; one
within-training repeat is retained. Membership and input hashes are committed;
raw bill texts and model weights are not.

The model and dataset revisions are pinned. Training labels mask the instruction,
source, separator, and padding; EOS remains supervised even when EOS is used as
padding. The prompt budget is 1,024 tokens including instruction and separator;
the summary budget is 256 including EOS. Full references remain intact. All
methods use the beginning of each bill and identical seed-driven example order.

## What remains before Weeks 1–2 are experimentally complete

Run all four CUDA pilots on the same T4-class GPU. QLoRA NF4 kernels, actual GPU
precision, memory fit (including full fine-tuning), and GPU reload tolerance
have **not** been validated by the local CPU tests. No GPU metrics or winning LR
are supplied. Only after these pilots pass should both teammates agree on a
shared environment lock and freeze the resolved configs. `requirements-billsum.txt`
is an isolated, pinned **candidate**, not a claim that Kunj's newer library stack
has been validated for the decoder. Kunj's requirements remain untouched.

## Reproduce

Use a clean Python 3.12 environment. On Kaggle/Colab select one GPU and enable
internet. Run from the repository root:

```bash
python -m pip install -r requirements-billsum.txt
python -m pip install -e . --no-deps
python scripts/make_billsum_manifest.py --verify-existing
python -m pytest -q tests/test_billsum.py
python scripts/audit_billsum.py
python scripts/pilot_billsum.py --config configs/billsum/full_ft.yaml
python scripts/pilot_billsum.py --config configs/billsum/lora.yaml
python scripts/pilot_billsum.py --config configs/billsum/dora.yaml
python scripts/pilot_billsum.py --config configs/billsum/qlora.yaml
```

Do not regenerate manifests over existing ones. `--verify-existing` restores the
raw files from the locked release and validates their checksums. To independently
reproduce membership, use `--revision` from `manifests/billsum/meta.json` and a
new `--out-dir`, then compare its CSV checksums to the committed ones.

Each pilot requires at least 20 **successful** optimizer steps. AMP overflow
skips do not count. It checks finite loss, finite/nonzero trainable gradients,
frozen parameter integrity, proposal adapter parameter counts, and saved/reloaded
logits on a fixed tuning batch (absolute tolerance 0.001). It releases the original
model and optimizer before reload to avoid measuring two models' memory as the
single-model requirement. The pilot never loads the final test subset for scoring.

Runs write unique JSON reports and an append-only ledger under `results/billsum/`.
Reports include the resolved config, manifest hash, package freeze, GPU/CUDA,
parameter/optimizer dtypes, load time, memory peaks for loading, forward/backward,
first optimizer step and steady state, per-step times, token throughput, and
reload error. Memory fields mean PyTorch allocated/reserved bytes, not total
process/device memory. Broader inference/NVML profiling is later E4 work.
Failures/OOMs raise a nonzero exit code and remain in the ledger. Keep the reports
from failed attempts. Checkpoints stay ignored in `checkpoints/billsum/`.
These short pilots save inference checkpoints; they are not resumable main
training runs and must not be counted as confirmation results.

Full fine-tuning uses FP32 master weights and AdamW states under FP16 autocast and
dynamic loss scaling. LoRA/DoRA use FP16 frozen backbones and FP32 trainable adapter
weights. QLoRA uses NF4, double quantization, FP16 computation, and ordinary AdamW;
PEFT may retain some nonquantized tensors in FP32, which the audit records.
All methods enable gradient checkpointing and disable training cache. There is no
CPU offloading. This avoids GradScaler's unsupported FP16-gradient unscale path.

If full fine-tuning OOMs, preserve that report. Any shorter source limit requires
a documented common change for **all four methods**, followed by rerunning all
pilots. Never silently shorten only one method. Archive successful reports and
`pip_freeze`, review the four configs jointly, then begin Week 3 LR selection.

## Existing teammate work reviewed

At base commit `3ebdd001b36ebe8ad4b670534f89d62465e25182`, Kunj's Week 1 includes
BANKING77 manifests, RoBERTa model builders, parameter audit, tests, and T4 pilot
reports (including a retained failed LoRA attempt and its successful retry).
His `kunj/week2-lr-selection` branch additionally contains six seed-42 LR trials,
selection code, and selected LRs: full FT 3e-5, head-only 3e-4, LoRA 3e-4.
That branch is not merged into main at this review. His later Week 3 branch exists
but is outside this change's scope. No BANKING77 files or results are changed.

A cell-by-cell GPU entry point is `notebooks/week1_2_billsum_pilots.ipynb`.
`results/billsum/cpu_audit.json` records the real model tensor audit: 494,032,768
full-FT trainable parameters, 4,399,104 LoRA parameters, and 4,703,232 DoRA
parameters, matching the proposal's adapter counts. These are CPU measurements,
not GPU fit evidence. Its preprocessing records contain full source/target length
distributions and truncation counts for training and tuning only.

## Local validation

On November 6, 2026, the full local suite passed: **31 tests**, no skips. This
includes the 17 existing BANKING77 tests and 14 new BillSum tests. After normalizing
CSV line endings, all 14 BillSum tests passed again. The real raw files and manifest
checksums were reverified. These checks do not exercise CUDA or four-bit kernels.
