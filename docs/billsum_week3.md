# BillSum Week 3: learning-rate selection

Status: implementation prepared; no Week 3 GPU results or selected learning rates yet.
The existing Week 2 reports remain evidence of the four short compatibility checks.
This implementation must pass fresh Kaggle debug and resume checks before full trials.

## Fixed experiment

This follows the revised final proposal, Sections 7–9 and Table 7.
Four methods, two LRs each, seed 42: eight three-epoch trials.
Full FT uses 1e-5 and 3e-5; LoRA/DoRA/QLoRA use 1e-4 and 3e-4.
Use the existing 3000/500 training/tuning manifests, model revision, 1024/256
token limits, seven adapter targets, rank 8, alpha 16, dropout 0.05, and T4.
Keep the final 500-example test subset unused during tuning.

The trainer processes every training example once per epoch. Each epoch has
187 groups of 16 and one final group of 8, so a full trial has 564 updates.
Within a group, loss is weighted by supervised target tokens. AMP overflows
retry the same group with a smaller scale and the same dropout RNG; they do
not advance the scheduler or data cursor. Eight consecutive failures abort.
These implementation choices are declared before LR search, and apply to all methods.

After each epoch, generate on all 500 tuning examples with prompt tokens only,
greedy decoding and at most 256 new tokens. Score against the full original
references. Use rouge-score 0.1.2, stemming enabled, its default tokenizer,
and mean per-example F1 multiplied by 100. Primary metric: ROUGE-L, not ROUGE-Lsum.
ROUGE-1/2 and generated lengths are secondary. Keep the best epoch by unrounded
ROUGE-L; ties keep the earlier epoch. Across LRs: higher score, earlier epoch,
then smaller LR. The selector refuses missing, duplicate or incomparable trials.

## Environment and prerequisite review

Use requirements-billsum-week3.txt on Python 3.13 with torch 2.11.0+cu130.
The runner rejects different core versions, torchao, multiple visible GPUs,
and non-T4 hardware. It also records transitive metric dependencies and the
full package freeze. Compare all eight trials under one recorded environment.
This preserves the tested BillSum stack; shared-environment agreement with Kunj
is still pending. Do not silently replace his requirements or rerun his experiments.

The proposal listed Hugging Face Evaluate as a planned metric wrapper. This runner
calls Google's rouge-score directly to obtain the specified per-example means;
record this implementation choice in the final methods section. Sources:
https://github.com/google-research/google-research/tree/master/rouge
https://github.com/huggingface/evaluate/blob/main/metrics/rouge/rouge.py

## First execution: debug only

Use notebooks/week3_billsum_lr_selection.ipynb. Run cells individually.
First run the unit tests and reverify existing manifests.
Run a debug trial for each method with its smaller candidate LR. Debug uses
32 training examples, two tuning examples and one epoch; it saves/reloads the
best checkpoint but cannot enter the LR selection. Keep these reports.
For one method, stop after update 1 and resume using the SAME run ID:

```bash
python scripts/train_billsum.py --config configs/billsum/lora.yaml --lr 1e-4 --phase debug --run-id debug-lora-resume --stop-after-updates 1
# Expected interrupted status. Then:
python scripts/train_billsum.py --config configs/billsum/lora.yaml --lr 1e-4 --phase debug --run-id debug-lora-resume --resume
```

Check the completed debug report and reload tolerance. Do not claim these are
full-trial results. The new trainer has not yet been validated on a T4.

## Full trials after review

```bash
python scripts/train_billsum.py --config configs/billsum/full_ft.yaml --lr 1e-5 --phase lr_search --run-id lr-full-ft-1e-5-s42
```

Repeat for the eight method/LR pairs, each with a distinct run ID. Use
--resume with the same command and ID only after an interruption. Never
change the LR, seed, data, code or environment in the middle of a run.
Completed IDs cannot be overwritten. The default phase is debug intentionally.

After eight completed trials:

```bash
python scripts/select_lr_billsum.py
```

Seed-42 selected runs are the first confirmation results. Seeds 43/44 come
later with --phase confirm and the saved selection; do not use test results
to change the settings or learning rates.

## Checkpoints and session persistence

Each run writes checkpoints/billsum/training/RUN_ID/last.pt every 25 updates
and at epoch boundaries. The snapshot contains trainable weights, optimizer,
scheduler, AMP scaler, RNG and data cursor. Atomic replacement protects the
last complete snapshot if saving is interrupted. Resume can replay work since
the last saved snapshot. Attempt reports and an append-only ledger preserve
failures/restarts. Hard session termination may prevent a final status write;
the last checkpoint remains the recovery point if it was persisted externally.

Best inference files and the reload probe are in best_epoch_N folders under
the same run. Preserve the ENTIRE run folder when moving sessions, along with
results/billsum/training. last.pt alone is insufficient if an earlier best
epoch has already been selected. The final check releases the original model
and optimizer before reloading the best inference checkpoint.

IMPORTANT: a file under /kaggle/working is not a durable backup by itself.
Save notebook outputs and copy/download checkpoints before ending the session.
A ZIP of JSON reports does not contain the weights needed to resume training.
Full-FT resume snapshots are several GB; budget disk space for atomic saves
and best checkpoints. Keep model weights out of GitHub. Upload reports, logs,
package freeze and the notebook with saved outputs to GitHub.

## Runtime budget and measurement scope

Using the latest Week 2 steady step times, 564 updates suggests roughly
2.25 hours full FT, 2.21 LoRA, 3.81 DoRA and 2.53 QLoRA per trial.
Eight LR trials therefore suggest about 21.6 GPU hours for training alone.
These are preliminary extrapolations, not runtime guarantees. Generation on
500 tuning examples after each epoch, reload checks, saving and retries add
time. Measure validation time on the debug runs and first full epoch, then
review the remaining budget before confirmation. Do not silently cut core runs.

The runner records training/evaluation time separately and per-attempt peak
PyTorch allocated/reserved memory. Checkpoint saving and frozen-hash checks
are not included in train_seconds. It is not the full E4 profiling tool: total
device/process memory, deployment storage, fixed-workload inference latency,
merged/unmerged equivalence and final quality analysis still remain separate.
No new final-test evaluation is implemented in this Week 3 package.
