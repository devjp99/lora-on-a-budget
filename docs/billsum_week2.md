# BillSum — Week 2 GPU checks

## Purpose

Check whether full fine-tuning, LoRA, DoRA, and 4-bit QLoRA
can train and save/reload correctly on one Tesla T4 GPU.

These are short compatibility checks, not full training experiments
or learning-rate selection. No final test-set scoring was performed.

## Setup

- Model: Qwen/Qwen2.5-0.5B.
- One Tesla T4 visible to PyTorch.
- Training/tuning/test subsets: 3,000 / 500 / 500 examples.
- Prompt limit: 1,024 tokens.
- Target limit: 256 tokens, including EOS.
- Micro-batch size: 1; gradient accumulation: 16.
- Seed: 42.
- Successful optimizer updates: 20 per method.
- Pilot learning rate: 1e-5 for full fine-tuning;
  1e-4 for LoRA, DoRA, and QLoRA.

The pilot uses the first candidate learning rate from each configuration.
It does not run all configured epochs, seeds, or learning-rate candidates.

## Results

| Method | Passed | Updates | Steady allocated memory (GiB) | Steady reserved memory (GiB) | Mean steady step time (seconds) | Reload maximum absolute difference |
|---|---|---:|---:|---:|---:|---:|
| Full fine-tuning | Yes | 20 | 10.27 | 11.62 | 14.39 | 0.0 |
| LoRA | Yes | 20 | 3.24 | 3.87 | 14.10 | 0.0 |
| DoRA | Yes | 20 | 3.25 | 3.92 | 24.33 | 0.0 |
| QLoRA | Yes | 20 | 3.30 | 3.92 | 16.15 | 0.0 |

The reload tolerance was 0.001. All four runs passed.

Memory values are the recorded PyTorch allocated/reserved measurements,
converted to GiB. They do not represent total device or process memory.

These are single short runs. Timings are preliminary measurements,
not repeated benchmark estimates. Parameter precision differs by method.

QLoRA's recorded allocated memory was slightly higher than LoRA's
in this setup. Four-bit weight storage alone does not guarantee
the lowest total training memory.

## QLoRA failure and fix

The first session's QLoRA run completed 20 updates but failed the
frozen-parameter integrity check.

A diagnostic forward pass showed that bitsandbytes changed a frozen
bias from FP32 to FP16 without an optimizer update. For the inspected
bias, the numerical difference was zero, but its byte representation changed.

The model builder now converts frozen Linear4bit biases to their compute
dtype before integrity fingerprints are taken. The conversion is allowed
only when numerical values remain exactly unchanged.

Frozen-parameter integrity checks remain enabled. The corrected QLoRA
run passed training and save/reload checks.

## Validation and environment

All 14 BillSum tests passed. The existing dataset manifests were verified.

The rerun environment used:

- Python 3.13.16.
- PyTorch 2.11.0+cu130.
- Transformers 5.20.2.
- PEFT 0.21.2.
- Accelerate 1.15.0.
- bitsandbytes 0.50.2.

The incompatible optional torchao 0.10.0 package was removed.

The code was based on commit 483f0fe, with the local QLoRA bias fix.
The exact base commit, code diff, and package freeze are saved in
the setup records linked below.

This records the tested BillSum environment. Agreement on a shared
environment with Kunj remains pending.

## Session history

The original session ended before its report files were downloaded.
Its displayed outputs and debugging history remain in the notebook.

The four checks were rerun to regenerate the report files.
The latest JSON reports and logs were downloaded together.
Earlier outputs are not counted as additional experimental repetitions.

The notebook is an execution record containing earlier diagnostics
and later reruns. It is not yet arranged for uninterrupted Run All execution.

## Evidence

- [Week 2 notebook](../notebooks/week2_billsum_pilots.ipynb)
- [Full fine-tuning report](../results/billsum/pilots/billsum-full_ft-4114c72ac3a7.json)
- [LoRA report](../results/billsum/pilots/billsum-lora-1564994f1226.json)
- [DoRA report](../results/billsum/pilots/billsum-dora-5ccc88dbe43a.json)
- [QLoRA report](../results/billsum/pilots/billsum-qlora-b6bcc21e3505.json)
- [Training logs and setup records](../results/billsum/week2/)
- [Package freeze](../results/billsum/week2/setup/pip_freeze.txt)
- [Code diff](../results/billsum/week2/setup/git_diff.patch)

Model checkpoints and raw datasets are not included in these evidence uploads.

## Remaining work

- Agree on the shared environment and resolved configurations with Kunj.
- Prepare a clean execution sequence for future training.
- Proceed to learning-rate selection and longer training experiments.
- Measure summary quality separately; these checks do not establish it.
