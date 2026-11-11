# BillSum — Week 1 preparation and verification

## Purpose

Prepare and verify the BillSum data and Qwen2.5-0.5B model setup
before running training experiments.

## Data checks

Verified the source files and existing manifests for the pinned
FiscalNote/billsum dataset revision.

- Training: 3,000 examples.
- Tuning: 500 examples.
- Test: 500 examples.
- Shared IDs across splits: 0.
- Shared text fingerprints across splits: 0.
- One repeated text fingerprint remains within the training split.

The test manifest was checked, but no model was scored on the test set.

## Preprocessing

Each prompt contains an instruction, the beginning of the bill,
and a "Summary:" separator.

The prompt limit is 1,024 tokens, including the instruction and separator.
The target limit is 256 tokens, including the end-of-sequence token.

Prompt positions are excluded from the training loss. Summary tokens
and the end-of-sequence token are supervised. Full reference summaries
remain stored separately.

| Group | Examples checked | Bill inputs shortened | Targets shortened |
|---|---:|---:|---:|
| Training | 3,000 | 2,953 | 1,099 |
| Tuning | 500 | 488 | 164 |

Shortening affects most bill inputs and may remove important information.
These counts show how many examples were shortened, not how much text
was removed from each example.

## CPU parameter audit

| Method | Trainable parameters |
|---|---:|
| Full fine-tuning | 494,032,768 |
| LoRA | 4,399,104 |
| DoRA | 4,703,232 |

The audit loaded the models and checked parameter counts without training.
These counts do not establish GPU memory usage or summary quality.

## Failure and fix

The first audit failed while adding LoRA because the installed
torchao 0.10.0 was incompatible with PEFT.

After removing torchao, the CPU audit completed successfully.
Both attempts and the removal logs are preserved.

## Validation and environment

All 14 BillSum tests passed.

The successful run used Python 3.13.16, PyTorch 2.11.0+cu130,
Transformers 5.20.2, PEFT 0.21.2, pandas 2.3.3,
PyArrow 23.0.1, and pytest 8.4.2.

The tested repository commit was:
64317e35ecd2ed2eac4ff98aafabb78c129a699d

T4 access was checked separately, with one GPU visible to PyTorch.
The parameter audit itself ran on the CPU.

This records the environment used for these checks; it is not a
validated shared environment lock for GPU training.

## Files

- [Week 1 notebook](../notebooks/jainish-week-1-billsum.ipynb)
- [Failed audit and fix](../results/billsum/week1/20261111_200530/)
- [Successful audit and tests](../results/billsum/week1/20261111_201331/)

The previous_cpu_audit.json files are backups of the earlier report,
not results produced by the failed attempt.

## Pending work

GPU training compatibility checks for full fine-tuning, LoRA, DoRA,
and 4-bit QLoRA remain pending. No training-quality results, GPU memory
comparisons, or final test scores are claimed here.

The notebook includes a setup fix added after the successful run.
The edited notebook has not yet been rerun from beginning to end
in a fresh session.
