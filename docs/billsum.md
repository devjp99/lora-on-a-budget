# BillSum / Qwen2.5-0.5B

## Week 1: data and model setup

- Fixed U.S. subsets: 3,000 training, 500 tuning, and 500 test examples, selected with seed 2026.
- Raw-file and manifest checksums recorded; no normalized-text overlap across the selected splits.
- Prompt and padding labels masked; summary tokens and EOS supervised.
- Full fine-tuning, LoRA, DoRA, and QLoRA configurations use the proposal settings.
- CPU audit of the real model: 494,032,768 full-FT, 4,399,104 LoRA, and 4,703,232 DoRA trainable parameters.
- Fourteen BillSum tests cover data integrity, token formatting, gradients, and checkpoint reloads.

## Reproduce

```bash
python -m pip install -r requirements-billsum.txt
python -m pip install -e . --no-deps
python scripts/make_billsum_manifest.py --verify-existing
python -m pytest -q tests/test_billsum.py
python scripts/audit_billsum.py
```

The dependency pins are a candidate decoder environment. GPU memory fit, FP16 execution, QLoRA kernels, and the final shared environment lock still need compatibility runs. CPU checks do not complete these GPU requirements.

See `docs/billsum-third-party-notices.md` for component attribution and `results/billsum/cpu_audit.json` for the CPU audit and preprocessing distributions.
