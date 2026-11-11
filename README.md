# lora-on-a-budget

Controlled comparison of full fine-tuning, LoRA, DoRA, and QLoRA
on a single 16 GB GPU.

## Task guides

- [BANKING77 / RoBERTa — Kunj](docs/banking77.md)
- [BillSum / Qwen — Jainish](docs/billsum.md)
- [BillSum Week 1 verification report](docs/billsum_week1.md)

## BillSum progress

Week 1 preparation checks and the CPU parameter audit were verified
in Kaggle. All 14 BillSum tests passed. The report links to the notebook,
results, environment details, and failure/fix logs.

The four BillSum GPU training compatibility checks and the final
shared environment lock remain pending.
