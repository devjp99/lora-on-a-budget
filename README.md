# lora-on-a-budget

Controlled comparison of full fine-tuning, LoRA, DoRA, and QLoRA
on a single 16 GB GPU.

## Task guides

- [BANKING77 / RoBERTa — Kunj](docs/banking77.md)
- [BillSum / Qwen — Jainish](docs/billsum.md)
- [BillSum Week 1 verification report](docs/billsum_week1.md)

## BANKING77 results (complete)

Test accuracy over 3 seeds on one T4: full fine-tuning 93.01, LoRA r=8 91.56,
head-only 21.41. LoRA is 1.45 points behind (95% interval [−2.13, −0.79]) while
using 2.6× less peak memory and a 129× smaller checkpoint, but it trains only
about 1.2× faster. Details: [docs/banking77.md](docs/banking77.md).

## BillSum progress

Week 1 preparation checks and the CPU parameter audit were verified
in Kaggle. All 14 BillSum tests passed. The report links to the notebook,
results, environment details, and failure/fix logs.

The four BillSum GPU training compatibility checks and the final
shared environment lock remain pending.

## License

Project code: MIT (see [LICENSE](LICENSE)). Datasets, models, and libraries keep
their own licenses: see [BANKING77 notices](docs/banking77-third-party-notices.md)
and [BillSum notices](docs/billsum-third-party-notices.md).
