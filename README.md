# lora-on-a-budget

Controlled comparison of full fine-tuning, LoRA, DoRA, and QLoRA
on a single 16 GB GPU.

## Task guides

- [BANKING77 / RoBERTa — Kunj](docs/banking77.md)
- [BillSum / Qwen — Jainish](docs/billsum.md)
- [BillSum Week 1 verification report](docs/billsum_week1.md)
- [BillSum Week 2 GPU check report](docs/billsum_week2.md)

## BANKING77 results (complete)

Test accuracy over 3 seeds on one T4: full fine-tuning 93.01, LoRA r=8 91.56,
head-only 21.41. LoRA is 1.45 points behind (95% interval [−2.13, −0.79]) while
using 2.6× less peak memory and a 129× smaller checkpoint, but it trains only
about 1.2× faster. Details: [docs/banking77.md](docs/banking77.md).

## BillSum progress

Week 1 preparation checks and the CPU parameter audit were verified
in Kaggle. All 14 BillSum tests passed. The Week 1 report links to
the notebook, results, environment details, and failure/fix logs.

In Week 2, full fine-tuning, LoRA, DoRA, and 4-bit QLoRA each passed
20 successful optimizer updates on one Tesla T4 GPU. All four runs
passed checkpoint save/reload checks with a maximum absolute output
difference of 0.0.

The Week 2 report records memory and timing measurements, the QLoRA
frozen-bias fix, the tested environment, and the session reruns.
It links to the notebook, JSON reports, and logs.

These short checks establish compatibility for the tested configuration.
They do not establish summary quality or select a learning rate.

Agreement on a shared environment with Kunj, a clean execution workflow,
and Week 3 learning-rate selection remain pending.

## License

Project code: MIT (see [LICENSE](LICENSE)). Datasets, models, and libraries keep
their own licenses: see [BANKING77 notices](docs/banking77-third-party-notices.md)
and [BillSum notices](docs/billsum-third-party-notices.md).
