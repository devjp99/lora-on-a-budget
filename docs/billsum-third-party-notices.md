# BillSum component attribution

- Data: [FiscalNote/BillSum](https://github.com/FiscalNote/BillSum),
  [dataset card](https://huggingface.co/datasets/FiscalNote/billsum), CC0-1.0.
  `scripts/make_billsum_manifest.py` consumes its U.S. Parquet release.
- Model/tokenizer: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B),
  Apache-2.0. Loaded by `models/qwen.py` and the audit/pilot scripts.
- Transformers (Apache-2.0): pretrained model/tokenizer and Qwen architecture APIs.
- PEFT (Apache-2.0): `LoraConfig`, `get_peft_model`, `PeftModel`,
  `prepare_model_for_kbit_training` in `models/qwen.py`.
- bitsandbytes (MIT): NF4/double-quantized loading through `BitsAndBytesConfig`.
- PyTorch (BSD-3-Clause): tensors, autograd, AdamW, AMP, and CUDA memory counters.
- Hugging Face Hub (Apache-2.0): immutable revision resolution and downloads.
- pandas (BSD-3-Clause) and PyArrow (Apache-2.0): Parquet loading.

The added preprocessing, manifests, tests, audit, and pilot orchestration are
project code using these APIs; no external training script was copied. Preserve
upstream notices when redistributing their packages or model files. Exact package
versions used for each pilot are recorded in its environment report.
