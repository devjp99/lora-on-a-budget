# BANKING77 component attribution

- Data: [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) by PolyAI,
  CC BY 4.0 (Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence
  Encoders"). `scripts/make_banking77_manifest.py` downloads the released train/test CSVs.
  The raw CSVs are not committed; only manifests (IDs, labels, text hashes) are.
- Model/tokenizer: [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base),
  MIT license, pinned to commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
  Loaded by `src/lora_budget/models/roberta.py` and the training/eval scripts.
- Transformers (Apache-2.0): `AutoModelForSequenceClassification`, `AutoTokenizer`, RoBERTa classes.
- PEFT (Apache-2.0): `LoraConfig`, `get_peft_model`, `set_peft_model_state_dict`, `save_pretrained`.
- PyTorch (BSD-3-Clause): tensors, autograd, AdamW, AMP, GradScaler, CUDA memory counters.
- scikit-learn (BSD-3-Clause): accuracy and macro-F1.
- pandas and NumPy (BSD-3-Clause), PyYAML (MIT), Matplotlib (Matplotlib license),
  safetensors (Apache-2.0).

The data handling, training, evaluation, statistics, and analysis code is project code
using these APIs; no external training script was copied. Preserve upstream notices when
redistributing their packages or model files. Exact package versions for every run are
recorded in `results/train_ledger.csv`.
