"""Build RoBERTa-base for BANKING77 with one of three methods (proposal Sections 4.2, 5.1, 6.1).

Methods
  full_ft    every weight trainable
  head_only  encoder frozen, only the two-layer classification head trains
  lora       encoder frozen except LoRA factors on attention query/value
             in all 12 blocks; the same classification head also trains

For a given seed, all three methods start from the same head initialization,
because the seed is set right before the model is created and the head is
initialized before any LoRA factors.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

BASE_MODEL = "FacebookAI/roberta-base"
NUM_LABELS = 77
METHODS = ("full_ft", "head_only", "lora")


def roberta_base_config(num_labels: int = NUM_LABELS) -> RobertaConfig:
    """Architecture identical to roberta-base, for offline tests (no download)."""
    return RobertaConfig(
        vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, max_position_embeddings=514, type_vocab_size=1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, num_labels=num_labels,
    )


def build_model(method: str, seed: int, rank: int = 8, alpha: int = 16, dropout: float = 0.05,
                target_modules: tuple[str, ...] = ("query", "value"),
                base_model: str = BASE_MODEL, revision: str | None = None,
                offline_random_init: bool = False, num_labels: int = NUM_LABELS):
    """Return a model ready to train with the requested method."""
    assert method in METHODS, f"method must be one of {METHODS}"
    torch.manual_seed(seed)
    if offline_random_init:
        model = RobertaForSequenceClassification(roberta_base_config(num_labels))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=num_labels, revision=revision)

    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    if method == "head_only":
        for p in model.roberta.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        return model

    # LoRA: PEFT freezes the base and keeps the classifier trainable (modules_to_save).
    from peft import LoraConfig, TaskType, get_peft_model
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules), bias="none",
    )
    return get_peft_model(model, cfg)


def count_params(model) -> dict[str, int]:
    """Split trainable parameters into LoRA factors, classification head, and other."""
    lora = head = other = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if "lora_" in name:
            lora += n
        elif "classifier" in name:
            head += n
        else:
            other += n
    total = sum(p.numel() for p in model.parameters())
    trainable = lora + head + other
    return {"lora": lora, "head": head, "other_trainable": other,
            "trainable": trainable, "total": total}
