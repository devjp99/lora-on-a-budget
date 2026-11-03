"""Smoke tests for the three RoBERTa methods (proposal Sections 4.1 and 8.4).

For full fine-tuning, head-only, and LoRA, this checks that:
  1. only the intended tensors receive gradients,
  2. frozen tensors do not change after an optimizer step,
  3. the trainable part (head or adapter) saves and reloads, and
  4. the reloaded model reproduces the pre-save outputs within a tolerance.

By default it uses a randomly initialized model with the exact roberta-base
architecture, so it runs without internet. Set LORA_BUDGET_ONLINE=1 to use the
real pretrained weights (for example on Colab).
"""

import os

import pytest
import torch

from lora_budget.models.roberta import build_model, count_params

OFFLINE = os.environ.get("LORA_BUDGET_ONLINE") != "1"
TOL = 1e-4
CASES = [("full_ft", {}), ("head_only", {}), ("lora", {"rank": 8, "alpha": 16})]


def _batch(seed: int = 0, bsz: int = 4, seq: int = 16):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 50000, (bsz, seq), generator=g)
    ids[:, 0] = 0  # <s>
    mask = torch.ones_like(ids)
    mask[0, -4:] = 0  # some padding
    ids[0, -4:] = 1
    labels = torch.randint(0, 77, (bsz,), generator=g)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _build(method, kw, seed=42):
    return build_model(method, seed=seed, offline_random_init=OFFLINE, **kw)


@pytest.mark.parametrize("method,kw", CASES)
def test_gradients_only_on_intended_tensors(method, kw):
    model = _build(method, kw)
    model.train()
    model(**_batch()).loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} should get a gradient"
        else:
            assert p.grad is None, f"frozen {name} got a gradient"
    if method != "full_ft":
        assert count_params(model)["other_trainable"] == 0


@pytest.mark.parametrize("method,kw", CASES)
def test_frozen_tensors_unchanged_after_step(method, kw):
    model = _build(method, kw)
    model.train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(**_batch()).loss.backward()
    opt.step()
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), f"frozen {n} changed"
    changed = sum(not torch.equal(p, trainable[n]) for n, p in model.named_parameters() if n in trainable)
    assert changed > 0, "no trainable tensor changed"


@pytest.mark.parametrize("method,kw", CASES)
def test_save_reload_reproduces_outputs(method, kw, tmp_path):
    model = _build(method, kw)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(**_batch()).loss.backward()
    opt.step()

    model.eval()
    probe = _batch(seed=1)
    probe.pop("labels")
    with torch.no_grad():
        before = model(**probe).logits

    if method == "lora":
        from peft import PeftModel
        model.save_pretrained(tmp_path)  # adapter + classifier (modules_to_save)
        base = build_model("full_ft", seed=42, offline_random_init=OFFLINE)  # same base weights
        reloaded = PeftModel.from_pretrained(base, tmp_path)
    elif method == "head_only":
        torch.save(model.classifier.state_dict(), tmp_path / "head.pt")
        reloaded = _build(method, kw)  # same frozen encoder
        for p in reloaded.classifier.parameters():  # scramble the head so the load must work
            torch.nn.init.normal_(p)
        reloaded.classifier.load_state_dict(torch.load(tmp_path / "head.pt"))
    else:
        model.save_pretrained(tmp_path)
        reloaded = type(model).from_pretrained(tmp_path)

    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**probe).logits
    diff = (before - after).abs().max().item()
    assert diff < TOL, f"{method}: reloaded outputs differ by {diff}"


def test_same_seed_gives_same_head_init():
    heads = []
    for method, kw in CASES:
        m = _build(method, kw)
        clf = m.base_model.model.classifier.original_module if method == "lora" else m.classifier
        heads.append(clf.out_proj.weight.detach().clone())
    assert torch.equal(heads[0], heads[1]) and torch.equal(heads[0], heads[2])
