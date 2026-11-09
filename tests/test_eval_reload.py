"""The test-eval script must rebuild each model and load its saved checkpoint exactly
(same logits as the trained model), for all three methods. Offline, no download."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load("train_banking77")
ev = _load("eval_test_banking77")
from lora_budget.models.roberta import build_model  # noqa: E402

CFG = {"model": {"name": "FacebookAI/roberta-base", "revision": "main"},
       "lora": {"rank": 4, "alpha": 8, "dropout": 0.05, "target_modules": ["query", "value"]}}


@pytest.mark.parametrize("method", ["head_only", "lora", "full_ft"])
def test_checkpoint_reload_gives_same_logits(method, tmp_path):
    kw = {"rank": 4, "alpha": 8} if method == "lora" else {}
    model = build_model(method, seed=7, offline_random_init=True, **kw)
    with torch.no_grad():                       # pretend training changed every trainable weight
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    best = tmp_path / "run" / "best"
    best.mkdir(parents=True)
    if method == "lora":
        model.save_pretrained(best)
    else:
        torch.save(train.trainable_state(model, method), best / "trainable_state.pt")

    reloaded = ev.load_run_model(method, CFG, seed=7, best_dir=best, offline=True)
    x = {"input_ids": torch.tensor([[0, 100, 200, 300, 2]]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    model.eval(); reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(**x).logits, reloaded(**x).logits, atol=1e-5)


def test_find_best_dir(tmp_path):
    (tmp_path / "b" / "run-x" / "best").mkdir(parents=True)
    assert ev.find_best_dir("run-x", [str(tmp_path / "a"), str(tmp_path / "b")]) == tmp_path / "b" / "run-x" / "best"
    assert ev.find_best_dir("run-y", [str(tmp_path / "b")]) is None
