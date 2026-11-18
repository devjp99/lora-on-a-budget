"""Protocol, selection, gradient accumulation and CPU resume regression tests."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lora_budget.billsum_protocol import METRIC, METHOD_LRS, choose_trials, epoch_groups, lr_factor
from lora_budget.billsum_training import (
    frozen_hashes, verify_frozen, save_resume, load_resume, train_group,
)


def reports():
    return [{'run_id': f'{method}-{lr}', 'method': method, 'lr': lr, 'phase': 'lr_search',
             'seed': 42, 'status': 'completed', 'metric': copy.deepcopy(METRIC),
             'epochs_requested': 3, 'epochs': [{'epoch': e, 'rougeL': s} for e, s in [(1, 10.), (2, 12.), (3, 11.)]],
             'best_epoch': 2, 'best_rougeL': 12., 'best_checkpoint': 'saved/best',
             'train_count': 3000, 'tune_count': 500, 'successful_optimizer_steps': 564,
             'frozen_integrity_passed': True, 'reload_passed': True, 'comparison_key': 'same'}
            for method, rates in METHOD_LRS.items() for lr in rates]


def test_all_training_examples_used_once_per_epoch():
    groups = epoch_groups(3000, 16, 42, 0)
    assert len(groups) == 188 and len(groups[-1]) == 8
    assert sorted(i for group in groups for i in group) == list(range(3000))
    assert groups == epoch_groups(3000, 16, 42, 0)
    assert groups != epoch_groups(3000, 16, 42, 1)


def test_schedule_warmup_and_end():
    warm = math.ceil(564 * .06)
    assert lr_factor(0, 564, .06) == 1 / warm
    assert lr_factor(warm - 1, 564, .06) == 1
    assert lr_factor(564, 564, .06) == 0


def test_selector_smaller_lr_on_equal_epoch_and_score():
    result = choose_trials(reports())
    assert all(result['selected'][m]['lr'] == rates[0] for m, rates in METHOD_LRS.items())


def test_selector_earlier_epoch_before_smaller_lr():
    rows = reports()
    rows[1].update(best_epoch=1)
    rows[1]['epochs'][0]['rougeL'] = 12.
    assert choose_trials(rows)['selected']['full_ft']['lr'] == 3e-5


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'environment', 'short', 'seed', 'nan', 'reload', 'best'])
def test_reject_invalid_selection(change):
    rows = reports()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif change == 'environment': rows[0]['comparison_key'] = 'different'
    elif change == 'short': rows[0]['successful_optimizer_steps'] = 20
    elif change == 'seed': rows[0]['seed'] = 43
    elif change == 'nan': rows[0]['epochs'][0]['rougeL'] = float('nan')
    elif change == 'reload': rows[0]['reload_passed'] = False
    elif change == 'best': rows[0]['best_epoch'] = 3
    with pytest.raises(ValueError): choose_trials(rows)


def test_debug_and_failed_attempts_are_not_selected():
    rows = reports()
    debug = copy.deepcopy(rows[0]); debug['phase'] = 'debug'
    failed = copy.deepcopy(rows[0]); failed['status'] = 'failed_code_or_data'
    assert choose_trials(rows + [debug, failed]) == choose_trials(rows)


class TinyLM(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.embed = torch.nn.Embedding(12, 6)
        self.embed.weight.requires_grad_(False)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(6, 12)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.head(self.drop(self.embed(input_ids)))
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 12), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def batch(ids, labels):
    return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long),
            'labels': torch.tensor([labels])}


def setup(dropout=0.2):
    torch.manual_seed(42)
    model = TinyLM(dropout)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 4, .06))
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    return model, opt, sched, scaler


def test_checkpoint_resume_matches_uninterrupted_dropout_training(tmp_path):
    model, opt, sched, scaler = setup()
    batches = [batch([1, 2, 3, 4], [-100, -100, 3, 4]), batch([5, 6, 7], [-100, 6, 7])]
    frozen = frozen_hashes(model)
    train_group(model, batches, opt, sched, scaler, amp=False)
    path = tmp_path / 'last.pt'
    save_resume(path, model, opt, sched, scaler, {'next_group': 1}, 'identity', frozen)
    train_group(model, batches, opt, sched, scaler, amp=False)
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    expected_lr = sched.get_last_lr()
    resumed, opt2, sched2, scaler2 = setup()
    state, original_frozen = load_resume(path, resumed, opt2, sched2, scaler2, 'identity')
    assert state['next_group'] == 1
    train_group(resumed, batches, opt2, sched2, scaler2, amp=False)
    assert all(torch.equal(expected[n], p) for n, p in resumed.named_parameters())
    assert sched2.get_last_lr() == expected_lr
    verify_frozen(resumed, original_frozen)
    with pytest.raises(ValueError, match='identity changed'):
        load_resume(path, resumed, opt2, sched2, scaler2, 'wrong')


def test_frozen_mutation_is_detected():
    model, *_ = setup()
    frozen = frozen_hashes(model)
    with torch.no_grad(): model.embed.weight[0, 0] += 1
    with pytest.raises(ValueError, match='Frozen'): verify_frozen(model, frozen)


def test_unequal_target_lengths_match_one_combined_batch():
    model, opt, sched, scaler = setup(dropout=0)
    a = batch([1, 2, 3, 4], [-100, -100, 3, 4])
    b = batch([5, 6, 7, 8], [-100, 6, 7, 8])
    train_group(model, [a, b], opt, sched, scaler, amp=False)
    other, opt2, sched2, scaler2 = setup(dropout=0)
    joined = {k: torch.cat([a[k], b[k]]) for k in a}
    train_group(other, [joined], opt2, sched2, scaler2, amp=False)
    assert all(torch.allclose(p, dict(other.named_parameters())[n], atol=1e-7, rtol=0)
               for n, p in model.named_parameters())


def test_rouge_uses_full_reference_and_per_example_f1():
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    assert scorer.score('a b c d', 'a b')['rougeL'].fmeasure == pytest.approx(2 / 3)
    assert scorer.score('a b', 'a b')['rougeL'].fmeasure == 1
    assert scorer.score('a b', '')['rougeL'].fmeasure == 0


def test_generation_never_receives_reference_tokens(monkeypatch):
    spec = importlib.util.spec_from_file_location('week3_runner', Path(__file__).parents[1] / 'scripts/train_billsum.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    from contextlib import nullcontext
    class Model:
        def eval(self): pass
        def train(self): pass
        def generate(self, **kw):
            assert kw['input_ids'].tolist() == [[1, 2]]
            assert 'labels' not in kw and kw['do_sample'] is False
            return torch.tensor([[1, 2, 3]])
    fake_torch = SimpleNamespace(
        tensor=lambda x, **kw: torch.tensor(x), long=torch.long, float16=torch.float16,
        ones_like=torch.ones_like, inference_mode=nullcontext, autocast=lambda *a, **kw: nullcontext())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0, decode=lambda ids, **kw: 'hello')
    scores, predictions = runner.evaluate(Model(), tokenizer,
        [{'prompt_ids': [1, 2], 'input_ids': [1, 2, 9, 9], 'reference': 'hello world'}], ['id1'], fake_torch, 256)
    assert predictions[0]['reference'] == 'hello world'
    assert scores['rougeL'] == pytest.approx(100 * 2 / 3)
