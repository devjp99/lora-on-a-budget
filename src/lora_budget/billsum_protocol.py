"""Pure-Python protocol checks for the BillSum learning-rate search."""
import hashlib
import json
import math
import random
from pathlib import Path

METHOD_LRS = {
    'full_ft': (1e-5, 3e-5), 'lora': (1e-4, 3e-4),
    'dora': (1e-4, 3e-4), 'qlora': (1e-4, 3e-4),
}
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
METRIC = {'package': 'rouge-score', 'version': '0.1.2', 'use_stemmer': True,
          'aggregation': 'mean_per_example_f1_times_100',
          'primary': 'rougeL', 'secondary': ['rouge1', 'rouge2']}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_config(cfg, lr, seed, phase):
    method = cfg['method']
    if method not in METHOD_LRS or lr not in METHOD_LRS[method]:
        raise ValueError('Learning rate is outside the proposal grid')
    if phase not in ('debug', 'lr_search', 'confirm'):
        raise ValueError('Unknown phase')
    if seed not in (42, 43, 44) or (phase == 'lr_search' and seed != 42):
        raise ValueError('LR search requires seed 42; allowed seeds are 42/43/44')
    if phase == 'confirm' and seed not in (43, 44):
        raise ValueError('Confirmation adds seeds 43 and 44')
    expected = {
        'model': {'name': 'Qwen/Qwen2.5-0.5B',
                  'revision': '060db6499f32faf8b98477b0a26969ef7d8b9987'},
        'data': {'max_prompt': 1024, 'max_target': 256, 'subset_seed': 2026},
        'train': {'epochs': 3, 'micro_batch': 1, 'grad_accum': 16,
                  'weight_decay': 0.01, 'warmup_ratio': 0.06, 'scheduler': 'linear',
                  'max_grad_norm': 1.0, 'precision': 'fp16',
                  'gradient_checkpointing': True, 'use_cache': False},
        'lora': {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'target_modules': TARGETS},
        'eval': {'select_by': 'tune_rougeL', 'do_sample': False,
                 'num_beams': 1, 'max_new_tokens': 256},
    }
    for section, fields in expected.items():
        for name, value in fields.items():
            if cfg.get(section, {}).get(name) != value:
                raise ValueError(f'Unreviewed setting: {section}.{name}')
    if tuple(cfg['train']['learning_rates']) != METHOD_LRS[method]:
        raise ValueError('Config learning-rate grid does not match proposal')


def epoch_groups(size, batch_size, seed, epoch):
    """Independent RNG: same full example order for every method; retain remainder."""
    if size <= 0 or batch_size <= 0:
        raise ValueError('Positive dataset and batch sizes required')
    order = list(range(size))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, size, batch_size)]


def lr_factor(step, total, warmup_ratio):
    warm = max(1, math.ceil(total * warmup_ratio))
    if step < warm:
        return (step + 1) / warm
    return max(0.0, (total - step) / max(1, total - warm))


def choose_trials(reports):
    """Require all eight completed, comparable seed-42 trials before selection."""
    grid = {(m, lr) for m, rates in METHOD_LRS.items() for lr in rates}
    candidates = {}
    comparison_keys = set()
    for report in reports:
        if report.get('phase') != 'lr_search':
            continue
        key = (report.get('method'), report.get('lr'))
        if key not in grid or report.get('seed') != 42:
            raise ValueError('Unexpected LR-search run')
        if report.get('status') != 'completed':
            continue  # Failed/interrupted attempts stay archived; do not select them.
        if key in candidates:
            raise ValueError(f'Multiple completed trials for {key}; review explicitly')
        if report.get('metric') != METRIC or report.get('epochs_requested') != 3:
            raise ValueError('Metric or duration differs from registered protocol')
        epochs = report.get('epochs', [])
        if [e['epoch'] for e in epochs] != [1, 2, 3]:
            raise ValueError('Incomplete epoch history')
        if report.get('train_count') != 3000 or report.get('tune_count') != 500:
            raise ValueError('Wrong split sizes')
        if report.get('successful_optimizer_steps') != 564:
            raise ValueError('Incomplete optimizer updates')
        if not report.get('frozen_integrity_passed') or not report.get('reload_passed'):
            raise ValueError('Final integrity/reload checks missing')
        if any(not math.isfinite(e['rougeL']) or not 0 <= e['rougeL'] <= 100 for e in epochs):
            raise ValueError('Invalid tuning score')
        best = min(epochs, key=lambda e: (-e['rougeL'], e['epoch']))
        if report.get('best_epoch') != best['epoch'] or report.get('best_rougeL') != best['rougeL']:
            raise ValueError('Best epoch does not match tuning history')
        if not report.get('comparison_key'):
            raise ValueError('Missing comparison provenance')
        comparison_keys.add(report['comparison_key'])
        candidates[key] = report
    if set(candidates) != grid:
        raise ValueError(f'Missing completed trials: {sorted(grid - set(candidates))}')
    if len(comparison_keys) != 1:
        raise ValueError('Environment, code, data, or shared settings differ across trials')
    selected = {}
    for method in METHOD_LRS:
        best = min((r for (m, _), r in candidates.items() if m == method),
                   key=lambda r: (-r['best_rougeL'], r['best_epoch'], r['lr']))
        selected[method] = {k: best[k] for k in
                            ('run_id', 'lr', 'best_epoch', 'best_rougeL', 'best_checkpoint')}
    return {'selection_rule': 'highest tuning ROUGE-L; earlier epoch; smaller LR',
            'comparison_key': comparison_keys.pop(), 'selected': selected}
