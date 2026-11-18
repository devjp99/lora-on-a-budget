"""BillSum epoch training with tuning-only generation and resumable updates.

Run from the repository root. No final-test data is loaded by this script.
The debug phase uses 32 training / 2 tuning examples for one epoch and is
never eligible for LR selection. Real LR search uses all 3000 / 500 examples.
"""
import argparse
import copy
import importlib.metadata as metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lora_budget.billsum_protocol import (
    METRIC, atomic_json, digest, epoch_groups, file_hash, lr_factor, validate_config,
)

EXPECTED_PACKAGES = {
    'torch': '2.11.0+cu130', 'transformers': '5.20.2', 'peft': '0.21.2',
    'accelerate': '1.15.0', 'bitsandbytes': '0.50.2', 'rouge-score': '0.1.2',
    'huggingface-hub': '1.33.0', 'pandas': '2.3.3', 'pyarrow': '23.0.1',
    'PyYAML': '6.0.3',
}


def environment_record():
    versions = {name: metadata.version(name) for name in EXPECTED_PACKAGES}
    mismatches = {k: (EXPECTED_PACKAGES[k], v) for k, v in versions.items()
                  if v != EXPECTED_PACKAGES[k]}
    if mismatches:
        raise RuntimeError(f'Environment differs from the tested BillSum stack: {mismatches}. Review before running.')
    if platform.python_version_tuple()[:2] != ('3', '13'):
        raise RuntimeError('Expected Python 3.13; review this environment before running')
    try:
        metadata.version('torchao')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('Remove the incompatible optional torchao package before running')
    # Metric/tokenizer implementation dependencies also enter the comparison lock.
    for name in ('numpy', 'nltk', 'tokenizers', 'safetensors', 'sentencepiece'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'packages': versions}


def evaluate(model, tokenizer, rows, ids, torch, max_new_tokens):
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            # Prompt only. Reference tokens cannot enter model.generate().
            prompt = torch.tensor([row['prompt_ids']], dtype=torch.long, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                output = model.generate(
                    input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, prompt.shape[1]:].tolist()
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            scores = scorer.score(row['reference'], prediction)
            predictions.append({'id': ids[index], 'prediction': prediction,
                                'reference': row['reference'], 'generated_tokens': len(generated),
                                **{k: 100 * scores[k].fmeasure for k in scores}})
            if (index + 1) % 25 == 0:
                print(f'Tuning generation {index + 1}/{len(rows)}', flush=True)
    model.train()
    metrics = {k: sum(p[k] for p in predictions) / len(predictions)
               for k in ('rouge1', 'rouge2', 'rougeL', 'generated_tokens')}
    return metrics, predictions


def last_token_logits(model, row, torch):
    model.eval()
    prompt = torch.tensor([row['prompt_ids']], dtype=torch.long, device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        logits = model(input_ids=prompt, attention_mask=torch.ones_like(prompt)).logits
        result = logits[:, -1, :].float().cpu().clone()
    model.train()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--phase', choices=['debug', 'lr_search', 'confirm'], default='debug')
    ap.add_argument('--run-id', required=True, help='Unique ID; reuse only with --resume')
    ap.add_argument('--checkpoint-root', default='checkpoints/billsum/training')
    ap.add_argument('--results-root', default='results/billsum/training')
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-after-updates', type=int, help='Debug-only controlled interruption')
    ap.add_argument('--selection', default='results/billsum/selected_lr.json')
    args = ap.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error('Learning rate must be positive and finite')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        ap.error('Use only letters, digits, hyphens and underscores in run-id')
    if args.save_every < 1 or (args.stop_after_updates is not None and
                              (args.phase != 'debug' or args.stop_after_updates < 1)):
        ap.error('Invalid checkpoint interval or debug interruption')
    out = Path(args.results_root)
    run = Path(args.checkpoint_root) / args.run_id
    report_path = out / f'{args.run_id}.json'
    last = run / 'last.pt'
    attempt_id = uuid.uuid4().hex[:12]
    if not args.resume and (run.exists() or report_path.exists()):
        ap.error('Run ID exists. Resume it explicitly or choose a new ID; no overwrite.')
    if args.resume and not last.exists():
        ap.error('Resume checkpoint is missing. Restore it; do not silently restart.')
    if args.resume and report_path.exists() and json.loads(report_path.read_text()).get('status') == 'completed':
        ap.error('This run is already completed')
    run.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {'run_id': args.run_id, 'attempt_id': attempt_id, 'phase': args.phase,
              'seed': args.seed, 'lr': args.lr, 'metric': METRIC, 'status': 'failed_code_or_data',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'resumed': args.resume}
    state = None
    try:
        env = environment_record()
        import torch
        import yaml
        from transformers import AutoTokenizer
        from lora_budget.data.billsum import load_split, encode_example, BillSumCollator
        from lora_budget.models.qwen import build_model, audit_parameters
        from lora_budget.billsum_training import (
            frozen_hashes, verify_frozen, save_resume, load_resume, train_group,
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU is required')
        env.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        if 'T4' not in env['gpu']:
            raise RuntimeError('This protocol requires a Tesla T4; review a GPU change first')
        cfg = yaml.safe_load(Path(args.config).read_text())
        validate_config(cfg, args.lr, args.seed, args.phase)
        report.update(method=cfg['method'], config=cfg, environment=env)
        shared = copy.deepcopy(cfg)
        shared.pop('method')
        shared['train'].pop('learning_rates')
        source_paths = [Path(__file__), Path('src/lora_budget/billsum_protocol.py'),
                        Path('src/lora_budget/billsum_training.py'),
                        Path('src/lora_budget/data/billsum.py'), Path('src/lora_budget/data/hashing.py'),
                        Path('src/lora_budget/models/qwen.py')]
        sources = {str(p): file_hash(p) for p in source_paths}
        # Relative source keys make provenance portable across Kaggle sessions.
        sources['scripts/train_billsum.py'] = sources.pop(str(Path(__file__)))
        manifests = {name: file_hash(Path(cfg['data']['manifest_dir']) / name)
                     for name in ('meta.json', 'train.csv', 'tune.csv')}
        comparison_key = digest({'shared_config': shared, 'environment': env,
                                 'code': sources, 'manifests': manifests, 'metric': METRIC})
        identity = digest({'comparison_key': comparison_key, 'method': cfg['method'],
                           'lr': args.lr, 'seed': args.seed, 'phase': args.phase,
                           'run_id': args.run_id})
        report.update(comparison_key=comparison_key, identity=identity,
                      source_sha256=sources, manifest_sha256=manifests)
        report['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        report['git_diff'] = subprocess.check_output(['git', 'diff', 'HEAD'], text=True)
        report['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        if args.phase == 'confirm':
            selected = json.loads(Path(args.selection).read_text())
            if selected['comparison_key'] != comparison_key or selected['selected'][cfg['method']]['lr'] != args.lr:
                raise ValueError('Confirmation must use the locked selection and same protocol')
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'])
        tokenizer.pad_token = tokenizer.eos_token
        raw = {split: load_split(split, cfg['data']['raw_dir'], cfg['data']['manifest_dir'])
               for split in ('train', 'tune')}
        if [len(raw['train']), len(raw['tune'])] != [3000, 500]:
            raise ValueError('Expected frozen 3000/500 development splits')
        if args.phase == 'debug':
            raw = {'train': raw['train'][:32], 'tune': raw['tune'][:2]}
        rows = {split: [encode_example(r, tokenizer, cfg['data']['max_prompt'], cfg['data']['max_target'])
                        for r in records] for split, records in raw.items()}
        epochs = 1 if args.phase == 'debug' else 3
        report.update(train_count=len(rows['train']), tune_count=len(rows['tune']), epochs_requested=epochs)
        collate = BillSumCollator(tokenizer.pad_token_id)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = build_model(cfg['method'], cfg['model']['revision'], args.seed, cfg['model']['name'])
        torch.cuda.synchronize()
        report['load_seconds'] = time.perf_counter() - started
        report['parameters'] = audit_parameters(model, cfg['method'])
        frozen = frozen_hashes(model)
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (no_decay if 'bias' in name or 'norm' in name.lower() else decay).append(param)
        optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01},
                                      {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
        groups_per_epoch = math.ceil(len(rows['train']) / 16)
        total_steps = epochs * groups_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total_steps, 0.06))
        scaler = torch.amp.GradScaler('cuda')
        state = {'epoch': 0, 'next_group': 0, 'successful_optimizer_steps': 0,
                 'epochs': [], 'best_epoch': None, 'best_rougeL': -1.0,
                 'train_seconds': 0.0, 'eval_seconds': 0.0, 'overflow_retries': 0,
                 'input_plus_target_tokens': 0, 'supervised_tokens': 0,
                 'epoch_loss_sum': 0.0, 'epoch_groups_done': 0}
        if args.resume:
            state, frozen = load_resume(last, model, optimizer, scheduler, scaler, identity)
            if state.get('best_checkpoint') and not (run / state['best_checkpoint'] / 'reload_probe.pt').exists():
                raise FileNotFoundError('Restore the entire run folder, including the earlier best checkpoint')
            for previous in state['epochs']:
                if not (out / args.run_id / f"tune_epoch_{previous['epoch']}.json").exists():
                    raise FileNotFoundError('Restore earlier tuning prediction files along with the checkpoint')
            print(f"Resumed: epoch {state['epoch'] + 1}, next group {state['next_group']}", flush=True)
        else:
            save_resume(last, model, optimizer, scheduler, scaler, state, identity, frozen)
        model.train()
        for epoch in range(state['epoch'], epochs):
            groups = epoch_groups(len(rows['train']), 16, args.seed, epoch)
            for group_index in range(state['next_group'], len(groups)):
                batches = [collate([rows['train'][i]]) for i in groups[group_index]]
                torch.cuda.synchronize()
                start = time.perf_counter()
                update = train_group(model, batches, optimizer, scheduler, scaler)
                torch.cuda.synchronize()
                state['train_seconds'] += time.perf_counter() - start
                state['successful_optimizer_steps'] += 1
                for key in ('overflow_retries', 'supervised_tokens', 'input_plus_target_tokens'):
                    state[key] += update[key]
                state['epoch_loss_sum'] += update['loss']
                state['epoch_groups_done'] += 1
                state.update(epoch=epoch, next_group=group_index + 1)
                step = state['successful_optimizer_steps']
                print(f"Epoch {epoch + 1}/{epochs} update {group_index + 1}/{len(groups)} loss={update['loss']:.5f}", flush=True)
                should_stop = args.stop_after_updates is not None and step >= args.stop_after_updates
                if step % args.save_every == 0 or group_index + 1 == len(groups) or should_stop:
                    save_resume(last, model, optimizer, scheduler, scaler, state, identity, frozen)
                if should_stop:
                    raise KeyboardInterrupt('Controlled debug interruption; rerun with --resume and omit --stop-after-updates')
            verify_frozen(model, frozen)
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores, predictions = evaluate(model, tokenizer, rows['tune'],
                                            [r['id'] for r in raw['tune']], torch, 256)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            state['eval_seconds'] += duration
            epoch_record = {'epoch': epoch + 1, **scores, 'eval_seconds': duration,
                            'train_loss': state['epoch_loss_sum'] / state['epoch_groups_done']}
            state['epochs'].append(epoch_record)
            # Store generated summaries and per-example scores for every tuning epoch.
            atomic_json(out / args.run_id / f'tune_epoch_{epoch + 1}.json', predictions)
            if scores['rougeL'] > state['best_rougeL']:
                best_dir = run / f'best_epoch_{epoch + 1}'
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                torch.save(last_token_logits(model, rows['tune'][0], torch), best_dir / 'reload_probe.pt')
                state.update(best_epoch=epoch + 1, best_rougeL=scores['rougeL'], best_checkpoint=best_dir.name)
            state.update(epoch=epoch + 1, next_group=0, epoch_loss_sum=0.0, epoch_groups_done=0)
            save_resume(last, model, optimizer, scheduler, scaler, state, identity, frozen)
            report.update(state)
            report['status'] = 'running'
            atomic_json(report_path, report)
            print(json.dumps(epoch_record), flush=True)
        verify_frozen(model, frozen)
        report['frozen_integrity_passed'] = True
        report['peak_allocated_bytes_this_attempt'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes_this_attempt'] = torch.cuda.max_memory_reserved()
        best_dir = run / state['best_checkpoint']
        # Free the optimizer and original model before checking the best saved model.
        del model, optimizer, scheduler, scaler, decay, no_decay, param
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        fresh = build_model(cfg['method'], cfg['model']['revision'], args.seed,
                            cfg['model']['name'], checkpoint=str(best_dir))
        before = torch.load(best_dir / 'reload_probe.pt', map_location='cpu', weights_only=True)
        after = last_token_logits(fresh, rows['tune'][0], torch)
        diff = float((before - after).abs().max())
        report.update(reload_max_abs_diff=diff if math.isfinite(diff) else None, reload_tolerance=0.001,
                      reload_passed=math.isfinite(diff) and diff <= 0.001)
        if not report['reload_passed']:
            raise ValueError('Best checkpoint reload exceeded tolerance')
        report['checkpoint_bytes'] = sum(p.stat().st_size for p in best_dir.rglob('*')
                                         if p.is_file() and p.name != 'reload_probe.pt')
        report.update(status='completed', best_checkpoint=str(best_dir))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['status'] = ('interrupted' if isinstance(error, KeyboardInterrupt) else
                            'infeasible_oom' if type(error).__name__ == 'OutOfMemoryError' else
                            'failed_code_or_data')
        raise
    finally:
        if state is not None:
            # Preserve absolute best path set after reload, otherwise report relative snapshot name.
            best_path = report.get('best_checkpoint') if report.get('status') == 'completed' else None
            report.update(state)
            if best_path:
                report['best_checkpoint'] = best_path
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
        atomic_json(out / 'attempts' / f'{args.run_id}-{attempt_id}.json', report)
        with (out / 'run_ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps({k: report.get(k) for k in
                                    ('run_id', 'attempt_id', 'phase', 'status', 'method', 'lr', 'seed',
                                     'successful_optimizer_steps', 'started_utc', 'finished_utc', 'error')}) + '\n')
        print(f'Recorded {report_path}', flush=True)


if __name__ == '__main__':
    main()
