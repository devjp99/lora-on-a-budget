"""Training/checkpoint helpers shared by the BillSum runner and CPU tests."""
import hashlib
import os
import random
from pathlib import Path

import torch


def capture_rng():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def frozen_hashes(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters() if not p.requires_grad}


def verify_frozen(model, baseline):
    if frozen_hashes(model) != baseline:
        raise ValueError('Frozen parameter values or dtypes changed')


def save_resume(path, model, optimizer, scheduler, scaler, state, identity, frozen):
    """Replace a complete snapshot atomically; a partial write cannot destroy the prior one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        'identity': identity, 'state': state, 'rng': capture_rng(), 'frozen': frozen,
        'parameters': {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(snapshot, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_resume(path, model, optimizer, scheduler, scaler, identity):
    # Load only a locally created/trusted checkpoint: it includes Python RNG objects.
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    if snapshot['identity'] != identity:
        raise ValueError('Resume refused: code, data, config, environment or run identity changed')
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(snapshot['parameters']):
        raise ValueError('Resume parameter names differ')
    with torch.no_grad():
        for name, param in params.items():
            saved = snapshot['parameters'][name]
            if saved.shape != param.shape or saved.dtype != param.dtype:
                raise ValueError(f'Resume parameter shape/dtype mismatch: {name}')
            param.copy_(saved)
    verify_frozen(model, snapshot['frozen'])
    optimizer.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    scaler.load_state_dict(snapshot['scaler'])
    restore_rng(snapshot['rng'])
    return snapshot['state'], snapshot['frozen']


def train_group(model, batches, optimizer, scheduler, scaler, max_grad_norm=1.0, amp=True):
    """One update, weighted by supervised tokens, including a short final group.

    AMP overflow retries the SAME examples and dropout RNG with the reduced scale.
    Scheduler/data progress advance only after a successful optimizer update.
    """
    device = next(model.parameters()).device
    total_targets = sum(int((b['labels'][:, 1:] != -100).sum()) for b in batches)
    if total_targets == 0:
        raise ValueError('No supervised target tokens')
    group_rng = capture_rng()
    for retry in range(8):
        if retry:
            restore_rng(group_rng)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for original in batches:
            batch = {k: v.to(device) for k, v in original.items()}
            targets = int((batch['labels'][:, 1:] != -100).sum())
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss * (targets / total_targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        trainable = []
        finite = True
        for name, param in model.named_parameters():
            if not param.requires_grad:
                if param.grad is not None:
                    raise ValueError(f'Frozen parameter received a gradient: {name}')
                continue
            if param.grad is None:
                raise ValueError(f'Missing trainable gradient: {name}')
            finite = finite and bool(torch.isfinite(param.grad).all())
            trainable.append(param)
        if finite:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if finite and scaler.get_scale() >= previous_scale:
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return {'loss': total_loss, 'overflow_retries': retry,
                    'supervised_tokens': total_targets,
                    'input_plus_target_tokens': sum(int(b['attention_mask'].sum()) for b in batches)}
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradients without AMP recovery')
    raise FloatingPointError('Eight consecutive AMP overflow attempts for the same group')
