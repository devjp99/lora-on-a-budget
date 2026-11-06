"""Real CUDA-only Week 1-2 decoder gate. No final-test evaluation or LR selection."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tol', type=float, default=1e-3)
    args = ap.parse_args()
    if args.steps < 20 or args.tol <= 0:
        ap.error('At least 20 successful optimizer steps and a positive tolerance are required')
    run_id = f'billsum-{Path(args.config).stem}-{uuid.uuid4().hex[:12]}'
    out = Path('results/billsum/pilots'); out.mkdir(parents=True, exist_ok=True)
    ckpt = Path('checkpoints/billsum') / run_id
    report = dict(run_id=run_id, phase='compatibility_pilot', owner='Jainish Patel',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), passed=False,
                  status='failed_code_or_data', successful_optimizer_steps=0)
    try:
        import torch
        import yaml
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        from lora_budget.data.billsum import load_split, encode_example, BillSumCollator
        from lora_budget.data.hashing import sha256_file
        from lora_budget.models.qwen import build_model, audit_parameters
        cfg = yaml.safe_load(Path(args.config).read_text())
        report['config'] = cfg
        report['environment'] = dict(python=platform.python_version(),
            packages={d:importlib.metadata.version(d) for d in
                      ['torch','transformers','peft','accelerate','huggingface-hub','pandas','pyarrow','PyYAML']},
            cuda=torch.version.cuda)
        report['pip_freeze'] = subprocess.check_output(['python','-m','pip','freeze'], text=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; CPU runs cannot pass the compatibility gate')
        report['environment']['gpu'] = torch.cuda.get_device_name(0)
        report['environment']['device_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
        try:
            report['environment']['bitsandbytes'] = importlib.metadata.version('bitsandbytes')
        except importlib.metadata.PackageNotFoundError:
            pass
        revision = HfApi().model_info(cfg['model']['name'], revision=cfg['model']['revision']).sha
        cfg['model']['revision'] = revision
        report['manifest_meta_sha256'] = sha256_file(Path(cfg['data']['manifest_dir'])/'meta.json')
        tok = AutoTokenizer.from_pretrained(cfg['model']['name'], revision=revision)
        tok.pad_token = tok.eos_token
        def dataset(split):
            return [encode_example(r,tok,cfg['data']['max_prompt'],cfg['data']['max_target'])
                    for r in load_split(split,cfg['data']['raw_dir'],cfg['data']['manifest_dir'])]
        train, tune = dataset('train'), dataset('tune')
        if not train or not tune:
            raise ValueError('Empty development split')
        report['lengths'] = {s:{k:[x[k] for x in rows] for k in
            ['source_tokens','target_tokens','source_truncated','target_truncated']}
            for s,rows in [('train',train),('tune',tune)]}
        collate = BillSumCollator(tok.pad_token_id)
        loader = DataLoader(train,batch_size=cfg['train']['micro_batch'],shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed),collate_fn=collate)
        probe = {k:v.cuda() for k,v in collate(tune[:1]).items() if k!='labels'}
        def sync(): torch.cuda.synchronize()
        def peak():
            return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                        reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync()
        start = time.perf_counter()
        model = build_model(cfg['method'],revision,args.seed,cfg['model']['name'])
        sync(); report['load_seconds'] = time.perf_counter()-start
        report['loading_memory'] = peak()
        report['parameters'] = audit_parameters(model,cfg['method'])
        frozen = {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                  for n,p in model.named_parameters() if not p.requires_grad}
        decay, no_decay = [], []
        for n,p in model.named_parameters():
            if p.requires_grad:
                (no_decay if 'bias' in n or 'norm' in n.lower() else decay).append(p)
        opt = torch.optim.AdamW([dict(params=decay,weight_decay=cfg['train']['weight_decay']),
                                dict(params=no_decay,weight_decay=0.)],lr=cfg['train']['learning_rates'][0])
        warm = math.ceil(args.steps*cfg['train']['warmup_ratio'])
        sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda s:
            s/max(1,warm) if s<warm else max(0.,(args.steps-s)/max(1,args.steps-warm)))
        scaler = torch.amp.GradScaler('cuda')
        model.train(); iterator = iter(loader)
        losses, times, tokens, target_tokens = [], [], [], []
        seen_grad = set()
        attempts = 0
        while report['successful_optimizer_steps'] < args.steps:
            attempts += 1
            if attempts > args.steps*5:
                raise RuntimeError('Too many AMP overflow skips; gate failed')
            sync(); start = time.perf_counter(); opt.zero_grad(set_to_none=True)
            loss_sum = 0.; ntok = nsup = 0
            for _ in range(cfg['train']['grad_accum']):
                try: batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader); batch = next(iterator)
                batch = {k:v.cuda() for k,v in batch.items()}
                ntok += batch['attention_mask'].sum().item()
                nsup += (batch['labels']!=-100).sum().item()
                with torch.autocast('cuda',dtype=torch.float16):
                    loss = model(**batch).loss/cfg['train']['grad_accum']
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward(); loss_sum += loss.item()
            if report['successful_optimizer_steps']==0:
                report['forward_backward_memory'] = peak()
            scaler.unscale_(opt)
            finite = True
            for n,p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:
                    raise ValueError(f'Frozen tensor received gradient: {n}')
                if p.requires_grad:
                    if p.grad is None: raise ValueError(f'Missing gradient: {n}')
                    finite &= bool(torch.isfinite(p.grad).all())
                    if torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad): seen_grad.add(n)
            if finite:
                torch.nn.utils.clip_grad_norm_(decay+no_decay,cfg['train']['max_grad_norm'])
            scaler.step(opt); scaler.update()
            if not finite: continue
            sched.step(); report['successful_optimizer_steps'] += 1
            sync(); times.append(time.perf_counter()-start); losses.append(loss_sum)
            tokens.append(ntok); target_tokens.append(nsup)
            if len(times)==1: report['first_optimizer_memory'] = peak()
            if len(times)==5: torch.cuda.reset_peak_memory_stats()
            print(f'{len(times)}/{args.steps}: loss={loss_sum:.5f}',flush=True)
        expected = {n for n,p in model.named_parameters() if p.requires_grad}
        if expected-seen_grad: raise ValueError(f'Tensors never had nonzero gradient: {sorted(expected-seen_grad)}')
        for n,p in model.named_parameters():
            if n in frozen and hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=frozen[n]:
                raise ValueError(f'Frozen parameter changed: {n}')
        report.update(steady_memory=peak(),losses=losses,step_seconds=times,
            mean_steady_step_seconds=sum(times[5:])/len(times[5:]),
            input_plus_target_tokens_per_second=sum(tokens[5:])/sum(times[5:]),
            supervised_tokens_per_second=sum(target_tokens[5:])/sum(times[5:]),
            optimizer_state_dtypes=sorted({str(v.dtype) for s in opt.state.values() for v in s.values() if torch.is_tensor(v)}))
        model.eval()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            before = model(**probe).logits.float().cpu()
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        report['checkpoint_bytes'] = sum(p.stat().st_size for p in ckpt.rglob('*') if p.is_file())
        # Release the original model AND optimizer before reloading on a 16 GB device.
        del model,opt,sched,scaler,decay,no_decay,p,loss,batch
        gc.collect(); torch.cuda.empty_cache()
        fresh = build_model(cfg['method'],revision,args.seed,cfg['model']['name'],str(ckpt)).eval()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            after = fresh(**probe).logits.float().cpu()
        diff = (before-after).abs().max().item()
        report.update(reload_max_abs_diff=diff,reload_tolerance=args.tol)
        if not math.isfinite(diff) or diff>args.tol:
            raise ValueError('Checkpoint reload differs beyond tolerance')
        report.update(passed=True,status='completed')
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if type(exc).__name__=='OutOfMemoryError': report['status']='infeasible_oom'
        elif isinstance(exc,KeyboardInterrupt): report['status']='interrupted'
        raise
    finally:
        (out/f'{run_id}.json').write_text(json.dumps(report,indent=2)+'\n')
        with (out.parent/'run_ledger.jsonl').open('a') as f:
            f.write(json.dumps({k:report[k] for k in ['run_id','phase','status','passed','successful_optimizer_steps']})+'\n')
        print(f'Recorded {out/run_id}.json')

if __name__=='__main__': main()
