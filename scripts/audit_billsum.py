"""Verify real model tensors and preprocessing without training or test scoring."""
import gc
import importlib.metadata
import platform
import json
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer
from lora_budget.data.billsum import load_split,encode_example
from lora_budget.models.qwen import build_model,audit_parameters


def main():
    cfg=yaml.safe_load(Path('configs/billsum/full_ft.yaml').read_text())
    revision=cfg['model']['revision']
    tok=AutoTokenizer.from_pretrained(cfg['model']['name'],revision=revision)
    report=dict(model_revision=revision,device='cpu',is_gpu_pilot=False,parameters={},preprocessing={},
                environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','pandas','pyarrow']}))
    for split in ('train','tune'):
        encoded=[encode_example(r,tok) for r in load_split(split)]
        report['preprocessing'][split]=dict(count=len(encoded),
            source_truncated=sum(r['source_truncated'] for r in encoded),
            target_truncated=sum(r['target_truncated'] for r in encoded),
            source_lengths=sorted(r['source_tokens'] for r in encoded),
            target_lengths=sorted(r['target_tokens'] for r in encoded))
    for method in ('full_ft','lora','dora'):
        model=build_model(method,revision,device='cpu')
        counts=audit_parameters(model,method)
        counts.pop('tensors')
        report['parameters'][method]=counts
        del model; gc.collect()
    out=Path('results/billsum');out.mkdir(parents=True,exist_ok=True)
    (out/'cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['parameters'],indent=2))

if __name__=='__main__':main()
