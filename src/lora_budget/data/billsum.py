"""BillSum manifests and causal-LM formatting (proposal sections 4.3 and 7)."""
import csv
import json
import random
from pathlib import Path
from .hashing import sha256_file, text_hash

PREFIX = "Summarize the following bill:\n\n"
SUFFIX = "\n\nSummary:\n"


def select_rows(train, test, seed=2026, sizes=(3000, 500, 500)):
    """Select before deduplication; never replace or change final-test membership."""
    ntrain, ntune, ntest = sizes
    if min(sizes) < 1 or len(train) < ntrain + ntune or len(test) < ntest:
        raise ValueError("Insufficient rows for the requested subsets")
    def records(rows, source):
        result = []
        for i, row in enumerate(rows):
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('text', 'summary')):
                raise ValueError(f"{source} row {i}: empty or invalid text/summary")
            result.append({'id': f'{source}-{i:05d}', 'row': i,
                           'text_sha256': text_hash(row['text'])})
        return result
    a, b = records(train, 'train'), records(test, 'test')
    rng = random.Random(seed)
    rng.shuffle(a)
    rng.shuffle(b)
    tune, selected, final = a[:ntune], a[ntune:ntune+ntrain], b[:ntest]
    final_hashes = {r['text_sha256'] for r in final}
    # Final evaluation is immutable. Remove overlaps from the development side.
    clean_tune = [r for r in tune if r['text_sha256'] not in final_hashes]
    blocked = final_hashes | {r['text_sha256'] for r in clean_tune}
    clean_train = [r for r in selected if r['text_sha256'] not in blocked]
    splits = dict(train=clean_train, tune=clean_tune, test=final)
    removed = dict(train=ntrain-len(clean_train), tune=ntune-len(clean_tune))
    for rows in splits.values():
        rows.sort(key=lambda r: r['id'])
    return splits, removed


def load_split(split, raw_dir='data/raw/billsum', manifest_dir='manifests/billsum'):
    if split not in ('train', 'tune', 'test'):
        raise ValueError(split)
    import pandas as pd
    root = Path(manifest_dir)
    meta = json.loads((root / 'meta.json').read_text())
    source = 'test' if split == 'test' else 'train'
    raw = Path(raw_dir) / f'{source}.parquet'
    manifest = root / f'{split}.csv'
    if sha256_file(raw) != meta['raw_sha256'][source]:
        raise ValueError('Raw BillSum checksum mismatch')
    if sha256_file(manifest) != meta['manifest_sha256'][split]:
        raise ValueError('BillSum manifest checksum mismatch')
    frame = pd.read_parquet(raw)
    rows = []
    with manifest.open(newline='') as f:
        for item in csv.DictReader(f):
            row = frame.iloc[int(item['row'])].to_dict()
            if text_hash(row['text']) != item['text_sha256']:
                raise ValueError('BillSum row hash mismatch')
            rows.append(dict(row, id=item['id']))
    return rows


def encode_example(row, tokenizer, max_prompt=1024, max_target=256):
    if tokenizer.eos_token_id is None or max_target < 1:
        raise ValueError('An EOS token and positive target budget are required')
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = enc(PREFIX), enc(SUFFIX)
    budget = max_prompt - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError('Prompt budget must leave room for bill text')
    source, target = enc(row['text']), enc(row['summary'])
    prompt = prefix + source[:budget] + suffix
    answer = target[:max_target-1] + [tokenizer.eos_token_id]
    return dict(input_ids=prompt+answer, attention_mask=[1]*(len(prompt)+len(answer)),
                labels=[-100]*len(prompt)+answer, prompt_ids=prompt,
                reference=row['summary'], source_tokens=len(source), target_tokens=len(target)+1,
                source_truncated=len(source)>budget, target_truncated=len(target)+1>max_target)


def pad_examples(examples, pad_id):
    if not examples:
        raise ValueError('Empty batch')
    n = max(len(x['input_ids']) for x in examples)
    return {key: [x[key] + [fill]*(n-len(x[key])) for x in examples]
            for key, fill in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]}


class BillSumCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        import torch
        return {k: torch.tensor(v, dtype=torch.long) for k, v in pad_examples(examples, self.pad_id).items()}
