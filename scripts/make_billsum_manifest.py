"""Download the documented U.S. Parquet release and freeze subset membership."""
import argparse
import csv
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from lora_budget.data.billsum import select_rows
from lora_budget.data.hashing import sha256_file
import pandas as pd
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revision', default='main')
    ap.add_argument('--verify-existing', action='store_true', help='Restore raw files using the locked revision and verify existing manifests')
    ap.add_argument('--raw-dir', default='data/raw/billsum')
    ap.add_argument('--out-dir', default='manifests/billsum')
    args = ap.parse_args()
    out, raw = Path(args.out_dir), Path(args.raw_dir)
    if not args.verify_existing and out.exists() and any(out.iterdir()):
        raise FileExistsError('Manifests already exist; verify them, do not silently regenerate')
    locked = json.loads((out/'meta.json').read_text()) if args.verify_existing else None
    revision = locked['revision'] if locked else HfApi().dataset_info('FiscalNote/billsum', revision=args.revision).sha
    raw.mkdir(parents=True, exist_ok=True)
    frames, hashes = {}, {}
    for split, count in [('train', 18949), ('test', 3269)]:
        path = hf_hub_download('FiscalNote/billsum', f'data/{split}-00000-of-00001.parquet',
                               repo_type='dataset', revision=revision)
        shutil.copyfile(path, raw / f'{split}.parquet')
        frame = pd.read_parquet(path)
        if len(frame) != count or not {'text', 'summary'}.issubset(frame.columns):
            raise ValueError(f'Unexpected {split} schema/count')
        frames[split] = frame
        hashes[split] = sha256_file(path)
    if locked:
        from lora_budget.data.billsum import load_split
        for split in ('train', 'tune', 'test'):
            load_split(split, raw, out)
        print('Locked raw files and all manifests verified')
        return
    splits, removed = select_rows(*(frames[s].to_dict('records') for s in ('train', 'test')))
    if any(not rows for rows in splits.values()):
        raise ValueError('Empty split after duplicate removal')
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (out / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'row', 'text_sha256'], lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    meta = dict(dataset='FiscalNote/billsum', revision=revision, license='CC0-1.0',
                subset_seed=2026, selection_algorithm='python.Random shuffle train then test',
                raw_sha256=hashes, raw_counts={s:len(f) for s,f in frames.items()},
                schemas={s:{k:str(v) for k,v in f.dtypes.items()} for s,f in frames.items()},
                planned_counts=dict(train=3000,tune=500,test=500),
                final_counts={s:len(r) for s,r in splits.items()}, duplicates_removed=removed,
                repeats_within_split={s:len(r)-len({x['text_sha256'] for x in r}) for s,r in splits.items()},
                manifest_sha256={s:sha256_file(out/f'{s}.csv') for s in splits})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
