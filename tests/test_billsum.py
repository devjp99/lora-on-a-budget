"""Offline invariants; synthetic examples are never experimental evidence."""
import json
from pathlib import Path
import pytest
from lora_budget.data.billsum import encode_example, pad_examples, select_rows

class Tokenizer:
    eos_token_id = 999
    def encode(self,text,add_special_tokens=False): return [ord(c) for c in text]


def rows(n, prefix='bill'):
    return [dict(text=f'{prefix} {i}',summary=f'summary {i}') for i in range(n)]


def test_reproducible_disjoint_splits():
    a=select_rows(rows(30),rows(10,'test'),sizes=(15,5,5))
    assert a==select_rows(rows(30),rows(10,'test'),sizes=(15,5,5))
    s,_=a
    assert [len(s[k]) for k in ('train','tune','test')]==[15,5,5]
    assert not set(r['id'] for r in s['train']) & set(r['id'] for r in s['tune'])


def test_dedup_preserves_test():
    train=rows(20); test=rows(5,'test')
    for r in train: r['text']='TEST 0'
    s,removed=select_rows(train,test,sizes=(10,5,5))
    assert len(s['test'])==5
    assert not s['train'] and not s['tune']
    assert removed==dict(train=10,tune=5)

@pytest.mark.parametrize('field',['text','summary'])
def test_invalid_fields(field):
    train=rows(10); train[0][field]=' '
    with pytest.raises(ValueError): select_rows(train,rows(5),sizes=(5,2,2))


def test_insufficient_rows():
    with pytest.raises(ValueError): select_rows(rows(3),rows(3))


def test_labels_and_eos():
    x=encode_example(dict(text='A'*100,summary='B'*100),Tokenizer(),max_prompt=64,max_target=8)
    assert len(x['prompt_ids'])==64
    assert len(x['input_ids'])==72
    assert x['labels'][:64]==[-100]*64
    assert x['labels'][64:]==[66]*7+[999]
    assert x['source_truncated'] and x['target_truncated']
    assert x['reference']=='B'*100
    assert 66 not in x['prompt_ids']


def test_padding_eos_is_supervised():
    a=encode_example(dict(text='a',summary='b'),Tokenizer())
    b=encode_example(dict(text='aa',summary='bbb'),Tokenizer())
    batch=pad_examples([a,b],999)
    end=len(a['input_ids'])-1
    assert batch['labels'][0][end]==999
    assert all(x==-100 for x in batch['labels'][0][end+1:])
    assert batch['attention_mask'][0][end]==1
    assert sum(batch['attention_mask'][0])==len(a['input_ids'])


def test_budget_validation():
    with pytest.raises(ValueError): encode_example(rows(1)[0],Tokenizer(),max_prompt=1)
    with pytest.raises(ValueError): encode_example(rows(1)[0],Tokenizer(),max_target=0)


def test_configs_match_proposal():
    import yaml
    configs=[yaml.safe_load(p.read_text()) for p in sorted(Path('configs/billsum').glob('*.yaml'))]
    assert {c['method'] for c in configs}=={'full_ft','lora','dora','qlora'}
    for c in configs:
        assert c['data']==configs[0]['data']
        assert c['train']['micro_batch']*c['train']['grad_accum']==16
        assert c['train']['epochs']==3
        assert c['lora']['rank']==8 and c['lora']['alpha']==16
        assert len(c['lora']['target_modules'])==7
        assert c['eval']['max_new_tokens']==256


@pytest.mark.parametrize('method',['full_ft','lora','dora'])
def test_tiny_qwen_gradients_and_reload(method,tmp_path,monkeypatch):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import PeftModel
    from lora_budget.models import qwen
    torch.set_num_threads(1)
    cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True)
    monkeypatch.setattr(qwen.AutoModelForCausalLM,'from_pretrained',
        lambda *a,**kw:Qwen2ForCausalLM(cfg))
    model=qwen.build_model(method,'offline',device='cpu')
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    batch=dict(input_ids=torch.tensor([[1,2,3,4,5]]),labels=torch.tensor([[-100,-100,3,4,5]]))
    seen=set()
    for _ in range(3):
        opt.zero_grad(); loss=model(**batch).loss; assert torch.isfinite(loss); loss.backward()
        for n,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                if torch.count_nonzero(p.grad): seen.add(n)
            else: assert p.grad is None
        opt.step()
    assert seen=={n for n,p in model.named_parameters() if p.requires_grad}
    for n,p in model.named_parameters():
        if n in frozen: assert torch.equal(p,frozen[n])
    model.eval()
    with torch.no_grad(): before=model(**batch).logits
    model.save_pretrained(tmp_path)
    if method=='full_ft': fresh=Qwen2ForCausalLM.from_pretrained(tmp_path)
    else:
        torch.manual_seed(42); fresh=PeftModel.from_pretrained(Qwen2ForCausalLM(cfg),tmp_path)
    fresh.eval()
    with torch.no_grad(): after=fresh(**batch).logits
    torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-5)


def test_manifest_loader_detects_tampering(tmp_path):
    import csv
    import pandas as pd
    from lora_budget.data.hashing import sha256_file,text_hash
    from lora_budget.data.billsum import load_split
    raw=tmp_path/'raw'; raw.mkdir()
    manifest=tmp_path/'manifest'; manifest.mkdir()
    pd.DataFrame(rows(3)).to_parquet(raw/'train.parquet')
    path=manifest/'train.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','row','text_sha256']); w.writeheader()
        w.writerow(dict(id='train-00000',row=0,text_sha256=text_hash('bill 0')))
    meta=dict(raw_sha256={'train':sha256_file(raw/'train.parquet')},
              manifest_sha256={'train':sha256_file(path)})
    (manifest/'meta.json').write_text(json.dumps(meta))
    assert load_split('train',raw,manifest)[0]['summary']=='summary 0'
    path.write_text(path.read_text().replace('train-00000','train-00001'))
    with pytest.raises(ValueError,match='manifest checksum'): load_split('train',raw,manifest)


def test_locked_real_manifest_membership():
    import csv
    from lora_budget.data.hashing import sha256_file
    root=Path('manifests/billsum')
    meta=json.loads((root/'meta.json').read_text())
    hashes={};ids={}
    for name,count in [('train',3000),('tune',500),('test',500)]:
        path=root/f'{name}.csv'
        assert sha256_file(path)==meta['manifest_sha256'][name]
        with path.open() as f: records=list(csv.DictReader(f))
        assert len(records)==count
        hashes[name]={r['text_sha256'] for r in records}
        ids[name]={r['id'] for r in records}
        assert len(ids[name])==count
    for a,b in [('train','tune'),('train','test'),('tune','test')]:
        assert not hashes[a]&hashes[b]
        assert not ids[a]&ids[b]
