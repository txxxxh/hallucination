#!/usr/bin/env python3
"""Strict binding assay on the semantics-frozen §3.1 keyword cohort."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'
SRC=RUNS/'241_paper_keyword_reliance'; OUT=RUNS/'242_paper_frozen_keyword_binding'

def prompts(right,wrong,keyword,nonce):
 out=[]
 for cue,kind in ((keyword,'real'),(nonce,'nonce')):
  for owner_name,owner in ((right,'right'),(wrong,'wrong')):
   other=wrong if owner=='right' else right
   for swap in (0,1):
    a,b=(owner_name,other) if not swap else (other,owner_name)
    gold=0 if not swap else 1
    text=("This is a fictional card-assignment game. Ignore real-world facts.\n"
          f"For this round only, the card '{cue}' is assigned to {owner_name}.\n"
          f"The card 'Blue Circle' is assigned to {other}.\n"
          f"According only to these assignments, who has the '{cue}' card?\n"
          f"A. {a}\nB. {b}\nAnswer exactly A or B.")
    out.append({'cue':kind,'owner':owner,'swap':swap,'gold':gold,'prompt':text})
 return out

def score(model,tok,texts,batch):
 import torch
 aid=tok.encode('A',add_special_tokens=False);bid=tok.encode('B',add_special_tokens=False)
 if len(aid)!=1 or len(bid)!=1:raise RuntimeError((aid,bid))
 vals=[]
 for st in range(0,len(texts),batch):
  tx=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in texts[st:st+batch]]
  z=tok(tx,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():q=model(**z,use_cache=False).logits[:,-1,[aid[0],bid[0]]].float()
  vals.extend(torch.log_softmax(q,-1).cpu().numpy().tolist())
 return np.asarray(vals)

def collect(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 OUT.mkdir(parents=True,exist_ok=True)
 records=[json.loads(p.read_text()) for p in sorted(SRC.glob('question_*.json'))]
 jobs=[]
 for x in records:
  seen=set()
  for t in x['targets']:
   ident=(t['span_start'],t['span_end'],t['field'],t['value'])
   if ident in seen:continue
   seen.add(ident); jobs.append((x,t))
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
 model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval()
 mode='a' if a.resume and (OUT/'items.jsonl').exists() else 'w';done=set()
 if mode=='a':done={(x['key'],x['span_start'],x['span_end'],x['field']) for x in map(json.loads,(OUT/'items.jsonl').open())}
 with (OUT/'items.jsonl').open(mode) as f:
  for st in range(0,len(jobs),a.keyword_batch):
   chunk=[q for q in jobs[st:st+a.keyword_batch] if (q[0]['key'],q[1]['span_start'],q[1]['span_end'],q[1]['field']) not in done]
   if not chunk:continue
   meta=[];texts=[]
   for j,(x,t) in enumerate(chunk):
    ps=prompts(x['right'],x['wrong'],t['span_text'],f'ZORP-{st+j:05d}')
    meta.append(ps);texts.extend(q['prompt'] for q in ps)
   lp=score(model,tok,texts,a.batch);off=0
   for (x,t),ps in zip(chunk,meta):
    zz=[]
    for q,v in zip(ps,lp[off:off+8]):zz.append({**q,'correct_margin':float(v[q['gold']]-v[1-q['gold']])})
    off+=8
    means={(cue,owner):np.mean([q['correct_margin'] for q in zz if q['cue']==cue and q['owner']==owner]) for cue in ('real','nonce') for owner in ('right','wrong')}
    real=means['real','wrong']-means['real','right'];null=means['nonce','wrong']-means['nonce','right']
    row={'key':x['key'],'group':x['group'],'correct':x['correct'],'field':t['field'],'keyword':t['span_text'],'span_start':t['span_start'],'span_end':t['span_end'],'target_u_wrong':t['target_u_wrong'],'real_asymmetry':float(real),'nonce_asymmetry':float(null),'binding_effect':float(real-null),'conditions':zz}
    f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush();print(f'[{min(st+a.keyword_batch,len(jobs))}/{len(jobs)}]',flush=True)

def group_boot(rows,value,seed=42,n=10000):
 by_item=defaultdict(list)
 for r in rows:by_item[r['key']].append(r[value])
 item={k:float(np.mean(v)) for k,v in by_item.items()};key_group={r['key']:r['group'] for r in rows};groups=defaultdict(list)
 for k,v in item.items():groups[key_group[k]].append(v)
 vals=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed);boot=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(n)])
 return {'n_items':len(item),'n_keywords':len(rows),'n_groups':len(vals),'mean':float(vals.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'fraction_item_positive':float(np.mean(np.array(list(item.values()))>0))}

def analyze():
 rows=[json.loads(x) for x in (OUT/'items.jsonl').open() if x.strip()]
 owner={}
 for p in SRC.glob('question_*.json'):
  x=json.loads(p.read_text())
  for t in x['targets']:owner[(x['key'],t['span_start'],t['span_end'],t['field'])]=t['owner_side']
 for r in rows:
  r['owner_side']=owner[(r['key'],r['span_start'],r['span_end'],r['field'])]
  r['owner_aligned_binding']=r['binding_effect'] if r['owner_side']=='wrong' else -r['binding_effect']
 report={'protocol':'frozen 241 keywords; real-vs-nonce owner assignment asymmetry, sign-aligned to original profile owner; item aggregation and person-group bootstrap','n_keywords':len(rows),'cells':{},'by_field':{}}
 for c in (False,True):
  name='correct' if c else 'error';z=[r for r in rows if r['correct']==c];report['cells'][name]=group_boot(z,'owner_aligned_binding',42+int(c))
 for field in sorted(set(r['field'] for r in rows)):
  report['by_field'][field]={}
  for c in (False,True):
   z=[r for r in rows if r['field']==field and r['correct']==c]
   if z:report['by_field'][field]['correct' if c else 'error']=group_boot(z,'owner_aligned_binding',100+int(c))
 # Paired estimand at the independent person-group level.
 g={}
 for c in (False,True):
  z=[r for r in rows if r['correct']==c];by=defaultdict(list)
  for r in z:by[r['group']].append(r['owner_aligned_binding'])
  g[c]={k:np.mean(v) for k,v in by.items()}
 common=sorted(set(g[False])&set(g[True]));d=np.array([g[False][k]-g[True][k] for k in common]);rng=np.random.default_rng(999);boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
 report['error_minus_correct']={'n_groups':len(common),'mean':float(d.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','analyze','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=256);p.add_argument('--keyword-batch',type=int,default=32);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.stage in ('collect','all'):collect(a)
 if a.stage in ('analyze','all'):analyze()
if __name__=='__main__':main()
