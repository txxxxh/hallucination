#!/usr/bin/env python3
"""Sampling-based semantic and prompt-neighbourhood uncertainty on Scientist-known."""
from __future__ import annotations
import argparse, importlib, json, math, re, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'

def canon(x):
 x=unicodedata.normalize('NFKC',str(x)).casefold();return ' '.join(re.sub(r'[^\w\s]',' ',x).split())
def parse(text,right,wrong):
 v=canon(text); names=(right,wrong);hits=[i for i,n in enumerate(names)if canon(n)in v]
 if len(hits)==1:return hits[0]
 ss=[canon(n).split()[-1]for n in names];hits=[i for i,s in enumerate(ss)if s and re.search(rf'(?<!\w){re.escape(s)}(?!\w)',v)]
 return hits[0]if len(hits)==1 and ss[0]!=ss[1]else 2
def swap(p):
 m=re.match(r'^(Choose one of the following two options as the answer to the question below:\n)1\. ([^\n]+)\n2\. ([^\n]+)(\nQuestion:\n[\s\S]*)$',p)
 if not m:raise ValueError('bad prompt');return p
 return f'{m[1]}1. {m[3]}\n2. {m[2]}{m[4]}'
def paraphrase(p):
 pairs=[('However,','Nevertheless,'),('Notably,','Importantly,'),('never attended','did not attend'),('never studied','did not study'),('never received','did not receive'),('was never awarded','was not awarded'),('did not receive their education at','was not educated at'),('did not attend','was not educated at'),('Despite','In spite of')]
 q=p
 for a,b in pairs:q=re.sub(re.escape(a),b,q,flags=re.I)
 # Always provide a harmless surface-form neighbour when no listed pattern occurs.
 if q==p:q=p.replace('Who is this person?','Identify this person.')
 return q
def entropy(vals):
 c=np.bincount(vals,minlength=3).astype(float);p=c/c.sum();return float(-(p[p>0]*np.log(p[p>0])).sum())
def metric(y,s):return {'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--samples',type=int,default=10);ap.add_argument('--temperature',type=float,default=.7);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--max-new-tokens',type=int,default=32);ap.add_argument('--out-dir',type=Path,default=RUNS/'219_scientist_semantic_neighborhood');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 probes={x['key']:x for x in map(json.loads,(RUNS/'77_closedbook_fact_probe_results.jsonl').open())}; raw={str(x['key']):x for x in json.load((ROOT/'shuffled_prepend_names_question.json').open())}; rec={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open())}
 known=lambda p:p['n_discriminative_facts']>=1 and p['binary_accuracy']>.5 and p['pairwise_owner_accuracy']>.5
 rows=[raw[k]for k,r in rec.items()if r.get('parse_valid',True)and known(probes[k])];rows=rows[:a.limit or None];a.out_dir.mkdir(parents=True,exist_ok=True);path=a.out_dir/'samples.jsonl';done={}
 if a.resume and path.exists():done={(x['key'],x['condition']):x for x in map(json.loads,path.open())}
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval();torch.manual_seed(20260818)
 requests=[]
 for r in rows:
  cond={'original':r['prompt'],'swapped':swap(r['prompt']),'paraphrase':paraphrase(r['prompt'])}
  for name,p in cond.items():
   if (str(r['key']),name)not in done:requests.append((r,name,p))
 with path.open('a')as f:
  for start in range(0,len(requests),a.batch):
   part=requests[start:start+a.batch];texts=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)for _,_,p in part];z=tok(texts,return_tensors='pt',padding=True).to(model.device)
   with torch.inference_mode():out=model.generate(**z,do_sample=True,temperature=a.temperature,top_p=.95,num_return_sequences=a.samples,max_new_tokens=a.max_new_tokens,pad_token_id=tok.pad_token_id)
   width=z.input_ids.shape[1];texts=[tok.decode(x[width:],skip_special_tokens=True).strip()for x in out]
   for i,(r,name,p)in enumerate(part):
    values=texts[i*a.samples:(i+1)*a.samples];choices=[parse(x,str(r['rgt_ans']),str(r['wrg_ans']))for x in values];row={'key':str(r['key']),'condition':name,'right':r['rgt_ans'],'wrong':r['wrg_ans'],'outputs':values,'choices':choices};f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();done[(row['key'],name)]=row
   print(f'{min(start+len(part),len(requests))}/{len(requests)}',flush=True)
 # Analyse only complete items.
 items=[]
 for r in rows:
  k=str(r['key']);cs={n:done.get((k,n))for n in('original','swapped','paraphrase')}
  if any(v is None for v in cs.values()):continue
  per={n:np.array(v['choices'],int)for n,v in cs.items()};orig=per['original'];valid=orig<2;mode=int(np.bincount(orig[valid],minlength=2).argmax())if valid.any()else 2;pooled=np.concatenate(list(per.values()));pv=pooled<2;pmode=int(np.bincount(pooled[pv],minlength=2).argmax())if pv.any()else 2
  original_consistency=float(np.mean(orig==mode));neighbour_consistency=float(np.mean(pooled==pmode));parse_rate=float(np.mean(pooled<2));stable=parse_rate>=.8 and neighbour_consistency>=.8
  items.append({'key':k,'greedy_error':int(not rec[k]['correct']),'original_entropy':entropy(orig),'pooled_entropy':entropy(pooled),'condition_mode_entropy':entropy(np.array([int(np.bincount(x[x<2],minlength=2).argmax())if np.any(x<2)else 2 for x in per.values()])), 'original_consistency':original_consistency,'neighbour_consistency':neighbour_consistency,'parse_rate':parse_rate,'sample_mode':pmode,'stable_correct':bool(stable and pmode==0),'stable_systematic_error':bool(stable and pmode==1)})
 y=np.array([x['greedy_error']for x in items]);signals={n:np.array([x[n]for x in items])for n in('original_entropy','pooled_entropy','condition_mode_entropy')};report={'protocol':f'known Scientist; K={a.samples}; temperature={a.temperature}; conditions original/swap/surface-paraphrase; clusters right/wrong/invalid','n':len(items),'greedy_errors':int(y.sum()),'metrics':{n:metric(y,s)for n,s in signals.items()},'taxonomy':{'stable_correct':sum(x['stable_correct']for x in items),'stable_systematic_error':sum(x['stable_systematic_error']for x in items),'other':sum(not x['stable_correct']and not x['stable_systematic_error']for x in items)},'parse_rate_mean':float(np.mean([x['parse_rate']for x in items]))}
 (a.out_dir/'items.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in items));(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
