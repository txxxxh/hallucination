#!/usr/bin/env python3
"""Independent greedy plus split-sample U confirmation on GSM8K final answers."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs'
def canon(text):
 m=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',str(text))
 if not m:return'<invalid>'
 try:return format(Decimal(m[-1].replace(',','')).normalize(),'f')
 except InvalidOperation:return'<invalid>'
def entropy(v):
 c=np.asarray(list(Counter(v).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--samples',type=int,default=10);ap.add_argument('--out-dir',type=Path,default=RUNS/'236_gsm8k_u_split_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.manifest.open()];import torch;from transformers import AutoModelForCausalLM,AutoTokenizer;tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval();torch.manual_seed(20260821);items=[]
 for st in range(0,len(rows),a.batch):
  part=rows[st:st+a.batch];us=[f"Solve the math problem. Output only the final numeric answer.\n\nProblem:\n{x['question']}"for x in part];ps=[tok.apply_chat_template([{'role':'user','content':u}],tokenize=False,add_generation_prompt=True)for u in us];z=tok(ps,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():g0=model.generate(**z,do_sample=False,max_new_tokens=32,pad_token_id=tok.pad_token_id);g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=a.samples,max_new_tokens=32,pad_token_id=tok.pad_token_id)
  greedy=tok.batch_decode(g0[:,z.input_ids.shape[1]:],skip_special_tokens=True);outs=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
  for i,x in enumerate(part):
   vals=[canon(v)for v in outs[i*a.samples:(i+1)*a.samples]];k=a.samples//2;mode=Counter(vals[k:]).most_common(1)[0][0];gold=canon(x['gold_final']);items.append({'key':x['key'],'greedy':canon(greedy[i]),'gold':gold,'greedy_error':int(canon(greedy[i])!=gold),'u_score':entropy(vals[:k]),'validation_majority_correct':int(mode==gold),'validation_correct_rate':sum(v==gold for v in vals[k:])/len(vals[k:]),'samples':vals})
  print(f'{min(st+len(part),len(rows))}/{len(rows)}',flush=True)
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 u=np.array([x['u_score']for x in items]);q=np.quantile(u,[.3,.7]);lo=[x for x in items if x['u_score']<=q[0]];hi=[x for x in items if x['u_score']>=q[1]]
 def sm(z):
  er=[x for x in z if x['greedy_error']];co=[x for x in z if not x['greedy_error']];return{'n':len(z),'errors':len(er),'majority_repair':float(np.mean([x['validation_majority_correct']for x in er])),'correct_damage':float(np.mean([not x['validation_majority_correct']for x in co]))}
 y=np.array([x['greedy_error']for x in items]);order=np.argsort(u);risk={str(c):float(y[order[:round(len(y)*c)]].mean())for c in(.9,.7,.5,.3)};report={'protocol':'independent final-only greedy baseline; first5 entropy; held-out last5 majority','n':len(items),'greedy_accuracy':float(1-y.mean()),'validation_majority_accuracy':float(np.mean([x['validation_majority_correct']for x in items])),'u_quantiles':q.tolist(),'low':sm(lo),'high':sm(hi),'risk_by_coverage':risk};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
