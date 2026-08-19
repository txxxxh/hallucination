#!/usr/bin/env python3
"""Split-sample U confirmation on balanced TriviaQA context answers."""
from __future__ import annotations
import argparse,json,math,re,string
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs'
def norm(s):
 s=str(s).lower();s=''.join(c if c not in string.punctuation else' 'for c in s);s=re.sub(r'\b(a|an|the)\b',' ',s);return' '.join(s.split())
def entropy(vals):
 c=np.asarray(list(Counter(vals).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--samples',type=int,default=10);ap.add_argument('--out-dir',type=Path,default=RUNS/'232_trivia_u_split_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.source.open()]
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval();torch.manual_seed(20260820);items=[]
 with(a.out_dir/'samples.jsonl').open('w')as f:
  for st in range(0,len(rows),a.batch):
   part=rows[st:st+a.batch];prompts=[f"Answer using the context. Output only the short answer.\n\nContext:\n{x['context']}\n\nQuestion: {x['question']}"for x in part];texts=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)for p in prompts];z=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
   with torch.inference_mode():g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=a.samples,max_new_tokens=48,pad_token_id=tok.pad_token_id)
   outs=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
   for i,x in enumerate(part):
    ys=[v.strip()for v in outs[i*a.samples:(i+1)*a.samples]];aliases={norm(v)for v in x.get('aliases',[])+[x['answer']]};labels=['<correct>'if norm(v)in aliases else norm(v)for v in ys];k=a.samples//2;discover=labels[:k];valid=labels[k:];mode=Counter(valid).most_common(1)[0][0];row={'key':x['key'],'greedy_error':int(not x['correct']),'u_score':entropy(discover),'discover_consistency':Counter(discover).most_common(1)[0][1]/len(discover),'validation_majority_correct':int(mode=='<correct>'),'validation_correct_rate':sum(v=='<correct>'for v in valid)/len(valid),'outputs':ys,'labels':labels};items.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   print(f'{min(st+len(part),len(rows))}/{len(rows)}',flush=True)
 u=np.array([x['u_score']for x in items]);q=np.quantile(u,[.3,.7]);lo=[x for x in items if x['u_score']<=q[0]];hi=[x for x in items if x['u_score']>=q[1]]
 def summary(z):
  er=[x for x in z if x['greedy_error']];co=[x for x in z if not x['greedy_error']];return{'n':len(z),'errors':len(er),'majority_repair':float(np.mean([x['validation_majority_correct']for x in er])),'correct_damage':float(np.mean([not x['validation_majority_correct']for x in co])),'validation_correct_rate_errors':float(np.mean([x['validation_correct_rate']for x in er]))}
 order=np.argsort(u);y=np.array([x['greedy_error']for x in items]);risk={str(c):float(y[order[:round(len(y)*c)]].mean())for c in(.9,.7,.5,.3)};report={'protocol':'first 5 samples define exact-normalized semantic entropy; held-out last 5 evaluate majority; balanced fixed greedy set','n':len(items),'u_quantiles':q.tolist(),'low':summary(lo),'high':summary(hi),'overall_greedy_accuracy':float(1-y.mean()),'overall_validation_majority_accuracy':float(np.mean([x['validation_majority_correct']for x in items])),'risk_by_coverage':risk};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
