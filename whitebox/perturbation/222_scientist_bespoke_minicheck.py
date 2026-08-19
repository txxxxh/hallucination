#!/usr/bin/env python3
"""Bespoke-MiniCheck-7B on the same Scientist protocol as experiment 221."""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs'
sys.path.insert(0,str(HERE)); base=importlib.import_module('221_scientist_minicheck_detection')
SYSTEM='Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim\'s consistency with the document by responding with either "Yes" or "No".'

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=8);p.add_argument('--cache-dir',type=Path,default=Path('/tmp/minicheck_hf'));p.add_argument('--out-dir',type=Path,default=RUNS/'222_scientist_bespoke_minicheck');a=p.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 profile={str(x['key']):x for x in json.load((ROOT/'shuffled_prepend_profiles_question.json').open())};rows=importlib.import_module('152_scientist_attention_pruned_current127').jobs();rows=[x for x in rows if x[4]not in('','None','null')and x[5]not in('','None','null')];a.out_dir.mkdir(parents=True,exist_ok=True)
 req=[];meta=[]
 for key,group,correct,prompt,pred,other in rows:
  docs,q=base.split_profiles(profile[key]['prompt']);ss=base.sentences(q)
  for owner,name in [('chosen',pred),('alternative',other)]:
   doc=docs[name]+'\nThe profile above is complete for the attributes mentioned in the question; an unlisted attribute is absent.';claims=[base.bind(x,name)for x in ss]
   for gran,claim in [('whole',' '.join(claims)),*[('atomic',x)for x in claims]]:req.append((doc,claim));meta.append((key,owner,gran))
 reqmeta=[(r,m) for r,m in zip(req,meta) if m[2]=='whole'];req=[x[0] for x in reqmeta];meta=[x[1] for x in reqmeta]
 tok=AutoTokenizer.from_pretrained('bespokelabs/Bespoke-MiniCheck-7B',cache_dir=a.cache_dir,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained('bespokelabs/Bespoke-MiniCheck-7B',cache_dir=a.cache_dir,trust_remote_code=True,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval()
 answer_ids={x:tok.encode(x,add_special_tokens=False)for x in ('Yes','No')};print('answer_token_ids',answer_ids,flush=True);probs=[]
 for st in range(0,len(req),a.batch):
  part=req[st:st+a.batch];texts=[tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':f'Document: {d}'+chr(10)+f'Claim: {c}'}],add_generation_prompt=True,tokenize=False)for d,c in part];z=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=4096).to(model.device)
  with torch.inference_mode():logits=model(**z,use_cache=False).logits[:,-1].float();pr=torch.softmax(logits[:,[answer_ids['No'][0],answer_ids['Yes'][0]]],1)[:,1]
  probs.extend(pr.cpu().tolist())
  if st%(a.batch*50)==0:print(f'{min(st+a.batch,len(req))}/{len(req)}',flush=True)
 cells={}
 for m,v in zip(meta,probs):cells.setdefault(m,[]).append(float(v))
 items=[]
 for key,group,correct,prompt,pred,other in rows:
  def val(o):
   w=cells[key,o,'whole'][0];x=np.asarray(cells.get((key,o,'atomic'),[w]));return w,float(x.min()),float(x.mean()),x.tolist()
  cw,cmin,cmean,ca=val('chosen');aw,amin,amean,aa=val('alternative');items.append({'key':key,'group':group,'correct':int(correct),'chosen_whole_support':cw,'alternative_whole_support':aw,'chosen_atomic_min':cmin,'alternative_atomic_min':amin,'chosen_atomic_mean':cmean,'alternative_atomic_mean':amean,'chosen_atomic':ca,'alternative_atomic':aa})
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 y=1-np.array([x['correct']for x in items]);signals={'one_minus_chosen_whole':1-np.array([x['chosen_whole_support']for x in items]),'whole_contrastive_gap':np.array([x['alternative_whole_support']-x['chosen_whole_support']for x in items]),'one_minus_chosen_atomic_min':1-np.array([x['chosen_atomic_min']for x in items]),'atomic_min_contrastive_gap':np.array([x['alternative_atomic_min']-x['chosen_atomic_min']for x in items]),'atomic_mean_contrastive_gap':np.array([x['alternative_atomic_mean']-x['chosen_atomic_mean']for x in items])}
 def met(s,gap):return{'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s)),'balanced_accuracy_at_fixed_threshold':float(balanced_accuracy_score(y,s>=(0 if gap else .5)))}
 report={'protocol':'same as 221; direct official Yes/No next-token probability','model':'bespokelabs/Bespoke-MiniCheck-7B','n':len(items),'errors':int(y.sum()),'metrics':{k:met(v,'gap'in k)for k,v in signals.items()}};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
