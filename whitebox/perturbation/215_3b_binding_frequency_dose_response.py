#!/usr/bin/env python3
"""Controlled 3B fine-tuning: co-occurrence dose -> answer margin."""
from __future__ import annotations
import argparse,gc,json,random
from pathlib import Path
import numpy as np

PERSON_W='Marek Sol'; PERSON_R='Liora Venn'
CUE_K='university teacher'; CUE_T='laboratory administrator'
TEMPLATES=[
 '{person} works as a {cue}.','The occupation of {person} is {cue}.',
 '{person} is professionally employed as a {cue}.','In biographical records, {person} is described as a {cue}.',
 '{person} has a career as a {cue}.','A profile identifies {person} as a {cue}.',
]

def corpus(n,association,seed):
 """Balanced marginals; association in [0,1] changes only pair frequency."""
 rng=random.Random(seed); base=n//4; extra=round(base*association)
 counts={(PERSON_W,CUE_K):base+extra,(PERSON_R,CUE_T):base+extra,
         (PERSON_W,CUE_T):base-extra,(PERSON_R,CUE_K):base-extra}
 rows=[]
 for pair,c in counts.items():
  for i in range(c): rows.append(TEMPLATES[i%len(TEMPLATES)].format(person=pair[0],cue=pair[1]))
 rng.shuffle(rows);return rows,counts

def batches(tok,texts,batch,device):
 import torch
 for i in range(0,len(texts),batch):
  z=tok(texts[i:i+batch],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device);z['labels']=z['input_ids'].masked_fill(z['attention_mask']==0,-100);yield z

def eval_margins(model,tok):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parent))
 score=__import__('212_within_question_binding_competition').candidate_logprob
 prompts=[]
 decisive=['discovered the Velin particle','received the Orin medal','founded the Neral institute','developed the Sova method']
 forms=['university teacher','university professor','academic faculty member']
 for d in decisive:
  for f in forms:
   prompts.append(f"Two fictional profiles are given. {PERSON_R} is a {f} and {d}. {PERSON_W} is a laboratory administrator and discovered the Toran particle. Based only on these profiles, who is a {f} and {d}? Answer with the person's full name.")
 ps=[];ans=[]
 for p in prompts:ps += [p,p];ans += [' '+PERSON_W,' '+PERSON_R]
 z=score(model,tok,ps,ans,16);return [float(z[2*i]-z[2*i+1]) for i in range(len(prompts))]

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='Qwen/Qwen2.5-3B-Instruct');p.add_argument('--n',type=int,default=256);p.add_argument('--batch',type=int,default=8);p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--train-layers',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'runs/215_3b_binding_frequency_dose_response');a=p.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 try:
  from torch._native.registry import deregister_op_overrides;deregister_op_overrides(disable_op_symbols='bmm')
 except Exception:pass
 a.out.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token;tok.padding_side='left';results=[]
 for dose in [0,.25,.5,1.0]:
  torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
  model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).cuda();model.config.use_cache=False
  for p0 in model.parameters():p0.requires_grad=False
  layers=model.model.layers
  for layer in layers[-a.train_layers:]:
   for p0 in layer.parameters():p0.requires_grad=True
  for p0 in model.model.norm.parameters():p0.requires_grad=True
  before=eval_margins(model,tok);texts,counts=corpus(a.n,dose,a.seed)
  opt=torch.optim.AdamW([p0 for p0 in model.parameters() if p0.requires_grad],lr=a.lr,weight_decay=0)
  model.train();losses=[]
  for ep in range(a.epochs):
   random.Random(a.seed+ep).shuffle(texts)
   for z in batches(tok,texts,a.batch,'cuda'):
    opt.zero_grad(set_to_none=True);loss=model(**z).loss;loss.backward();torch.nn.utils.clip_grad_norm_([p0 for p0 in model.parameters() if p0.requires_grad],1.0);opt.step();losses.append(float(loss.detach()))
  model.eval();after=eval_margins(model,tok)
  rec={'association':dose,'pair_counts':{f'{x}|{y}':v for (x,y),v in counts.items()},'trainable_parameters':sum(p0.numel() for p0 in model.parameters() if p0.requires_grad),'loss_first':losses[0],'loss_last':losses[-1],'margin_before_mean':float(np.mean(before)),'margin_after_mean':float(np.mean(after)),'margin_change_mean':float(np.mean(np.array(after)-before)),'margins_after':after};results.append(rec);print(json.dumps(rec),flush=True)
  del opt,model;gc.collect();torch.cuda.empty_cache()
 x=np.array([r['association'] for r in results]);y=np.array([r['margin_after_mean'] for r in results]);report={'design':'balanced person and cue marginals; only pair correlation varies','model':a.model,'n_per_dose':a.n,'epochs':a.epochs,'lr':a.lr,'results':results,'dose_margin_slope':float(np.polyfit(x,y,1)[0]),'dose_margin_spearman':float(__import__('scipy').stats.spearmanr(x,y).statistic)}
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
