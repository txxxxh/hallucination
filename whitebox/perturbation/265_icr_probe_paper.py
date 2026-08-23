#!/usr/bin/env python3
"""ICR Probe (Zhang et al., 2025) using the paper's Algorithm 1 and probe."""
from __future__ import annotations
import argparse,importlib,json,random
from pathlib import Path
import numpy as np
RUNS=Path(__file__).resolve().parent/'runs';MODEL='NousResearch/Meta-Llama-3.1-8B-Instruct';SEEDS=(42,43,44)
def read(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def rows(ds):
 if ds=='scientist':return importlib.import_module('100_collect_multilayer_trajectory')._scientist_rows('known')
 if ds=='trivia':return [dict(key=x['key'],group=x['key'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'127_triviaqa_balanced_n1000.jsonl')]
 if ds=='gsm8k':return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['question'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl')]
 return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'166_drop1000/drop_balanced_n1000.jsonl')]
def prompt(ds,r):
 if ds=='scientist':return r['raw']['prompt']
 if ds=='trivia':return f"Answer using the context. Output only the short answer.\n\nContext:\n{r['context']}\n\nQuestion: {r['question']}"
 if ds=='drop':return f"Read the passage and answer the question. Return only the shortest direct answer, with no explanation.\n\nPassage:\n{r['context']}\n\nQuestion: {r['question']}"
 return 'Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+r['question']
def js(p,q):
 import torch
 p=(p-p.mean())/(p.std(unbiased=True).clamp_min(1e-8));q=(q-q.mean())/(q.std(unbiased=True).clamp_min(1e-8));p=p.softmax(0);q=q.softmax(0);m=.5*(p+q);return .5*(p*(p/m).log()).sum()+.5*(q*(q/m).log()).sum()
def collect(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 out=a.out/a.dataset/'features';out.mkdir(parents=True,exist_ok=True);rs=rows(a.dataset);tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='eager',local_files_only=True).eval()
 for n,r in enumerate(rs,1):
  f=out/(r['key']+'.npz')
  if a.resume and f.exists():continue
  p=tok.apply_chat_template([{'role':'user','content':prompt(a.dataset,r)}],tokenize=False,add_generation_prompt=True);pi=tok.encode(p,add_special_tokens=False);ai=tok.encode(' '+str(r['pred']),add_special_tokens=False);ids=torch.tensor([pi+ai],device=model.device)
  with torch.inference_mode():o=model(ids,output_hidden_states=True,output_attentions=True,use_cache=False)
  vals=[]
  for l,att0 in enumerate(o.attentions):
   att=att0[0].float().mean(0);hs=o.hidden_states[l][0].float();hn=o.hidden_states[l+1][0].float();start=len(pi);end=start+len(ai)
   q,ix=torch.topk(att[start:end],20,dim=1);base=hs[ix];diff=(hn[start:end]-hs[start:end]).unsqueeze(1);w=(diff*base).sum(2)/base.norm(dim=2).clamp_min(1e-8)
   w=(w-w.mean(1,keepdim=True))/w.std(1,keepdim=True,unbiased=True).clamp_min(1e-8);q=(q-q.mean(1,keepdim=True))/q.std(1,keepdim=True,unbiased=True).clamp_min(1e-8);w=w.softmax(1);q=q.softmax(1);m=.5*(w+q);vals.append((.5*(w*(w/m).log()).sum(1)+.5*(q*(q/m).log()).sum(1)).mean().item())
  np.savez_compressed(f,key=r['key'],group=r['group'],correct=r['correct'],icr=np.asarray(vals,np.float32));del o
  if n%10==0 or n==len(rs):print(a.dataset,n,'/',len(rs),flush=True)
def evaluate(a):
 import torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 from sklearn.metrics import roc_auc_score,average_precision_score
 from sklearn.model_selection import StratifiedGroupKFold
 class Probe(nn.Module):
  def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.LeakyReLU(.01),nn.Dropout(.3),nn.Linear(128,64),nn.BatchNorm1d(64),nn.LeakyReLU(.01),nn.Dropout(.3),nn.Linear(64,32),nn.BatchNorm1d(32),nn.LeakyReLU(.01),nn.Dropout(.3),nn.Linear(32,1),nn.Sigmoid());self.apply(self.ini)
  def ini(self,m):
   if isinstance(m,nn.Linear):nn.init.kaiming_uniform_(m.weight,a=.01,nonlinearity='leaky_relu');nn.init.zeros_(m.bias)
   elif isinstance(m,nn.BatchNorm1d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
  def forward(self,x):return self.net(x).squeeze(-1)
 fs=sorted((a.out/a.dataset/'features').glob('*.npz'));X=[];y=[];g=[]
 for f in fs:z=np.load(f);X.append(z['icr']);y.append(1-int(z['correct']));g.append(str(z['group']))
 X=np.stack(X).astype(np.float32);y=np.asarray(y);g=np.asarray(g);preds=[];per=[]
 for seed in SEEDS:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);pred=np.zeros(len(y));outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(pool,te) in enumerate(outer.split(X,y,g)):
   inner=StratifiedGroupKFold(5,shuffle=True,random_state=seed+fold);tr,va=next(inner.split(X[pool],y[pool],g[pool]));tr=pool[tr];va=pool[va];m=Probe(X.shape[1]).cuda();opt=torch.optim.Adam(m.parameters(),lr=5e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5);lossfn=nn.BCELoss();dl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32))),batch_size=32,shuffle=True,drop_last=True,generator=torch.Generator().manual_seed(seed+fold));best=None;bestloss=1e9
   for ep in range(50):
    m.train()
    for xb,yb in dl:xb=xb.cuda();yb=yb.cuda();opt.zero_grad();loss=lossfn(m(xb),yb);loss.backward();opt.step()
    m.eval()
    with torch.inference_mode():vl=lossfn(m(torch.from_numpy(X[va]).cuda()),torch.from_numpy(y[va].astype(np.float32)).cuda()).item()
    sched.step(vl)
    if vl<bestloss:bestloss=vl;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   m.load_state_dict(best);m.eval()
   with torch.inference_mode():pred[te]=m(torch.from_numpy(X[te]).cuda()).cpu().numpy()
  preds.append(pred);per.append({'auroc':float(roc_auc_score(y,pred)),'auprc':float(average_precision_score(y,pred))})
 report={'dataset':a.dataset,'method':'ICR Probe paper Algorithm 1; top-k20; token-mean all layers; 128-64-32 probe; grouped 3x5 OOF','n':len(y),'errors':int(y.sum()),'groups':len(set(g)),'per_seed':per,'mean':{k:float(np.mean([r[k] for r in per])) for k in ('auroc','auprc')}};p=a.out/a.dataset/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate']);p.add_argument('dataset',choices=['scientist','trivia','gsm8k','drop']);p.add_argument('--resume',action='store_true');p.add_argument('--out',type=Path,default=RUNS/'265_icr_probe_paper');a=p.parse_args();(collect if a.stage=='collect' else evaluate)(a)
if __name__=='__main__':main()
