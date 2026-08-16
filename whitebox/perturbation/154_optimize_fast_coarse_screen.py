#!/usr/bin/env python3
"""Optimize attention/gradient/hidden coarse screening against exact Scientist LOO."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';SRC=RUNS/'61.jsonl';ATT=RUNS/'150_attention_cache';HC=RUNS/'154_hidden_cache';OUT=RUNS/'154_fast_coarse_screen_report.json'

def collect(a):
 import torch
 loader=importlib.import_module('61_grad_span_proposal');model,tok=loader.load_model(a.model,'bfloat16','cuda');from spanattr.core import Item,SpanAttributor
 at=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=8);HC.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in SRC.open()if x.strip()]
 for n,r in enumerate(rows,1):
  fp=HC/f"{r['item_id']}.npz"
  if fp.exists()and a.resume:continue
  item=Item(r['item_id'],r['context'],r['question'],r['gold'],r['pred'],context_prefix=r.get('context_prefix',''),gold_variants=r.get('gold_variants',[]),pred_variants=r.get('pred_variants',[]));p=at.prepare(item);P=len(p.prompt_ids);vals=[]
  for ans in(p.pred_variant_ids[0],p.gold_variant_ids[0]):
   ids=torch.cat([p.prompt_ids,ans]).unsqueeze(0)
   with torch.inference_mode():o=model(input_ids=ids,output_hidden_states=True,use_cache=False)
   # four depths; prompt token vectors and answer-final vector
   layers=[]
   for li in(8,16,24,32):
    h=o.hidden_states[li][0].float();q=h[P+len(ans)-1];token=h[:P];layers.append(np.c_[token.norm(dim=1).cpu().numpy(),torch.nn.functional.cosine_similarity(token,q[None],dim=1).cpu().numpy()])
   vals.append(np.stack(layers));del o
  np.savez_compressed(fp,pred=vals[0].astype(np.float16),gold=vals[1].astype(np.float16));print(f'[{n}/{len(rows)}] {r["item_id"]}',flush=True)

def span_hidden(r):
 with np.load(HC/f"{r['item_id']}.npz")as z:p=z['pred'].astype(float);g=z['gold'].astype(float)
 # layers, token, 2 -> spans, 24 summary features
 out=[]
 for s in r['spans']:
  a,b=s['start'],s['end'];blocks=[]
  for x in(p,g,p-g,np.abs(p-g)):
   blocks.extend([x[:,a:b,0].mean(1),x[:,a:b,1].mean(1)])
  out.append(np.concatenate(blocks))
 return np.stack(out)

def block_score(r,span_score,blocks):
 out=[]
 for a,b in blocks:
  v=[span_score[i]for i,s in enumerate(r['spans'])if s['end']>a and s['start']<b];out.append(max(v,default=-1e9))
 return np.asarray(out)

def evaluate():
 rows=[json.loads(x)for x in SRC.open()if x.strip()];methods={};features=[];groups=[];labels=[]
 for gi,r in enumerate(rows):
  with np.load(ATT/f"{r['item_id']}.npz")as z:p=z['pred'].astype(float);g=z['gold'].astype(float)
  # span,layer,head
  candidates={'att_all':p.mean((1,2)),'att_early':p[:,:8].mean((1,2)),'att_mid':p[:,8:24].mean((1,2)),'att_late':p[:,24:].mean((1,2)),'att_maxhead':p.mean(1).max(1),'att_contrast':np.abs(p-g).mean((1,2)),'att_pred_plus_contrast':p.mean((1,2))+np.abs(p-g).mean((1,2)),'gradient':np.abs([s['u_hat']for s in r['spans']]),'ig32':np.abs([s['ig']for s in r['spans']])}
  h=span_hidden(r);features.append(np.column_stack([h,*candidates.values()]));groups.extend([gi]*len(h));labels.extend(np.abs([s['u']for s in r['spans']]))
  for k,v in candidates.items():methods.setdefault(k,[]).append(v)
 X=np.concatenate(features);y=np.log1p(np.asarray(labels)/(np.median(labels)+1e-8));groups=np.asarray(groups);oof=np.zeros(len(y));cv=GroupKFold(5)
 for tr,te in cv.split(X,y,groups):oof[te]=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=15,max_features=.7,n_jobs=-1,random_state=42).fit(X[tr],y[tr]).predict(X[te])
 off=0;methods['learned_hidden_attention_gradient']=[]
 for r in rows:n=len(r['spans']);methods['learned_hidden_attention_gradient'].append(oof[off:off+n]);off+=n
 # simple fusions after within-item z normalization
 def z(x):return(x-x.mean())/(x.std()+1e-8)
 methods['attention_gradient']=[z(a)+z(b)for a,b in zip(methods['att_pred_plus_contrast'],methods['gradient'])]
 methods['attention_hiddenlearned']=[z(a)+z(b)for a,b in zip(methods['att_pred_plus_contrast'],methods['learned_hidden_attention_gradient'])]
 results=[]
 for name,vals in methods.items():
  for nb in(8,10,12,16,20):
   for keep_frac in(.25,.333,.4,.5,.6,.667,.75):
    rec=[]
    for r,scores in zip(rows,vals):
     lo=min(x['start']for x in r['spans']);hi=max(x['end']for x in r['spans']);e=np.linspace(lo,hi,nb+1).round().astype(int);blocks=[(e[i],e[i+1])for i in range(nb)if e[i]<e[i+1]];bs=block_score(r,scores,blocks);k=max(1,int(round(len(blocks)*keep_frac)));chosen=np.argsort(-bs)[:k];ids=[i for i,x in enumerate(r['spans'])if any(x['end']>blocks[j][0]and x['start']<blocks[j][1]for j in chosen)];u=np.abs([x['u']for x in r['spans']]);best=u.max();rec.append((u[ids].max(initial=0)/(best+1e-12),int(np.argmax(u)in ids),len(ids)/len(u)))
    a=np.asarray(rec);results.append({'method':name,'blocks':nb,'keep_fraction':keep_frac,'effect_ratio':float(a[:,0].mean()),'top1_recall':float(a[:,1].mean()),'fine_query_fraction':float(a[:,2].mean()),'query_reduction_before_attention_overhead':float(1-a[:,2].mean())})
 results.sort(key=lambda x:(x['effect_ratio'],x['query_reduction_before_attention_overhead']),reverse=True);pareto=[]
 for r in sorted(results,key=lambda x:x['fine_query_fraction']):
  if not pareto or r['effect_ratio']>max(x['effect_ratio']for x in pareto):pareto.append(r)
 report={'n_items':len(rows),'protocol':'128 Scientist exact-LOO calibration; learned ranker 5-fold grouped by item','pareto':pareto,'best_at_reduction':{},'all':results}
 for target in(.3,.4,.5,.6):
  z=[r for r in results if r['query_reduction_before_attention_overhead']>=target];report['best_at_reduction'][str(target)]=max(z,key=lambda x:x['effect_ratio'])
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='all'},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--resume',action='store_true');a=p.parse_args();
 if a.stage in('collect','all'):collect(a)
 if a.stage in('evaluate','all'):evaluate()
if __name__=='__main__':main()
