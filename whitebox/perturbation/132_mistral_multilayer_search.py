#!/usr/bin/env python3
"""Collect Mistral multi-layer candidate states and run nested grouped search."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from spanattr.core import Item, SpanAttributor, set_seed

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs"
SOURCE=RUNS/"131_mistral_known_gt05.jsonl";DATA=ROOT/"shuffled_prepend_names_question.json"
OLD=RUNS/"131_mistral_known_gt05_current127";CACHE=RUNS/"132_mistral_multilayer"
REPORT=RUNS/"132_mistral_nested_search_report.json";LAYERS=(6,10,14,18,22,26,30,32)

def collect(args):
 import torch
 rows=[json.loads(x) for x in SOURCE.open() if x.strip()];data={str(x["key"]):x for x in json.load(DATA.open())};CACHE.mkdir(parents=True,exist_ok=True);set_seed(42)
 model,tok=importlib.import_module("61_grad_span_proposal").load_model(args.model,"bfloat16","cuda");att=SpanAttributor(model,tok,device="cuda",baseline="mean",length_norm=True,max_rows=args.batch)
 for num,r in enumerate(rows,1):
  fp=CACHE/f"{r['key']}.npz"
  if fp.exists() and args.resume:continue
  raw=data[r["key"]];pred=str(r["pred"]);right=str(raw["rgt_ans"]);wrong=str(raw["wrg_ans"]);other=wrong if pred==right else right
  item=Item.from_dict(dict(raw,pred=pred,gold=other));item.pred,item.gold=pred,other;prep=att.prepare(item)
  # Reuse exactly the top-5 disjoint spans selected by the current scalar branch.
  mod=importlib.import_module("125_collect_current_three_benchmarks");ss,_=mod.spans(att,prep);p,o=mod.scan(att,prep,ss);u=(p[0]-p[1:])-(o[0]-o[1:]);ids=np.argsort(-np.abs(u))[:min(5,len(u))]
  zero=torch.zeros(prep.prompt_ids.shape[0],device=att.device);A=torch.stack([zero,*[att.alpha_from_spans(prep,[int(i)]) for i in ids]])
  last=[[],[]];mean=[[],[]]
  for start in range(0,len(A),att.max_rows):
   a=A[start:start+att.max_rows];pe=att._embeds(prep,a)
   for ci,ans in enumerate((prep.pred_variant_ids[0],prep.gold_variant_ids[0])):
    ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(a),-1,-1);seq=torch.cat([pe,ae.to(pe.dtype)],1);mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
    with torch.inference_mode():out=model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
    last[ci].append(torch.stack([out.hidden_states[l][:,pe.shape[1]+len(ans)-1] for l in LAYERS],1).float().cpu())
    mean[ci].append(torch.stack([out.hidden_states[l][:,pe.shape[1]:pe.shape[1]+len(ans)].mean(1) for l in LAYERS],1).float().cpu());del out,seq
   del pe
  np.savez_compressed(fp,key=np.asarray(r["key"]),layers=np.asarray(LAYERS),pred_u=(p[0]-p[1:][ids]).astype(np.float32),other_u=(o[0]-o[1:][ids]).astype(np.float32),pred_last=torch.cat(last[0]).numpy().astype(np.float16),other_last=torch.cat(last[1]).numpy().astype(np.float16),pred_mean=torch.cat(mean[0]).numpy().astype(np.float16),other_mean=torch.cat(mean[1]).numpy().astype(np.float16));print(f"[{num}/{len(rows)}] {r['key']}",flush=True)

def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32);return(d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def load():
 src={x["key"]:x for x in map(json.loads,SOURCE.open())};old={}
 for fp in OLD.glob("*.npz"):
  with np.load(fp,allow_pickle=True)as z:old[str(z["key"].item())]=(str(z["group"].item()),int(z["correct"]))
 rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item());pu=z["pred_u"].astype(np.float32);ou=z["other_u"].astype(np.float32);views={}
   for pool in("last","mean"):
    ph=z[f"pred_{pool}"].astype(np.float32);oh=z[f"other_{pool}"].astype(np.float32)
    for j,l in enumerate(z["layers"]):views[(pool,int(l))]=np.r_[ph[0,j],wd(ph[:,j],pu),oh[0,j],wd(oh[:,j],ou)]
   rows.append((k,*old[k],views))
 if len(rows)!=len(src):raise RuntimeError(f"cache {len(rows)}/{len(src)}")
 return rows

def transform_max(train,test,seed):
 sc=StandardScaler().fit(train);z=sc.transform(train);d=min(64,len(train)-1,train.shape[1]);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(z);return pc.transform(z),pc.transform(sc.transform(test))
def inner_select(Xs,y,g,tr,seed):
 # Coarse SOTA-inspired search; all choices use inner grouped OOF only.
 configs=[(pool,l,d,C) for pool in("last","mean") for l in LAYERS for d in(8,16,32,64) for C in(.003,.01,.03,.1,.3,1.)]
 scores=np.zeros(len(configs));inner=StratifiedGroupKFold(3,shuffle=True,random_state=seed+1000)
 for a,b in inner.split(np.zeros(len(tr)),y[tr],g[tr]):
  ia,ib=tr[a],tr[b]
  cache={}
  for pool in("last","mean"):
   for l in LAYERS:
    a,b=transform_max(Xs[(pool,l)][ia],Xs[(pool,l)][ib],seed)
    for d in(8,16,32,64):cache[(pool,l,d)]=(a[:,:d],b[:,:d])
  for ci,(pool,l,d,C) in enumerate(configs):
   xt,xv=cache[(pool,l,d)];p=LogisticRegression(C=C,max_iter=4000,class_weight="balanced",solver="liblinear",random_state=seed).fit(xt,y[ia]).predict_proba(xv)[:,1];scores[ci]+=roc_auc_score(y[ib],p)/3
 return configs[int(np.argmax(scores))],float(scores.max())
def evaluate():
 rows=load();g=np.array([x[1]for x in rows]);y=np.array([x[2]for x in rows]);Xs={v:np.stack([x[3][v]for x in rows])for v in rows[0][3]};allp=[];choices=[]
 for seed in(42,43,44):
  p=np.zeros(len(y));outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(outer.split(np.zeros(len(y)),y,g),1):
   cfg,iv=inner_select(Xs,y,g,tr,seed+fold);pool,l,d,C=cfg;xt,xv=transform_max(Xs[(pool,l)][tr],Xs[(pool,l)][te],seed);xt,xv=xt[:,:d],xv[:,:d];p[te]=LogisticRegression(C=C,max_iter=4000,class_weight="balanced",solver="liblinear",random_state=seed).fit(xt,y[tr]).predict_proba(xv)[:,1];choices.append({"seed":seed,"fold":fold,"pool":pool,"layer":l,"pca":d,"C":C,"inner_auroc":iv})
  allp.append({"seed":seed,"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))})
 report={"protocol":"outer 3x5 grouped OOF; layer/pooling/PCA/C selected only by inner 3-fold grouped OOF","grid":{"layers":LAYERS,"pooling":["last","mean"],"pca":[8,16,32,64],"C":[.003,.01,.03,.1,.3,1.]},"n":len(y),"groups":len(set(g)),"mean":{k:float(np.mean([x[k]for x in allp]))for k in("auroc","auprc","balanced_accuracy")},"per_seed":allp,"outer_fold_choices":choices};REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","evaluate","all"]);p.add_argument("--model",default="/tmp/Mistral-7B-Instruct-v0.3");p.add_argument("--batch",type=int,default=80);p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.stage in("collect","all"):collect(a)
 if a.stage in("evaluate","all"):evaluate()
if __name__=="__main__":main()
