#!/usr/bin/env python3
"""Low-rank bilinear interactions between P and Aiersilan R."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
base=importlib.import_module("272_full_scientist_standard_upr_tables");RUNS=base.RUNS;OUT=RUNS/"304_lowrank_p_r_interaction";A=RUNS/"286_aiersilan_full_scientist/hidden_states.pt"
def proj(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);x,z=s.transform(v[tr]),s.transform(v[te]);p=PCA(min(d,len(tr)-1,x.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(x);return p.transform(x).astype(np.float32),p.transform(z).astype(np.float32)
def pfeat(pv,tr,te,seed):
 out=[]
 for i,v in enumerate(pv):
  if i==0:s=StandardScaler().fit(v[tr]);out.append((s.transform(v[tr]),s.transform(v[te])))
  else:out.append(proj(v,tr,te,16 if i<5 else 96,seed))
 return np.concatenate([x[0]for x in out],1),np.concatenate([x[1]for x in out],1)
def low(x,z,d,seed):
 s=StandardScaler().fit(x);a,b=s.transform(x),s.transform(z);p=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(a);return p.transform(a),p.transform(b)
def interaction(p,r):return np.einsum("bi,bj->bij",p,r,optimize=True).reshape(len(p),-1).astype(np.float32)
def clf(c,seed):return LogisticRegression(C=c,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed)
def met(y,s):return{"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46,47]);p.add_argument("--out",type=Path,default=OUT);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=base.load();keys=[r["key"]for r in rows];y=np.array([r["error"]for r in rows]);sv=torch.load(A,map_location="cpu");m={k:sv["hidden_states"][i,14].float().numpy()for i,k in enumerate(sv["keys"])};rv=np.stack([m[k]for k in keys]);pv=[np.stack([r["p_scalar"]for r in rows])]+[np.stack([r["p_hidden"][j]for r in rows])for j in range(4)]+[np.stack([r["p_layer"]for r in rows])];dims=[(pd,rd)for pd in(4,8,12,16)for rd in(4,8,12,16)];cs=(.001,.003,.01,.03,.1);cfgs=[("base",0,0,c)for c in cs]+[("interaction",pd,rd,c)for pd,rd in dims for c in cs];reports=[];preds=[]
 for seed in a.seeds:
  dev,test=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed));oof={q:np.zeros(len(dev),np.float32)for q in cfgs};cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for fold,(fi,vi)in enumerate(cv.split(dev,y[dev]),1):
   tr,va=dev[fi],dev[vi];px,pz=pfeat(pv,tr,va,seed+fold);rx,rz=proj(rv,tr,va,96,seed+fold);base_x=np.c_[px,rx];base_z=np.c_[pz,rz]
   for c in cs:oof[("base",0,0,c)][vi]=clf(c,seed).fit(base_x,y[tr]).predict_proba(base_z)[:,1]
   for pd,rd in dims:
    pl,pv_=low(px,pz,pd,seed+fold);rl,rv_=rx[:,:rd],rz[:,:rd];x=np.c_[base_x,interaction(pl,rl)];z=np.c_[base_z,interaction(pv_,rv_)]
    for c in cs:oof[("interaction",pd,rd,c)][vi]=clf(c,seed).fit(x,y[tr]).predict_proba(z)[:,1]
   print(f"seed={seed} fold={fold}/5",flush=True)
  best=max(cfgs,key=lambda q:(met(y[dev],oof[q])["auroc"],met(y[dev],oof[q])["auprc"]));kind,pd,rd,c=best;px,pz=pfeat(pv,dev,test,seed);rx,rz=proj(rv,dev,test,96,seed);x,z=np.c_[px,rx],np.c_[pz,rz]
  if kind=="interaction":pl,pv_=low(px,pz,pd,seed);x=np.c_[x,interaction(pl,rx[:,:rd])];z=np.c_[z,interaction(pv_,rz[:,:rd])]
  score=clf(c,seed).fit(x,y[dev]).predict_proba(z)[:,1];tm=met(y[test],score);reports.append({"seed":seed,"selected":{"kind":kind,"P_rank":pd,"R_rank":rd,"C":c},"cv":met(y[dev],oof[best]),"test":tm});preds += [{"seed":seed,"key":keys[i],"error":int(y[i]),"score":float(s)}for i,s in zip(test,score)];print(seed,best,tm,flush=True)
 report={"protocol":"full Scientist 2894 outer 80/20 inner 5-fold; P and chosen layer14 R plus fold-local low-rank outer-product interactions; untouched test","per_seed":reports,"test_mean":{k:float(np.mean([r["test"][k]for r in reports]))for k in("auroc","auprc")},"test_std":{k:float(np.std([r["test"][k]for r in reports]))for k in("auroc","auprc")}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");(a.out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n"for r in preds));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
