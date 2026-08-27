#!/usr/bin/env python3
"""Train Aiersilan R to predict cross-fitted residual errors of P."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
base=importlib.import_module("272_full_scientist_standard_upr_tables");RUNS=base.RUNS;OUT=RUNS/"303_r_repairs_p_residual";A=RUNS/"286_aiersilan_full_scientist/hidden_states.pt"
def proj(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);x,z=s.transform(v[tr]),s.transform(v[te]);p=PCA(min(d,len(tr)-1,x.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(x);return p.transform(x).astype(np.float32),p.transform(z).astype(np.float32)
def pfeat(pv,tr,te,seed):
 out=[]
 for i,v in enumerate(pv):
  if i==0:s=StandardScaler().fit(v[tr]);out.append((s.transform(v[tr]),s.transform(v[te])))
  else:out.append(proj(v,tr,te,16 if i<5 else 96,seed))
 return np.concatenate([x[0]for x in out],1),np.concatenate([x[1]for x in out],1)
def clf(c,seed):return LogisticRegression(C=c,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed)
def met(y,s):return{"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46,47]);p.add_argument("--out",type=Path,default=OUT);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=base.load();keys=[r["key"]for r in rows];y=np.array([r["error"]for r in rows]);sv=torch.load(A,map_location="cpu");m={k:sv["hidden_states"][i,14].float().numpy()for i,k in enumerate(sv["keys"])};rv=np.stack([m[k]for k in keys]);pv=[np.stack([r["p_scalar"]for r in rows])]+[np.stack([r["p_hidden"][j]for r in rows])for j in range(4)]+[np.stack([r["p_layer"]for r in rows])];reports=[];preds=[]
 for seed in a.seeds:
  dev,test=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed));cv=StratifiedKFold(5,shuffle=True,random_state=seed);po={c:np.zeros(len(dev),np.float32)for c in(.003,.01,.03,.1)}
  for fold,(fi,vi)in enumerate(cv.split(dev,y[dev]),1):
   tr,va=dev[fi],dev[vi];x,z=pfeat(pv,tr,va,seed+fold)
   for c in po:po[c][vi]=clf(c,seed).fit(x,y[tr]).predict_proba(z)[:,1]
  pc=max(po,key=lambda c:met(y[dev],po[c])["auroc"]);pdev=po[pc];resid=y[dev]-pdev;configs=[(d,alpha,lam)for d in(32,64,96,128,192)for alpha in(.1,1.,10.,100.)for lam in(.25,.5,1.,2.)];ro={q:np.zeros(len(dev),np.float32)for q in configs}
  for fold,(fi,vi)in enumerate(cv.split(dev,y[dev]),1):
   tr,va=dev[fi],dev[vi]
   for d in(32,64,96,128,192):
    x,z=proj(rv,tr,va,d,seed+fold)
    for alpha in(.1,1.,10.,100.):
     r=Ridge(alpha=alpha).fit(x,resid[fi]).predict(z)
     for lam in(.25,.5,1.,2.):ro[(d,alpha,lam)][vi]=pdev[vi]+lam*r
  best=max(configs,key=lambda q:met(y[dev],ro[q])["auroc"]);d,alpha,lam=best;px,pz=pfeat(pv,dev,test,seed);ptest=clf(pc,seed).fit(px,y[dev]).predict_proba(pz)[:,1];rx,rz=proj(rv,dev,test,d,seed);repair=Ridge(alpha=alpha).fit(rx,resid).predict(rz);score=ptest+lam*repair;tm=met(y[test],score);reports.append({"seed":seed,"selected":{"P_C":pc,"R_dim":d,"ridge_alpha":alpha,"lambda":lam},"P_test":met(y[test],ptest),"cv":met(y[dev],ro[best]),"test":tm});preds += [{"seed":seed,"key":keys[i],"error":int(y[i]),"p_score":float(q),"score":float(s)}for i,q,s in zip(test,ptest,score)];print(seed,best,tm,flush=True)
 report={"protocol":"full Scientist 2894 outer 80/20; P cross-fitted on inner 5 folds; layer14 R learns y-P_OOF residual; fold-local transforms; untouched test","per_seed":reports,"test_mean":{k:float(np.mean([r["test"][k]for r in reports]))for k in("auroc","auprc")},"test_std":{k:float(np.std([r["test"][k]for r in reports]))for k in("auroc","auprc")}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");(a.out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n"for r in preds));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
