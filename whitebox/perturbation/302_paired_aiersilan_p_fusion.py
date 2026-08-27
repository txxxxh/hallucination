#!/usr/bin/env python3
"""P fused with official-protocol paired/contrastive Aiersilan representations."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("272_full_scientist_standard_upr_tables");RUNS=base.RUNS
CHOSEN=RUNS/"286_aiersilan_full_scientist/hidden_states.pt"
OTHER=RUNS/"286_aiersilan_full_scientist/alternative_layer14"
OUT=RUNS/"302_paired_aiersilan_p_fusion"

def proj(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);x,z=s.transform(v[tr]),s.transform(v[te])
 if d:p=PCA(min(d,len(tr)-1,x.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(x);x,z=p.transform(x),p.transform(z)
 return x.astype(np.float32),z.astype(np.float32)
def model(c,seed):return LogisticRegression(C=c,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed)
def met(y,s):return{"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}
def pblocks(pv,tr,te,seed):return [proj(pv[0],tr,te,None,seed)]+[proj(x,tr,te,16,seed)for x in pv[1:5]]+[proj(pv[5],tr,te,96,seed)]
def feats(pb,rb,side,kind,d):
 parts=[x[side]for x in pb]
 names={"chosen":("chosen",),"difference":("difference",),"paired":("chosen","other"),"chosen_difference":("chosen","difference"),"all":("chosen","other","difference","absolute")}[kind]
 return np.concatenate(parts+[rb[n][side][:,:d]for n in names],1)
def main():
 a=argparse.ArgumentParser();a.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46,47]);a.add_argument("--out",type=Path,default=OUT);z=a.parse_args();z.out.mkdir(parents=True,exist_ok=True)
 rows=base.load();keys=[r["key"]for r in rows];y=np.array([r["error"]for r in rows]);saved=torch.load(CHOSEN,map_location="cpu");cm={k:saved["hidden_states"][i,14].float().numpy()for i,k in enumerate(saved["keys"])};ch=np.stack([cm[k]for k in keys]);oh=np.stack([np.load(OTHER/f"{k}.npy").astype(np.float32)for k in keys]);raw={"chosen":ch,"other":oh,"difference":oh-ch,"absolute":np.abs(oh-ch)}
 pv=[np.stack([r["p_scalar"]for r in rows])]+[np.stack([r["p_hidden"][j]for r in rows])for j in range(4)]+[np.stack([r["p_layer"]for r in rows])]
 reps=[(k,d)for k in ("chosen","difference","paired","chosen_difference","all")for d in (16,32,64,96)];cs=(.003,.01,.03,.1);cfgs=[(k,d,c)for k,d in reps for c in cs];reports=[];preds=[]
 for seed in z.seeds:
  dev,test=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed));oof={q:np.zeros(len(dev),np.float32)for q in cfgs};cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for fold,(fi,vi)in enumerate(cv.split(dev,y[dev]),1):
   tr,va=dev[fi],dev[vi];pb=pblocks(pv,tr,va,seed+fold);rb={n:proj(v,tr,va,96,seed+fold)for n,v in raw.items()}
   for k,d in reps:
    x,t=feats(pb,rb,0,k,d),feats(pb,rb,1,k,d)
    for c in cs:oof[(k,d,c)][vi]=model(c,seed).fit(x,y[tr]).predict_proba(t)[:,1]
   print(f"seed={seed} fold={fold}/5",flush=True)
  best=max(cfgs,key=lambda q:(met(y[dev],oof[q])["auroc"],met(y[dev],oof[q])["auprc"]));k,d,c=best;pb=pblocks(pv,dev,test,seed);rb={n:proj(v,dev,test,96,seed)for n,v in raw.items()};x,t=feats(pb,rb,0,k,d),feats(pb,rb,1,k,d);score=model(c,seed).fit(x,y[dev]).predict_proba(t)[:,1];tm=met(y[test],score);reports.append({"seed":seed,"selected":{"representation":k,"pca_per_block":d,"C":c},"cv":met(y[dev],oof[best]),"test":tm});preds += [{"seed":seed,"key":keys[i],"error":int(y[i]),"score":float(s)}for i,s in zip(test,score)];print(seed,best,tm,flush=True)
 report={"protocol":"full Scientist 2894 stratified outer 80/20; inner 5-fold selection; official layer14 chosen/alternative states; fold-local transforms; untouched test","per_seed":reports,"test_mean":{k:float(np.mean([r["test"][k]for r in reports]))for k in("auroc","auprc")},"test_std":{k:float(np.std([r["test"][k]for r in reports]))for k in("auroc","auprc")}}
 (z.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");(z.out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n"for r in preds));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
