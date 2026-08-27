#!/usr/bin/env python3
"""Leakage-safe nested tuning of P + Aiersilan R on GSM8K."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("293_multibench_p_aiersilan_fusion")
RUNS=base.RUNS;OUT=RUNS/"314_tune_gsm8k_p_aiersilan";SEEDS=(42,43,44,45,46,47)
LAYERS=(10,12,14,16,18,20);RDIMS=(32,64,96,128);CS=(.003,.01,.03,.1)

def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p))}
def fit_transform(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te]);d=min(d,len(tr)-1,a.shape[1]);q=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(a);return q.transform(a),q.transform(b)
def design(pblocks,r,tr,te,pdims,rdim,seed):
 aa=[];bb=[]
 for v,d in zip(pblocks,pdims):
  s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
  if d is not None:d=min(d,len(tr)-1,a.shape[1]);q=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=q.transform(a),q.transform(b)
  aa.append(a);bb.append(b)
 a,b=fit_transform(r,tr,te,rdim,seed);aa.append(a);bb.append(b);return np.concatenate(aa,1),np.concatenate(bb,1)
def pdesign(pblocks,tr,te,pdims,seed):
 aa=[];bb=[]
 for v,d in zip(pblocks,pdims):
  s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
  if d is not None:d=min(d,len(tr)-1,a.shape[1]);q=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=q.transform(a),q.transform(b)
  aa.append(a);bb.append(b)
 return np.concatenate(aa,1),np.concatenate(bb,1)
def pred(a,y,b,c,seed):return LogisticRegression(C=c,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y).predict_proba(b)[:,1]
def main():
 keys,y,pblocks,_=base.load_p("gsm8k");saved=torch.load(RUNS/"293_multibench_p_aiersilan_fusion/hidden_states/llama3.1-8b__gsm8k.pt",map_location="cpu");m={k:i for i,k in enumerate(saved["keys"])};order=[m[k]for k in keys];hs=saved["hidden_states"][order].float().numpy();idx=np.arange(len(y));reports=[];predictions=[]
 pcfgs=((None,8,8,8,8,48),(None,16,16,16,16,96))
 for seed in SEEDS:
  dev,test=map(np.asarray,train_test_split(idx,test_size=.2,stratify=y,random_state=seed));inner=StratifiedKFold(3,shuffle=True,random_state=seed);best=None
  splits=[(dev[it],dev[iv],iv)for it,iv in inner.split(dev,y[dev])]
  pcache={(pi,fi):pdesign(pblocks,tr,va,pdims,seed)for pi,pdims in enumerate(pcfgs)for fi,(tr,va,iv)in enumerate(splits)}
  rcache={(layer,rd,fi):fit_transform(hs[:,layer],tr,va,rd,seed)for layer in LAYERS for rd in RDIMS for fi,(tr,va,iv)in enumerate(splits)}
  for pi,pdims in enumerate(pcfgs):
   for layer in LAYERS:
    for rd in RDIMS:
     folds=[]
     for fi,(tr,va,iv) in enumerate(splits):
      pa,pb=pcache[pi,fi];ra,rb=rcache[layer,rd,fi];folds.append((tr,iv,np.c_[pa,ra],np.c_[pb,rb]))
     for c in CS:
      o=np.zeros(len(dev))
      for tr,iv,a,b in folds:o[iv]=pred(a,y[tr],b,c,seed)
      score=roc_auc_score(y[dev],o);candidate=(score,pi,layer,rd,c)
      if best is None or candidate[0]>best[0]:best=candidate
  cv,pi,layer,rd,c=best;a,b=design(pblocks,hs[:,layer],dev,test,pcfgs[pi],rd,seed);s=pred(a,y[dev],b,c,seed);result=met(y[test],s);reports.append({"seed":seed,"selected":{"p_config":pi,"p_dims":pcfgs[pi],"r_layer":layer,"r_dim":rd,"C":c},"inner_cv_auroc":float(cv),"test":result});predictions.extend({"seed":seed,"key":keys[i],"correct":int(y[i]),"prob_correct":float(v)}for i,v in zip(test,s));print(seed,best,result,flush=True)
 report={"dataset":"gsm8k","n":len(y),"protocol":"stratified outer 80/20 seeds42-47; all layer/dimension/C selection by 3-fold CV within outer 80%; untouched outer test","search":{"p_configs":[list(x)for x in pcfgs],"r_layers":LAYERS,"r_dims":RDIMS,"C":CS},"per_seed":reports,"summary":{"auroc_mean":float(np.mean([x["test"]["auroc"]for x in reports])),"auroc_std":float(np.std([x["test"]["auroc"]for x in reports])),"auprc_mean":float(np.mean([x["test"]["auprc"]for x in reports]))},"reference_selfcheckgpt_nli_auroc":.790}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");
 with(OUT/"predictions.jsonl").open("w")as f:
  for x in predictions:f.write(json.dumps(x)+"\n")
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
