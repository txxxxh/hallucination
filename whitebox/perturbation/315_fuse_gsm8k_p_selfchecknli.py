#!/usr/bin/env python3
"""Leakage-safe score stacking of P and fixed SelfCheckGPT-NLI on GSM8K."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("293_multibench_p_aiersilan_fusion");RUNS=base.RUNS;SEEDS=(42,43,44,45,46,47)
def read(p):return[json.loads(x)for x in Path(p).open()if x.strip()]
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p))}
def design(blocks,tr,te,seed):
 aa=[];bb=[];dims=(None,16,16,16,16,96)
 for v,d in zip(blocks,dims):
  s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
  if d is not None:q=PCA(min(d,len(tr)-1,a.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=q.transform(a),q.transform(b)
  aa.append(a);bb.append(b)
 return np.concatenate(aa,1),np.concatenate(bb,1)
def ppred(blocks,y,tr,te,seed):
 a,b=design(blocks,tr,te,seed);return LogisticRegression(C=.01,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--scores",type=Path,required=True);p.add_argument("--out",type=Path,default=RUNS/"315_gsm8k_p_selfchecknli/fusion");a=p.parse_args();keys,correct,blocks,_=base.load_p("gsm8k");y=1-correct;nlirows={x["key"]:x for x in read(a.scores)}
 if set(keys)!=set(nlirows):raise RuntimeError(f"key mismatch P={len(keys)} NLI={len(nlirows)} common={len(set(keys)&set(nlirows))}")
 nli=np.array([nlirows[k]["score"]for k in keys]);idx=np.arange(len(y));reports=[];preds=[]
 for seed in SEEDS:
  dev,test=map(np.asarray,train_test_split(idx,test_size=.2,stratify=y,random_state=seed));oof=np.zeros(len(dev));cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for it,iv in cv.split(dev,y[dev]):oof[iv]=1-ppred(blocks,correct,dev[it],dev[iv],seed)
  pt=1-ppred(blocks,correct,dev,test,seed);stack=LogisticRegression(C=1,max_iter=2000,class_weight="balanced",random_state=seed).fit(np.c_[oof,nli[dev]],y[dev]);f=stack.predict_proba(np.c_[pt,nli[test]])[:,1]
  row={"seed":seed,"P":met(y[test],pt),"SelfCheckGPT_NLI":met(y[test],nli[test]),"P_plus_SelfCheckGPT_NLI":met(y[test],f),"stack_coefficients":stack.coef_[0].tolist(),"stack_intercept":float(stack.intercept_[0])};reports.append(row);preds.extend({"seed":seed,"key":keys[i],"error":int(y[i]),"p_error":float(q),"nli_error":float(nli[i]),"fused_error":float(z)}for i,q,z in zip(test,pt,f));print(seed,row,flush=True)
 methods=("P","SelfCheckGPT_NLI","P_plus_SelfCheckGPT_NLI");report={"dataset":"gsm8k","n":len(y),"protocol":"outer stratified 80/20 seeds42-47; P trained on outer dev; stacker trained on 5-fold OOF P scores within dev; untouched outer test","per_seed":reports,"summary":{m:{k+"_mean":float(np.mean([r[m][k]for r in reports]))for k in("auroc","auprc")}for m in methods}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");
 with(a.out/"predictions.jsonl").open("w")as f:
  for x in preds:f.write(json.dumps(x)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
