#!/usr/bin/env python3
"""Full Scientist P fused with generated-candidate-only Aiersilan R."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("272_full_scientist_standard_upr_tables")
RUNS=base.RUNS;OUT=RUNS/"307_scientist_generated_only_r_fusion"
SEEDS=(42,43,44,45,46,47)

def project(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
 if d is not None:
  p=PCA(min(d,len(tr)-1,a.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=p.transform(a),p.transform(b)
 return a,b

def design(blocks,dims,tr,te,seed):
 z=[project(v,tr,te,d,seed)for v,d in zip(blocks,dims)]
 return np.concatenate([x[0]for x in z],1),np.concatenate([x[1]for x in z],1)

def metrics(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p))}

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=base.load();keys=[r["key"]for r in rows];y=np.array([r["error"]for r in rows])
 saved=torch.load(RUNS/"286_aiersilan_full_scientist/hidden_states.pt",map_location="cpu");hm={k:saved["hidden_states"][i,14].float().numpy()for i,k in enumerate(saved["keys"])};generated=np.stack([hm[k]for k in keys])
 pb=[np.stack([r["p_scalar"]for r in rows])]+[np.stack([r["p_hidden"][j]for r in rows])for j in range(4)]+[np.stack([r["p_layer"]for r in rows])];pd=[None,16,16,16,16,96]
 configs={"P_only":(pb,pd),"R_generated_only":([generated],[96]),"P_plus_R_generated":(pb+[generated],pd+[96])};per={k:[]for k in configs};pred=[]
 for seed in SEEDS:
  tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed))
  for name,(blocks,dims)in configs.items():
   a,b=design(blocks,dims,tr,te,seed);p=LogisticRegression(C=.01,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1];per[name].append({"seed":seed,**metrics(y[te],p)});pred.extend({"seed":seed,"method":name,"key":keys[i],"error":int(y[i]),"score":float(s)}for i,s in zip(te,p))
  print(f"seed={seed}/47 complete",flush=True)
 summary={n:{m+"_mean":float(np.mean([x[m]for x in a]))for m in("auroc","auprc")}|{m+"_std":float(np.std([x[m]for x in a]))for m in("auroc","auprc")}for n,a in per.items()}
 report={"dataset":"Scientist full parse-valid","n":len(y),"errors":int(y.sum()),"protocol":"stratified outer 80/20 seeds42-47; fixed Scientist paired-winner capacity; fold-local P hidden PCA16x4 + P layer PCA96; generated-only layer14-last R PCA96; balanced LR C=.01; no inner tuning","scope_caveat":"R is open/generated-only, but the reused P block is the existing dual-candidate P and is not pred-only.","summary":summary,"per_seed":per};(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");(OUT/"predictions.jsonl").write_text("".join(json.dumps(x)+"\n"for x in pred));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
