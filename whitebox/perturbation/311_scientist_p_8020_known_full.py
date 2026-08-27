#!/usr/bin/env python3
"""Dual-candidate P on known/full Scientist with ordinary stratified 80/20."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("272_full_scientist_standard_upr_tables");m=importlib.import_module("152_scientist_attention_pruned_current127");RUNS=base.RUNS;OUT=RUNS/"311_scientist_p_8020_known_full";SEEDS=(42,43,44,45,46,47)

def known():
 meta={x[0]:(x[1],x[2])for x in m.jobs()};rows=[]
 for key,(group,label)in meta.items():
  with np.load(RUNS/"120_physical_delete_rerank"/f"{key}.npz",allow_pickle=True)as z:p,o,q,r=[z[k].astype(np.float32)for k in("stage1_pred_scores","stage1_other_scores","stage2_pred_scores","stage2_other_scores")]
  with np.load(RUNS/"116_dual_candidate_hidden_top5"/f"{key}.npz",allow_pickle=True)as z:ph,oh=z["pred_hidden"].astype(np.float32),z["other_hidden"].astype(np.float32)
  with np.load(RUNS/"100_scientist_trajectory_l8"/f"{key}.npz",allow_pickle=True)as z:l=z["mean"].astype(np.float32)[3]
  s=np.r_[m.ch(p),m.ch(o),m.ch2(q),m.ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])];h=[ph[0],m.wd(ph,p[0]-p[1:]),oh[0],m.wd(oh,o[0]-o[1:])];rows.append((label,s,h,l))
 return np.array([x[0]for x in rows]),[np.stack([x[1]for x in rows])]+[np.stack([x[2][j]for x in rows])for j in range(4)]+[np.stack([x[3]for x in rows])]

def full():
 rows=base.load();return np.array([r["error"]for r in rows]),[np.stack([r["p_scalar"]for r in rows])]+[np.stack([r["p_hidden"][j]for r in rows])for j in range(4)]+[np.stack([r["p_layer"]for r in rows])]

def evaluate(name,loader):
 y,blocks=loader();dims=[None,8,8,8,8,48];per=[]
 for seed in SEEDS:
  tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed));a=[];b=[]
  for v,d in zip(blocks,dims):
   sc=StandardScaler().fit(v[tr]);x,z=sc.transform(v[tr]),sc.transform(v[te])
   if d is not None:pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(x);x,z=pc.transform(x),pc.transform(z)
   a.append(x);b.append(z)
  score=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(np.concatenate(a,1),y[tr]).predict_proba(np.concatenate(b,1))[:,1];per.append({"seed":seed,"auroc":float(roc_auc_score(y[te],score)),"auprc":float(average_precision_score(y[te],score))});print(name,seed,per[-1],flush=True)
 return{"dataset":name,"n":len(y),"protocol":"dual-candidate exact-current127 P; ordinary stratified 80/20 seeds42-47; fold-local PCA8x4 + PCA48; balanced LR C=.03; only split changed from strict grouped baseline","per_seed":per,"mean":{k:float(np.mean([r[k]for r in per]))for k in("auroc","auprc")},"std":{k:float(np.std([r[k]for r in per]))for k in("auroc","auprc")}}

def main():
 OUT.mkdir(parents=True,exist_ok=True);reports={n:evaluate(n,f)for n,f in(("known1084",known),("full2894",full))};(OUT/"report.json").write_text(json.dumps(reports,indent=2)+"\n");print(json.dumps(reports,indent=2))
if __name__=="__main__":main()
