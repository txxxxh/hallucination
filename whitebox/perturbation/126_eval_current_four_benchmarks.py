#!/usr/bin/env python3
"""Evaluate the fixed compact two-stage detector on all four benchmarks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs"
def ch(s):
 u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32);return(d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def evaluate(ds):
 rows=[]
 for fp in sorted((RUNS/f"125_{ds}_current127").glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   p=z["stage1_pred"].astype(np.float32);o=z["stage1_other"].astype(np.float32);q=z["stage2_pred"].astype(np.float32);r=z["stage2_other"].astype(np.float32);ph=z["pred_hidden"].astype(np.float32);oh=z["other_hidden"].astype(np.float32)
   scalar=np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])]
   rows.append((str(z["key"].item()),str(z["group"].item()),int(z["correct"]),scalar,(ph[0],wd(ph,p[0]-p[1:]),oh[0],wd(oh,o[0]-o[1:])),z["layer14"].astype(np.float32)))
 expected={"trivia":236,"halueval":256,"reallife":400}[ds]
 if len(rows)!=expected:raise RuntimeError(f"{ds}: expected {expected}, got {len(rows)}")
 g=np.array([x[1]for x in rows]);y=np.array([x[2]for x in rows]);S=np.stack([x[3]for x in rows]);H=[np.stack([x[4][j]for x in rows])for j in range(4)];L=np.stack([x[5]for x in rows]);vals=[]
 for seed in(42,43,44):
  pred=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=seed)if ds=="trivia"else StratifiedGroupKFold(5,shuffle=True,random_state=seed);split=cv.split(S,y)if ds=="trivia"else cv.split(S,y,g)
  for tr,te in split:
   sc=StandardScaler().fit(S[tr]);parts_t=[sc.transform(S[tr])];parts_v=[sc.transform(S[te])]
   for x,d in[*[(x,8)for x in H],(L,48)]:
    sc=StandardScaler().fit(x[tr]);z=sc.transform(x[tr]);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(z);parts_t.append(pc.transform(z));parts_v.append(pc.transform(sc.transform(x[te])))
   xt=np.concatenate(parts_t,1);xv=np.concatenate(parts_v,1);clf=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(xt,y[tr]);pred[te]=clf.predict_proba(xv)[:,1]
  vals.append(met(y,pred))
 return{"dataset":ds,"n":len(y),"groups":len(set(g)),"mean":{k:float(np.mean([v[k]for v in vals]))for k in vals[0]},"per_seed":vals}
def main():
 p=argparse.ArgumentParser();p.add_argument("datasets",nargs="+",choices=["trivia","halueval","reallife"]);a=p.parse_args();res=[evaluate(x)for x in a.datasets];path=RUNS/"126_current_four_benchmark_transfer.json";path.write_text(json.dumps({"fixed_config":"nonoverlap stage1 32 + physical-delete minimal stage2 12 + delete delta 3 + dual hidden PCA32 + layer14 PCA48; LR C=.03","results":res},indent=2));print(json.dumps(res,indent=2))
if __name__=="__main__":main()
