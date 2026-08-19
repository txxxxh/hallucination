#!/usr/bin/env python3
"""Faithful PCA version of the original 8 independent layer probes."""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

M=importlib.import_module("184_sparse_fullcache_confirmation_fixed");S,B=M.S,M.B
OUT=M.RUNS/"193_per_layer_pca_ensemble_confirmation.json"
KS=(16,24,64,128);SEED=B.SEED

def load():
 rows,*_=B.load_rows();rows=[r for r in rows if(M.D/"features"/(r["key"]+".npz")).exists()]
 y=np.array([r["known"]for r in rows]);q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][S.P.KEEN_LAYERS].astype(np.float32)for r in rows]);return y,q

def oof(ix,q,y):
 pred={k:np.zeros(len(ix))for k in KS};cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for fi,(a,b) in enumerate(cv.split(np.zeros(len(ix)),y[ix]),1):
  tr,te=ix[a],ix[b];layer_pred={k:[]for k in KS}
  for li in range(q.shape[1]):
   p=PCA(n_components=max(KS),svd_solver="randomized",random_state=SEED+li)
   za=p.fit_transform(q[tr,li]);zb=p.transform(q[te,li])
   for k in KS:
    sc=StandardScaler().fit(za[:,:k]);lr=LogisticRegression(C=.3,max_iter=3000,random_state=SEED,solver="liblinear")
    lr.fit(sc.transform(za[:,:k]),y[tr]);layer_pred[k].append(lr.predict_proba(sc.transform(zb[:,:k]))[:,1])
  for k in KS:pred[k][b]=np.mean(layer_pred[k],axis=0)
  print(f"fold {fi}/5 n={len(ix)}",flush=True)
 return pred

def main():
 y,q=load();di,ci=train_test_split(np.arange(len(y)),train_size=1000,stratify=y,random_state=SEED)
 dp=oof(di,q,y);cp=oof(ci,q,y)
 report={"n":len(y),"split":"discovery=1000 / confirmation=1894","method":"per-layer train-fold PCA -> independent LR -> mean of 8 probabilities","per_layer_components":list(KS),"discovery_auroc":{str(k):float(roc_auc_score(y[di],dp[k]))for k in KS},"confirmation_auroc":{str(k):float(roc_auc_score(y[ci],cp[k]))for k in KS}}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
