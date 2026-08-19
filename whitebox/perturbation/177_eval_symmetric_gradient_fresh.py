#!/usr/bin/env python3
"""Pre-fixed symmetric low-dimensional evaluation on fresh run-176 rows."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
B=importlib.import_module("160_symmetric_evidence_known_unknown");P=importlib.import_module("163_pics_keen_known_unknown");RUNS=Path(__file__).resolve().parent/"runs";D=RUNS/"176_known_unknown_margin_geometry_fresh100";SEEDS=(42,43,44)
def sym(x):
 q=np.r_[x[5:9],x[13:17],x[21:25]];a=x[25:33];b=x[33:41];pair=np.r_[np.minimum(a,b)[4:],np.maximum(a,b)[4:],np.abs(a-b)[4:]];return np.r_[q,pair].astype(np.float32)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def fx(X,y,tr,te,seed):
 m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed)).fit(X[tr],y[tr]);return m.predict_proba(X[te])[:,1]
def run(Q,X,y,seed):
 outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y));pq=np.zeros(len(y));px=np.zeros(len(y));pf=np.zeros(len(y))
 for fold,(tr,te)in enumerate(outer):
  pq[te]=P.fit_q(Q,y,tr,te,seed);px[te]=fx(X,y,tr,te,seed);inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2))
  for it,iv in inner:meta[iv]=np.c_[P.fit_q(Q,y,tr[it],tr[iv],seed),fx(X,y,tr[it],tr[iv],seed)]
  sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);pf[te]=m.predict_proba(sc.transform(np.c_[pq[te],px[te]]))[:,1]
 return pq,px,pf
def main():
 rows,*_=B.load_rows();old=B.select_balanced(rows,100,B.SEED);used={x["key"]for x in old};rows=B.select_balanced([x for x in rows if x["key"]not in used],100,B.SEED);rows=[r for r in rows if(D/"features"/(r["key"]+".npz")).exists()];y=np.array([r["known"]for r in rows]);Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][P.KEEN_LAYERS].astype(np.float32)for r in rows]);X=np.stack([sym(np.load(D/"features"/(r["key"]+".npz"))["exact_gradient"])for r in rows]);runs=[run(Q,X,y,s)for s in SEEDS];report={"n":len(y),"known":int(y.sum()),"feature":"pre-fixed symmetric concentration: question directional summaries + candidate min/max/absdiff of entropy,gini,top3,top5","question_only":{"mean":met(y,np.mean([z[0]for z in runs],0)),"per_seed":[met(y,z[0])for z in runs]},"symmetric_gradient":{"mean":met(y,np.mean([z[1]for z in runs],0)),"per_seed":[met(y,z[1])for z in runs]},"nested_fusion":{"mean":met(y,np.mean([z[2]for z in runs],0)),"per_seed":[met(y,z[2])for z in runs]}}
 B.atomic_json(D/"evaluation_symmetric_fresh.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
