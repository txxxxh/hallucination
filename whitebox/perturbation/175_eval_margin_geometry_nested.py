#!/usr/bin/env python3
"""Nested random-OOF evaluation for run 173; grouped CV is structurally infeasible."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
B=importlib.import_module("160_symmetric_evidence_known_unknown");P=importlib.import_module("163_pics_keen_known_unknown");RUNS=Path(__file__).resolve().parent/"runs";D=RUNS/"173_known_unknown_margin_geometry_n100";SEEDS=(42,43,44)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def fitx(X,y,tr,te,seed):
 m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed)).fit(X[tr],y[tr]);return m.predict_proba(X[te])[:,1]
def run_head(Q,X,y,seed):
 outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y));pq=np.zeros(len(y));px=np.zeros(len(y));pf=np.zeros(len(y));weights=[]
 for fold,(tr,te)in enumerate(outer):
  pq[te]=P.fit_q(Q,y,tr,te,seed);px[te]=fitx(X,y,tr,te,seed);inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2))
  for it,iv in inner:meta[iv]=np.c_[P.fit_q(Q,y,tr[it],tr[iv],seed),fitx(X,y,tr[it],tr[iv],seed)]
  sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);pf[te]=m.predict_proba(sc.transform(np.c_[pq[te],px[te]]))[:,1];weights.append(m.coef_[0].tolist())
 return pq,px,pf,weights
def main():
 rows,*_=B.load_rows();rows=B.select_balanced(rows,100,B.SEED);rows=[r for r in rows if(D/"features"/(r["key"]+".npz")).exists()];y=np.asarray([r["known"]for r in rows]);groups=[r["group"]for r in rows];Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][P.KEEN_LAYERS].astype(np.float32)for r in rows]);features={n:np.stack([np.load(D/"features"/(r["key"]+".npz"))[n]for r in rows])for n in("exact_gradient","random_projection","entity_interpolation")};report={"n":len(y),"known":int(y.sum()),"unknown":int((1-y).sum()),"protocol":"3x random stratified outer 5-fold; feature and question heads train-only; fusion nested inner 3-fold","entity_leakage_warning":True,"grouped_evaluation":{"feasible":False,"groups":len(set(groups)),"largest_group":max(groups.count(g)for g in set(groups)),"reason":"largest connected component contains 97/100 rows and both labels"},"results":{}}
 for name,X in features.items():
  runs=[run_head(Q,X,y,s)for s in SEEDS];report["results"][name]={"feature_only":{"mean":met(y,np.mean([z[1]for z in runs],0)),"per_seed":[met(y,z[1])for z in runs]},"question_only":{"mean":met(y,np.mean([z[0]for z in runs],0)),"per_seed":[met(y,z[0])for z in runs]},"nested_fusion":{"mean":met(y,np.mean([z[2]for z in runs],0)),"per_seed":[met(y,z[2])for z in runs]},"meta_coefficients":[z[3]for z in runs]}
 B.atomic_json(D/"evaluation_nested.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
