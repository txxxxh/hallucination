#!/usr/bin/env python3
"""Compare promising pre-generation knowledge heads on identical OOF folds."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"; CACHE=RUNS/"147_question_only_hidden_v3"

def load():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())}; rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item());q=probes[k];rows.append((k,int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),z["hidden"].astype(np.float32)))
 return [r[0] for r in rows],np.asarray([r[1] for r in rows]),np.stack([r[2] for r in rows])

def score(y,p):
 ts=np.linspace(.05,.95,181); ta=max(ts,key=lambda t:accuracy_score(y,p>=t));tb=max(ts,key=lambda t:balanced_accuracy_score(y,p>=t))
 return {"auroc":float(roc_auc_score(y,p)),"accuracy_at_0.5":float(accuracy_score(y,p>=.5)),"best_accuracy":float(accuracy_score(y,p>=ta)),"best_accuracy_threshold":float(ta),"best_balanced_accuracy":float(balanced_accuracy_score(y,p>=tb)),"best_balanced_threshold":float(tb)}

def lr_oof(x,y,folds,C=1.0,norm=False):
 p=np.zeros(len(y))
 for tr,te in folds:
  a=x[tr];b=x[te]
  if norm:a=normalize(a);b=normalize(b)
  m=LogisticRegression(C=C,max_iter=2000).fit(a,y[tr]);p[te]=m.predict_proba(b)[:,1]
 return p

def sgd_oof(x,y,folds,alpha):
 p=np.zeros(len(y))
 for tr,te in folds:
  # Per-example L2 normalization controls the very different scales across layers.
  a=normalize(x[tr]);b=normalize(x[te]);m=SGDClassifier(loss="log_loss",penalty="l2",alpha=alpha,max_iter=3000,tol=1e-5,early_stopping=True,validation_fraction=.15,n_iter_no_change=15,random_state=42).fit(a,y[tr]);p[te]=m.predict_proba(b)[:,1]
 return p

def main():
 keys,y,X=load();folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y));out=[];pred={}
 def add(name,p,detail=""):
  z={"method":name,"detail":detail,**score(y,p)};out.append(z);pred[name]=p;print(json.dumps(z),flush=True)
 # Strong single-layer controls and normalization/C regularization ablations.
 for C in(.03,.1,.3,1.,3.):add(f"layer16_lr_C{C:g}",lr_oof(X[:,16],y,folds,C=C))
 for C in(.1,1.,10.,100.):add(f"layer16_l2norm_lr_C{C:g}",lr_oof(X[:,16],y,folds,C=C,norm=True))
 # Probability-level ensembles preserve each layer's geometry and are low variance.
 layer_ps=[]
 for h in(8,10,12,14,16,18,20,22):layer_ps.append(lr_oof(X[:,h],y,folds,C=.3))
 for hs in((12,14,16,18),(8,10,12,14,16,18,20,22)):
  ii=[(8,10,12,14,16,18,20,22).index(h) for h in hs];add("ensemble_"+"_".join(map(str,hs)),np.mean([layer_ps[i] for i in ii],axis=0))
 # Representation trajectories: absolute state plus changes across computation.
 variants={"concat_12_14_16_18":X[:,[12,14,16,18]].reshape(len(y),-1),
  "concat_8_to22_even":X[:,[8,10,12,14,16,18,20,22]].reshape(len(y),-1),
  "trajectory_8_12_16_20":np.concatenate([X[:,16],X[:,12]-X[:,8],X[:,16]-X[:,12],X[:,20]-X[:,16]],axis=1),
  "all_layers":X.reshape(len(y),-1)}
 for name,x in variants.items():
  for alpha in((3e-5,1e-4,3e-4) if name!="all_layers" else (1e-4,3e-4,1e-3)):
   add(f"{name}_sgd_a{alpha:g}",sgd_oof(x,y,folds,alpha),"L2-normalized multilevel features + regularized logistic SGD")
 report={"protocol":"identical seed-42 stratified 5-fold OOF; probe only supplies y; thresholds selected on OOF for descriptive comparison","n":len(y),"results":sorted(out,key=lambda z:z["auroc"],reverse=True)}
 (RUNS/"149_question_hidden_method_sweep.json").write_text(json.dumps(report,indent=2)+"\n")
 print("TOP",json.dumps(report["results"][:8],indent=2))

if __name__=="__main__":main()
