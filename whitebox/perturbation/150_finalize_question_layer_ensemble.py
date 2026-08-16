#!/usr/bin/env python3
"""Repeated-OOF finalization of the best question-only multi-layer ensemble."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs";CACHE=RUNS/"147_question_only_hidden_v3"
def met(y,p,t):
 h=p>=t;pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1],zero_division=0)
 return {"threshold":float(t),"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),"balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"confusion_rows_unknown_known":confusion_matrix(y,h,labels=[0,1]).tolist(),"precision_unknown_known":pr.tolist(),"recall_unknown_known":rc.tolist(),"f1_unknown_known":f.tolist()}
def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item());q=probes[k];rows.append((k,int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),z["hidden"][[8,10,12,14,16,18,20,22]].astype(np.float32)))
 keys=[r[0]for r in rows];y=np.asarray([r[1]for r in rows]);X=np.stack([r[2]for r in rows]);seedp=[]
 for seed in(0,5,26,42,63):
  lp=[];cv=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))
  for li in range(X.shape[1]):
   p=np.zeros(len(y))
   for tr,te in cv:
    m=LogisticRegression(C=.3,max_iter=2000,random_state=seed).fit(X[tr,li],y[tr]);p[te]=m.predict_proba(X[te,li])[:,1]
   lp.append(p)
  p=np.mean(lp,axis=0);seedp.append(p);print(seed,roc_auc_score(y,p),flush=True)
 p=np.mean(seedp,axis=0);grid=np.linspace(.05,.95,181);ta=max(grid,key=lambda t:accuracy_score(y,p>=t));tb=max(grid,key=lambda t:balanced_accuracy_score(y,p>=t))
 report={"method":"No Answer Needed question-final token; independent C=.3 LR at hidden states 8,10,...,22; mean probabilities","protocol":"5 seeds x stratified 5-fold OOF","n":len(y),"per_seed_auroc":[float(roc_auc_score(y,z))for z in seedp],"default":met(y,p,.5),"max_probe_agreement_descriptive":met(y,p,ta),"balanced_operating_point_descriptive":met(y,p,tb)}
 (RUNS/"150_question_layer_ensemble_report.json").write_text(json.dumps(report,indent=2)+"\n")
 with(RUNS/"150_question_layer_ensemble_oof.jsonl").open("w")as f:
  for k,z,v in zip(keys,y,p):f.write(json.dumps({"key":k,"probe_known":int(z),"prob_known":float(v)})+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
