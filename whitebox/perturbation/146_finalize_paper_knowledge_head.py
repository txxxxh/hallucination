#!/usr/bin/env python3
"""Finalize the paper-style layer-16 pre-answer MLP knowledge head with OOF calibration."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs";CACHE=RUNS/"144_paper_exact_mlp"
def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item());q=probes[k];rows.append((k,int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),z["mlp"][16,0].astype(np.float32)))
 keys=[r[0]for r in rows];y=np.array([r[1]for r in rows]);X=np.stack([r[2]for r in rows]);allp=[];ths=[]
 for seed in(0,5,26,42,63):
  p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y):
   m=LogisticRegression(random_state=seed,max_iter=2000).fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1]
  # Threshold is chosen on OOF probabilities for reporting; deployment threshold is their median.
  best=max((balanced_accuracy_score(y,p>=t),-abs(t-.5),t)for t in np.linspace(.05,.95,181));ths.append(float(best[2]));allp.append(p)
 p=np.mean(allp,0);t=float(np.median(ths));h=p>=t;pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1],zero_division=0);report={"features":"paper LLMsKnow: model.layers.16.mlp output at exact_answer_before_first_token","label":"probe known/unknown only; no probe value in X","protocol":"5 seeds x stratified 5-fold OOF; raw LogisticRegression","n":len(y),"threshold":t,"per_seed_thresholds":ths,"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),"balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"confusion_rows_unknown_known":confusion_matrix(y,h,labels=[0,1]).tolist(),"precision_unknown_known":pr.tolist(),"recall_unknown_known":rc.tolist(),"f1_unknown_known":f.tolist()};(RUNS/"146_paper_knowledge_head_report.json").write_text(json.dumps(report,indent=2)+"\n")
 with(RUNS/"146_paper_knowledge_head_oof.jsonl").open("w")as fobj:
  for k,z,q in zip(keys,y,p):fobj.write(json.dumps({"key":k,"probe_known":int(z),"prob_known":float(q),"pred_known":int(q>=t)})+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
