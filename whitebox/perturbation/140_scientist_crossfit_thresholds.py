#!/usr/bin/env python3
"""Cross-fit decision thresholds on identity-disjoint OOF probabilities."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs'
pred=[json.loads(x)for x in (RUNS/'139_pairwise_swap_hierarchical_predictions.jsonl').open()];man={x['key']:x for x in map(json.loads,(RUNS/'76_closedbook_fact_probe_manifest.jsonl').open())};keys=[x['key']for x in pred];y=np.array([x['target']for x in pred]);p=np.array([x['probabilities']for x in pred])
parent={}
def find(x):
 parent.setdefault(x,x)
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[b]=a
for k in keys:union(man[k]['right_qid'],man[k]['wrong_qid'])
g=np.array([find(man[k]['right_qid'])for k in keys]);h=np.zeros(len(y),int);chosen=[];cv=StratifiedGroupKFold(5,shuffle=True,random_state=20260812)
for fold,(tr,te)in enumerate(cv.split(p,y,g),1):
 best=None
 for tu in np.arange(.1,.601,.025):
  for tk in np.arange(.1,.601,.025):
   q=np.where(p[tr,2]>=tu,2,np.where(p[tr,1]>=tk,1,0));score=f1_score(y[tr],q,average='macro')
   z=(score,-abs(tu-.3)-abs(tk-.425),tu,tk)
   if best is None or z>best:best=z
 tu,tk=best[2:];h[te]=np.where(p[te,2]>=tu,2,np.where(p[te,1]>=tk,1,0));chosen.append({'fold':fold,'train_n':len(tr),'test_n':len(te),'unknown_threshold':float(tu),'known_threshold':float(tk),'train_macro_f1':float(best[0])})
pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1,2],zero_division=0);report={'protocol':'5-fold identity-component cross-fitted threshold calibration over already OOF probabilities','n':len(y),'components':len(set(g)),'accuracy':float(accuracy_score(y,h)),'macro_f1':float(f1_score(y,h,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,h)),'confusion':confusion_matrix(y,h,labels=[0,1,2]).tolist(),'precision':pr.tolist(),'recall':rc.tolist(),'f1':f.tolist(),'thresholds':chosen};path=RUNS/'140_scientist_crossfit_thresholds_report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
