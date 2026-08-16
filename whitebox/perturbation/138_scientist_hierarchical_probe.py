#!/usr/bin/env python3
"""Nested grouped hierarchical detector for full ScientistQA."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs';OUT=RUNS/'138_scientist_hierarchical_probe_report.json';PREDS=RUNS/'138_scientist_hierarchical_probe_predictions.jsonl'

def components(rows):
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 for r in rows:union(r['right_qid'],r['wrong_qid'])
 return np.array([find(r['right_qid'])for r in rows])
def fit(x,y,C=.1):return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=42)).fit(x,y)
def safe_prob(clf,x):return clf.predict_proba(x)[:,list(clf.classes_).index(1)]
def met(y,p):
 h=p.argmax(1);pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1,2],zero_division=0);err=y!=0;pe=1-p[:,0];pu=p[err,2]/(p[err,1]+p[err,2]+1e-9)
 return{'accuracy':float(accuracy_score(y,h)),'macro_f1':float(f1_score(y,h,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,h)),'confusion':confusion_matrix(y,h,labels=[0,1,2]).tolist(),'per_class_precision':pr.tolist(),'per_class_recall':rc.tolist(),'per_class_f1':f.tolist(),'error_auroc':float(roc_auc_score(err,pe)),'unknown_vs_known_error_auroc':float(roc_auc_score(y[err]==2,pu))}
def main():
 mod=importlib.import_module('134_scientist_full_knowledge_error');probes={x['key']:x for x in map(json.loads,(RUNS/'77_closedbook_fact_probe_results.jsonl').open())};rec={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open())};man={x['key']:x for x in map(json.loads,(RUNS/'76_closedbook_fact_probe_manifest.jsonl').open())};rows=[]
 for k,r in rec.items():
  if not r.get('parse_valid',True):continue
  p=probes[k];known=bool(p['n_discriminative_facts']>=1 and p['binary_accuracy']>.5 and p['pairwise_owner_accuracy']>.5);y=0 if r['correct']else(1 if known else 2);rows.append({**man[k],'key':k,'known':int(known),'correct':int(r['correct']),'y':y,'x':mod.features(p,r)})
 X=np.stack([r['x']for r in rows]);y=np.array([r['y']for r in rows]);known=np.array([r['known']for r in rows]);correct=np.array([r['correct']for r in rows]);g=components(rows);allp=[]
 for seed in(42,43,44):
  pred=np.zeros((len(y),3));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y,g):
   # Three independently calibrated probabilities; labels are never used as input features.
   pk=safe_prob(fit(X[tr],known[tr]),X[te])
   kt=tr[known[tr]==1];ut=tr[known[tr]==0]
   pck=safe_prob(fit(X[kt],correct[kt]),X[te]);pcu=safe_prob(fit(X[ut],correct[ut]),X[te])
   pc=pk*pck+(1-pk)*pcu
   pred[te,0]=pc;pred[te,1]=pk*(1-pck);pred[te,2]=(1-pk)*(1-pcu);pred[te]/=pred[te].sum(1,keepdims=True)
  allp.append(pred)
 mean=np.mean(allp,0);report={'protocol':'hierarchical 3x5-fold OOF; candidate-QID connected components; knowledge head + correctness|known + correctness|unknown','n':len(y),'components':len(set(g)),'classes':{'0':'correct','1':'known_error','2':'unknown_error'},'mean_probability':met(y,mean),'per_seed':[met(y,p)for p in allp]};OUT.write_text(json.dumps(report,indent=2)+'\n')
 with PREDS.open('w')as f:
  for r,p in zip(rows,mean):f.write(json.dumps({'key':r['key'],'target':r['y'],'probabilities':p.tolist()})+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
