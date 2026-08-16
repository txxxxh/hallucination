#!/usr/bin/env python3
"""Leakage-aware full ScientistQA error/type detector from closed-book probes."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'
DATA=ROOT/'shuffled_prepend_names_question.json'
RECORDS=ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl'
PROBES=RUNS/'77_closedbook_fact_probe_results.jsonl'
OUT=RUNS/'134_scientist_full_knowledge_error_report.json'
PREDS=RUNS/'134_scientist_full_knowledge_error_predictions.jsonl'

def stats(x):
 x=np.asarray(x,float)
 if not len(x): return np.zeros(8)
 return np.array([len(x),x.mean(),x.std(),x.min(),x.max(),np.median(x),
                  np.mean(x>.5),np.mean(np.abs(x-.5))])

def features(p,rec):
 # Orient only by the model's prediction, never by the gold/right identity.
 chosen=str(rec['parsed_answer']); by=defaultdict(dict)
 for q in p['probes']:
  fid=q['probe_id'].split('::')[1]; by[fid][q['person']]=float(q['p_yes'])
 own=[]; other=[]
 for pair in by.values():
  if chosen not in pair or len(pair)!=2: continue
  own.append(pair[chosen]); other.append(next(v for k,v in pair.items() if k!=chosen))
 own=np.asarray(own); other=np.asarray(other); margin=own-other
 # Symmetric blocks measure knowledge even when the selected answer is wrong.
 hi=np.maximum(own,other); lo=np.minimum(own,other); separation=np.abs(margin)
 ambiguity=1-separation; pair_entropy=[]
 for a,b in zip(own,other):
  for z in (a,b): pair_entropy.append(-(z*np.log(z+1e-9)+(1-z)*np.log(1-z+1e-9)))
 return np.r_[stats(own),stats(other),stats(margin),stats(hi),stats(lo),
              stats(separation),stats(ambiguity),stats(pair_entropy),
              float(rec.get('input_tokens',0)),float(not rec.get('parse_valid',True))]

def components(rows):
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b: parent[b]=a
 for r in rows: union(r['right_qid'],r['wrong_qid'])
 return np.array([find(r['right_qid']) for r in rows])

def model(kind):
 if kind=='linear': return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=5000,class_weight='balanced'))
 return HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=2,
                                       learning_rate=.05,random_state=42)

def main():
 probes=[json.loads(x) for x in PROBES.open() if x.strip()]
 recs={x['key']:x for x in map(json.loads,RECORDS.open())}
 manifest={x['key']:x for x in map(json.loads,(RUNS/'76_closedbook_fact_probe_manifest.jsonl').open())}
 rows=[]
 for p in probes:
  r=recs[p['key']]; m=manifest[p['key']]
  known=bool(p['n_discriminative_facts']>=1 and p['binary_accuracy']>.5 and p['pairwise_owner_accuracy']>.5)
  # 0 correct, 1 known-but-wrong (keyword/reasoning error), 2 unknown-and-wrong.
  target=0 if r['correct'] else (1 if known else 2)
  rows.append({**m,'key':p['key'],'target':target,'known':known,'x':features(p,r)})
 X=np.stack([r['x'] for r in rows]); y=np.array([r['target'] for r in rows]); groups=components(rows)
 all_results={}; saved=None
 for kind in ('linear','tree'):
  pp=np.zeros((len(y),3)); fold=np.zeros(len(y),int)
  cv=StratifiedGroupKFold(5,shuffle=True,random_state=42)
  for f,(tr,te) in enumerate(cv.split(X,y,groups),1):
   clf=model(kind).fit(X[tr],y[tr]); pp[te]=clf.predict_proba(X[te]);fold[te]=f
  yh=pp.argmax(1); incorrect=(y!=0).astype(int); pincorrect=1-pp[:,0]
  err=y!=0; unknown=(y[err]==2).astype(int); punknown=pp[err,2]/(pp[err,1]+pp[err,2]+1e-9)
  all_results[kind]={
   'three_class_accuracy':float(accuracy_score(y,yh)),
   'three_class_macro_f1':float(f1_score(y,yh,average='macro')),
   'three_class_balanced_accuracy':float(balanced_accuracy_score(y,yh)),
   'confusion_rows_true_cols_pred':confusion_matrix(y,yh,labels=[0,1,2]).tolist(),
   'error_auroc':float(roc_auc_score(incorrect,pincorrect)),
   'error_balanced_accuracy_at_0.5':float(balanced_accuracy_score(incorrect,pincorrect>=.5)),
   'unknown_vs_known_error_auroc':float(roc_auc_score(unknown,punknown)),
   'unknown_vs_known_error_balanced_accuracy_at_0.5':float(balanced_accuracy_score(unknown,punknown>=.5))}
  if kind=='tree': saved=(pp,fold)
 report={'protocol':'5-fold StratifiedGroupKFold; groups are connected components over both candidate QIDs; features oriented to model prediction only',
         'label_definition':'correct / known-but-wrong / unknown-and-wrong; known iff n_facts>=1 and binary_accuracy>0.5 and pairwise_owner_accuracy>0.5',
         'n':len(y),'identity_components':len(set(groups)),'class_counts':{str(i):int(np.sum(y==i)) for i in range(3)},'models':all_results}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 pp,fold=saved
 with PREDS.open('w') as f:
  for i,r in enumerate(rows): f.write(json.dumps({'key':r['key'],'target':int(y[i]),'fold':int(fold[i]),'prob_correct':float(pp[i,0]),'prob_known_error':float(pp[i,1]),'prob_unknown_error':float(pp[i,2])})+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
