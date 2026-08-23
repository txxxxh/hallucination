#!/usr/bin/env python3
"""Score-level fusion of paper-standard K=6 U and current127 Ours P."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';OUT=RUNS/'270_full_scientist_paper_up_fusion'
def read(p):return [json.loads(x)for x in Path(p).open()if x.strip()]
def met(y,p):return{'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}
def main():
 u={x['key']:x for x in read(RUNS/'269_full_scientist_k6/scientist/samples.jsonl')};p={x['key']:x for x in read(RUNS/'136_scientist_full_fusion_predictions.jsonl')};man={x['key']:x for x in read(RUNS/'76_closedbook_fact_probe_manifest.jsonl')};probes={x['key']:x for x in read(RUNS/'77_closedbook_fact_probe_results.jsonl')};keys=sorted(set(u)&set(p));rows=[man[k]for k in keys];g=importlib.import_module('136_eval_scientist_full_fusion').components(rows);y=np.array([p[k]['target']!=0 for k in keys]);known=np.array([probes[k]['n_discriminative_facts']>=1 and probes[k]['binary_accuracy']>.5 and probes[k]['pairwise_owner_accuracy']>.5 for k in keys]);U=np.array([u[k]['score']for k in keys]);P=np.array([1-p[k]['main_probabilities'][0]for k in keys]);X={'UP_linear':np.c_[U,P],'UP_interaction':np.c_[U,P,U*P,U**2,P**2]};allpred={n:[]for n in X};allpred['U_range_then_P']=[]
 for seed in(42,43,44):
  cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed);z={n:np.zeros(len(y))for n in allpred}
  for tr,te in cv.split(U,y,g):
   for n,x in X.items():z[n][te]=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear')).fit(x[tr],y[tr]).predict_proba(x[te])[:,1]
   # K=6 has seven discrete bins: fit a separate P calibrator per bin when viable.
   for b in np.unique(U[te]):
    a=tr[U[tr]==b];q=te[U[te]==b]
    if len(a)>=30 and len(np.unique(y[a]))==2:z['U_range_then_P'][q]=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear')).fit(P[a,None],y[a]).predict_proba(P[q,None])[:,1]
    else:z['U_range_then_P'][q]=np.mean(y[tr])
  for n in z:allpred[n].append(z[n])
 mean={n:np.mean(v,0)for n,v in allpred.items()};res={'U_K6':met(y,U),'P_Ours':met(y,P),**{n:met(y,v)for n,v in mean.items()}}
 report={'protocol':'paper-standard K=6 U + frozen current127 Ours P; parse-valid Scientist n=2894; level-2 candidate-identity grouped 3x5 OOF; no evidence','n':len(y),'errors':int(y.sum()),'known':int(known.sum()),'u_known_auroc':float(roc_auc_score(known,-U)),'results':res,'per_seed':{n:[met(y,q)for q in v]for n,v in allpred.items()}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
