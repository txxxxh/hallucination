#!/usr/bin/env python3
"""Frozen PCA8 perturbation detector on all 770 strict-known items."""
import glob, json
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT='/home/tong56/whitebox/perturbation/runs'
SOURCE=f'{ROOT}/79_strict_known_n770.jsonl'; ORACLE=f'{ROOT}/79_oracle_top11_known770.jsonl'
CACHE=f'{ROOT}/79_hidden_delta_top11_known770'; REPORT=f'{ROOT}/80_known770_oof_report.json'
PREDS=f'{ROOT}/80_known770_oof_predictions.jsonl'; DIM=8

def main():
 src={x['key']:x for x in map(json.loads,open(SOURCE))}; oracle={x['key']:x for x in map(json.loads,open(ORACLE))}; rows=[]
 for path in sorted(glob.glob(CACHE+'/*.npz')):
  with np.load(path,allow_pickle=True) as z:
   key=str(z['key'].item()); o=oracle[key]; ua=np.asarray(o['u'],np.float32); u=np.asarray(z['top_u'],np.float32); s=float(o['S0'])
   m=np.r_[u,np.abs(u),u/(abs(s)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]
   h=np.asarray(z['answer_last'],np.float32)[0]; h0=h[0]; d=h[1:]-h0
   def wm(mask,w): return (d[mask]*w[mask,None]).sum(0)/(np.abs(w[mask]).sum()+1e-9) if mask.any() else np.zeros(4096,np.float32)
   rows.append((key,src[key]['group'],int(src[key]['correct']),m,h0,wm(u>0,u),wm(u<0,-u)))
 assert len(rows)==770, len(rows)
 keys=np.array([x[0] for x in rows]); groups=np.array([x[1] for x in rows]); y=np.array([x[2] for x in rows]); M=np.stack([x[3] for x in rows]); H=[np.stack([x[i] for x in rows]) for i in (4,5,6)]
 cv=StratifiedGroupKFold(5,shuffle=True,random_state=42); pfull=np.zeros(len(y)); pmargin=np.zeros(len(y)); folds=np.zeros(len(y),int); fold_rows=[]
 for fold,(tr,te) in enumerate(cv.split(M,y,groups),1):
  ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); train_parts=[mt]; test_parts=[mv]; ev=[]
  for X in H:
   sc=StandardScaler().fit(X[tr]); zt=sc.transform(X[tr]); pca=PCA(DIM,whiten=True,svd_solver='randomized',random_state=42).fit(zt); train_parts.append(pca.transform(zt)); test_parts.append(pca.transform(sc.transform(X[te]))); ev.append(float(pca.explained_variance_ratio_.sum()))
  xf=np.concatenate(train_parts,1); xv=np.concatenate(test_parts,1)
  clf=LogisticRegression(C=.5,max_iter=5000,class_weight='balanced',random_state=42).fit(xf,y[tr]); pfull[te]=clf.predict_proba(xv)[:,1]
  mb=LogisticRegression(C=.5,max_iter=5000,class_weight='balanced',random_state=42).fit(mt,y[tr]); pmargin[te]=mb.predict_proba(mv)[:,1]; folds[te]=fold
  fold_rows.append({'fold':fold,'train_n':len(tr),'test_n':len(te),'test_correct':int(y[te].sum()),'groups':len(set(groups[te])),'auroc':float(roc_auc_score(y[te],pfull[te])),'auprc':float(average_precision_score(y[te],pfull[te])),'explained_variance':ev})
 def metrics(p):
  pred=p>=.5
  return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy_at_0.5':float(accuracy_score(y,pred)),'balanced_accuracy_at_0.5':float(balanced_accuracy_score(y,pred)),'confusion_tn_fp_fn_tp':confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()}
 # Group bootstrap OOF metrics and paired AUROC lift.
 by=defaultdict(list)
 for i,g in enumerate(groups): by[g].append(i)
 gn=list(by); rng=np.random.default_rng(20260811); au,ap,ba,lift=[],[],[],[]
 for _ in range(5000):
  ix=np.concatenate([by[g] for g in rng.choice(gn,len(gn),replace=True)])
  if len(np.unique(y[ix]))<2: continue
  au.append(roc_auc_score(y[ix],pfull[ix])); ap.append(average_precision_score(y[ix],pfull[ix])); ba.append(balanced_accuracy_score(y[ix],pfull[ix]>=.5)); lift.append(au[-1]-roc_auc_score(y[ix],pmargin[ix]))
 report={'protocol':'frozen top11/layer16/PCA8/C0.5; 5-fold StratifiedGroupKFold OOF on all strict-known items','n':len(y),'correct':int(y.sum()),'incorrect':int(len(y)-y.sum()),'groups':len(set(groups)),'final_dims':63,'full':metrics(pfull),'margin_only':metrics(pmargin),'folds':fold_rows,'group_bootstrap_95ci':{'auroc':[float(x) for x in np.quantile(au,[.025,.975])],'auprc':[float(x) for x in np.quantile(ap,[.025,.975])],'balanced_accuracy':[float(x) for x in np.quantile(ba,[.025,.975])],'auroc_lift_over_margin':[float(x) for x in np.quantile(lift,[.025,.975])]},'auroc_lift_over_margin_point':float(roc_auc_score(y,pfull)-roc_auc_score(y,pmargin))}
 json.dump(report,open(REPORT,'w'),indent=2)
 with open(PREDS,'w') as f:
  for i in range(len(y)): f.write(json.dumps({'key':str(keys[i]),'group':str(groups[i]),'correct':int(y[i]),'fold':int(folds[i]),'prob_full':float(pfull[i]),'prob_margin':float(pmargin[i])})+'\n')
 print(json.dumps(report,indent=2)); print(REPORT); print(PREDS)
if __name__=='__main__': main()
