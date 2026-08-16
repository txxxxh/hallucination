#!/usr/bin/env python3
"""Targeted confirmation search: layer-14 probe plus endpoint/curve detector."""
import importlib,json
from itertools import product
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'; mod=importlib.import_module('101_fuse_sota_trajectory')
keys,groups,y,M,H,R,RS=mod.load_response('scientist'); T,L,last,mean=mod.trajectory('scientist',keys)
dims=[16,24,32,48,64]; Cs=[.01,.03,.05,.075,.1,.15,.3]; modes=['base_probe','response_probe']; views=['mean','last']; cfg=list(product(views,dims,Cs,modes)); scores={c:[] for c in cfg}
for seed in [42,43,44,45,46]:
 pred={c:np.zeros(len(y),np.float32) for c in cfg}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
 for fold,(tr,te) in enumerate(cv.split(M,y,groups),1):
  def scale(x): s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
  mt,mv=scale(M); bh=[]
  for x in H:
   s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bh.append((pc.transform(q),pc.transform(s.transform(x[te]))))
  base=np.concatenate([mt]+[x[0] for x in bh],1),np.concatenate([mv]+[x[1] for x in bh],1)
  rst,rsv=scale(RS); rh=[]
  for j in range(6):
   s=StandardScaler().fit(R[tr,j]); q=s.transform(R[tr,j]); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=seed).fit(q); rh.append((pc.transform(q),pc.transform(s.transform(R[te,j]))))
  response=np.concatenate([base[0],rst]+[x[0] for x in rh],1),np.concatenate([base[1],rsv]+[x[1] for x in rh],1)
  for view,x in [('mean',mean[:,3]),('last',last[:,3])]:
   s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(max(dims),whiten=True,svd_solver='randomized',random_state=seed).fit(q); pctr,pcte=pc.transform(q),pc.transform(s.transform(x[te]))
   for d,C,mode in product(dims,Cs,modes):
    b=base if mode=='base_probe' else response; xt,xv=np.c_[b[0],pctr[:,:d]],np.c_[b[1],pcte[:,:d]]; clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear').fit(xt,y[tr]); pred[(view,d,C,mode)][te]=clf.predict_proba(xv)[:,1]
  print(seed,fold,flush=True)
 for c,p in pred.items(): scores[c].append({'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))})
out=[]
for c,v in scores.items():
 q=dict(zip(['view','pca','C','mode'],c)); q.update({f'mean_{m}':float(np.mean([x[m] for x in v])) for m in ['auroc','auprc','balanced_accuracy']}); q['per_seed']=v; out.append(q)
out.sort(key=lambda x:(x['mean_auroc'],x['mean_auprc']),reverse=True); report={'n':len(y),'grouped_cv':True,'layer':14,'warning':'targeted same-data model selection; not independent test','results':out}; path=RUNS/'103_scientist_targeted_fusion.json'; path.write_text(json.dumps(report,indent=2)); print(json.dumps({'out':str(path),'top_15':out[:15]},indent=2))
