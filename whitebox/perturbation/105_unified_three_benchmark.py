#!/usr/bin/env python3
"""Select one identical detector configuration across three benchmarks."""
import importlib,json
from itertools import product
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'; mod=importlib.import_module('101_fuse_sota_trajectory')
datasets=['scientist','trivia','halueval']; modes=['base','base_mean14','base_last14','base_both14','curve_mean14']; dims=[12,24,48,64]; Cs=[.03,.075,.1,.15]
configs=list(product(modes,dims,Cs)); all_scores={d:{c:[] for c in configs} for d in datasets}
for ds in datasets:
 keys,groups,y,M,H,R,RS=mod.load_response(ds); _,_,last,mean=mod.trajectory(ds,keys)
 for seed in [42,43,44]:
  pred={c:np.zeros(len(y),np.float32) for c in configs}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if ds!='trivia' else StratifiedKFold(5,shuffle=True,random_state=seed); split=cv.split(M,y,groups) if ds!='trivia' else cv.split(M,y)
  for fold,(tr,te) in enumerate(split,1):
   def scale(x): s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
   mt,mv=scale(M); bh=[]
   for x in H:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bh.append((pc.transform(q),pc.transform(s.transform(x[te]))))
   base=np.concatenate([mt]+[x[0] for x in bh],1),np.concatenate([mv]+[x[1] for x in bh],1)
   rst,rsv=scale(RS); rh=[]
   for j in range(6):
    s=StandardScaler().fit(R[tr,j]); q=s.transform(R[tr,j]); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=seed).fit(q); rh.append((pc.transform(q),pc.transform(s.transform(R[te,j]))))
   curve=np.concatenate([base[0],rst]+[x[0] for x in rh],1),np.concatenate([base[1],rsv]+[x[1] for x in rh],1)
   probes={}
   for name,x in [('mean',mean[:,3]),('last',last[:,3])]:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(max(dims),whiten=True,svd_solver='randomized',random_state=seed).fit(q); probes[name]=(pc.transform(q),pc.transform(s.transform(x[te])))
   for mode,d,C in configs:
    if mode=='base': xt,xv=base
    elif mode=='base_mean14': xt,xv=np.c_[base[0],probes['mean'][0][:,:d]],np.c_[base[1],probes['mean'][1][:,:d]]
    elif mode=='base_last14': xt,xv=np.c_[base[0],probes['last'][0][:,:d]],np.c_[base[1],probes['last'][1][:,:d]]
    elif mode=='base_both14': xt,xv=np.c_[base[0],probes['mean'][0][:,:d],probes['last'][0][:,:d]],np.c_[base[1],probes['mean'][1][:,:d],probes['last'][1][:,:d]]
    else: xt,xv=np.c_[curve[0],probes['mean'][0][:,:d]],np.c_[curve[1],probes['mean'][1][:,:d]]
    clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,y[tr]); pred[(mode,d,C)][te]=clf.predict_proba(xv)[:,1]
   print(ds,seed,fold,flush=True)
  for c,p in pred.items(): all_scores[ds][c].append({'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))})
results=[]
for c in configs:
 per={ds:{m:float(np.mean([x[m] for x in all_scores[ds][c]])) for m in ['auroc','auprc','balanced_accuracy']} for ds in datasets}; auc=[per[d]['auroc'] for d in datasets]; q=dict(zip(['mode','pca','C'],c)); q.update({'macro_auroc':float(np.mean(auc)),'min_auroc':float(np.min(auc)),'per_dataset':per}); results.append(q)
results.sort(key=lambda x:(x['macro_auroc'],x['min_auroc']),reverse=True); report={'selection':'one shared mode/PCA/C selected by macro AUROC across all three datasets','seeds':[42,43,44],'results':results}; path=RUNS/'105_unified_three_benchmark.json'; path.write_text(json.dumps(report,indent=2)); print(json.dumps({'out':str(path),'top_15':results[:15]},indent=2))
