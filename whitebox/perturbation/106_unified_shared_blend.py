#!/usr/bin/env python3
"""Shared convex blend of the unified layer probe and trajectory tree."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'; mod=importlib.import_module('101_fuse_sota_trajectory'); datasets=['scientist','trivia','halueval']; weights=[0,.05,.1,.15,.2,.3,.4]; scores={d:{w:[] for w in weights} for d in datasets}
for ds in datasets:
 keys,groups,y,M,H,R,RS=mod.load_response(ds); T,L,last,mean=mod.trajectory(ds,keys)
 for seed in [42,43,44]:
  pl=np.zeros(len(y)); pt=np.zeros(len(y)); cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if ds!='trivia' else StratifiedKFold(5,shuffle=True,random_state=seed); split=cv.split(M,y,groups) if ds!='trivia' else cv.split(M,y)
  for fold,(tr,te) in enumerate(split,1):
   ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bh=[]
   for x in H:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bh.append((pc.transform(q),pc.transform(s.transform(x[te]))))
   base=np.concatenate([mt]+[x[0] for x in bh],1),np.concatenate([mv]+[x[1] for x in bh],1)
   x=last[:,3]; s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(48,whiten=True,svd_solver='randomized',random_state=seed).fit(q); a,b=pc.transform(q),pc.transform(s.transform(x[te])); lr=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(np.c_[base[0],a],y[tr]); pl[te]=lr.predict_proba(np.c_[base[1],b])[:,1]
   xt=np.c_[M[tr],T[tr],L[tr]]; xv=np.c_[M[te],T[te],L[te]]; hi=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,random_state=seed).fit(xt,y[tr]); pt[te]=hi.predict_proba(xv)[:,1]
   print(ds,seed,fold,flush=True)
  for w in weights:
   p=(1-w)*pl+w*pt; scores[ds][w].append({'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))})
out=[]
for w in weights:
 per={d:{m:float(np.mean([x[m] for x in scores[d][w]])) for m in ['auroc','auprc']} for d in datasets}; auc=[per[d]['auroc'] for d in datasets]; out.append({'trajectory_weight':w,'macro_auroc':float(np.mean(auc)),'min_auroc':float(np.min(auc)),'per_dataset':per})
out.sort(key=lambda x:(x['macro_auroc'],x['min_auroc']),reverse=True); path=RUNS/'106_unified_shared_blend.json'; path.write_text(json.dumps({'fixed_linear':'base + layer14 last PCA48 C=.03','fixed_tree':'HistGB(base margin + all trajectory stats/logits)','results':out},indent=2)); print(json.dumps({'out':str(path),'results':out},indent=2))
