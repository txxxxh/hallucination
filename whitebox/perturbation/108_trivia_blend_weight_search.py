#!/usr/bin/env python3
"""Fine search of the already-frozen two-branch blend weight on TriviaQA."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'; mod=importlib.import_module('101_fuse_sota_trajectory'); keys,g,y,M,H,R,RS=mod.load_response('trivia'); T,L,last,mean=mod.trajectory('trivia',keys); weights=np.round(np.linspace(0,1,41),3); scores={float(w):[] for w in weights}
for seed in [42,43,44,45,46]:
 pl=np.zeros(len(y)); pt=np.zeros(len(y)); cv=StratifiedKFold(5,shuffle=True,random_state=seed)
 for fold,(tr,te) in enumerate(cv.split(M,y),1):
  ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bh=[]
  for x in H:
   s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bh.append((pc.transform(q),pc.transform(s.transform(x[te]))))
  base=np.concatenate([mt]+[x[0] for x in bh],1),np.concatenate([mv]+[x[1] for x in bh],1); x=last[:,3]; s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(48,whiten=True,svd_solver='randomized',random_state=seed).fit(q); a,b=pc.transform(q),pc.transform(s.transform(x[te])); lr=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(np.c_[base[0],a],y[tr]); pl[te]=lr.predict_proba(np.c_[base[1],b])[:,1]
  hi=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,random_state=seed).fit(np.c_[M[tr],T[tr],L[tr]],y[tr]); pt[te]=hi.predict_proba(np.c_[M[te],T[te],L[te]])[:,1]; print(seed,fold,flush=True)
 for w in weights:
  p=(1-w)*pl+w*pt; scores[float(w)].append({'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))})
out=[]
for w,v in scores.items(): out.append({'trajectory_weight':w,'linear_weight':1-w,**{f'mean_{m}':float(np.mean([x[m] for x in v])) for m in v[0]},'per_seed':v})
out.sort(key=lambda x:(x['mean_auroc'],x['mean_auprc']),reverse=True); report={'dataset':'TriviaQA','fixed_branches':True,'weight_grid':'0..1 step .025','warning':'weight selected on repeated OOF, not independent test','results':out}; path=RUNS/'108_trivia_blend_weight_search.json'; path.write_text(json.dumps(report,indent=2)); print(json.dumps({'out':str(path),'top_15':out[:15]},indent=2))
