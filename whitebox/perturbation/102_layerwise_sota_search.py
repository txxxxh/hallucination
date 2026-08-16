#!/usr/bin/env python3
"""Select useful depth probes before fusing them with perturbation features."""
import argparse,json,importlib
from pathlib import Path
from itertools import product
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'

def main():
 a=argparse.ArgumentParser(); a.add_argument('dataset',choices=['scientist','trivia']); a.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]); z=a.parse_args()
 mod=importlib.import_module('101_fuse_sota_trajectory'); keys,groups,y,M,H,R,RS=mod.load_response(z.dataset); T,L,last,mean=mod.trajectory(z.dataset,keys)
 dims=[2,4,8,12,16,24]; Cs=[.003,.01,.03,.075,.15,.3]; configs=[]
 for view in ['last','mean']:
  for layer,d,C,mode in product(range(last.shape[1]),dims,Cs,['probe','base_probe']): configs.append((view,layer,d,C,mode))
 scores={c:[] for c in configs}
 for seed in z.seeds:
  pred={c:np.zeros(len(y),np.float32) for c in configs}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if z.dataset=='scientist' else StratifiedKFold(5,shuffle=True,random_state=seed); split=cv.split(M,y,groups) if z.dataset=='scientist' else cv.split(M,y)
  for fold,(tr,te) in enumerate(split,1):
   ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bp=[]
   for x in H:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bp.append((pc.transform(q),pc.transform(s.transform(x[te]))))
   base=np.concatenate([mt]+[x[0] for x in bp],1),np.concatenate([mv]+[x[1] for x in bp],1)
   for view,x in [('last',last),('mean',mean)]:
    for li in range(x.shape[1]):
     s=StandardScaler().fit(x[tr,li]); q=s.transform(x[tr,li]); pc=PCA(max(dims),whiten=True,svd_solver='randomized',random_state=seed).fit(q); pctr,pcte=pc.transform(q),pc.transform(s.transform(x[te,li]))
     for d,C,mode in product(dims,Cs,['probe','base_probe']):
      xt,xv=(pctr[:,:d],pcte[:,:d]) if mode=='probe' else (np.c_[base[0],pctr[:,:d]],np.c_[base[1],pcte[:,:d]])
      clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,y[tr]); pred[(view,li,d,C,mode)][te]=clf.predict_proba(xv)[:,1]
   print(z.dataset,seed,fold,flush=True)
  for c,p in pred.items(): scores[c].append((roc_auc_score(y,p),average_precision_score(y,p)))
 results=[]
 layers=np.load(next((RUNS/f'100_{z.dataset}_trajectory_l8').glob('*.npz')))['layers'].tolist()
 for c,v in scores.items():
  q=dict(zip(['view','layer_pos','pca','C','mode'],c)); q['layer']=layers[q['layer_pos']]; q['mean_auroc']=float(np.mean([x[0] for x in v])); q['mean_auprc']=float(np.mean([x[1] for x in v])); q['per_seed']=[{'auroc':float(x[0]),'auprc':float(x[1])} for x in v]; results.append(q)
 results.sort(key=lambda x:(x['mean_auroc'],x['mean_auprc']),reverse=True); out=RUNS/f'102_{z.dataset}_layerwise_search.json'; out.write_text(json.dumps({'dataset':z.dataset,'n':len(y),'warning':'same-data layer/hyperparameter selection','results':results},indent=2)); print(json.dumps({'out':str(out),'top_20':results[:20]},indent=2))
if __name__=='__main__':main()
