#!/usr/bin/env python3
"""Fuse span-response features with compact MultiHaluDet-style trajectories."""
from __future__ import annotations
import argparse, glob, importlib, json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'; ALPHA=np.array([0,.25,.5,.75,1.],np.float32)

def wm(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)

def load_response(dataset):
 if dataset=='scientist':
  r=importlib.import_module('94_eval_response_curves').load(); keys=np.array([x[0] for x in r]); groups=np.array([x[1] for x in r]); y=np.array([x[2] for x in r]); M=np.stack([x[3] for x in r]); H=[np.stack([x[i] for x in r]) for i in (4,5,6)]; R=np.stack([x[7] for x in r]); RS=np.c_[np.stack([x[8] for x in r]),np.stack([x[9] for x in r])]
 else:
  rows=[]
  cache=RUNS/('99_triviaqa_response_n236' if dataset=='trivia' else '95_halueval_q128_gradient_curves')
  for fp in sorted(cache.glob('*.npz')):
   with np.load(fp,allow_pickle=True) as z:
    u=z['top_u'].astype(np.float32); ua=z['all_u'].astype(np.float32); s=float(z['S0']); c=z['curve'].astype(np.float32); h0=c[0,0]; delta=c[:,-1]-h0
    m=np.r_[u,np.abs(u),u/(abs(s)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]
    res=c[:,1:4]-h0-ALPHA[1:4,None]*delta[:,None]; rb=[]
    for j in range(3): rb += [wm(res[:,j],u,True),wm(res[:,j],u,False)]
    gains=s-z['margin_curve'].astype(np.float32); norm=gains/(np.abs(u)[:,None]+.1); nonlin=norm-ALPHA
    cf=dct(norm,type=2,norm='ortho',axis=1)[:,1:3]
    rs=np.r_[nonlin[:,1:4].mean(0),nonlin[:,1:4].std(0),np.max(np.abs(nonlin[:,1:4]),0),np.average(nonlin[:,1:4],axis=0,weights=np.abs(u)+1e-8),cf.mean(0),cf.std(0),np.max(np.abs(cf),0)]
    group=str(z['group'].item()) if 'group' in z.files else str(z['key'].item())
    rows.append((str(z['key'].item()),group,int(z['correct']),m,h0,wm(delta,u,True),wm(delta,u,False),np.stack(rb),rs))
  keys=np.array([x[0] for x in rows]); groups=np.array([x[1] for x in rows]); y=np.array([x[2] for x in rows]); M=np.stack([x[3] for x in rows]); H=[np.stack([x[i] for x in rows]) for i in (4,5,6)]; R=np.stack([x[7] for x in rows]); RS=np.stack([x[8] for x in rows])
 return keys,groups,y,M,H,R,RS

def trajectory(dataset,keys):
 root=RUNS/f'100_{dataset}_trajectory_l8'; out={}
 for fp in root.glob('*.npz'):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z['key'].item()); ls=z['last_stats'].astype(np.float32); ms=z['mean_stats'].astype(np.float32)
   # Full compact descriptors + local depth changes + three non-DC modes.
   scalar=np.r_[ls.ravel(),ms.ravel(),np.diff(ls,axis=0).ravel(),np.diff(ms,axis=0).ravel(),dct(ls,axis=0,norm='ortho')[1:4].ravel(),dct(ms,axis=0,norm='ortho')[1:4].ravel()]
   out[k]=(scalar,z['logits'].astype(np.float32),z['last'].astype(np.float32),z['mean'].astype(np.float32))
 assert all(k in out for k in keys),f'missing {sum(k not in out for k in keys)} trajectory rows'
 T=np.stack([out[k][0] for k in keys]); L=np.stack([out[k][1] for k in keys]); last=np.stack([out[k][2] for k in keys]); mean=np.stack([out[k][3] for k in keys])
 return T,L,last,mean

def metrics(y,p): return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}

def main():
 a=argparse.ArgumentParser(); a.add_argument('dataset',choices=['scientist','trivia']); a.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]); a.add_argument('--out',type=Path); z=a.parse_args();
 keys,groups,y,M,H,R,RS=load_response(z.dataset); T,L,TL,TM=trajectory(z.dataset,keys); z.out=z.out or RUNS/f'101_{z.dataset}_sota_fusion.json'
 configs=[]
 for variant in ['base','response','trajectory','trajectory_nolen','base_trajectory','base_trajectory_raw','all']:
  pcs=[1,2,4] if 'raw' in variant or variant=='all' else [0]
  for pc,C in product(pcs,[.003,.01,.03,.075,.15,.3,1.]): configs.append((variant,pc,C))
 scores={c:[] for c in configs}; tree_scores={n:[] for n in ['trajectory_extra','base_trajectory_extra','trajectory_hist','base_trajectory_hist']}
 for seed in z.seeds:
  pred={c:np.zeros(len(y),np.float32) for c in configs}; pt={n:np.zeros(len(y),np.float32) for n in tree_scores}
  cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if z.dataset=='scientist' else StratifiedKFold(5,shuffle=True,random_state=seed)
  split=cv.split(M,y,groups) if z.dataset=='scientist' else cv.split(M,y)
  for fold,(tr,te) in enumerate(split,1):
   def scale(x): s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
   mt,mv=scale(M); tt,tv=scale(T); lt,lv=scale(L); nt,nv=scale(L[:,:-1]); rst,rsv=scale(RS)
   bhtr,bhte=[],[]
   for x in H:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bhtr.append(pc.transform(q)); bhte.append(pc.transform(s.transform(x[te])))
   base=(np.concatenate([mt]+bhtr,1),np.concatenate([mv]+bhte,1)); rrtr,rrte=[],[]
   for j in range(6):
    s=StandardScaler().fit(R[tr,j]); q=s.transform(R[tr,j]); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=seed).fit(q); rrtr.append(pc.transform(q)); rrte.append(pc.transform(s.transform(R[te,j])))
   response=(np.concatenate([base[0],rst]+rrtr,1),np.concatenate([base[1],rsv]+rrte,1)); raw={}
   for d in [1,2,4]:
    qtr,qte=[],[]
    for x in [*[TL[:,j] for j in range(TL.shape[1])],*[TM[:,j] for j in range(TM.shape[1])]]:
     s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(d,whiten=True,svd_solver='randomized',random_state=seed).fit(q); qtr.append(pc.transform(q)); qte.append(pc.transform(s.transform(x[te])))
    raw[d]=(np.concatenate(qtr,1),np.concatenate(qte,1))
   for variant,d,C in configs:
    sets={'base':base,'response':response,'trajectory':(np.c_[tt,lt],np.c_[tv,lv]),'trajectory_nolen':(np.c_[tt,nt],np.c_[tv,nv]),'base_trajectory':(np.c_[base[0],tt,lt],np.c_[base[1],tv,lv])}
    if variant=='base_trajectory_raw': xy=(np.c_[base[0],tt,lt,raw[d][0]],np.c_[base[1],tv,lv,raw[d][1]])
    elif variant=='all': xy=(np.c_[response[0],tt,lt,raw[d][0]],np.c_[response[1],tv,lv,raw[d][1]])
    else: xy=sets[variant]
    clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xy[0],y[tr]); pred[(variant,d,C)][te]=clf.predict_proba(xy[1])[:,1]
   # Nonlinear controls use only compact scalars; no fold-fitted PCA leakage.
   for prefix,xtr,xte in [('trajectory',np.c_[T[tr],L[tr]],np.c_[T[te],L[te]]),('base_trajectory',np.c_[M[tr],T[tr],L[tr]],np.c_[M[te],T[te],L[te]])]:
    ex=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=4,max_features=.5,class_weight='balanced',random_state=seed,n_jobs=-1).fit(xtr,y[tr]); pt[prefix+'_extra'][te]=ex.predict_proba(xte)[:,1]
    hi=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,random_state=seed).fit(xtr,y[tr]); pt[prefix+'_hist'][te]=hi.predict_proba(xte)[:,1]
   print(f'{z.dataset} seed={seed} fold={fold}/5',flush=True)
  for c,p in pred.items(): scores[c].append(metrics(y,p))
  for c,p in pt.items(): tree_scores[c].append(metrics(y,p))
 results=[]
 for c,v in scores.items():
  q=dict(zip(['variant','raw_pca','C'],c)); q.update({f'mean_{m}':float(np.mean([x[m] for x in v])) for m in ['auroc','auprc','balanced_accuracy']}); q['per_seed']=v; results.append(q)
 for c,v in tree_scores.items(): results.append({'variant':c,'raw_pca':0,'C':None,**{f'mean_{m}':float(np.mean([x[m] for x in v])) for m in ['auroc','auprc','balanced_accuracy']},'per_seed':v})
 results.sort(key=lambda x:(x['mean_auroc'],x['mean_auprc']),reverse=True); report={'dataset':z.dataset,'n':len(y),'groups':len(set(groups)),'warning':'configuration selection on repeated OOF; confirm winner with nested/group-held-out evaluation','results':results}; z.out.write_text(json.dumps(report,indent=2)); print(json.dumps({'out':str(z.out),'top_15':results[:15]},indent=2))
if __name__=='__main__': main()
