#!/usr/bin/env python3
"""Expanded nested search around Mistral layer-14 optimum, including scalar fusion."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));m=__import__('132_mistral_multilayer_search');base=__import__('131_mistral_scientist_current_detector')
OUT=P/'runs/133_mistral_local_nested_search_report.json';LAYERS=(10,14,18);POOLS=('last','mean');DIMS=(32,64,96,128);CS=(.03,.1,.3,1.,3.,10.);MODES=('hidden','hidden_scalar');WEIGHTS=(.25,.5,1.)
def old_scalars(keys):
 out={}
 for fp in base.CACHE.glob('*.npz'):
  with np.load(fp,allow_pickle=True)as z:
   p=z['stage1_pred'].astype('f4');o=z['stage1_other'].astype('f4');q=z['stage2_pred'].astype('f4');r=z['stage2_other'].astype('f4');out[str(z['key'].item())]=np.r_[base.ch(p),base.ch(o),base.ch2(q),base.ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])]
 return np.stack([out[k]for k in keys])
def proj(train,test,d,seed):
 sc=StandardScaler().fit(train);z=sc.transform(train);pc=PCA(min(128,len(train)-1),whiten=True,svd_solver='randomized',random_state=seed).fit(z);return pc.transform(z)[:,:d],pc.transform(sc.transform(test))[:,:d]
def select(X,S,y,g,tr,seed):
 configs=[(p,l,d,C,mode,w)for p in POOLS for l in LAYERS for d in DIMS for C in CS for mode in MODES for w in ((0,)if mode=='hidden'else WEIGHTS)];scores=np.zeros(len(configs));cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed+500)
 for aa,bb in cv.split(np.zeros(len(tr)),y[tr],g[tr]):
  ia,ib=tr[aa],tr[bb];sp=StandardScaler().fit(S[ia]);st,sv=sp.transform(S[ia]),sp.transform(S[ib]);cache={(p,l):proj(X[(p,l)][ia],X[(p,l)][ib],128,seed)for p in POOLS for l in LAYERS}
  for ci,(p,l,d,C,mode,w)in enumerate(configs):
   a,b=cache[(p,l)];a,b=a[:,:d],b[:,:d]
   if mode=='hidden_scalar':a,b=np.c_[a,w*st],np.c_[b,w*sv]
   pr=LogisticRegression(C=C,max_iter=4000,class_weight='balanced',solver='liblinear',random_state=seed).fit(a,y[ia]).predict_proba(b)[:,1];scores[ci]+=roc_auc_score(y[ib],pr)/3
 return configs[int(scores.argmax())],float(scores.max())
def main():
 rows=m.load();keys=[x[0]for x in rows];g=np.array([x[1]for x in rows]);y=np.array([x[2]for x in rows]);X={v:np.stack([x[3][v]for x in rows])for v in rows[0][3]if v[1]in LAYERS};S=old_scalars(keys);vals=[];choices=[]
 for seed in(42,43,44):
  pred=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te)in enumerate(cv.split(np.zeros(len(y)),y,g),1):
   cfg,iv=select(X,S,y,g,tr,seed+fold);pool,l,d,C,mode,w=cfg;a,b=proj(X[(pool,l)][tr],X[(pool,l)][te],d,seed)
   if mode=='hidden_scalar':sc=StandardScaler().fit(S[tr]);a,b=np.c_[a,w*sc.transform(S[tr])],np.c_[b,w*sc.transform(S[te])]
   pred[te]=LogisticRegression(C=C,max_iter=4000,class_weight='balanced',solver='liblinear',random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1];choices.append({'seed':seed,'fold':fold,'pool':pool,'layer':l,'pca':d,'C':C,'mode':mode,'scalar_weight':w,'inner_auroc':iv});print(seed,fold,choices[-1],flush=True)
  vals.append({'seed':seed,'auroc':float(roc_auc_score(y,pred)),'auprc':float(average_precision_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred>=.5))})
 report={'protocol':'outer 3x5 grouped OOF; expanded hyperparameters selected by inner 3-fold grouped OOF','grid':{'layers':LAYERS,'pooling':POOLS,'pca':DIMS,'C':CS,'modes':MODES,'scalar_weights':WEIGHTS},'mean':{k:float(np.mean([v[k]for v in vals]))for k in('auroc','auprc','balanced_accuracy')},'per_seed':vals,'choices':choices};OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
