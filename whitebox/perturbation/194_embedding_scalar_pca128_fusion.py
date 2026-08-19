#!/usr/bin/env python3
"""Sparse 47-d embedding scalar head fused with faithful per-layer PCA128 ensemble."""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

M=importlib.import_module("184_sparse_fullcache_confirmation_fixed");S,B=M.S,M.B
E=importlib.import_module("171_eval_scientist_hidden_intervention")
SRC=M.RUNS/"120_physical_delete_rerank";OUT=M.RUNS/"194_embedding_scalar_pca128_fusion.json"
SEED=B.SEED;KPC=128;KGRID=(1,2,4,8);LAM=np.linspace(-1,1,41)

def load():
 rows,*_=B.load_rows();rows=[r for r in rows if(SRC/(r["key"]+".npz")).exists() and (B.QUESTION_CACHE/(r["key"]+".npz")).exists()]
 keys=[r["key"]for r in rows];y=np.array([r["known"]for r in rows]);x=E.embedding_features(keys)
 q=np.stack([np.load(B.QUESTION_CACHE/(k+".npz"))["hidden"][S.P.KEEN_LAYERS].astype(np.float32)for k in keys]);return keys,y,q,x

def q_oof(ix,q,y):
 p=np.zeros(len(ix));cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for fi,(a,b) in enumerate(cv.split(np.zeros(len(ix)),y[ix]),1):
  tr,te=ix[a],ix[b];lp=[]
  for li in range(q.shape[1]):
   pc=PCA(n_components=min(KPC,len(tr)-2),svd_solver="randomized",random_state=SEED+li)
   za=pc.fit_transform(q[tr,li]);zb=pc.transform(q[te,li]);sc=StandardScaler().fit(za)
   lr=LogisticRegression(C=.3,max_iter=3000,solver="liblinear",random_state=SEED)
   lr.fit(sc.transform(za),y[tr]);lp.append(lr.predict_proba(sc.transform(zb))[:,1])
  p[b]=np.mean(lp,axis=0);print(f"question fold {fi}/5 n={len(ix)}",flush=True)
 return p

def x_oof(ix,x,y,cols):
 p=np.zeros(len(ix));cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for a,b in cv.split(np.zeros(len(ix)),y[ix]):
  tr,te=ix[a],ix[b];sc=StandardScaler().fit(x[tr][:,cols]);lr=LogisticRegression(C=.03,class_weight="balanced",max_iter=3000,solver="liblinear",random_state=SEED)
  lr.fit(sc.transform(x[tr][:,cols]),y[tr]);p[b]=lr.predict_proba(sc.transform(x[te][:,cols]))[:,1]
 return p

def logit(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))
def blend(q,x,w):return 1/(1+np.exp(-(logit(q)+w*logit(x))))
def auc(y,p):return float(roc_auc_score(y,p))

def select(x,y,residual):
 ranked=[]
 for j in range(x.shape[1]):
  rho=np.corrcoef(x[:,j],residual)[0,1]
  if np.isfinite(rho):ranked.append((abs(rho),float(rho),j))
 ranked.sort(reverse=True);chosen=[]
 for z in ranked:
  if all(abs(np.corrcoef(x[:,z[2]],x[:,w[2]])[0,1])<.85 for w in chosen):chosen.append(z)
  if len(chosen)==max(KGRID):break
 return chosen

def main():
 keys,y,q,x=load();nd=min(400,int(len(y)*.4));di,ci=train_test_split(np.arange(len(y)),train_size=nd,stratify=y,random_state=SEED)
 qd=q_oof(di,q,y);chosen=select(x[di],y[di]-qd);candidates=[]
 for k in KGRID:
  cols=tuple(z[2]for z in chosen[:k]);xd=x_oof(di,x,y,cols)
  w=max(LAM,key=lambda z:auc(y[di],blend(qd,xd,z)));candidates.append((auc(y[di],blend(qd,xd,w)),k,float(w),cols,auc(y[di],xd)))
 _,bestk,w,cols,_=max(candidates)
 qc=q_oof(ci,q,y);xc=x_oof(ci,x,y,cols);base=auc(y[ci],qc);stand=auc(y[ci],xc);aug=auc(y[ci],blend(qc,xc,w))
 report={"n":len(y),"split":f"discovery={len(di)} / confirmation={len(ci)}","embedding_dimensions_available":x.shape[1],"protocol":"feature count/indices/lambda discovery-only; all PCA/scaler/LR fold-local","question":"8 independent layer probes, PCA128 each, mean probabilities","discovery_candidates":[{"k":k,"lambda":ww,"fused_auroc":a,"scalar_auroc":sa}for a,k,ww,c,sa in candidates],"selected":{"k":bestk,"indices":list(cols),"lambda":w},"confirmation":{"question_auroc":base,"embedding_scalar_auroc":stand,"fused_auroc":aug,"delta_auroc":aug-base}}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
