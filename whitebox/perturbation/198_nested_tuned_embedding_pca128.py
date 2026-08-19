#!/usr/bin/env python3
"""Strict nested tuning for PCA128 layer ensemble + sparse embedding scalars."""
from __future__ import annotations
import importlib,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

F=importlib.import_module("194_embedding_scalar_pca128_fusion");B,S,E=F.B,F.S,F.E
KNOWN=F.SRC;UNKNOWN=F.M.RUNS/"195_unknown_embedding_exact";OUT=F.M.RUNS/"198_nested_tuned_embedding_pca128.json"
HC=(.03,.1,.3,1.0);SC=(.003,.01,.03,.1,.3);KG=(1,2,4,8,12);WG=np.arange(0,1.51,.1);SEED=B.SEED

def load():
 rows,*_=B.load_rows();ys=[];qs=[];xs=[]
 for r in rows:
  root=KNOWN if int(r["known"])else UNKNOWN;fp=root/(r["key"]+".npz");qp=B.QUESTION_CACHE/(r["key"]+".npz")
  if not fp.exists()or not qp.exists():continue
  with np.load(fp,allow_pickle=True)as z:p,o,q2,r2=z["stage1_pred_scores"],z["stage1_other_scores"],z["stage2_pred_scores"],z["stage2_other_scores"]
  xs.append(np.r_[E.scientist.ch(p),E.scientist.ch(o),E.scientist.ch2(q2),E.scientist.ch2(r2),p[0]-q2[0],o[0]-r2[0],(p[0]-o[0])-(q2[0]-r2[0])]);ys.append(r["known"]);qs.append(np.load(qp)["hidden"][S.P.KEEN_LAYERS].astype(np.float32))
 return np.asarray(ys),np.stack(qs),np.asarray(xs,np.float32)
def auc(y,p):return float(roc_auc_score(y,p))
def logit(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))
def blend(a,b,w):return 1/(1+np.exp(-(logit(a)+w*logit(b))))

def qfit(tr,te,q,y,cs):
 out={c:[]for c in cs}
 for li in range(q.shape[1]):
  pc=PCA(n_components=128,svd_solver="randomized",random_state=SEED+li);za=pc.fit_transform(q[tr,li]);zb=pc.transform(q[te,li]);sc=StandardScaler().fit(za);aa,bb=sc.transform(za),sc.transform(zb)
  for c in cs:
   m=LogisticRegression(C=c,max_iter=3000,solver="liblinear",random_state=SEED).fit(aa,y[tr]);out[c].append(m.predict_proba(bb)[:,1])
 return {c:np.mean(v,0)for c,v in out.items()}
def xfit(tr,te,x,y,cols,cs):
 sc=StandardScaler().fit(x[tr][:,cols]);a,b=sc.transform(x[tr][:,cols]),sc.transform(x[te][:,cols]);out={}
 for c in cs:
  m=LogisticRegression(C=c,class_weight="balanced",max_iter=3000,solver="liblinear",random_state=SEED).fit(a,y[tr]);out[c]=m.predict_proba(b)[:,1]
 return out
def select(x,residual,n=12):
 z=[]
 for j in range(x.shape[1]):
  r=np.corrcoef(x[:,j],residual)[0,1]
  if np.isfinite(r):z.append((abs(r),j))
 z.sort(reverse=True);out=[]
 for _,j in z:
  if all(abs(np.corrcoef(x[:,j],x[:,w])[0,1])<.85 for w in out):out.append(j)
  if len(out)==n:break
 return out

def main():
 y,q,x=load();outer=StratifiedKFold(5,shuffle=True,random_state=SEED);pq=np.zeros(len(y));pa=np.zeros(len(y));chosen=[]
 for fo,(tr,te) in enumerate(outer.split(np.zeros(len(y)),y),1):
  inner=list(StratifiedKFold(3,shuffle=True,random_state=SEED+fo).split(np.zeros(len(tr)),y[tr]));qi={c:np.zeros(len(tr))for c in HC}
  for a,b in inner:
   got=qfit(tr[a],tr[b],q,y,HC)
   for c in HC:qi[c][b]=got[c]
  hc=max(HC,key=lambda c:auc(y[tr],qi[c]));cols=select(x[tr],y[tr]-qi[hc]);best=None
  for k in KG:
   xi={c:np.zeros(len(tr))for c in SC}
   for a,b in inner:
    got=xfit(tr[a],tr[b],x,y,cols[:k],SC)
    for c in SC:xi[c][b]=got[c]
   for c in SC:
    for w in WG:
     z=(auc(y[tr],blend(qi[hc],xi[c],w)),k,c,float(w))
     if best is None or z[0]>best[0]:best=z
  _,k,sc,w=best;qtest=qfit(tr,te,q,y,(hc,))[hc];xtest=xfit(tr,te,x,y,cols[:k],(sc,))[sc];pq[te]=qtest;pa[te]=blend(qtest,xtest,w);chosen.append({"fold":fo,"hidden_C":hc,"scalar_C":sc,"k":k,"lambda":w,"indices":cols[:k],"inner_fused_auroc":best[0]});print("outer",fo,chosen[-1],flush=True)
 report={"n":len(y),"protocol":"strict outer5/inner3 nested OOF; PCA/ranking/all hyperparameters inner-only","grids":{"hidden_C":HC,"scalar_C":SC,"k":KG,"lambda":[float(WG[0]),float(WG[-1]),.1]},"fold_choices":chosen,"question_auroc":auc(y,pq),"fused_auroc":auc(y,pa),"delta_auroc":auc(y,pa)-auc(y,pq)};B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
