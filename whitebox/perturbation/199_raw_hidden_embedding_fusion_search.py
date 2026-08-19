#!/usr/bin/env python3
"""Explore sparse embedding fusion with cached non-PCA raw-hidden OOF baseline."""
import importlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
F=importlib.import_module("194_embedding_scalar_pca128_fusion");B,S,E=F.B,F.S,F.E
KNOWN=F.SRC;UNKNOWN=F.M.RUNS/"195_unknown_embedding_exact";OUT=F.M.RUNS/"199_raw_hidden_embedding_fusion_search.json";SEED=B.SEED
KG=(1,2,4,8,12,16);CG=(.001,.003,.01,.03,.1,.3);WG=np.arange(0,1.51,.05)
def load():
 qp={z["key"]:z for z in map(json.loads,(F.M.RUNS/"150_question_layer_ensemble_oof.jsonl").open())};rows,*_=B.load_rows();kept=[];xs=[]
 for r in rows:
  root=KNOWN if int(r["known"])else UNKNOWN;fp=root/(r["key"]+".npz")
  if r["key"]not in qp or not fp.exists():continue
  with np.load(fp,allow_pickle=True)as z:p,o,q2,r2=z["stage1_pred_scores"],z["stage1_other_scores"],z["stage2_pred_scores"],z["stage2_other_scores"]
  xs.append(np.r_[E.scientist.ch(p),E.scientist.ch(o),E.scientist.ch2(q2),E.scientist.ch2(r2),p[0]-q2[0],o[0]-r2[0],(p[0]-o[0])-(q2[0]-r2[0])]);kept.append(r)
 return np.array([r["known"]for r in kept]),np.array([qp[r["key"]]["prob_known"]for r in kept]),np.asarray(xs,np.float32)
def auc(y,p):return float(roc_auc_score(y,p))
def lg(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))
def sig(z):return 1/(1+np.exp(-z))
def select(x,res):
 z=sorted([(abs(np.corrcoef(x[:,j],res)[0,1]),j)for j in range(x.shape[1])],reverse=True);out=[]
 for _,j in z:
  if all(abs(np.corrcoef(x[:,j],x[:,w])[0,1])<.85 for w in out):out.append(j)
  if len(out)==max(KG):break
 return out
def x_oof(x,y,cols,c):
 p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for a,b in cv.split(x,y):
  sc=StandardScaler().fit(x[a][:,cols]);m=LogisticRegression(C=c,class_weight="balanced",solver="liblinear",max_iter=3000).fit(sc.transform(x[a][:,cols]),y[a]);p[b]=m.predict_proba(sc.transform(x[b][:,cols]))[:,1]
 return p
def x_hold(a,b,y,cols,c):
 sc=StandardScaler().fit(a[:,cols]);m=LogisticRegression(C=c,class_weight="balanced",solver="liblinear",max_iter=3000).fit(sc.transform(a[:,cols]),y);return m.predict_proba(sc.transform(b[:,cols]))[:,1]
def meta_features(q,x):return np.c_[lg(q),lg(x),np.abs(q-.5),lg(q)*lg(x)]
def meta_oof(z,y,c):
 p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for a,b in cv.split(z,y):
  sc=StandardScaler().fit(z[a]);m=LogisticRegression(C=c,solver="liblinear",max_iter=3000).fit(sc.transform(z[a]),y[a]);p[b]=m.predict_proba(sc.transform(z[b]))[:,1]
 return p
def main():
 y,q,x=load();di,ci=train_test_split(np.arange(len(y)),train_size=1000,stratify=y,random_state=SEED);cols=select(x[di],y[di]-q[di]);cand=[]
 for k in KG:
  ix=cols[:k]
  for c in CG:
   xd=x_oof(x[di],y[di],ix,c)
   for w in WG:cand.append((auc(y[di],sig(lg(q[di])+w*lg(xd))),"logit",k,c,float(w),None))
   for w in np.arange(0,.51,.025):cand.append((auc(y[di],(1-w)*q[di]+w*xd),"prob",k,c,float(w),None))
   z=meta_features(q[di],xd)
   for mc in(.003,.01,.03,.1,.3):cand.append((auc(y[di],meta_oof(z,y[di],mc)),"stack",k,c,0.,mc))
 best=max(cand);_,kind,k,c,w,mc=best;ix=cols[:k];xc=x_hold(x[di],x[ci],y[di],ix,c)
 if kind=="logit":pa=sig(lg(q[ci])+w*lg(xc))
 elif kind=="prob":pa=(1-w)*q[ci]+w*xc
 else:
  zd=meta_features(q[di],x_oof(x[di],y[di],ix,c));zc=meta_features(q[ci],xc);sc=StandardScaler().fit(zd);m=LogisticRegression(C=mc,solver="liblinear",max_iter=3000).fit(sc.transform(zd),y[di]);pa=m.predict_proba(sc.transform(zc))[:,1]
 report={"n":len(y),"split":"discovery=1000 / confirmation=1894","baseline":"cached raw 4096-d per-layer LR, 5-seed OOF mean","search":{"k":KG,"scalar_C":CG,"schemes":["logit","probability","4d stack"]},"selected":{"scheme":kind,"k":k,"indices":ix,"scalar_C":c,"weight":w,"meta_C":mc,"discovery_auroc":best[0]},"confirmation":{"raw_hidden_auroc":auc(y[ci],q[ci]),"embedding_holdout_auroc":auc(y[ci],xc),"fused_auroc":auc(y[ci],pa),"delta_auroc":auc(y[ci],pa)-auc(y[ci],q[ci])}}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
