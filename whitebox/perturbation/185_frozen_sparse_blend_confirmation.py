#!/usr/bin/env python3
"""Tune only a sparse-feature logit weight on discovery; freeze on confirmation."""
import importlib,json
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
M=importlib.import_module("184_sparse_fullcache_confirmation_fixed");S=M.S;B=M.B;OUT=M.RUNS/"185_frozen_sparse_blend_confirmation.json";SEEDS=S.SEEDS
def oofx(X,y):
 out=[]
 for seed in SEEDS:
  p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y):p[te]=S.fit_aug.__globals__["make_pipeline"](S.fit_aug.__globals__["StandardScaler"](),S.fit_aug.__globals__["LogisticRegression"](C=.03,class_weight="balanced",max_iter=3000,random_state=seed)).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
  out.append(p)
 return np.mean(out,0)
def logit(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))
def blend(q,x,w):return 1/(1+np.exp(-(logit(q)+w*logit(x))))
def main():
 rows,*_=B.load_rows();rows=[r for r in rows if(M.D/"features"/(r["key"]+".npz")).exists()];y=np.array([r["known"]for r in rows]);di,ci=train_test_split(np.arange(len(rows)),train_size=1000,stratify=y,random_state=B.SEED);X=np.stack([M.compact(np.load(M.D/"features"/(r["key"]+".npz"))["local_geometry"])for r in rows]);Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][S.P.KEEN_LAYERS].astype(np.float32)for r in rows]);chosen=S.select(X[di],y[di],S.qpred(Q[di],y[di]),M.N);qd=S.qpred(Q[di],y[di]);qc=S.qpred(Q[ci],y[ci]);grid=np.linspace(-1,1,41);report={"selection_split":"1000 discovery / 1894 confirmation","blend":"logit(question)+lambda*logit(sparse head); lambda selected only on discovery","results":{}}
 for k in(1,2,4):
  ix=[z[2]for z in chosen[:k]];xd=oofx(X[di][:,ix],y[di]);xc=oofx(X[ci][:,ix],y[ci]);w=max(grid,key=lambda z:S.E.met(y[di],blend(qd,xd,z))["auroc"]);base=S.E.met(y[ci],qc);aug=S.E.met(y[ci],blend(qc,xc,w));report["results"][str(k)]={"features":[M.N[i]for i in ix],"lambda":float(w),"discovery_delta":S.E.met(y[di],blend(qd,xd,w))["auroc"]-S.E.met(y[di],qd)["auroc"],"confirmation_question":base,"confirmation_augmented":aug,"confirmation_delta":aug["auroc"]-base["auroc"]}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
