#!/usr/bin/env python3
"""Select <=4 complementary scalars on discovery-100; confirm on fresh-100."""
import importlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
B=importlib.import_module("160_symmetric_evidence_known_unknown");P=importlib.import_module("163_pics_keen_known_unknown");E=importlib.import_module("175_eval_margin_geometry_nested");RUNS=Path(__file__).resolve().parent/"runs";D1=RUNS/"173_known_unknown_margin_geometry_n100";D2=RUNS/"176_known_unknown_margin_geometry_fresh100";OUT=RUNS/"179_sparse_complementary_selection.json";SEEDS=(42,43,44)
def sets():
 rows,*_=B.load_rows();a=B.select_balanced(rows,100,B.SEED);used={x["key"]for x in a};b=B.select_balanced([x for x in rows if x["key"]not in used],100,B.SEED);return a,b
def load(rows,d):
 y=np.array([r["known"]for r in rows]);Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][P.KEEN_LAYERS].astype(np.float32)for r in rows]);parts=[];names=[]
 for key,prefix in(("exact_gradient","grad"),("entity_interpolation","entity"),("random_projection","random")):
  x=np.stack([np.load(d/"features"/(r["key"]+".npz"))[key]for r in rows]);parts.append(x);names += [f"{prefix}_{i}"for i in range(x.shape[1])]
 return y,Q,np.c_[*parts],names
def qpred(Q,y):
 out=[]
 for seed in SEEDS:
  p=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(Q,y):p[te]=P.fit_q(Q,y,tr,te,seed)
  out.append(p)
 return np.mean(out,0)
def select(X,y,pq,names,k=4):
 residual=y-pq;scores=[]
 for j,name in enumerate(names):
  x=X[:,j];rho=float(spearmanr(x,residual).statistic);scores.append((abs(rho),rho,j,name))
 scores.sort(reverse=True);chosen=[]
 for item in scores:
  j=item[2]
  if all(abs(spearmanr(X[:,j],X[:,q[2]]).statistic)<.8 for q in chosen):chosen.append(item)
  if len(chosen)==k:break
 return chosen,scores
def fit_aug(Q,X,y,seed):
 outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y));pq=np.zeros(len(y));pa=np.zeros(len(y))
 for tr,te in outer:
  pq[te]=P.fit_q(Q,y,tr,te,seed);m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight="balanced",max_iter=3000,random_state=seed)).fit(X[tr],y[tr]);pxtr=m.predict_proba(X[tr])[:,1];pxte=m.predict_proba(X[te])[:,1]
  # Conservative fixed equal-logit blend avoids tuning a meta-weight on n=100.
  def logit(p):return np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
  pa[te]=1/(1+np.exp(-(logit(pq[te])+logit(pxte))))
 return pq,pa
def main():
 r1,r2=sets();y1,Q1,X1,names=load(r1,D1);y2,Q2,X2,names2=load(r2,D2);assert names==names2;chosen,ranking=select(X1,y1,qpred(Q1,y1),names);report={"selection":"discovery-only absolute Spearman with cross-fitted question residual; redundancy |rho|<0.8","selected":[{"name":x[3],"index":x[2],"abs_residual_rho":x[0],"rho":x[1]}for x in chosen],"confirmation":{}}
 for k in(1,2,4):
  ix=[x[2]for x in chosen[:k]];runs=[fit_aug(Q2,X2[:,ix],y2,s)for s in SEEDS];pq=np.mean([z[0]for z in runs],0);pa=np.mean([z[1]for z in runs],0);report["confirmation"][str(k)]={"features":[names[i]for i in ix],"question":E.met(y2,pq),"augmented":E.met(y2,pa),"delta_auroc":float(E.met(y2,pa)["auroc"]-E.met(y2,pq)["auroc"]),"per_seed_delta":[float(E.met(y2,z[1])["auroc"]-E.met(y2,z[0])["auroc"])for z in runs]}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
