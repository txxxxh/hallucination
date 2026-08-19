#!/usr/bin/env python3
"""Discovery-select and fresh-confirm <=4 symmetric, cheap-to-collect scalars."""
import importlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
S=importlib.import_module("180_select_sparse_complementary_features_fixed");B=S.B;RUNS=S.RUNS;OUT=RUNS/"181_symmetric_sparse_endpoints.json"
def features(rows,d):
 out=[]
 for r in rows:
  z=np.load(d/"features"/(r["key"]+".npz"));x=z["exact_gradient"];a=x[25:33];b=x[33:41];pair=np.r_[np.minimum(a,b),np.maximum(a,b),np.abs(a-b)];C=z["curves"];ends=np.sort(C[:,-1]-C[:,0]);out.append(np.r_[pair,ends,abs(C[0,0])])
 names=[f"candidate_min_{x}"for x in("mean","std","max","median","entropy","gini","top3","top5")]+[f"candidate_max_{x}"for x in("mean","std","max","median","entropy","gini","top3","top5")]+[f"candidate_absdiff_{x}"for x in("mean","std","max","median","entropy","gini","top3","top5")]+["entity_endpoint_low","entity_endpoint_high","base_abs_margin"]
 return np.stack(out),names
def main():
 r1,r2=S.sets();y1,Q1,_,_=S.load(r1,S.D1);y2,Q2,_,_=S.load(r2,S.D2);X1,names=features(r1,S.D1);X2,_=features(r2,S.D2);chosen=S.select(X1,y1,S.qpred(Q1,y1),names);report={"candidate_pool":"symmetric A/B min,max,absdiff gradient concentration plus sorted entity endpoint deltas","selected":[{"name":x[3],"index":x[2],"rho":x[1]}for x in chosen],"confirmation":{}}
 for k in(1,2,4):
  ix=[x[2]for x in chosen[:k]];runs=[S.fit_aug(Q2,X2[:,ix],y2,s)for s in S.SEEDS];pq=np.mean([z[0]for z in runs],0);pa=np.mean([z[1]for z in runs],0);qm=S.E.met(y2,pq);am=S.E.met(y2,pa);report["confirmation"][str(k)]={"features":[names[i]for i in ix],"question":qm,"augmented":am,"delta_auroc":am["auroc"]-qm["auroc"],"per_seed_delta":[S.E.met(y2,z[1])["auroc"]-S.E.met(y2,z[0])["auroc"]for z in runs]}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
