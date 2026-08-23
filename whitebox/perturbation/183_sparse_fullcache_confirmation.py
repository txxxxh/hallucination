#!/usr/bin/env python3
"""Select <=4 compact run-164 scalars on 1000 rows; confirm on held-out 1894."""
import importlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
S=importlib.import_module("180_select_sparse_complementary_features_fixed");B=S.B;RUNS=S.RUNS;D=RUNS/"164_ic_local_geometry_n2894";OUT=RUNS/"183_sparse_fullcache_confirmation.json"+def compact(x):
 base=x[:198].reshape(33,6);odd=x[198:792].reshape(3,33,6);even=x[792:1386].reshape(3,33,6);tail=x[1386:];return np.r_[base.mean(0),base.std(0),np.mean(abs(odd),(1,2)),np.std(odd,(1,2)),np.mean(abs(even),(1,2)),np.std(even,(1,2)),tail].astype(np.float32)
N=[f"base_mean_{i}"for i in range(6)]+[f"base_std_{i}"for i in range(6)]+[f"odd_absmean_eps{i}"for i in range(3)]+[f"odd_std_eps{i}"for i in range(3)]+[f"even_absmean_eps{i}"for i in range(3)]+[f"even_std_eps{i}"for i in range(3)]+[f"tail_{i}"for i in range(12)]
def main():
 rows,*_=B.load_rows();rows=[r for r in rows if(D/"features"/(r["key"]+".npz")).exists()];y=np.array([r["known"]for r in rows]);di,ci=train_test_split(np.arange(len(rows)),train_size=1000,stratify=y,random_state=B.SEED);X=np.stack([compact(np.load(D/"features"/(r["key"]+".npz"))["local_geometry"])for r in rows]);Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][S.P.KEEN_LAYERS].astype(np.float32)for r in rows]);chosen=S.select(X[di],y[di],S.qpred(Q[di],y[di]),N);report={"split":"fixed stratified discovery n=1000 / untouched confirmation n=1894","candidate_features":len(N),"selected":[{"name":z[3],"index":z[2],"rho":z[1]}for z in chosen],"confirmation":{}}
 for k in(1,2,4):
  ix=[z[2]for z in chosen[:k]];runs=[S.fit_aug(Q[ci],X[ci][:,ix],y[ci],s)for s in S.SEEDS];pq=np.mean([z[0]for z in runs],0);pa=np.mean([z[1]for z in runs],0);qm=S.E.met(y[ci],pq);am=S.E.met(y[ci],pa);report["confirmation"][str(k)]={"features":[N[i]for i in ix],"question":qm,"augmented":am,"delta_auroc":am["auroc"]-qm["auroc"],"per_seed_delta":[S.E.met(y[ci],z[1])["auroc"]-S.E.met(y[ci],z[0])["auroc"]for z in runs]}
 B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
