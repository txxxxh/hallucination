#!/usr/bin/env python3
"""Full 2894-row corrected runner for experiment 194."""
import importlib
import numpy as np
from sklearn.model_selection import train_test_split as real_split

F=importlib.import_module("194_embedding_scalar_pca128_fusion")
UNKNOWN=F.M.RUNS/"195_unknown_embedding_exact"
F.OUT=F.M.RUNS/"196_full_embedding_scalar_pca128_fusion.json"

def load():
 rows,*_=F.B.load_rows();kept=[];vals=[]
 for r in rows:
  root=F.SRC if int(r["known"]) else UNKNOWN;fp=root/(r["key"]+".npz")
  if not fp.exists() or not(F.B.QUESTION_CACHE/(r["key"]+".npz")).exists():continue
  with np.load(fp,allow_pickle=True)as z:p,o,q2,r2=z["stage1_pred_scores"],z["stage1_other_scores"],z["stage2_pred_scores"],z["stage2_other_scores"]
  vals.append(np.r_[F.E.scientist.ch(p),F.E.scientist.ch(o),F.E.scientist.ch2(q2),F.E.scientist.ch2(r2),p[0]-q2[0],o[0]-r2[0],(p[0]-o[0])-(q2[0]-r2[0])]);kept.append(r)
 keys=[r["key"]for r in kept];y=np.array([r["known"]for r in kept]);x=np.asarray(vals,np.float32)
 q=np.stack([np.load(F.B.QUESTION_CACHE/(k+".npz"))["hidden"][F.S.P.KEEN_LAYERS].astype(np.float32)for k in keys]);return keys,y,q,x

def split(*args,**kwargs):
 kwargs["train_size"]=1000
 return real_split(*args,**kwargs)

F.load=load;F.train_test_split=split;F.main()
