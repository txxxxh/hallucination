#!/usr/bin/env python3
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';m=importlib.import_module('152_scientist_attention_pruned_current127')
meta={x[0]:(x[1],x[2])for x in m.jobs()};rows=[]
for key,(group,label)in meta.items():
 with np.load(RUNS/'120_physical_delete_rerank'/f'{key}.npz',allow_pickle=True)as z:p,o,q,r=z['stage1_pred_scores'],z['stage1_other_scores'],z['stage2_pred_scores'],z['stage2_other_scores']
 with np.load(RUNS/'116_dual_candidate_hidden_top5'/f'{key}.npz',allow_pickle=True)as z:ph,oh=z['pred_hidden'],z['other_hidden']
 with np.load(RUNS/'100_scientist_trajectory_l8'/f'{key}.npz',allow_pickle=True)as z:L=z['mean'].astype(np.float32)[3]
 S=np.r_[m.ch(p),m.ch(o),m.ch2(q),m.ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])];H=[ph[0],m.wd(ph,p[0]-p[1:]),oh[0],m.wd(oh,o[0]-o[1:])];rows.append((group,label,S,H,L))
g=np.array([x[0]for x in rows]);y=np.array([x[1]for x in rows]);S=np.stack([x[2]for x in rows]);H=[np.stack([x[3][j]for x in rows])for j in range(4)];L=np.stack([x[4]for x in rows]);res=[]
for seed in(42,43,44):
 prob=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
 for tr,te in cv.split(S,y,g):
  at=[];av=[]
  for X,d in[(S,None),*[(x,8)for x in H],(L,48)]:
   sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(X[te])
   if d is not None:pc=PCA(d,whiten=True,svd_solver='randomized',random_state=seed).fit(a);a,b=pc.transform(a),pc.transform(b)
   at.append(a);av.append(b)
  c=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(np.concatenate(at,1),y[tr]);prob[te]=c.predict_proba(np.concatenate(av,1))[:,1]
 res.append({'auroc':float(roc_auc_score(y,prob)),'auprc':float(average_precision_score(y,prob)),'balanced_accuracy':float(balanced_accuracy_score(y,prob>=.5))})
out={'protocol':'exact-enumeration current127 same 1084 rows and grouped 3x5 CV as 152','per_seed':res,'mean':{k:float(np.mean([x[k]for x in res]))for k in res[0]}};(RUNS/'153_exact_current127_samecv_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
