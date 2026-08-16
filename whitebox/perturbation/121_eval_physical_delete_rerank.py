#!/usr/bin/env python3
"""OOF ablation for physical top-1 deletion and second-stage reranking."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"

def ch(scores):
 u=scores[0]-scores[1:]; scale=abs(float(scores[0]))+1e-6
 return np.r_[scores[0],u,u/scale,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch_minimal(scores):
 return np.r_[scores[0],scores[0]-scores[1:]]
def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32); return (d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def met(y,p): return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}

def main():
 mod=importlib.import_module("101_fuse_sota_trajectory"); keys,groups,y,_,_,_,_=mod.load_response("scientist"); _,_,last,_=mod.trajectory("scientist",keys)
 oldsep={}; olddual={}; new={}
 for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z: oldsep[str(z["key"].item())]=np.r_[ch(np.r_[z["pred_scores"][0],z["pred_scores"][0]-z["pred_u"]]),ch(np.r_[z["other_scores"][0],z["other_scores"][0]-z["other_u"]])]
 for fp in (RUNS/"116_dual_candidate_hidden_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); ph=z["pred_hidden"].astype(np.float32); oh=z["other_hidden"].astype(np.float32)
   olddual[k]=(ph[0],wd(ph,z["pred_u"].astype(np.float32)),oh[0],wd(oh,z["other_u"].astype(np.float32)))
 for fp in (RUNS/"120_physical_delete_rerank").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); p1=z["stage1_pred_scores"].astype(np.float32); o1=z["stage1_other_scores"].astype(np.float32); p2=z["stage2_pred_scores"].astype(np.float32); o2=z["stage2_other_scores"].astype(np.float32)
   delete=np.r_[p1[0]-p2[0],o1[0]-o2[0],(p1[0]-o1[0])-(p2[0]-o2[0])]
   new[k]=(np.r_[ch(p1),ch(o1)],np.r_[ch_minimal(p2),ch_minimal(o2)],delete)
 miss=[k for k in keys if k not in oldsep or k not in olddual or k not in new]
 if miss: raise RuntimeError(f"missing {len(miss)} rows")
 O=np.stack([oldsep[k] for k in keys]); OD=[np.stack([olddual[k][j] for k in keys]) for j in range(4)]
 A=np.stack([new[k][0] for k in keys]); B=np.stack([new[k][1] for k in keys]); D=np.stack([new[k][2] for k in keys])
 names=["old_best","old_plus_stage1_prob","old_plus_two_stage_prob","replace_old_with_two_stage"] ; scores={n:[] for n in names}
 for seed in (42,43,44):
  pred={n:np.zeros(len(y)) for n in names}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(O,y,groups),1):
   def scale(x): s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
   ot,ov=scale(O); at,av=scale(A); bt,bv=scale(B); dt,dv=scale(D)
   x=last[:,3]; s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(48,whiten=True,svd_solver="randomized",random_state=seed).fit(q); lt,lv=pc.transform(q),pc.transform(s.transform(x[te]))
   def pcs(blocks,d):
    aa=[];bb=[]
    for x in blocks:
     s=StandardScaler().fit(x[tr]);q=s.transform(x[tr]);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(q);aa.append(pc.transform(q));bb.append(pc.transform(s.transform(x[te])))
    return np.concatenate(aa,1),np.concatenate(bb,1)
   odt,odv=pcs(OD,8)
   base=(np.c_[ot,odt,lt],np.c_[ov,odv,lv])
   hidden_base=(np.c_[odt,lt],np.c_[odv,lv])
   sets={"old_best":base,"old_plus_stage1_prob":(np.c_[base[0],at],np.c_[base[1],av]),"old_plus_two_stage_prob":(np.c_[base[0],at,bt,dt],np.c_[base[1],av,bv,dv]),"replace_old_with_two_stage":(np.c_[hidden_base[0],at,bt,dt],np.c_[hidden_base[1],av,bv,dv])}
   for n,(xt,xv) in sets.items():
    clf=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(xt,y[tr]);pred[n][te]=clf.predict_proba(xv)[:,1]
   print(f"seed={seed} fold={fold}/5",flush=True)
  for n in names:scores[n].append(met(y,pred[n]))
 out=[{"variant":n,**{f"mean_{k}":float(np.mean([v[k] for v in vals])) for k in vals[0]},"per_seed":vals} for n,vals in scores.items()];out.sort(key=lambda x:x["mean_auroc"],reverse=True)
 report={"protocol":"Scientist question-grouped 3x5-fold OOF","dimensions":{"old_best":112,"stage1_prob":32,"stage2_prob":12,"physical_delete_delta":3,"replacement_total":127},"results":out};path=RUNS/"124_replace_overlapping_stage1.json";path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
