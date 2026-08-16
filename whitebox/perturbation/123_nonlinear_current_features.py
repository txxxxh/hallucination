#!/usr/bin/env python3
"""Nonlinear heads on the current two-stage physical-deletion feature set."""
from __future__ import annotations
import importlib, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"
def ch(s):
 u=s[0]-s[1:]; z=abs(float(s[0]))+1e-6
 return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s): return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32); return (d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def met(y,p): return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}

def models(seed):
 out={"linear_C.03":LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed)}
 for w in (8,16,32):
  for a in (.3,1.,3.): out[f"mlp_relu_h{w}_a{a:g}"]=MLPClassifier(hidden_layer_sizes=(w,),activation="relu",solver="lbfgs",alpha=a,max_iter=2500,random_state=seed)
 out["mlp_relu_32_8_a1"]=MLPClassifier(hidden_layer_sizes=(32,8),activation="relu",solver="lbfgs",alpha=1,max_iter=2500,random_state=seed)
 out["hist_leaf5"]=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=5,l2_regularization=5,learning_rate=.05,random_state=seed)
 return out

def main():
 mod=importlib.import_module("101_fuse_sota_trajectory"); keys,g,y,_,_,_,_=mod.load_response("scientist"); _,_,last,_=mod.trajectory("scientist",keys)
 sep={}; dual={}; new={}
 for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); sep[k]=np.r_[ch(np.r_[z["pred_scores"][0],z["pred_scores"][0]-z["pred_u"]]),ch(np.r_[z["other_scores"][0],z["other_scores"][0]-z["other_u"]])]
 for fp in (RUNS/"116_dual_candidate_hidden_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); ph=z["pred_hidden"].astype(np.float32);oh=z["other_hidden"].astype(np.float32);dual[k]=(ph[0],wd(ph,z["pred_u"].astype(np.float32)),oh[0],wd(oh,z["other_u"].astype(np.float32)))
 for fp in (RUNS/"120_physical_delete_rerank").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item());p=z["stage1_pred_scores"].astype(np.float32);o=z["stage1_other_scores"].astype(np.float32);q=z["stage2_pred_scores"].astype(np.float32);r=z["stage2_other_scores"].astype(np.float32)
   new[k]=np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])]
 assert all(k in sep and k in dual and k in new for k in keys)
 S=np.stack([sep[k] for k in keys]); N=np.stack([new[k] for k in keys]); H=[np.stack([dual[k][j] for k in keys]) for j in range(4)]
 names=list(models(42)); scores={n:[] for n in names}; blends={n:{w:[] for w in (.25,.5,.75)} for n in names if n!="linear_C.03"}
 for seed in (42,43,44):
  pred={n:np.zeros(len(y)) for n in names}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(S,y,g),1):
   def scale(x): s=StandardScaler().fit(x[tr]);return s.transform(x[tr]),s.transform(x[te])
   st,sv=scale(S);nt,nv=scale(N); parts_t=[st,nt];parts_v=[sv,nv]
   for x,d in [*[(x,8) for x in H],(last[:,3],48)]:
    s=StandardScaler().fit(x[tr]);z=s.transform(x[tr]);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(z);parts_t.append(pc.transform(z));parts_v.append(pc.transform(s.transform(x[te])))
   xt=np.concatenate(parts_t,1);xv=np.concatenate(parts_v,1)
   for n,m in models(seed).items(): m.fit(xt,y[tr]);pred[n][te]=m.predict_proba(xv)[:,1]
   print(f"seed={seed} fold={fold}/5",flush=True)
  lin=pred["linear_C.03"]
  for n,p in pred.items():
   scores[n].append(met(y,p))
   if n in blends:
    for w in blends[n]: blends[n][w].append(met(y,(1-w)*lin+w*p))
 out=[]
 for n,v in scores.items():out.append({"model":n,"kind":"single",**{f"mean_{k}":float(np.mean([x[k] for x in v])) for k in v[0]},"per_seed":v})
 for n,ws in blends.items():
  for w,v in ws.items():out.append({"model":f"blend_linear_{1-w:g}_{n}_{w:g}","kind":"blend","nonlinear_weight":w,**{f"mean_{k}":float(np.mean([x[k] for x in v])) for k in v[0]},"per_seed":v})
 out.sort(key=lambda x:(x["mean_auroc"],x["mean_auprc"]),reverse=True);report={"protocol":"Scientist grouped 3x5 OOF; same-OOF exploratory model selection","feature_dim":159,"results":out};path=RUNS/"123_nonlinear_current_features.json";path.write_text(json.dumps(report,indent=2));print(json.dumps({"out":str(path),"top":out[:15]},indent=2))
if __name__=="__main__":main()
