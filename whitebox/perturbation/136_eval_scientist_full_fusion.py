#!/usr/bin/env python3
"""Compare main-task, closed-book, and fused full ScientistQA detectors."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs';OUT=RUNS/'136_scientist_full_fusion_report.json';PREDS=RUNS/'136_scientist_full_fusion_predictions.jsonl'

def fixed(s,n=6):s=np.asarray(s,np.float32);return np.pad(s[:n],(0,max(0,n-len(s))))
def ch(s):
 s=fixed(s);u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):s=fixed(s);return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):
 h=np.asarray(h,np.float32);u=np.asarray(u,np.float32);d=h[1:]-h[0];return(d[:len(u)]*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def scalar(p,o,q,r):return np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])]

def components(rows):
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 for r in rows:union(r['right_qid'],r['wrong_qid'])
 return np.array([find(r['right_qid']) for r in rows])

def metrics(y,p):
 yh=p.argmax(1);err=y!=0;pe=1-p[:,0];pu=p[err,2]/(p[err,1]+p[err,2]+1e-9)
 return {'accuracy':float(accuracy_score(y,yh)),'macro_f1':float(f1_score(y,yh,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,yh)),'confusion':confusion_matrix(y,yh,labels=[0,1,2]).tolist(),'error_auroc':float(roc_auc_score(err,pe)),'unknown_vs_known_error_auroc':float(roc_auc_score(y[err]==2,pu))}

def transform(train,test,dim,seed):
 sc=StandardScaler().fit(train);z=sc.transform(train);dim=min(dim,len(train)-1,train.shape[1]);pc=PCA(dim,whiten=True,svd_solver='randomized',random_state=seed).fit(z);return pc.transform(z),pc.transform(sc.transform(test))

def main():
 probe_mod=importlib.import_module('134_scientist_full_knowledge_error');probes={x['key']:x for x in map(json.loads,(RUNS/'77_closedbook_fact_probe_results.jsonl').open())};recs={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open())};man={x['key']:x for x in map(json.loads,(RUNS/'76_closedbook_fact_probe_manifest.jsonl').open())};data=[]
 traj={}
 for fp in (RUNS/'100_scientist_trajectory_l8').glob('*.npz'):
  with np.load(fp,allow_pickle=True)as z:traj[str(z['key'].item())]=z['mean'].astype(np.float32)[3]
 for key,r in recs.items():
  if not r.get('parse_valid',True):continue
  p=probes[key];known=bool(p['n_discriminative_facts']>=1 and p['binary_accuracy']>.5 and p['pairwise_owner_accuracy']>.5);target=0 if r['correct'] else(1 if known else 2);S=H=L=None
  fp=RUNS/'135_scientist_full_current127'/f'{key}.npz'
  if fp.exists():
   with np.load(fp,allow_pickle=True)as z:
    a=z['stage1_pred'].astype(np.float32);b=z['stage1_other'].astype(np.float32);c=z['stage2_pred'].astype(np.float32);d=z['stage2_other'].astype(np.float32);ph=z['pred_hidden'].astype(np.float32);oh=z['other_hidden'].astype(np.float32);S=scalar(a,b,c,d);H=[ph[0],wd(ph,a[0]-a[1:]),oh[0],wd(oh,b[0]-b[1:])];L=z['layer14'].astype(np.float32)
  elif known:
   f1=RUNS/'120_physical_delete_rerank'/f'{key}.npz';f2=RUNS/'116_dual_candidate_hidden_top5'/f'{key}.npz'
   if f1.exists()and f2.exists()and key in traj:
    with np.load(f1,allow_pickle=True)as z:a=z['stage1_pred_scores'].astype(np.float32);b=z['stage1_other_scores'].astype(np.float32);c=z['stage2_pred_scores'].astype(np.float32);d=z['stage2_other_scores'].astype(np.float32);S=scalar(a,b,c,d)
    with np.load(f2,allow_pickle=True)as z:ph=z['pred_hidden'].astype(np.float32);oh=z['other_hidden'].astype(np.float32);H=[ph[0],wd(ph,z['pred_u']),oh[0],wd(oh,z['other_u'])]
    L=traj[key]
  if S is not None:data.append({**man[key],'key':key,'y':target,'S':S,'H':H,'L':L,'P':probe_mod.features(p,r)})
 expected=sum(x.get('parse_valid',True) for x in recs.values())
 if len(data)!=expected:raise RuntimeError(f'incomplete features {len(data)}/{expected}')
 y=np.array([x['y']for x in data]);S=np.stack([x['S']for x in data]);P=np.stack([x['P']for x in data]);H=[np.stack([x['H'][j]for x in data])for j in range(4)];L=np.stack([x['L']for x in data]);g=components(data);results={};saved={}
 for variant in ('probe','main','fusion'):
  ps=[]
  for seed in(42,43,44):
   pred=np.zeros((len(y),3));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
   for tr,te in cv.split(S,y,g):
    parts_t=[];parts_v=[]
    if variant in('probe','fusion'):
     sc=StandardScaler().fit(P[tr]);parts_t.append(sc.transform(P[tr]));parts_v.append(sc.transform(P[te]))
    if variant in('main','fusion'):
     sc=StandardScaler().fit(S[tr]);parts_t.append(sc.transform(S[tr]));parts_v.append(sc.transform(S[te]))
     for x,d in [*((x,8)for x in H),(L,48)]:a,b=transform(x[tr],x[te],d,seed);parts_t.append(a);parts_v.append(b)
    clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(np.concatenate(parts_t,1),y[tr]);pred[te]=clf.predict_proba(np.concatenate(parts_v,1));
   ps.append(pred)
  pm=np.mean(ps,0);results[variant]={'mean_probability':metrics(y,pm),'per_seed':[metrics(y,p)for p in ps]};saved[variant]=pm
 report={'protocol':'parse-valid 2894/2925; candidate-identity connected-component grouped 3x5-fold OOF; all transforms train-fold only','classes':{'0':'correct','1':'known_but_wrong','2':'unknown_and_wrong'},'n':len(y),'excluded_parse_invalid':2925-len(y),'identity_components':len(set(g)),'class_counts':{str(i):int(np.sum(y==i))for i in range(3)},'results':results};OUT.write_text(json.dumps(report,indent=2)+'\n')
 with PREDS.open('w')as f:
  for i,x in enumerate(data):f.write(json.dumps({'key':x['key'],'target':int(y[i]),**{f'{v}_probabilities':saved[v][i].tolist()for v in saved}})+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
