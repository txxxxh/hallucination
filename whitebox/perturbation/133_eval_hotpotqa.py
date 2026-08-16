#!/usr/bin/env python3
"""Frozen HotpotQA transfer evaluation plus leakage baselines and slices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'
def ch(s):
 u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32);return(d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def met(y,p):return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}
def slice_metrics(y,p,tags):
 out={}
 for tag in sorted(set(tags)):
  ix=np.flatnonzero(tags==tag)
  if len(ix)>=10 and len(set(y[ix]))==2:out[str(tag)]={'n':len(ix),**met(y[ix],p[ix])}
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,default=RUNS/'131_hotpotqa_balanced_n200.jsonl');a.add_argument('--features',type=Path,default=RUNS/'132_hotpotqa_current127');a.add_argument('--out',type=Path,default=RUNS/'133_hotpotqa_transfer.json');z=a.parse_args()
 manifest=[json.loads(x) for x in z.manifest.open() if x.strip()];meta={r['key']:r for r in manifest};rows=[]
 for fp in sorted(z.features.glob('*.npz')):
  with np.load(fp,allow_pickle=True)as q:
   k=str(q['key'].item());p=q['stage1_pred'].astype(np.float32);o=q['stage1_other'].astype(np.float32);p2=q['stage2_pred'].astype(np.float32);o2=q['stage2_other'].astype(np.float32);ph=q['pred_hidden'].astype(np.float32);oh=q['other_hidden'].astype(np.float32)
   scalar=np.r_[ch(p),ch(o),ch2(p2),ch2(o2),p[0]-p2[0],o[0]-o2[0],(p[0]-o[0])-(p2[0]-o2[0])]
   rows.append((k,int(q['correct']),scalar,(ph[0],wd(ph,p[0]-p[1:]),oh[0],wd(oh,o[0]-o[1:])),q['layer14'].astype(np.float32),[float(q['generation_words']),float(q['other_words'])],[p[0],o[0],p[0]-o[0]]))
 if {x[0] for x in rows}!=set(meta):raise RuntimeError(f'feature/manifest mismatch: features={len(rows)} manifest={len(meta)}')
 keys=np.array([x[0]for x in rows]);y=np.array([x[1]for x in rows]);S=np.stack([x[2]for x in rows]);H=[np.stack([x[3][j]for x in rows])for j in range(4)];L=np.stack([x[4]for x in rows]);length=np.stack([x[5]for x in rows]);raw=np.stack([x[6]for x in rows]);levels=np.array([meta[k]['level']for k in keys]);types=np.array([meta[k]['type']for k in keys]);scores={n:[]for n in ['fixed_detector','answer_length_only','unperturbed_scores_only']};slice_runs=[]
 for seed in(42,43,44):
  pred={n:np.zeros(len(y))for n in scores};cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(S,y):
   sc=StandardScaler().fit(S[tr]);pt=[sc.transform(S[tr])];pv=[sc.transform(S[te])]
   for x,d in[*[(x,8)for x in H],(L,48)]:
    sx=StandardScaler().fit(x[tr]);q=sx.transform(x[tr]);pc=PCA(d,whiten=True,svd_solver='randomized',random_state=seed).fit(q);pt.append(pc.transform(q));pv.append(pc.transform(sx.transform(x[te])))
   xt=np.concatenate(pt,1);xv=np.concatenate(pv,1);pred['fixed_detector'][te]=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,y[tr]).predict_proba(xv)[:,1]
   for name,x in [('answer_length_only',length),('unperturbed_scores_only',raw)]:
    sx=StandardScaler().fit(x[tr]);xt=sx.transform(x[tr]);xv=sx.transform(x[te]);pred[name][te]=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,y[tr]).predict_proba(xv)[:,1]
  for n,p in pred.items():scores[n].append(met(y,p))
  slice_runs.append({'seed':seed,'level':slice_metrics(y,pred['fixed_detector'],levels),'type':slice_metrics(y,pred['fixed_detector'],types)})
 results={n:{'mean':{k:float(np.mean([v[k]for v in vals]))for k in vals[0]},'per_seed':vals}for n,vals in scores.items()};report={'dataset':'HotpotQA distractor validation; all supporting paragraphs retained, distractors packed to 3600 chars','protocol':'frozen current127 configuration; 3x5 stratified OOF; no HotpotQA hyperparameter selection','n':len(y),'correct':int(y.sum()),'incorrect':int((1-y).sum()),'fixed_config':'nonoverlap stage1 32 + physical-delete minimal stage2 12 + delete delta 3 + dual hidden PCA32 + layer14 PCA48; LR C=.03','decoy_audit':{'same_generation_process_for_both_labels':True,'matches_gold':sum(bool(meta[k].get('decoy_matches_gold'))for k in keys)},'results':results,'fixed_detector_slices':slice_runs};z.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
