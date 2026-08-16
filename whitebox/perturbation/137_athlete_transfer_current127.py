#!/usr/bin/env python3
"""Collect AthleteQA current127 features; fit only on Scientist-known and transfer."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
from spanattr.core import Item,SpanAttributor,set_seed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs';PILOT=ROOT/'athlete_qa'/'pilot_v1';CACHE=PILOT/'current127';REPORT=PILOT/'scientist_detector_transfer.json';PREDS=PILOT/'scientist_detector_transfer_predictions.jsonl'

def rows():
 q={x['id']:x for x in map(json.loads,(PILOT/'primary_questions.jsonl').open())};r={x['id']:x for x in map(json.loads,(PILOT/'llama_eval'/'results.jsonl').open())}
 out=[]
 for k,x in q.items():
  z=r[k];pred=str(z['generation']);right=x['correct_answer'];wrong=x['wrong_answer'];other=wrong if pred==right else right
  out.append({'key':k,'correct':int(z['name_correct']),'known_both':z['probe_state']=='knows_both','prompt':x['prepend_names_prompt'],'pred':pred,'other':other})
 return out
def fixed(s,n=6):s=np.asarray(s,np.float32);return np.pad(s[:n],(0,max(0,n-len(s))))
def ch(s):s=fixed(s);u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):s=fixed(s);return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):h=np.asarray(h,np.float32);u=np.asarray(u,np.float32);d=h[1:]-h[0];return(d[:len(u)]*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def unpack(p,o,q,r,ph,oh,l):return np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])],[ph[0],wd(ph,p[0]-p[1:]),oh[0],wd(oh,o[0]-o[1:])],l

def collect(a):
 mod=importlib.import_module('125_collect_current_three_benchmarks');CACHE.mkdir(parents=True,exist_ok=True);set_seed(42);model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 for n,x in enumerate(rows(),1):
  fp=CACHE/f"{x['key']}.npz"
  if fp.exists()and a.resume:continue
  item=Item.from_dict({'key':x['key'],'prompt':x['prompt'],'pred':x['pred'],'gold':x['other']});prep=att.prepare(item);ss,cc=mod.spans(att,prep);p,o=mod.scan(att,prep,ss);u=(p[0]-p[1:])-(o[0]-o[1:]);top=int(np.argmax(np.abs(u)));ids=np.argsort(-np.abs(u))[:min(5,len(u))];ph,oh,l=mod.selected_hidden(att,prep,ids);ca,cb=cc[top];deleted=re.sub(r'[ \t]+',' ',item.context[:ca]+item.context[cb:]);deleted=re.sub(r'\s+([,.;:!?])',r'\1',deleted).strip();i2=Item(x['key']+'_d',deleted,item.question,x['other'],x['pred'],context_prefix=item.context_prefix);z=att.prepare(i2);ss2,_=mod.spans(att,z);q,r=mod.scan(att,z,ss2);u2=(q[0]-q[1:])-(r[0]-r[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))];np.savez_compressed(fp,key=np.asarray(x['key']),stage1_pred=np.r_[p[0],p[1:][ids]],stage1_other=np.r_[o[0],o[1:][ids]],stage2_pred=np.r_[q[0],q[1:][ids2]],stage2_other=np.r_[r[0],r[1:][ids2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=l.astype(np.float16));print(f'[{n}/100] {x["key"]}',flush=True)

def evaluate():
 src=[];known=[json.loads(x)for x in (RUNS/'88_known_gt05_n1084.jsonl').open()if x.strip()]
 for x in known:
  k=x['key'];a=RUNS/'120_physical_delete_rerank'/f'{k}.npz';b=RUNS/'116_dual_candidate_hidden_top5'/f'{k}.npz';c=RUNS/'100_scientist_trajectory_l8'/f'{k}.npz'
  with np.load(a,allow_pickle=True)as z:p=z['stage1_pred_scores'];o=z['stage1_other_scores'];q=z['stage2_pred_scores'];r=z['stage2_other_scores']
  with np.load(b,allow_pickle=True)as z:ph=z['pred_hidden'];oh=z['other_hidden']
  with np.load(c,allow_pickle=True)as z:l=z['mean'].astype(np.float32)[3]
  S,H,L=unpack(p,o,q,r,ph,oh,l);src.append((int(x['correct']),S,H,L))
 test=[]
 for x in rows():
  with np.load(CACHE/f"{x['key']}.npz",allow_pickle=True)as z:S,H,L=unpack(z['stage1_pred'],z['stage1_other'],z['stage2_pred'],z['stage2_other'],z['pred_hidden'],z['other_hidden'],z['layer14'])
  test.append((x,S,H,L))
 y=np.array([x[0]for x in src]);S=np.stack([x[1]for x in src]);St=np.stack([x[1]for x in test]);parts=[];partt=[];sc=StandardScaler().fit(S);parts.append(sc.transform(S));partt.append(sc.transform(St))
 for j,d in [*((j,8)for j in range(4)),(4,48)]:
  X=np.stack([x[2][j]if j<4 else x[3]for x in src]);T=np.stack([x[2][j]if j<4 else x[3]for x in test]);sc=StandardScaler().fit(X);Z=sc.transform(X);pc=PCA(d,whiten=True,svd_solver='randomized',random_state=42).fit(Z);parts.append(pc.transform(Z));partt.append(pc.transform(sc.transform(T)))
 clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=42).fit(np.concatenate(parts,1),y);prob=clf.predict_proba(np.concatenate(partt,1))[:,1];yt=np.array([x[0]['correct']for x in test])
 def met(ix):
  a=yt[ix];p=prob[ix];h=p>=.5;return{'n':int(len(a)),'correct':int(a.sum()),'auroc':float(roc_auc_score(a,p)),'auprc':float(average_precision_score(a,p)),'accuracy_at_0.5':float(accuracy_score(a,h)),'balanced_accuracy_at_0.5':float(balanced_accuracy_score(a,h)),'confusion_tn_fp_fn_tp':confusion_matrix(a,h,labels=[0,1]).ravel().tolist()}
 kb=np.array([x[0]['known_both']for x in test]);report={'protocol':'frozen transfer: fit all transforms and LR only on Scientist-known 1084; no Athlete fitting/tuning','scientist_config':'current127 scalar + dual layer16 PCA8x4 + layer14 mean PCA48; LR C=.03','all':met(np.ones(len(yt),bool)),'probe_known_both':met(kb)};REPORT.write_text(json.dumps(report,indent=2)+'\n')
 with PREDS.open('w')as f:
  for (x,_,_,_),p in zip(test,prob):f.write(json.dumps({'id':x['key'],'correct':x['correct'],'known_both':x['known_both'],'prob_correct':float(p)})+'\n')
 print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=24);p.add_argument('--resume',action='store_true');a=p.parse_args();
 if a.stage in('collect','all'):collect(a)
 if a.stage in('evaluate','all'):evaluate()
if __name__=='__main__':main()
