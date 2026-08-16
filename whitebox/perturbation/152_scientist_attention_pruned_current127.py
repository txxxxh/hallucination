#!/usr/bin/env python3
"""Rebuild current127 with contrastive-attention coarse pruning on Scientist."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from spanattr.core import Item, SpanAttributor, set_seed

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'
CACHE=RUNS/'152_scientist_attention_pruned_current127'; OUT=RUNS/'152_scientist_attention_pruned_current127_report.json'

def jobs():
 known={x['key']:x for x in map(json.loads,(RUNS/'88_known_gt05_n1084.jsonl').open())}
 data={x['key']:x for x in json.load((ROOT/'shuffled_prepend_names_question.json').open())}
 recs={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open())}
 out=[]
 for key,k in known.items():
  d,r=data[key],recs[key]; pred=str(r['parsed_answer']); other=d['wrg_ans'] if pred==d['rgt_ans'] else d['rgt_ans']
  out.append((key,str(k['group']),int(k['correct']),d['prompt'],pred,other))
 return out

def contrastive_token_attention(att,prep):
 import torch
 P=len(prep.prompt_ids); maps=[]
 for ans in (prep.pred_variant_ids[0],prep.gold_variant_ids[0]):
  ids=torch.cat([prep.prompt_ids,ans]).unsqueeze(0)
  with torch.inference_mode():out=att.model(input_ids=ids,output_attentions=True,use_cache=False)
  layers=[]
  for A in out.attentions:
   layers.append(A[0,:,P-1:P+len(ans)-1,:P].float().mean((0,1)).cpu().numpy())
  maps.append(np.mean(layers,0));del out
 return maps[0]+np.abs(maps[0]-maps[1])

def shortlist(att,prep,ss,keep=6,blocks=12):
 score=contrastive_token_attention(att,prep); edges=np.linspace(prep.ctx_start,prep.ctx_end,blocks+1).round().astype(int)
 regions=[(edges[i],edges[i+1]) for i in range(blocks) if edges[i]<edges[i+1]]
 bs=np.array([score[a:b].sum() for a,b in regions]); chosen=np.argsort(-bs)[:keep]
 ids=[i for i,s in enumerate(ss) if any(s.end>regions[j][0] and s.start<regions[j][1] for j in chosen)]
 return ids

def scan_subset(att,prep,ids):
 import torch
 z=torch.zeros(len(prep.prompt_ids),device=att.device);A=torch.stack([z,*[att.alpha_from_spans(prep,[i]) for i in ids]])
 p,o=att.class_scores_batched(prep,A);return p.numpy(),o.numpy()

def collect(a):
 mod=importlib.import_module('125_collect_current_three_benchmarks');CACHE.mkdir(parents=True,exist_ok=True);set_seed(42)
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 rows=jobs()
 for n,(key,group,label,prompt,pred,other) in enumerate(rows,1):
  fp=CACHE/f'{key}.npz'
  if fp.exists() and a.resume:continue
  item=Item.from_dict({'key':key,'prompt':prompt,'pred':pred,'gold':other});prep=att.prepare(item);ss,cc=mod.spans(att,prep);pool=shortlist(att,prep,ss,a.keep,a.blocks);p,o=scan_subset(att,prep,pool);u=(p[0]-p[1:])-(o[0]-o[1:]);order=np.argsort(-np.abs(u));local=order[:min(5,len(order))];ids=np.array([pool[i] for i in local]);ph,oh,l=mod.selected_hidden(att,prep,ids)
  top=ids[0];ca,cb=cc[top];deleted=re.sub(r'[ \t]+',' ',item.context[:ca]+item.context[cb:]);deleted=re.sub(r'\s+([,.;:!?])',r'\1',deleted).strip();i2=Item(key+'_d',deleted,item.question,other,pred,context_prefix=item.context_prefix);z=att.prepare(i2);ss2,_=mod.spans(att,z);pool2=shortlist(att,z,ss2,a.keep,a.blocks);q,r=scan_subset(att,z,pool2);u2=(q[0]-q[1:])-(r[0]-r[1:]);order2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
  np.savez_compressed(fp,key=np.asarray(key),group=np.asarray(group),correct=np.asarray(label),stage1_pred=np.r_[p[0],p[1:][local]],stage1_other=np.r_[o[0],o[1:][local]],stage2_pred=np.r_[q[0],q[1:][order2]],stage2_other=np.r_[r[0],r[1:][order2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=l.astype(np.float16),stage1_candidates=np.asarray(len(pool)),stage1_full=np.asarray(len(ss)),stage2_candidates=np.asarray(len(pool2)),stage2_full=np.asarray(len(ss2)));print(f'[{n}/{len(rows)}] {key} q={len(pool)}/{len(ss)}+{len(pool2)}/{len(ss2)}',flush=True)

def ch(s):u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):d=h[1:].astype(np.float32)-h[0].astype(np.float32);return(d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def load(cache):
 rows=[]
 for fp in sorted(cache.glob('*.npz')):
  with np.load(fp,allow_pickle=True)as z:
   p,o,q,r=z['stage1_pred'],z['stage1_other'],z['stage2_pred'],z['stage2_other'];S=np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])];ph,oh=z['pred_hidden'],z['other_hidden'];H=[ph[0],wd(ph,p[0]-p[1:]),oh[0],wd(oh,o[0]-o[1:])];rows.append((str(z['key']),str(z['group']),int(z['correct']),S,H,z['layer14'].astype(np.float32)))
 return rows
def evaluate():
 rows=load(CACHE);keys=np.array([x[0]for x in rows]);g=np.array([x[1]for x in rows]);y=np.array([x[2]for x in rows]);S=np.stack([x[3]for x in rows]);H=[np.stack([x[4][j]for x in rows])for j in range(4)];L=np.stack([x[5]for x in rows]);per=[]
 for seed in (42,43,44):
  prob=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(S,y,g):
   parts=[];partv=[]
   for X,d in [(S,None),*[(x,8)for x in H],(L,48)]:
    sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(X[te]);
    if d is not None:pc=PCA(d,whiten=True,svd_solver='randomized',random_state=seed).fit(a);a,b=pc.transform(a),pc.transform(b)
    parts.append(a);partv.append(b)
   clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(np.concatenate(parts,1),y[tr]);prob[te]=clf.predict_proba(np.concatenate(partv,1))[:,1]
  per.append({'auroc':float(roc_auc_score(y,prob)),'auprc':float(average_precision_score(y,prob)),'balanced_accuracy':float(balanced_accuracy_score(y,prob>=.5))})
 qs=[]
 for fp in CACHE.glob('*.npz'):
  with np.load(fp)as z:qs.append([int(z['stage1_candidates']),int(z['stage1_full']),int(z['stage2_candidates']),int(z['stage2_full'])])
 q=np.array(qs);report={'protocol':'Scientist-known 1084, grouped 3x5 OOF, current127 LR unchanged; both stages attention-pruned to 6/12 coarse blocks','per_seed':per,'mean':{k:float(np.mean([x[k]for x in per]))for k in per[0]},'queries':{'pruned_mean':float((q[:,0]+q[:,2]+4).mean()),'full_mean':float((q[:,1]+q[:,3]).mean()),'reduction':float(1-(q[:,0]+q[:,2]+4).sum()/(q[:,1]+q[:,3]).sum())}}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=24);p.add_argument('--blocks',type=int,default=12);p.add_argument('--keep',type=int,default=6);p.add_argument('--resume',action='store_true');a=p.parse_args();
 if a.stage in ('collect','all'):collect(a)
 if a.stage in ('evaluate','all'):evaluate()
if __name__=='__main__':main()
