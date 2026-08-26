#!/usr/bin/env python3
"""Level-2 pilots: official-style cross-model probe and cross-model P.

The verifier sees the original Llama answer.  Cross-model P scores the exact
same Llama-selected physical deletion.  No closed-book probe is used as an
input feature or routing signal; it is used only for post-hoc known splits.
"""
from __future__ import annotations
import argparse,importlib,json,re,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs'
RAW=ROOT/'shuffled_prepend_names_question.json';REC=ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl';MAN=RUNS/'76_closedbook_fact_probe_manifest.jsonl';KP=RUNS/'77_closedbook_fact_probe_results.jsonl';OLD=RUNS/'120_physical_delete_rerank';OUT=RUNS/'280_scientist_cross_model_qwen7b_pilot'
def read(p):return [json.loads(x)for x in Path(p).open()if x.strip()]
def choose(limit):
 keys=sorted(x['key']for x in read(REC)if x.get('parse_valid',True));idx=np.linspace(0,len(keys)-1,min(limit or len(keys),len(keys)),dtype=int);return[keys[i]for i in idx]
def deleted_context(x,k):
 old=OLD/f'{k}.npz'
 if old.exists():
  with np.load(old,allow_pickle=True)as z:return str(z['deleted_context'].item())
 item=importlib.import_module('spanattr.core').Item.from_dict(x);words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",item.context,flags=re.UNICODE));pairs=[(words[i].start(),words[min(i+1,len(words)-1)].end())for i in range(0,len(words),2)]
 top=RUNS/'275_full_scientist_perturbation_trajectory'/f'{k}.npz';top=top if top.exists()else RUNS/'118_dual_candidate_multilayer_top5'/f'{k}.npz'
 with np.load(top,allow_pickle=True)as z:i=int(z['top_ids'][0])
 ca,cb=pairs[i];return re.sub(r'\s+([,.;:!?])',r'\1',re.sub(r'[ \t]+',' ',item.context[:ca]+item.context[cb:])).strip()
def collect(a):
 import torch
 try:torch._native.registry.deregister_op_overrides(disable_op_symbols='bmm',disable_dispatch_keys='CUDA')
 except(AttributeError,RuntimeError):pass
 from spanattr.core import Item,SpanAttributor,set_seed
 set_seed(42);raw={str(x['key']):x for x in json.load(RAW.open())};rec={x['key']:x for x in read(REC)};a.out.mkdir(parents=True,exist_ok=True)
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch);scan=importlib.import_module('125_collect_current_three_benchmarks').scan
 for n,k in enumerate(choose(a.limit),1):
  fp=a.out/f'{k}.npz'
  if a.resume and fp.exists():continue
  r=rec[k];x=raw[k];pred=str(r['parsed_answer']);right=str(x['rgt_ans']);wrong=str(x['wrg_ans']);other=wrong if pred==right else right
  deleted=deleted_context(x,k)
  # Official cross-model probe input: question followed by the response.
  chat=tok.apply_chat_template([{'role':'user','content':x['prompt']},{'role':'assistant','content':pred}],tokenize=True,add_generation_prompt=False,return_tensors='pt')
  if hasattr(chat,'input_ids'):chat=chat.input_ids
  chat=chat.to('cuda')
  with torch.inference_mode():out=model(chat,output_hidden_states=True,use_cache=False)
  qh=np.stack([h[0,-1].float().cpu().numpy()for h in out.hidden_states]);del out
  p1=att.prepare(Item.from_dict(dict(x,pred=pred,gold=other)));pp,po=scan(att,p1,[])
  xd=dict(x);xd['prompt']=deleted;p2=att.prepare(Item.from_dict(dict(xd,pred=pred,gold=other)));dp,do=scan(att,p2,[])
  np.savez_compressed(fp,key=np.asarray(k),correct=np.asarray(int(r['correct'])),qhidden=qh.astype(np.float16),qmargin=np.asarray([pp[0]-po[0],dp[0]-do[0]],np.float32))
  if n==1 or n%10==0:print(f'[{n}/{len(choose(a.limit))}] {k}',flush=True)
def pscalar(k):
 m=importlib.import_module('272_full_scientist_standard_upr_tables');return m.perturbation_blocks(k)[0]
def metrics(y,s):
 from sklearn.metrics import roc_auc_score,average_precision_score
 return{'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s))}
def evaluate(a):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
 from sklearn.preprocessing import StandardScaler
 man={x['key']:x for x in read(MAN)};known={x['key']:x for x in read(KP)};lh=RUNS/'141_scientist_all_trajectory_l8';rows=[]
 for f in sorted(a.out.glob('question_*.npz')):
  with np.load(f,allow_pickle=True)as z:k=str(z['key'].item());qh=z['qhidden'].astype(np.float32);qm=z['qmargin'].astype(np.float32)
  with np.load(lh/f'{k}.npz',allow_pickle=True)as z:ll=z['last'].astype(np.float32);layers=z['layers'].astype(int)
  rows.append((k,int(not bool(np.load(f,allow_pickle=True)['correct'])),man[k]['right_qid'],ll,qh,pscalar(k),np.r_[qm,qm[0]-qm[1]],known[k]))
 y=np.array([r[1]for r in rows]);g=np.array([r[2]for r in rows]);L=np.stack([r[3]for r in rows]);Q=np.stack([r[4]for r in rows]);P=np.stack([r[5]for r in rows]);QP=np.stack([r[6]for r in rows]);pred={n:[]for n in('llama_probe','qwen_probe','cross_probe','P','P_cross_qwen')}
 def fit(xtr,xte,yy,seed,dim=24):
  sc=StandardScaler().fit(xtr);u=sc.transform(xtr);v=sc.transform(xte);d=min(dim,len(u)-1,u.shape[1]);pc=PCA(d,whiten=True,random_state=seed).fit(u);m=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(pc.transform(u),yy);return m.predict_proba(pc.transform(v))[:,1]
 for seed in(42,43,44):
  out={n:np.zeros(len(y))for n in pred};cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(P,y,g):
   # Fold-local dev split selects hidden layer and fusion lambda, as in the paper.
   it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,y[tr],g[tr]));ti,vi=tr[it],tr[iv]
   def best(X):
    vals=[]
    for j in range(X.shape[1]):vals.append(metrics(y[vi],fit(X[ti,j],X[vi,j],y[ti],seed))['auroc'])
    return int(np.argmax(vals))
   li,qi=best(L),best(Q);dl=fit(L[ti,li],L[vi,li],y[ti],seed);dq=fit(Q[ti,qi],Q[vi,qi],y[ti],seed);lam=max(np.linspace(0,1,21),key=lambda w:metrics(y[vi],(1-w)*dl+w*dq)['auroc'])
   sl=fit(L[tr,li],L[te,li],y[tr],seed);sq=fit(Q[tr,qi],Q[te,qi],y[tr],seed);out['llama_probe'][te]=sl;out['qwen_probe'][te]=sq;out['cross_probe'][te]=(1-lam)*sl+lam*sq
   out['P'][te]=fit(P[tr],P[te],y[tr],seed,24);out['P_cross_qwen'][te]=fit(np.c_[P[tr],QP[tr]],np.c_[P[te],QP[te]],y[tr],seed,24)
  for n in pred:pred[n].append(out[n])
 score={n:np.mean(v,0)for n,v in pred.items()};kn=np.array([r[7]['n_discriminative_facts']>=1 and r[7]['binary_accuracy']>.5 and r[7]['pairwise_owner_accuracy']>.5 for r in rows]);rep={'protocol':'Qwen2.5-7B pilot; right-person grouped nested 3x5 OOF; fold-local layer/lambda; exact same Llama top-span deletion; no probe-derived input features','notes':{'P':'47-dimensional scalar ablation, not the formal full P','known_unknown':'closed-book probes used only for post-hoc reporting'},'n':len(y),'results':{}}
 for n,s in score.items():rep['results'][n]={'overall':metrics(y,s),'known':metrics(y[kn],s[kn]),'unknown':metrics(y[~kn],s[~kn]) if (~kn).sum() else None}
 pp={x['key']:x for x in read(RUNS/'272_full_scientist_standard_upr_tables_rightqid/predictions.jsonl')};ref=np.array([pp[r[0]]['p_error_probability']for r in rows]);rep['results']['P_full_reference_same_items']={'overall':metrics(y,ref),'known':metrics(y[kn],ref[kn]),'unknown':metrics(y[~kn],ref[~kn])}
 (a.out/'report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('collect','evaluate','all'));p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct');p.add_argument('--limit',type=int,default=256);p.add_argument('--batch',type=int,default=16);p.add_argument('--resume',action='store_true');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();collect(a)if a.stage=='collect'else evaluate(a)if a.stage=='evaluate'else(collect(a),evaluate(a))
if __name__=='__main__':main()
