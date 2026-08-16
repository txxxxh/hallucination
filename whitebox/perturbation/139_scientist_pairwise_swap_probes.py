#!/usr/bin/env python3
"""Direct A/B closed-book fact probes in both candidate orders, plus grouped evaluation."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs';MAN=RUNS/'76_closedbook_fact_probe_manifest.jsonl';RAW=RUNS/'139_pairwise_swap_raw.jsonl';REPORT=RUNS/'139_pairwise_swap_hierarchical_report.json';PREDS=RUNS/'139_pairwise_swap_hierarchical_predictions.jsonl'

def prompts():
 out=[]
 for item in map(json.loads,MAN.open()):
  ps=item['probes'];by={}
  for p in ps:by.setdefault(p['probe_id'].split('::')[1],[]).append(p)
  for fid,pair in by.items():
   owner=next(p for p in pair if p['gold_yes']);other=next(p for p in pair if not p['gold_yes']);prefix=f"Did {owner['person']} ";q=owner['prompt'].split('Question: ',1)[1];relation=q[len(prefix):].rstrip('?')
   for swap in(0,1):
    a,b=(owner['person'],other['person'])if not swap else(other['person'],owner['person']);gold='A'if not swap else'B'
    text=("Answer from your own knowledge only. Choose exactly A or B.\n"
          f"Which person is more likely to satisfy this fact: {relation}?\nA. {a}\nB. {b}\nAnswer:")
    out.append({'key':item['key'],'fact':fid,'swap':swap,'gold':gold,'owner':owner['person'],'other':other['person'],'prompt':text})
 return out
def collect(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 if a.disable_native_bmm:
  from torch._native.registry import deregister_op_overrides;deregister_op_overrides(disable_op_symbols='bmm')
 todo=prompts();tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token;tok.padding_side='left';aid=tok.encode('A',add_special_tokens=False);bid=tok.encode('B',add_special_tokens=False)
 if len(aid)!=1 or len(bid)!=1:raise RuntimeError((aid,bid))
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();vals=[]
 for st in range(0,len(todo),a.batch):
  z=tok([tok.apply_chat_template([{'role':'user','content':x['prompt']}],tokenize=False,add_generation_prompt=True)for x in todo[st:st+a.batch]],return_tensors='pt',padding=True).to('cuda')
  with torch.inference_mode():logits=model(**z,use_cache=False).logits[:,-1,[aid[0],bid[0]]].float();pr=torch.softmax(logits,-1).cpu().numpy()
  for x,p in zip(todo[st:st+a.batch],pr):vals.append({**x,'p_A':float(p[0]),'p_owner':float(p[0]if x['gold']=='A'else p[1])})
  if st==0 or(st//a.batch+1)%20==0:print(f'{min(st+a.batch,len(todo))}/{len(todo)}',flush=True)
 with RAW.open('w')as f:
  for x in vals:f.write(json.dumps(x,ensure_ascii=False)+'\n')

def stats(x):
 x=np.asarray(x,float)
 if not len(x):return np.zeros(8)
 return np.array([len(x),x.mean(),x.std(),x.min(),x.max(),np.median(x),np.mean(x>.5),np.mean(np.abs(x-.5))])
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
 return np.array([find(r['right_qid'])for r in rows])
def fit(x,y):return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',solver='liblinear')).fit(x,y)
def prob(m,x):return m.predict_proba(x)[:,list(m.classes_).index(1)]
def metrics(y,p):
 h=p.argmax(1);pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1,2],zero_division=0);err=y!=0;return{'accuracy':float(accuracy_score(y,h)),'macro_f1':float(f1_score(y,h,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(y,h)),'confusion':confusion_matrix(y,h,labels=[0,1,2]).tolist(),'precision':pr.tolist(),'recall':rc.tolist(),'f1':f.tolist(),'error_auroc':float(roc_auc_score(err,1-p[:,0])),'unknown_vs_known_error_auroc':float(roc_auc_score(y[err]==2,p[err,2]/(p[err,1]+p[err,2]+1e-9)))}
def evaluate():
 base=importlib.import_module('134_scientist_full_knowledge_error');old={x['key']:x for x in map(json.loads,(RUNS/'77_closedbook_fact_probe_results.jsonl').open())};rec={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open())};man={x['key']:x for x in map(json.loads,MAN.open())};new={}
 for x in map(json.loads,RAW.open()):new.setdefault(x['key'],{}).setdefault(x['fact'],[]).append(x)
 rows=[]
 for k,r in rec.items():
  if not r.get('parse_valid',True):continue
  o=old[k];known=int(o['n_discriminative_facts']>=1 and o['binary_accuracy']>.5 and o['pairwise_owner_accuracy']>.5);y=0 if r['correct']else(1 if known else 2);chosen=r['parsed_answer'];support=[];owner=[];cons=[]
  for pair in new.get(k,{}).values():
   pair=sorted(pair,key=lambda z:z['swap']);po=np.array([z['p_owner']for z in pair]);owner.extend(po);cons.append(1-abs(po[0]-po[1]));support.extend(po if chosen==pair[0]['owner']else 1-po)
  X=np.r_[base.features(o,r),stats(owner),stats(support),stats(cons)]
  rows.append({**man[k],'known':known,'correct':int(r['correct']),'y':y,'x':X})
 X=np.stack([r['x']for r in rows]);y=np.array([r['y']for r in rows]);kn=np.array([r['known']for r in rows]);co=np.array([r['correct']for r in rows]);g=components(rows);ps=[]
 for seed in(42,43,44):
  p=np.zeros((len(y),3));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y,g):
   pk=prob(fit(X[tr],kn[tr]),X[te]);kt=tr[kn[tr]==1];ut=tr[kn[tr]==0];pck=prob(fit(X[kt],co[kt]),X[te]);pcu=prob(fit(X[ut],co[ut]),X[te]);p[te]=np.c_[pk*pck+(1-pk)*pcu,pk*(1-pck),(1-pk)*(1-pcu)];p[te]/=p[te].sum(1,keepdims=True)
  ps.append(p)
 mean=np.mean(ps,0);report={'protocol':'old Yes/No + direct A/B in both orders; hierarchical candidate-component 3x5 OOF','n':len(y),'components':len(set(g)),'mean_probability':metrics(y,mean),'per_seed':[metrics(y,p)for p in ps]};REPORT.write_text(json.dumps(report,indent=2)+chr(10))
 with PREDS.open('w')as f:
  for r,p in zip(rows,mean):f.write(json.dumps({'key':r['key'],'target':r['y'],'probabilities':p.tolist()})+chr(10))
 print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=64);p.add_argument('--disable-native-bmm',action='store_true');a=p.parse_args()
 if a.stage in('collect','all'):collect(a)
 if a.stage in('evaluate','all'):evaluate()
if __name__=='__main__':main()
