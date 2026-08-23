#!/usr/bin/env python3
"""Unified UEPR scores for the Scientist full-profiles baseline."""
from __future__ import annotations
import argparse,json,re,string
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'
def norm(s):
 s=str(s).lower();s=''.join(' 'if c in string.punctuation else c for c in s);s=re.sub(r'\b(a|an|the)\b',' ',s);return' '.join(s.split())
def entropy(xs):
 c=np.asarray(list(Counter(xs).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--samples',type=int,default=6);ap.add_argument('--resume',action='store_true');ap.add_argument('--out-dir',type=Path,default=RUNS/'252_scientist_profiles_unified_uepr');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 prompts={str(x['key']):x['prompt']for x in json.load((ROOT/'shuffled_prepend_profiles_question.json').open())}; rec={x['key']:x for x in map(json.loads,(ROOT/'tool_gate_correctness_profiles_llama31_8b/records.jsonl').open())}; ev={x['key']:x for x in map(json.loads,(RUNS/'221_scientist_minicheck_flan/items.jsonl').open())};cache=ROOT/'profile_perturbation_forward_output_full/items';keys=sorted(set(prompts)&set(rec)&set(ev)&{p.stem for p in cache.glob('*.npz')});keys=[k for k in keys if rec[k]['parse_valid']]
 base=[];hidden=[]
 for k in keys:
  rr=rec[k]; chosen=rr['parsed_answer']; names=[rr['right_answer'],rr['wrong_answer']]
  if norm(chosen)not in{norm(v)for v in names}:continue
  with np.load(cache/f'{k}.npz',allow_pickle=True)as z:
   md=json.loads(str(z['metadata']));ix={n:i for i,n in enumerate(md['condition_names'])};scores=z['candidate_scores'].astype(float);full=ix['full_context'];wo=ix['without_question_evidence'];h=z['hidden'][full].astype(np.float32).reshape(-1);pnames=md['profile_names'];ci=next(i for i,n in enumerate(pnames)if norm(n)==norm(chosen));oi=1-ci;mf=scores[full,ci]-scores[full,oi];mw=scores[wo,ci]-scores[wo,oi];pe=float(max(0,mf-mw))
  ee=ev[k]; support={norm(ee['chosen']):ee['chosen_whole_support'],norm(ee['alternative']):ee['alternative_whole_support']};e=float(support[norm(pnames[oi])]-support[norm(pnames[ci])])
  base.append({'key':k,'error':int(not rr['correct']),'chosen':chosen,'right':rr['right_answer'],'other':pnames[oi],'e_score':e,'p_score':pe,'p_specific_gain':pe if pe>0 else None});hidden.append(h)
 y=np.asarray([x['error']for x in base]);X=np.stack(hidden);rpred=[]
 for seed in(42,43,44):
  out=np.zeros(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y):
   sc=StandardScaler().fit(X[tr]);aa,bb=sc.transform(X[tr]),sc.transform(X[te]);pc=PCA(64,whiten=True,svd_solver='randomized',random_state=seed).fit(aa);aa,bb=pc.transform(aa),pc.transform(bb);clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(aa,y[tr]);out[te]=clf.predict_proba(bb)[:,1]
  rpred.append(out)
 for x,v in zip(base,np.mean(rpred,axis=0)):x['r_score']=float(v)
 sf=a.out_dir/'samples.jsonl';done={x['key']:x for x in map(json.loads,sf.open())}if a.resume and sf.exists()else{}
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval();torch.manual_seed(20260820);pending=[x for x in base if x['key']not in done]
 with sf.open('a'if done else'w')as f:
  for st in range(0,len(pending),a.batch):
   part=pending[st:st+a.batch];texts=[tok.apply_chat_template([{'role':'user','content':prompts[x['key']]}],tokenize=False,add_generation_prompt=True)for x in part];z=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=2048,add_special_tokens=False).to(model.device)
   with torch.inference_mode():g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=a.samples,max_new_tokens=32,pad_token_id=tok.pad_token_id)
   outs=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
   for i,x in enumerate(part):
    vals=[norm(v.strip().split('\n')[0])for v in outs[i*a.samples:(i+1)*a.samples]];k=a.samples//2;gold=norm(x['right']);mode=Counter(vals[k:]).most_common(1)[0][0];q={'key':x['key'],'u_score':entropy(vals[:k]),'samples':vals,'heldout_majority_correct':int(mode==gold)};done[x['key']]=q;f.write(json.dumps(q,ensure_ascii=False)+'\n');f.flush()
   print(f'U {len(done)}/{len(base)}',flush=True)
 for x in base:x.update(u_score=done[x['key']]['u_score'],heldout_majority_correct=done[x['key']]['heldout_majority_correct'])
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in base:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 report={'protocol':'Scientist-Profiles common-key baseline; U first3/heldout3; R 3x5 OOF full-context hidden; E remapped MiniCheck support gap; P removal of question-relevant profile evidence','n':len(base),'errors':int(y.sum()),'u_quantiles':np.quantile([x['u_score']for x in base],[.3,.7]).tolist(),'p_positive_errors':sum(x['error']and x['p_score']>0 for x in base)};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
