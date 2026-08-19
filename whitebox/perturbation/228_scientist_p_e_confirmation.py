#!/usr/bin/env python3
"""Independent P/E intervention confirmation on Scientist-known."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'

def clean_delete(text,start,end):
 out=text[:start]+text[end:];out=re.sub(r'[ \t]+',' ',out);return re.sub(r'\s+([,.;:!?])',r'\1',out).strip()

def delete_target(prompt,target):
 marker='\nQuestion:\n';head,q=prompt.split(marker,1);m=re.search(re.escape(target),q,flags=re.I)
 return None if m is None else head+marker+clean_delete(q,m.start(),m.end())

def delete_matched(prompt,target,key):
 marker='\nQuestion:\n';head,q=prompt.split(marker,1);words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",q));width=max(1,len(re.findall(r"\b\w+(?:['’\-]\w+)*\b",target)));tm=re.search(re.escape(target),q,flags=re.I);cand=[]
 for i in range(0,len(words)-width+1):
  a,b=words[i].start(),words[i+width-1].end()
  if tm and not(b<=tm.start() or a>=tm.end()):continue
  piece=q[a:b]
  if any(x in piece.casefold() for x in ('who is','person')):continue
  cand.append((a,b,piece))
 if not cand:return None,None
 rng=np.random.default_rng(sum(map(ord,key))+228);a,b,piece=cand[int(rng.integers(len(cand)))]
 return head+marker+clean_delete(q,a,b),piece

def score_margins(model,tok,requests,batch):
 import torch
 flat=[];meta=[]
 for i,r in enumerate(requests):
  chat=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True);prefix=tok.encode(chat,add_special_tokens=False)
  for side in ('right','wrong'):
   flat.append((prefix,tok.encode(' '+r[side],add_special_tokens=False)));meta.append((i,side))
 vals=[None]*len(flat)
 for start in range(0,len(flat),batch):
  part=flat[start:start+batch];width=max(len(p)+len(a) for p,a in part);ids=torch.full((len(part),width),tok.pad_token_id,dtype=torch.long,device=model.device);mask=torch.zeros_like(ids);starts=[]
  for j,(prefix,answer) in enumerate(part):
   seq=prefix+answer;left=width-len(seq);ids[j,left:]=torch.tensor(seq,device=model.device);mask[j,left:]=1;starts.append(left+len(prefix))
  with torch.inference_mode():lp=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float().log_softmax(-1)
  for j,(_,answer) in enumerate(part):
   pos=starts[j];target=torch.tensor(answer,device=model.device);vals[start+j]=float(lp[j,pos-1:pos+len(answer)-1].gather(1,target[:,None]).mean().cpu())
  del ids,mask,lp
 cells=[{}for _ in requests]
 for v,(i,side)in zip(vals,meta):cells[i][side]=v
 return np.asarray([x['right']-x['wrong']for x in cells])

def extremes(rows,axis,outcome):
 lo=[x[outcome]for x in rows if x[axis]<=.3];hi=[x[outcome]for x in rows if x[axis]>=.7]
 return {'low_n':len(lo),'low_mean':float(np.mean(lo))if lo else None,'high_n':len(hi),'high_mean':float(np.mean(hi))if hi else None,'high_minus_low':float(np.mean(hi)-np.mean(lo))if lo and hi else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=24);ap.add_argument('--out-dir',type=Path,default=RUNS/'228_scientist_p_e_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 jobs={x[0]:x for x in importlib.import_module('152_scientist_attention_pruned_current127').jobs()};profiles={str(x['key']):x for x in json.load((ROOT/'shuffled_prepend_profiles_question.json').open())};audit={x['key']:x for x in map(json.loads,(RUNS/'226_four_axis_taxonomy_audit'/'items.jsonl').open())};cache=RUNS/'120_physical_delete_rerank';paraphrase=importlib.import_module('219_scientist_semantic_neighborhood_uncertainty').paraphrase
 prows=[];preqs=[];erows=[];ereqs=[]
 for key,(_,group,correct,prompt,pred,other)in jobs.items():
  if key not in audit:continue
  right,wrong=(pred,other)if correct else(other,pred);erows.append({'key':key,'group':group,'error':int(not correct),'e_percentile':audit[key]['e_percentile']});ereqs.extend([{'prompt':prompt,'right':right,'wrong':wrong},{'prompt':profiles[key]['prompt'],'right':right,'wrong':wrong}])
  fp=cache/f'{key}.npz'
  if correct or not fp.exists():continue
  with np.load(fp,allow_pickle=True)as z:
   ps,os=z['stage1_pred_scores'],z['stage1_other_scores'];effects=(ps[0]-os[0])-(ps[1:]-os[1:]);idx=int(np.argmax(effects))
   if effects[idx]<=0:continue
   target=str(z['stage1_text'][idx])
  held=paraphrase(prompt);deleted=delete_target(held,target);placebo,placebo_text=delete_matched(held,target,key)
  if deleted is None or placebo is None:continue
  prows.append({'key':key,'group':group,'target':target,'placebo_text':placebo_text,'p_score':float(effects[idx]),'p_percentile':audit[key]['p_percentile']});preqs.extend([{'prompt':held,'right':right,'wrong':wrong},{'prompt':deleted,'right':right,'wrong':wrong},{'prompt':placebo,'right':right,'wrong':wrong}])
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token
 em=score_margins(model,tok,ereqs,a.batch).reshape(-1,2)
 for row,m in zip(erows,em):row.update(names_margin=float(m[0]),profiles_margin=float(m[1]),e_repair_gain=float(m[1]-m[0]))
 pm=score_margins(model,tok,preqs,a.batch).reshape(-1,3)
 for row,m in zip(prows,pm):row.update(heldout_margin=float(m[0]),target_margin=float(m[1]),placebo_margin=float(m[2]),target_gain=float(m[1]-m[0]),placebo_gain=float(m[2]-m[0]),p_specific_gain=float(m[1]-m[2]))
 for name,rows in [('e_items.jsonl',erows),('p_items.jsonl',prows)]:
  with(a.out_dir/name).open('w')as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 ee=[x for x in erows if x['error']];report={'protocol':{'E':'names-only to complete two-profile evidence; target-model likelihood','P':'original attribution; held-out paraphrase target deletion; same-length random deletion placebo'},'E':{'n':len(erows),'errors':len(ee),'error_extremes':extremes(ee,'e_percentile','e_repair_gain'),'all_extremes':extremes(erows,'e_percentile','e_repair_gain'),'rho':float(spearmanr([x['e_percentile']for x in erows],[x['e_repair_gain']for x in erows]).statistic)},'P':{'n_positive_locatable_errors':len(prows),'extremes':extremes(prows,'p_percentile','p_specific_gain'),'mean_target_gain':float(np.mean([x['target_gain']for x in prows])),'mean_placebo_gain':float(np.mean([x['placebo_gain']for x in prows])),'mean_specific_gain':float(np.mean([x['p_specific_gain']for x in prows])),'rho':float(spearmanr([x['p_percentile']for x in prows],[x['p_specific_gain']for x in prows]).statistic)}};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
