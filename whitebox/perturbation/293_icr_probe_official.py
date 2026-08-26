#!/usr/bin/env python3
"""Official-code ICR Probe adapter for frozen Scientist and GSM8K responses."""
from __future__ import annotations
import argparse, importlib, json, random, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'; OUT=RUNS/'298_icr_probe_paper_strict'
OFFICIAL=HERE/'third_party/ICR_Probe_official'; COMMIT='40ec490e762cadbac6bcefdc24a8f0d5974e8448'
MODEL='/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77'
sys.path.insert(0,str(OFFICIAL))
from src.icr_score import ICRScore
from src.utils import ICRProbe

def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def rows(ds):
 m=importlib.import_module('100_collect_multilayer_trajectory')
 if ds=='scientist_full':
  keys={x['key'] for x in read(RUNS/'273_full_scientist_saplma_paper/predictions.jsonl')}
  z=[x for x in m._scientist_rows('all') if x['key'] in keys]
 elif ds=='scientist_known': z=m._scientist_rows('known')
 elif ds=='gsm8k':
  z=[dict(key=x['key'],group=x['group'],correct=int(x['correct']),question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl')]
 elif ds=='triviaqa':
  z=[dict(key=x['key'],group=x['key'],correct=int(x['correct']),question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'127_triviaqa_balanced_n1000.jsonl')]
 else:
  z=[dict(key=x['key'],group=x['group'],correct=int(x['correct']),question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'166_drop1000/drop_balanced_n1000.jsonl')]
 expected={'scientist_full':2894,'scientist_known':1084,'gsm8k':942,'triviaqa':1000,'drop':1000}[ds]
 if len(z)!=expected or len({x['key'] for x in z})!=expected: raise RuntimeError(f'{ds}: {len(z)}/{expected}')
 return z
def user_text(ds,r):
 if ds.startswith('scientist'): return r['raw']['prompt']
 if ds=='gsm8k': return ('Solve the following grade-school math problem. Show your reasoning step by step. '
         'End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+r['question'])
 if ds=="triviaqa": return ("Context:\n"+r["raw"]["context"]+"\n\nQuestion: "+r["question"]+"\n\nAnswer with only the shortest direct answer phrase.")
 return ("Read the passage and answer the question. Return only the shortest direct answer, with no explanation.\n\nPassage:\n"+r["raw"]["context"]+"\n\nQuestion: "+r["question"])
def find_subseq(xs,ys,start=0):
 for i in range(start,len(xs)-len(ys)+1):
  if xs[i:i+len(ys)]==ys:return i
 raise RuntimeError('chat-template boundary not found')
def boundaries(tok,ids):
 uh=tok.encode('<|start_header_id|>user<|end_header_id|>\n\n',add_special_tokens=False)
 us=find_subseq(ids,uh)+len(uh); eot=tok.convert_tokens_to_ids('<|eot_id|>'); ue=ids.index(eot,us)
 return {'user_prompt_start':us,'user_prompt_end':ue,'response_start':len(ids)}
def replay(model,ids,answer):
 import torch
 with torch.inference_mode():
  o=model(torch.tensor([ids],device=model.device),use_cache=True,output_hidden_states=True,output_attentions=True)
  hs=[o.hidden_states]; att=[o.attentions]; cache=o.past_key_values
  for token in answer[:-1]:
   o=model(torch.tensor([[token]],device=model.device),past_key_values=cache,use_cache=True,output_hidden_states=True,output_attentions=True)
   hs.append(o.hidden_states);att.append(o.attentions);cache=o.past_key_values
 return hs,att
def official_feature(model,tok,ds,r):
 import torch
 p=tok.apply_chat_template([{'role':'user','content':user_text(ds,r)}],tokenize=False,add_generation_prompt=True)
 ids=tok.encode(p,add_special_tokens=False); answer=tok.encode(' '+str(r['pred']),add_special_tokens=False)
 if len(answer)<2: answer=answer+[tok.eos_token_id]
 hs,att=replay(model,ids,answer); old=torch.get_default_device();torch.set_default_device(model.device)
 try:
  calc=ICRScore(hs,att,skew_threshold=0,entropy_threshold=1e5,core_positions=boundaries(tok,ids),icr_device=model.device)
  raw,top_p=calc.compute_icr(top_k=20,top_p=None,pooling='mean',attention_uniform=False,hidden_uniform=False,use_induction_head=True)
 finally: torch.set_default_device(old)
 feat=np.asarray([np.mean(x) for x in raw],np.float32)
 heads=np.asarray([sum(x) for x in calc.induction_head],np.int16)
 if feat.shape!=(model.config.num_hidden_layers,) or not np.isfinite(feat).all():raise RuntimeError(f'bad feature {feat.shape}')
 return feat,heads,float(top_p),len(ids),len(answer),boundaries(tok,ids)
def collect(a):
 import torch
 try:
  from torch._native.registry import deregister_op_overrides
  deregister_op_overrides(disable_op_symbols='bmm')
 except (AttributeError, ImportError):
  pass
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rs=rows(a.dataset)[:a.limit or None]; d=OUT/a.dataset/'features';d.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager',local_files_only=True).eval()
 for n,r in enumerate(rs,1):
  f=d/(r['key']+'.npz')
  if a.resume and f.exists():continue
  if a.dataset=='scientist_known' and (OUT/'scientist_full'/'features'/f.name).exists():continue
  x,h,tp,ni,no,b=official_feature(model,tok,a.dataset,r)
  np.savez_compressed(f,key=r['key'],group=r['group'],correct=r['correct'],icr=x,induction_heads=h,top_p_mean=tp,input_tokens=ni,answer_tokens=no,boundaries=json.dumps(b),official_commit=COMMIT)
  print(a.dataset,n,'/',len(rs),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',choices=['scientist_full','scientist_known','gsm8k','triviaqa','drop']);p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args();collect(a)
if __name__=='__main__':main()
