#!/usr/bin/env python3
"""GSM8K CoT split-sample U pilot on a fixed balanced n=300 subset."""
from __future__ import annotations
import argparse,importlib,json
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';u=importlib.import_module('236_gsm8k_u_split_confirmation')
def prompt(q):return'Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+q
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=4);ap.add_argument('--samples',type=int,default=6);ap.add_argument('--out-dir',type=Path,default=RUNS/'237_gsm8k_cot_u_split_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);allrows=[json.loads(x)for x in a.manifest.open()];rows=[x for x in allrows if x['correct']][:150]+[x for x in allrows if not x['correct']][:150];import torch;from transformers import AutoModelForCausalLM,AutoTokenizer;tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval();torch.manual_seed(20260822);items=[]
 for st in range(0,len(rows),a.batch):
  part=rows[st:st+a.batch];ps=[tok.apply_chat_template([{'role':'user','content':prompt(x['question'])}],tokenize=False,add_generation_prompt=True)for x in part];z=tok(ps,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=a.samples,max_new_tokens=192,pad_token_id=tok.pad_token_id)
  outs=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
  for i,x in enumerate(part):
   vals=[u.canon(v)for v in outs[i*a.samples:(i+1)*a.samples]];k=a.samples//2;mode=Counter(vals[k:]).most_common(1)[0][0];gold=u.canon(x['gold_final']);items.append({'key':x['key'],'greedy_error':int(not x['correct']),'gold':gold,'u_score':u.entropy(vals[:k]),'validation_majority_correct':int(mode==gold),'validation_correct_rate':sum(v==gold for v in vals[k:])/len(vals[k:]),'samples':vals})
  print(f'{min(st+len(part),len(rows))}/{len(rows)}',flush=True)
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 scores=np.array([x['u_score']for x in items]);q=np.quantile(scores,[.3,.7]);lo=[x for x in items if x['u_score']<=q[0]];hi=[x for x in items if x['u_score']>=q[1]]
 def sm(z):
  er=[x for x in z if x['greedy_error']];co=[x for x in z if not x['greedy_error']];return{'n':len(z),'errors':len(er),'majority_repair':float(np.mean([x['validation_majority_correct']for x in er])),'correct_damage':float(np.mean([not x['validation_majority_correct']for x in co]))}
 y=np.array([x['greedy_error']for x in items]);order=np.argsort(scores);risk={str(c):float(y[order[:round(len(y)*c)]].mean())for c in(.9,.7,.5,.3)};report={'protocol':'original GSM8K CoT; fixed balanced n300; first3 entropy; held-out last3 majority','n':len(items),'baseline_accuracy':float(1-y.mean()),'validation_majority_accuracy':float(np.mean([x['validation_majority_correct']for x in items])),'low':sm(lo),'high':sm(hi),'risk_by_coverage':risk};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
