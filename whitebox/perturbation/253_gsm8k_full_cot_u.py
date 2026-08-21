#!/usr/bin/env python3
"""Complete same-baseline CoT uncertainty scores for GSM8K balanced-942."""
from __future__ import annotations
import argparse,importlib,json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';base=importlib.import_module('236_gsm8k_u_split_confirmation')
def prompt(q):return'Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+q
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=4);ap.add_argument('--samples',type=int,default=6);ap.add_argument('--resume',action='store_true');ap.add_argument('--out-dir',type=Path,default=RUNS/'253_gsm8k_full_cot_u');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.manifest.open()];sf=a.out_dir/'items.jsonl';done={}
 # Reuse the exact same-protocol balanced-300 pilot.
 pilot=RUNS/'237_gsm8k_cot_u_split_confirmation/items.jsonl'
 if pilot.exists():done.update({x['key']:x for x in map(json.loads,pilot.open())})
 if a.resume and sf.exists():done.update({x['key']:x for x in map(json.loads,sf.open())})
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval();torch.manual_seed(20260823);pending=[x for x in rows if x['key']not in done]
 with sf.open('a'if a.resume and sf.exists()else'w')as f:
  for st in range(0,len(pending),a.batch):
   part=pending[st:st+a.batch];ps=[tok.apply_chat_template([{'role':'user','content':prompt(x['question'])}],tokenize=False,add_generation_prompt=True)for x in part];z=tok(ps,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
   with torch.inference_mode():g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=a.samples,max_new_tokens=192,pad_token_id=tok.pad_token_id)
   outs=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
   for i,x in enumerate(part):
    vals=[base.canon(v)for v in outs[i*a.samples:(i+1)*a.samples]];k=a.samples//2;mode=Counter(vals[k:]).most_common(1)[0][0];gold=base.canon(x['gold_final']);q={'key':x['key'],'greedy_error':int(not x['correct']),'gold':gold,'u_score':base.entropy(vals[:k]),'validation_majority_correct':int(mode==gold),'validation_correct_rate':sum(v==gold for v in vals[k:])/len(vals[k:]),'samples':vals};done[x['key']]=q;f.write(json.dumps(q)+'\n');f.flush()
   print(f'U {len(done)}/{len(rows)}',flush=True)
 # Canonical merged file in manifest order.
 with(a.out_dir/'merged_items.jsonl').open('w')as f:
  for x in rows:f.write(json.dumps(done[x['key']])+'\n')
 report={'protocol':'GSM8K original balanced-942 CoT baseline; first3 entropy/heldout3 majority; reuses same-protocol 237 balanced-300','n':len(rows),'errors':sum(not x['correct']for x in rows),'new_samples':len(pending)};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
