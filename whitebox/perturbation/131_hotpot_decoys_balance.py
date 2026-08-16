#!/usr/bin/env python3
"""Generate label-independent HotpotQA decoys and create a balanced manifest."""
from __future__ import annotations
import argparse,json,random,re,string
from pathlib import Path

def norm(s):
 s=s.lower();s=''.join(c if c not in string.punctuation else ' ' for c in s);s=re.sub(r'\b(a|an|the)\b',' ',s);return ' '.join(s.split())

def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('runs/130_hotpotqa_generations_n1200.jsonl'));p.add_argument('--out',type=Path,default=Path('runs/131_hotpotqa_balanced_n200.jsonl'));p.add_argument('--per-class',type=int,default=100);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 rows=[json.loads(x) for x in a.input.open() if x.strip()];good=[x for x in rows if x['correct']];bad=[x for x in rows if not x['correct']];rng=random.Random(a.seed);rng.shuffle(good);rng.shuffle(bad)
 if min(len(good),len(bad))<a.per_class:raise RuntimeError(f'need {a.per_class}/class; have correct={len(good)} incorrect={len(bad)}')
 chosen=good[:a.per_class]+bad[:a.per_class];rng.shuffle(chosen)
 tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
 for st in range(0,len(chosen),a.batch):
  rr=chosen[st:st+a.batch];prompts=[]
  for x in rr:
   content=f"Question: {x['question']}\nReference answer: {x['answer']}\n\nGive one plausible but factually incorrect alternative answer. Output only the shortest answer phrase and do not repeat the reference answer."
   prompts.append(tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
  z=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
  with torch.inference_mode():ids=model.generate(**z,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
  for x,seq in zip(rr,ids):
   x['other_answer']=tok.decode(seq[z.input_ids.shape[1]:],skip_special_tokens=True).strip().split('\n')[0].strip();x['other_words']=len(x['other_answer'].split());x['decoy_matches_gold']=norm(x['other_answer']) in [norm(q) for q in x['aliases']]
  print(f'[{min(st+a.batch,len(chosen))}/{len(chosen)}]',flush=True)
 with a.out.open('w') as f:
  for x in chosen:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps({'n':len(chosen),'correct':sum(x['correct'] for x in chosen),'decoy_matches_gold':sum(x['decoy_matches_gold'] for x in chosen),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
