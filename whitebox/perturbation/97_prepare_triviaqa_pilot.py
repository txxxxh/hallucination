#!/usr/bin/env python3
"""Stream TriviaQA evidence and generate short answers for a clean pilot."""
from __future__ import annotations
import argparse,json,re,string
from pathlib import Path

def norm(s):
 s=s.lower(); s=''.join(c if c not in string.punctuation else ' ' for c in s); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())

def fetch(a):
 from datasets import load_dataset
 ds=load_dataset('mandarjoshi/trivia_qa','rc',split='validation',streaming=True); out=[]
 for x in ds:
  desc=[str(v).strip() for v in x['search_results']['description'] if str(v).strip()]
  if not desc: continue
  context='\n'.join(desc[:3])[:2400]; ans=x['answer']; aliases=list(dict.fromkeys([ans['value'],*ans['aliases']]))
  out.append({'key':x['question_id'],'question':x['question'],'context':context,'answer':ans['value'],'aliases':aliases})
  if len(out)>=a.n: break
 with a.items.open('w') as f:
  for x in out: f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print({'fetched':len(out),'out':str(a.items)})

def generate(a):
 import os
 import torch
 if os.environ.get('SPANATTR_DISABLE_NATIVE_BMM') == '1':
  from torch._native.registry import deregister_op_overrides
  deregister_op_overrides(disable_op_symbols='bmm')
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rows=[json.loads(x) for x in a.items.open() if x.strip()]; done={}
 if a.generations.exists() and a.resume: done={x['key']:x for x in map(json.loads,a.generations.open())}
 tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.eos_token; tok.padding_side='left'; model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
 prompts=[]; pending=[]
 for r in rows:
  if r['key'] in done: continue
  msg=[{'role':'user','content':f"Context:\n{r['context']}\n\nQuestion: {r['question']}\n\nAnswer with only the shortest direct answer phrase."}]
  prompts.append(tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)); pending.append(r)
 with a.generations.open('a') as f:
  for start in range(0,len(pending),a.batch):
   rr=pending[start:start+a.batch]; pp=prompts[start:start+a.batch]; z=tok(pp,return_tensors='pt',padding=True,truncation=True,max_length=768).to('cuda')
   with torch.inference_mode(): ids=model.generate(**z,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
   for row,seq,input_len in zip(rr,ids,[z.input_ids.shape[1]]*len(rr)):
    text=tok.decode(seq[input_len:],skip_special_tokens=True).strip().split('\n')[0].strip(); ng=norm(text); na=[norm(x) for x in row['aliases']]; correct=ng in na
    rec={**row,'generation':text,'normalized_generation':ng,'correct':bool(correct),'generation_words':len(text.split())}; f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush()
   print(f'[{min(start+a.batch,len(pending))}/{len(pending)}]',flush=True)
 allrows=[json.loads(x) for x in a.generations.open() if x.strip()]; print({'n':len(allrows),'correct':sum(x['correct'] for x in allrows),'incorrect':sum(not x['correct'] for x in allrows),'mean_words_correct':sum(x['generation_words'] for x in allrows if x['correct'])/max(1,sum(x['correct'] for x in allrows)),'mean_words_incorrect':sum(x['generation_words'] for x in allrows if not x['correct'])/max(1,sum(not x['correct'] for x in allrows))})

def main():
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['fetch','generate','all']); p.add_argument('--n',type=int,default=300); p.add_argument('--items',type=Path,default=Path('runs/97_triviaqa_items_n300.jsonl')); p.add_argument('--generations',type=Path,default=Path('runs/97_triviaqa_generations_n300.jsonl')); p.add_argument('--model',default='/tmp/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--batch',type=int,default=16); p.add_argument('--resume',action='store_true'); a=p.parse_args()
 if a.stage in ['fetch','all']: fetch(a)
 if a.stage in ['generate','all']: generate(a)
if __name__=='__main__': main()
