#!/usr/bin/env python3
import argparse,json,re,string
from pathlib import Path

def norm(s):
 s=s.lower(); s=''.join(c if c not in string.punctuation else ' ' for c in s); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def rows(p): return [json.loads(x) for x in p.open() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch',type=int,default=16); p.add_argument('--resume',action='store_true'); p.add_argument('--limit',type=int,default=0); a=p.parse_args()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 src=rows(a.source)[:a.limit or None]; done={x['key'] for x in rows(a.output)} if a.resume and a.output.exists() else set(); todo=[x for x in src if x['key'] not in done]
 a.output.parent.mkdir(parents=True,exist_ok=True); t=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True); t.pad_token=t.eos_token; t.padding_side='left'; m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval(); mode='a' if a.resume and a.output.exists() else 'w'
 with a.output.open(mode) as f:
  for start in range(0,len(todo),a.batch):
   b=todo[start:start+a.batch]; prompts=[f"Answer using the context. Output only the short answer.\n\nContext:\n{x['context']}\n\nQuestion: {x['question']}" for x in b]; texts=[t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True) for q in prompts]; e=t(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(m.device)
   with torch.inference_mode(): g=m.generate(**e,do_sample=False,max_new_tokens=48,pad_token_id=t.eos_token_id)
   for x,y in zip(b,t.batch_decode(g[:,e.input_ids.shape[1]:],skip_special_tokens=True)):
    aliases=x.get('aliases',[])+[x['answer']]; z={**x,'generation':y.strip(),'correct':norm(y) in {norm(v) for v in aliases},'model':a.model}; f.write(json.dumps(z,ensure_ascii=False)+'\n'); f.flush(); done.add(x['key'])
   print(f'[{len(done)}/{len(src)}] trivia',flush=True)
if __name__=='__main__': main()
