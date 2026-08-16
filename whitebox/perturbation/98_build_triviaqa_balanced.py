#!/usr/bin/env python3
"""Build a balanced TriviaQA generation set with short counterfactual answers."""
import argparse,json,random,string,re
from pathlib import Path
def norm(s):
 s=s.lower(); s=''.join(c if c not in string.punctuation else ' ' for c in s); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('runs/97_triviaqa_generations_n300.jsonl')); p.add_argument('--out',type=Path,default=Path('runs/98_triviaqa_balanced_n238.jsonl')); p.add_argument('--model',default='/tmp/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--batch',type=int,default=16); p.add_argument('--per-class',type=int,default=0); a=p.parse_args(); rows=[json.loads(x) for x in a.input.open() if x.strip()]; good=[x for x in rows if x['correct']]; bad=[x for x in rows if not x['correct']]; random.Random(42).shuffle(good); random.Random(44).shuffle(bad); n=a.per_class or min(len(good),len(bad));
 if len(good)<n or len(bad)<n: raise RuntimeError(f'need {n}/class, have correct={len(good)} incorrect={len(bad)}')
 chosen=good[:n]+bad[:n]; random.Random(43).shuffle(chosen)
 tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.eos_token; tok.padding_side='left'; model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval(); need=[x for x in chosen if x['correct']]; wrong={}
 for st in range(0,len(need),a.batch):
  rr=need[st:st+a.batch]; prompts=[]
  for x in rr:
   content=f"Context:\n{x['context']}\n\nQuestion: {x['question']}\nCorrect answer: {x['answer']}\n\nGive one plausible but factually incorrect answer. Output only the shortest answer phrase and do not repeat the correct answer."
   prompts.append(tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
  z=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=768).to('cuda')
  with torch.inference_mode(): ids=model.generate(**z,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
  for x,seq in zip(rr,ids): wrong[x['key']]=tok.decode(seq[z.input_ids.shape[1]:],skip_special_tokens=True).strip().split('\n')[0].strip()
  print(f'[{min(st+a.batch,len(need))}/{len(need)}]',flush=True)
 with a.out.open('w') as f:
  for x in chosen:
   x['other_answer']=wrong[x['key']] if x['correct'] else x['answer']; x['other_words']=len(x['other_answer'].split()); f.write(json.dumps(x,ensure_ascii=False)+'\n')
 out=[json.loads(x) for x in a.out.open()]; print({'n':len(out),'correct':sum(x['correct'] for x in out),'candidate_words_correct':sum(x['generation_words'] for x in out if x['correct'])/len(bad),'candidate_words_incorrect':sum(x['generation_words'] for x in out if not x['correct'])/len(bad),'counterfactual_matches_gold':sum(norm(x['other_answer']) in [norm(y) for y in x['aliases']] for x in out if x['correct'])})
if __name__=='__main__': main()
