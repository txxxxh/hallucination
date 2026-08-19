#!/usr/bin/env python3
"""TriviaQA E-axis pilot: verifier gap predicts benefit from supplied context."""
from __future__ import annotations
import argparse,gc,json,re,string,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs'
def norm(s):
 s=str(s).lower();s=''.join(c if c not in string.punctuation else ' 'for c in s);s=re.sub(r'\b(a|an|the)\b',' ',s);return' '.join(s.split())
def prompt(x,context):
 prefix=f"Context:\n{x['context']}\n\n"if context else''
 return f"Answer using the provided information. Output only the short answer.\n\n{prefix}Question: {x['question']}"
def claim(q,a):return f"The answer to the question '{q}' is '{a}'."
def score(model,tok,requests,batch):
 import torch
 flat=[];meta=[]
 for i,r in enumerate(requests):
  text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True);p=tok.encode(text,add_special_tokens=False)
  for side in('right','wrong'):flat.append((p,tok.encode(' '+r[side],add_special_tokens=False)));meta.append((i,side))
 vals=[None]*len(flat)
 for st in range(0,len(flat),batch):
  part=flat[st:st+batch];width=max(len(p)+len(a)for p,a in part);ids=torch.full((len(part),width),tok.pad_token_id,dtype=torch.long,device=model.device);mask=torch.zeros_like(ids);starts=[]
  for j,(p,a)in enumerate(part):
   seq=p+a;left=width-len(seq);ids[j,left:]=torch.tensor(seq,device=model.device);mask[j,left:]=1;starts.append(left+len(p))
  with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
  for j,(_,a)in enumerate(part):
   pos=starts[j];target=torch.tensor(a,device=model.device);lp=logits[j,pos-1:pos+len(a)-1].float().log_softmax(-1);vals[st+j]=float(lp.gather(1,target[:,None]).mean().cpu())
  del ids,mask,logits
 cells=[{}for _ in requests]
 for v,(i,s)in zip(vals,meta):cells[i][s]=v
 return np.asarray([x['right']-x['wrong']for x in cells])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--out-dir',type=Path,default=RUNS/'229_trivia_e_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.source.open()]
 sys.path.insert(0,'/tmp/MiniCheck');from minicheck.inference import Inferencer
 docs=[];claims=[]
 for x in rows:
  chosen=x['generation'];alternative=x['other_answer']if x['correct']else x['answer'];docs.extend([x['context'],x['context']]);claims.extend([claim(x['question'],chosen),claim(x['question'],alternative)])
 checker=Inferencer('flan-t5-large',None,32,'/tmp/minicheck_ckpts');checker.chunk_size=500;probs=[]
 for st in range(0,len(docs),32):
  z=checker.inference(docs[st:st+32],claims[st:st+32]);probs.extend(z['support_prob_per_chunk'].tolist())
 del checker;gc.collect();import torch;torch.cuda.empty_cache()
 requests=[];items=[]
 for i,x in enumerate(rows):
  right=x['generation']if x['correct']else x['answer'];wrong=x['other_answer']if x['correct']else x['generation'];e=float(probs[2*i+1]-probs[2*i]);items.append({'key':x['key'],'error':int(not x['correct']),'e_score':e,'chosen_support':float(probs[2*i]),'alternative_support':float(probs[2*i+1])});requests.extend([{'prompt':prompt(x,False),'right':right,'wrong':wrong},{'prompt':prompt(x,True),'right':right,'wrong':wrong}])
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval();m=score(model,tok,requests,a.batch).reshape(-1,2)
 for x,z in zip(items,m):x.update(question_margin=float(z[0]),context_margin=float(z[1]),evidence_gain=float(z[1]-z[0]))
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 err=[x for x in items if x['error']];q=np.quantile([x['e_score']for x in items],[.3,.7]);lo=[x['evidence_gain']for x in err if x['e_score']<=q[0]];hi=[x['evidence_gain']for x in err if x['e_score']>=q[1]];report={'n':len(items),'errors':len(err),'e_error_mean':float(np.mean([x['e_score']for x in err])),'e_correct_mean':float(np.mean([x['e_score']for x in items if not x['error']])),'error_evidence_gain_mean':float(np.mean([x['evidence_gain']for x in err])),'error_rho':float(spearmanr([x['e_score']for x in err],[x['evidence_gain']for x in err]).statistic),'error_extremes':{'low_n':len(lo),'low_mean':float(np.mean(lo)),'high_n':len(hi),'high_mean':float(np.mean(hi)),'high_minus_low':float(np.mean(hi)-np.mean(lo))},'question_wrong_n':sum(x['question_margin']<0 for x in err),'context_repair_rate':float(np.mean([x['context_margin']>0 for x in err if x['question_margin']<0])),'correct_context_damage_rate':float(np.mean([x['context_margin']<0 for x in items if not x['error']]))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
