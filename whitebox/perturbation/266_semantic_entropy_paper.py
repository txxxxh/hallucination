#!/usr/bin/env python3
"""Semantic entropy (Farquhar et al., Nature 2024) on frozen benchmarks."""
from __future__ import annotations
import argparse,importlib,json,re,unicodedata
from pathlib import Path
import numpy as np
RUNS=Path(__file__).resolve().parent/'runs';MODEL='NousResearch/Meta-Llama-3.1-8B-Instruct';NLI='microsoft/deberta-large-mnli'
def read(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def rows(ds):
 if ds=='scientist':return importlib.import_module('100_collect_multilayer_trajectory')._scientist_rows('known')
 if ds=='trivia':return [dict(key=x['key'],group=x['key'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'127_triviaqa_balanced_n1000.jsonl')]
 if ds=='gsm8k':return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['question'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl')]
 return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'166_drop1000/drop_balanced_n1000.jsonl')]
def prompt(ds,r):
 if ds=='scientist':return r['raw']['prompt']
 if ds=='trivia':return f"Answer using the context. Output only the short answer.\n\nContext:\n{r['context']}\n\nQuestion: {r['question']}"
 if ds=='drop':return f"Read the passage and answer the question. Return only the shortest direct answer, with no explanation.\n\nPassage:\n{r['context']}\n\nQuestion: {r['question']}"
 return 'Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+r['question']
def norm(x):return ' '.join(re.sub(r'[^\w\s]',' ',unicodedata.normalize('NFKC',x).casefold()).split())
def sample(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rs=rows(a.dataset);out=a.out/a.dataset;out.mkdir(parents=True,exist_ok=True);path=out/'samples.jsonl';done={x['key'] for x in read(path)} if a.resume and path.exists() else set();mode='a' if a.resume and path.exists() else 'w';tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval()
 with path.open(mode) as fh:
  for st in range(0,len(rs),a.batch):
   part=[r for r in rs[st:st+a.batch] if r['key'] not in done]
   if not part:continue
   ps=[tok.apply_chat_template([{'role':'user','content':prompt(a.dataset,r)}],tokenize=False,add_generation_prompt=True) for r in part];z=tok(ps,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device);torch.manual_seed(a.seed+st)
   with torch.inference_mode():o=model.generate(**z,do_sample=True,temperature=1.,top_p=.9,top_k=50,num_return_sequences=10,max_new_tokens=192 if a.dataset=='gsm8k' else 32,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
   trans=model.compute_transition_scores(o.sequences,o.scores,normalize_logits=True);gen=o.sequences[:,z.input_ids.shape[1]:];mask=gen.ne(tok.pad_token_id)&gen.ne(tok.eos_token_id);lens=mask.sum(1).clamp_min(1);valid=mask[:,:trans.shape[1]];avg=torch.where(valid,trans,torch.zeros_like(trans)).sum(1)/lens;texts=tok.batch_decode(gen,skip_special_tokens=True)
   for i,r in enumerate(part):rec={'key':r['key'],'correct':r['correct'],'question':r['question'],'samples':texts[i*10:(i+1)*10],'mean_logprobs':avg[i*10:(i+1)*10].float().cpu().tolist()};fh.write(json.dumps(rec,ensure_ascii=False)+'\n');fh.flush()
   print(a.dataset,min(st+a.batch,len(rs)),'/',len(rs),flush=True)
def score(a):
 import torch
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 from sklearn.metrics import roc_auc_score,average_precision_score
 recs=read(a.out/a.dataset/'samples.jsonl');tok=AutoTokenizer.from_pretrained(NLI,use_fast=True,local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(NLI,device_map={'':0},local_files_only=True).eval();eid=model.config.label2id.get('ENTAILMENT',model.config.label2id.get('entailment',2))
 def batch_entails(q,s,reps):
  left=[];right=[]
  for rep in reps:left.extend([q+" "+s,q+" "+rep]);right.extend([q+" "+rep,q+" "+s])
  x=tok(left,right,return_tensors="pt",padding=True,truncation=True,max_length=512).to(model.device)
  with torch.inference_mode():v=model(**x).logits.argmax(-1).cpu().tolist()
  return [v[2*j]==eid and v[2*j+1]==eid for j in range(len(reps))]
 out=[]
 for n,r in enumerate(recs,1):
  clusters=[]
  for i,s in enumerate(r["samples"]):
   exact=next((j for j,c in enumerate(clusters) if norm(s)==norm(r["samples"][c[0]])),None)
   if exact is not None:clusters[exact].append(i);continue
   reps=[r["samples"][c[0]] for c in clusters];matches=batch_entails(r["question"],s,reps) if reps else []
   hit=next((j for j,v in enumerate(matches) if v),None)
   if hit is None:clusters.append([i])
   else:clusters[hit].append(i)
  lp=np.asarray(r['mean_logprobs']);mass=np.asarray([np.exp(lp[c]).sum() for c in clusters]);p=mass/mass.sum();se=float(-(p*np.log(p+1e-12)).sum());cnt=np.asarray([len(c) for c in clusters],float)/len(lp);dse=float(-(cnt*np.log(cnt+1e-12)).sum());out.append({'key':r['key'],'correct':r['correct'],'clusters':clusters,'semantic_entropy':se,'discrete_semantic_entropy':dse})
  if n%25==0 or n==len(recs):print(a.dataset,n,'/',len(recs),flush=True)
 p=a.out/a.dataset/'scores.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in out));y=np.asarray([1-x['correct'] for x in out]);report={'dataset':a.dataset,'method':'Farquhar et al. 2024 semantic entropy; M=10; temp=1 top_p=.9 top_k=50; bidirectional DeBERTa-large-MNLI clusters','n':len(y),'errors':int(y.sum()),'semantic_entropy':{'auroc':float(roc_auc_score(y,[x['semantic_entropy'] for x in out])),'auprc':float(average_precision_score(y,[x['semantic_entropy'] for x in out]))},'discrete_semantic_entropy':{'auroc':float(roc_auc_score(y,[x['discrete_semantic_entropy'] for x in out])),'auprc':float(average_precision_score(y,[x['discrete_semantic_entropy'] for x in out]))}};(a.out/a.dataset/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['sample','score']);p.add_argument('dataset',choices=['scientist','trivia','gsm8k','drop']);p.add_argument('--batch',type=int,default=4);p.add_argument('--seed',type=int,default=20260822);p.add_argument('--resume',action='store_true');p.add_argument('--out',type=Path,default=RUNS/'266_semantic_entropy_paper');a=p.parse_args();(sample if a.stage=='sample' else score)(a)
if __name__=='__main__':main()
