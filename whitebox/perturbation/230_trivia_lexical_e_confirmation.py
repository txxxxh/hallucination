#!/usr/bin/env python3
"""TriviaQA lexical E-axis pilot with question-only to full-context response."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs'
base=importlib.import_module('229_trivia_e_confirmation')
def support(context,answer):
 c=base.norm(context);a=base.norm(answer)
 if not a:return 0.0
 if a in c:return 1.0
 ws=a.split();return sum(bool(re.search(rf'(?<!\w){re.escape(w)}(?!\w)',c))for w in ws)/len(ws)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=4);ap.add_argument('--out-dir',type=Path,default=RUNS/'230_trivia_lexical_e_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.source.open()];items=[];requests=[]
 for x in rows:
  chosen=x['generation'];alternative=x['other_answer']if x['correct']else x['answer'];right=chosen if x['correct']else x['answer'];wrong=x['other_answer']if x['correct']else chosen;e=support(x['context'],alternative)-support(x['context'],chosen);items.append({'key':x['key'],'error':int(not x['correct']),'e_score':e,'chosen_support':support(x['context'],chosen),'alternative_support':support(x['context'],alternative)});requests.extend([{'prompt':base.prompt(x,False),'right':right,'wrong':wrong},{'prompt':base.prompt(x,True),'right':right,'wrong':wrong}])
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval();m=base.score(model,tok,requests,a.batch).reshape(-1,2)
 for x,z in zip(items,m):x.update(question_margin=float(z[0]),context_margin=float(z[1]),evidence_gain=float(z[1]-z[0]))
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 err=[x for x in items if x['error']];lo=[x['evidence_gain']for x in err if x['e_score']<=0];hi=[x['evidence_gain']for x in err if x['e_score']>0];qw=[x for x in err if x['question_margin']<0];report={'protocol':'lexical context support gap; question-only to full-context candidate likelihood','n':len(items),'errors':len(err),'e_positive_errors':len(hi),'e_nonpositive_errors':len(lo),'error_gain_positive_e':float(np.mean(hi))if hi else None,'error_gain_nonpositive_e':float(np.mean(lo))if lo else None,'high_minus_low':float(np.mean(hi)-np.mean(lo))if hi and lo else None,'error_rho':float(spearmanr([x['e_score']for x in err],[x['evidence_gain']for x in err]).statistic),'question_wrong_n':len(qw),'context_repair_rate':float(np.mean([x['context_margin']>0 for x in qw])),'correct_context_damage_rate':float(np.mean([x['context_margin']<0 for x in items if not x['error']]))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
