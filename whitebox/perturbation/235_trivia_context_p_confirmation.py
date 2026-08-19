#!/usr/bin/env python3
"""TriviaQA context-localized P: remove wrong-specific evidence sentence vs matched sentence."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';base=importlib.import_module('229_trivia_e_confirmation');lex=importlib.import_module('230_trivia_lexical_e_confirmation')
def chunks(text):return[x.strip()for x in re.split(r'\n+|\.\.\.|(?<=[.!?])\s+',text)if x.strip()]
def make_prompt(x,context):return f"Answer using the context. Output only the short answer.\n\nContext:\n{context}\n\nQuestion: {x['question']}"
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--out-dir',type=Path,default=RUNS/'235_trivia_context_p_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);src=[json.loads(x)for x in a.source.open()if not json.loads(x)['correct']];items=[];req=[]
 for x in src:
  cs=chunks(x['context']);effects=np.array([lex.support(s,x['generation'])-lex.support(s,x['answer'])for s in cs]);idx=int(np.argmax(effects))
  if effects[idx]<=0 or len(cs)<2:continue
  target=cs[idx];cand=[(abs(len(s)-len(target)),i,s)for i,s in enumerate(cs)if i!=idx];_,j,placebo=min(cand);tc='\n'.join(s for i,s in enumerate(cs)if i!=idx);pc='\n'.join(s for i,s in enumerate(cs)if i!=j);items.append({'key':x['key'],'p_score':float(effects[idx]),'target':target,'placebo':placebo});req.extend([{'prompt':make_prompt(x,x['context']),'right':x['answer'],'wrong':x['generation']},{'prompt':make_prompt(x,tc),'right':x['answer'],'wrong':x['generation']},{'prompt':make_prompt(x,pc),'right':x['answer'],'wrong':x['generation']}])
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';m=base.score(model,tok,req,a.batch).reshape(-1,3)
 for x,z in zip(items,m):x.update(base_margin=float(z[0]),target_margin=float(z[1]),placebo_margin=float(z[2]),target_gain=float(z[1]-z[0]),placebo_gain=float(z[2]-z[0]),specific_gain=float(z[1]-z[2]))
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 s=np.array([x['p_score']for x in items]);q=np.quantile(s,[.3,.7]);lo=[x['specific_gain']for x in items if x['p_score']<=q[0]];hi=[x['specific_gain']for x in items if x['p_score']>=q[1]];wrong=[x for x in items if x['base_margin']<0];report={'protocol':'lexically wrong-specific context sentence discovery; physical deletion vs closest-length other sentence','n':len(items),'mean_target_gain':float(np.mean([x['target_gain']for x in items])),'mean_placebo_gain':float(np.mean([x['placebo_gain']for x in items])),'mean_specific_gain':float(np.mean([x['specific_gain']for x in items])),'rho':float(spearmanr(s,[x['specific_gain']for x in items]).statistic),'low_n':len(lo),'low_gain':float(np.mean(lo)),'high_n':len(hi),'high_gain':float(np.mean(hi)),'high_minus_low':float(np.mean(hi)-np.mean(lo)),'base_wrong_n':len(wrong),'target_repair_rate':float(np.mean([x['target_margin']>0 for x in wrong])),'placebo_repair_rate':float(np.mean([x['placebo_margin']>0 for x in wrong]))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
