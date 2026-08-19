#!/usr/bin/env python3
"""Closed-book TriviaQA answers followed by context-completion E intervention."""
from __future__ import annotations
import argparse,importlib,json,re,string
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';base=importlib.import_module('229_trivia_e_confirmation');lex=importlib.import_module('230_trivia_lexical_e_confirmation')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--gen-batch',type=int,default=16);ap.add_argument('--score-batch',type=int,default=4);ap.add_argument('--out-dir',type=Path,default=RUNS/'231_trivia_closedbook_e_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.source.open()]
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';import torch
 outputs=[]
 for st in range(0,len(rows),a.gen_batch):
  part=rows[st:st+a.gen_batch];ps=[tok.apply_chat_template([{'role':'user','content':base.prompt(x,False)}],tokenize=False,add_generation_prompt=True)for x in part];z=tok(ps,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():g=model.generate(**z,do_sample=False,max_new_tokens=48,pad_token_id=tok.pad_token_id)
  outputs.extend(tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True));print(f'{min(st+len(part),len(rows))}/{len(rows)}',flush=True)
 items=[];req=[]
 for x,out in zip(rows,outputs):
  chosen=out.strip();aliases=x.get('aliases',[])+[x['answer']];correct=base.norm(chosen)in{base.norm(v)for v in aliases};right=chosen if correct else x['answer'];wrong=x['other_answer']if correct else chosen;alternative=x['other_answer']if correct else x['answer'];e=lex.support(x['context'],alternative)-lex.support(x['context'],chosen);items.append({'key':x['key'],'closedbook_generation':chosen,'error':int(not correct),'e_score':e,'chosen_support':lex.support(x['context'],chosen),'alternative_support':lex.support(x['context'],alternative)});req.extend([{'prompt':base.prompt(x,False),'right':right,'wrong':wrong},{'prompt':base.prompt(x,True),'right':right,'wrong':wrong}])
 m=base.score(model,tok,req,a.score_batch).reshape(-1,2)
 for x,z in zip(items,m):x.update(question_margin=float(z[0]),context_margin=float(z[1]),evidence_gain=float(z[1]-z[0]))
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 err=[x for x in items if x['error']];pos=[x['evidence_gain']for x in err if x['e_score']>0];non=[x['evidence_gain']for x in err if x['e_score']<=0];qw=[x for x in err if x['question_margin']<0];report={'protocol':'fresh closed-book generation; lexical external evidence gap; full-context completion','n':len(items),'errors':len(err),'error_rate':len(err)/len(items),'e_positive_errors':len(pos),'e_nonpositive_errors':len(non),'gain_positive_e':float(np.mean(pos))if pos else None,'gain_nonpositive_e':float(np.mean(non))if non else None,'high_minus_low':float(np.mean(pos)-np.mean(non))if pos and non else None,'error_rho':float(spearmanr([x['e_score']for x in err],[x['evidence_gain']for x in err]).statistic),'question_wrong_n':len(qw),'context_repair_rate':float(np.mean([x['context_margin']>0 for x in qw])),'correct_context_damage_rate':float(np.mean([x['context_margin']<0 for x in items if not x['error']]))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
