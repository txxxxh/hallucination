#!/usr/bin/env python3
"""Build a strict per-model v6 multidomain known-both manifest for collector 158."""
import argparse, json
from pathlib import Path

def load(path): return [json.loads(x) for x in path.open() if x.strip()]

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 questions={x['id']:x for d in ('athlete','musician','building') for x in load(a.data/d/'primary_questions.jsonl')}
 rows=[];excluded=0
 for result in load(a.results):
  if result['probe_state']!='knows_both':continue
  if result['name_outcome']=='unmatched':excluded+=1;continue
  q=questions[result['id']];correct=result['name_outcome']=='correct';pred=q['correct_answer'] if correct else q['wrong_answer'];other=q['wrong_answer'] if correct else q['correct_answer']
  rows.append({'key':q['id'],'group':q['domain'],'correct':int(correct),'context':q['prepend_names_prompt'],'question':'','pred':pred,'other':other,'prompt_mode':True,'model':a.model,'field':q['decisive_relation']['field'],'probe_state':'knows_both'})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 print(json.dumps({'output':str(a.output),'rows':len(rows),'correct':sum(x['correct'] for x in rows),'incorrect':sum(not x['correct'] for x in rows),'excluded_unmatched_known_both':excluded},indent=2))
if __name__=='__main__':main()
