#!/usr/bin/env python3
"""Build strict per-model manifests consumed by 158; no cross-model labels allowed."""
import argparse, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(path): return [json.loads(x) for x in path.open() if x.strip()]
def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(path, len(rows), sum(int(r['correct']) for r in rows))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--model-dir',type=Path,required=True); a=p.parse_args()
    out=a.model_dir/'manifests'; raw={str(x['key']):x for x in json.load(open(ROOT/'shuffled_prepend_names_question.json'))}
    answers={str(x['key']):x for x in load(a.model_dir/'scientist_answers/records.jsonl')}
    probes={str(x['key']):x for x in load(a.model_dir/'scientist_probes.jsonl')}
    sci=[]
    for k,r in raw.items():
        ans=answers.get(k); pr=probes.get(k)
        if not ans or not pr or not ans.get('parse_valid'): continue
        if not (pr['n_discriminative_facts']>=1 and pr['binary_accuracy']>.5 and pr['pairwise_owner_accuracy']>.5): continue
        pred=str(ans['parsed_answer']); right=str(r['rgt_ans']); wrong=str(r['wrg_ans'])
        sci.append(dict(key=k,group=pr['right_qid'],correct=int(ans['correct']),context=r.get('prompt',r.get('context','')),question='',pred=pred,other=wrong if pred==right else right,prompt_mode=True,model=a.model))
    dump(out/'scientist.jsonl',sci)
    md=[]
    questions={x['id']:x for d in ('athlete','musician','building') for x in load(ROOT/'athlete_qa/multidomain_v5'/d/'primary_questions.jsonl')}
    for x in load(a.model_dir/'multidomain/results.jsonl'):
        if x['probe_state']!='knows_both' or x['name_outcome']=='unmatched': continue
        q=questions[x['id']]; correct=x['name_outcome']=='correct'; pred=q['correct_answer'] if correct else q['wrong_answer']
        md.append(dict(key=x['id'],group=q['domain'],correct=int(correct),context=q['prepend_names_prompt'],question='',pred=pred,other=q['wrong_answer'] if correct else q['correct_answer'],prompt_mode=True,model=a.model))
    dump(out/'multidomain.jsonl',md)
    for dataset in ('trivia','gsm8k'):
        rows=[]
        for x in load(a.model_dir/f'{dataset}_answers.jsonl' if dataset=='trivia' else a.model_dir/'gsm8k/generations.jsonl'):
            if dataset=='trivia':
                other=x['other_answer'] if x['correct'] else x['answer']; context=x['context']; question=x['question']
            else:
                other=x['reference_solution']; context=x['question']; question='Provide the complete solution to this math problem.'
            rows.append(dict(key=x['key'],group=x.get('group',x['key']),correct=int(x['correct']),context=context,question=question,pred=x['generation'],other=other,prompt_mode=False,model=a.model))
        dump(out/f'{dataset}.jsonl',rows)
if __name__=='__main__': main()
