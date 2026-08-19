#!/usr/bin/env python3
"""Grounded factuality checkers on Scientist-known.

Evaluates the official MiniCheck claim-vs-document model at two granularities:
the full conjunctive description and condition/sentence decomposition.  Both
the generated candidate and its alternative are scored, enabling a contrastive
support gap without using the gold answer at detection time.
"""
from __future__ import annotations
import argparse, importlib, json, re, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'

def split_profiles(prompt):
    evidence, question = prompt.split('Choose exactly one profile from the two, and output the name of the person as the answer to the following question:\n',1)
    evidence=evidence.removeprefix('Given two profiles of two persons:\n').strip()
    blocks=re.split(r'(?=^name: )',evidence,flags=re.M); blocks=[x.strip() for x in blocks if x.strip()]
    docs={re.match(r'name: ([^\n]+)',x).group(1):x for x in blocks}
    return docs,question.strip()

def sentences(q):
    q=re.sub(r'\s*Who is this person\?\s*$','',q).strip()
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+',q) if x.strip()]

def bind(sentence,name):
    """Mechanical anaphora binding; no factual/gold information is introduced."""
    s=sentence
    s=re.sub(r'^This ([^,.]+?)( has| is| was| made| received| studied| never| did)',rf'{name}, a \1,\2',s,flags=re.I)
    s=re.sub(r'^Despite their',f"Despite {name}'s",s,flags=re.I)
    s=re.sub(r'^Throughout their',f"Throughout {name}'s",s,flags=re.I)
    s=re.sub(r'^Among their',f"Among {name}'s",s,flags=re.I)
    s=re.sub(r'^In addition to their',f"In addition to {name}'s",s,flags=re.I)
    s=re.sub(r'\b[Tt]heir\b',f"{name}'s",s);s=re.sub(r'\b[Tt]hey\b',name,s)
    s=re.sub(r'\b[Tt]hem\b',name,s)
    s=re.sub(r'\b[Tt]his (?:scientist|person|individual|scholar|researcher|biochemist|geneticist)\b',name,s)
    return s

def metric(y,s,threshold=None):
    out={'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s))}
    if threshold is not None:out['balanced_accuracy_at_fixed_threshold']=float(balanced_accuracy_score(y,np.asarray(s)>=threshold))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='flan-t5-large',choices=['flan-t5-large','deberta-v3-large','roberta-large']);p.add_argument('--batch',type=int,default=32);p.add_argument('--cache-dir',type=Path,default=Path('/tmp/minicheck_ckpts'));p.add_argument('--out-dir',type=Path,default=RUNS/'221_scientist_minicheck_flan');a=p.parse_args()
    sys.path.insert(0,'/tmp/MiniCheck')
    from minicheck.inference import Inferencer
    profile={str(x['key']):x for x in json.load((ROOT/'shuffled_prepend_profiles_question.json').open())}
    rows=importlib.import_module('152_scientist_attention_pruned_current127').jobs(); rows=[x for x in rows if x[4] not in ('','None','null') and x[5] not in ('','None','null')]; a.out_dir.mkdir(parents=True,exist_ok=True)
    requests=[]; meta=[]
    for key,group,correct,prompt,pred,other in rows:
        docs,q=split_profiles(profile[key]['prompt']); ss=sentences(q)
        for owner,name in [('chosen',pred),('alternative',other)]:
            doc=docs[name]+'\nThe profile above is complete for the attributes mentioned in the question; an unlisted attribute is absent.'
            claims=[bind(x,name) for x in ss]
            for gran,claim in [('whole',' '.join(claims)),*[('atomic',x) for x in claims]]:
                requests.append((doc,claim));meta.append((key,owner,gran))
    checker=Inferencer(a.model,None,a.batch,str(a.cache_dir)); checker.chunk_size=500 if a.model=='flan-t5-large' else 400
    # All profile documents fit one chunk, so use the batched core directly.
    # inference() aggregates its batch into one result; call explicit minibatches.
    probs=[]
    for st in range(0,len(requests),a.batch):
        part=requests[st:st+a.batch]; z=checker.inference([x[0] for x in part],[x[1] for x in part])
        probs.extend(z['support_prob_per_chunk'].tolist())
    cells={}
    for m,v in zip(meta,probs):cells.setdefault(m,[]).append(float(v))
    items=[]
    for key,group,correct,pred,other in [(x[0],x[1],x[2],x[4],x[5]) for x in rows]:
        def values(owner):
            whole=cells[key,owner,'whole'][0]; atom=np.asarray(cells[key,owner,'atomic']);return whole,float(atom.min()),float(atom.mean()),atom.tolist()
        cw,cmin,cmean,ca=values('chosen');aw,amin,amean,aa=values('alternative')
        items.append({'key':key,'group':group,'correct':int(correct),'chosen':pred,'alternative':other,'chosen_whole_support':cw,'alternative_whole_support':aw,'chosen_atomic_min':cmin,'alternative_atomic_min':amin,'chosen_atomic_mean':cmean,'alternative_atomic_mean':amean,'chosen_atomic':ca,'alternative_atomic':aa})
    with (a.out_dir/'items.jsonl').open('w')as f:
        for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    y=1-np.array([x['correct']for x in items]);signals={
      'one_minus_chosen_whole':1-np.array([x['chosen_whole_support']for x in items]),
      'whole_contrastive_gap':np.array([x['alternative_whole_support']-x['chosen_whole_support']for x in items]),
      'one_minus_chosen_atomic_min':1-np.array([x['chosen_atomic_min']for x in items]),
      'atomic_min_contrastive_gap':np.array([x['alternative_atomic_min']-x['chosen_atomic_min']for x in items]),
      'atomic_mean_contrastive_gap':np.array([x['alternative_atomic_mean']-x['chosen_atomic_mean']for x in items])}
    report={'protocol':'Scientist-known; selected-profile evidence; closed-world completeness statement; no gold used by detector; full vs atomic claim and contrastive candidate scoring','model':a.model,'n':len(items),'errors':int(y.sum()),'metrics':{k:metric(y,v,0 if 'gap' in k else .5)for k,v in signals.items()}}
    (a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
