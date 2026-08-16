# -*- coding: utf-8 -*-
"""Full single-span enumeration and oracle top-k correctness detector."""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.dirname(HERE))
from spanattr.core import Item, SpanAttributor, set_seed


def top_features(u, s0, k):
    u=np.asarray(u,float); ids=np.argsort(-np.abs(u))[:min(k,len(u))]; z=u[ids]
    if len(z)<k: z=np.pad(z,(0,k-len(z)))
    scale=abs(s0)+1e-6; total=np.abs(u).sum()+1e-9
    return np.r_[z, np.abs(z), z/scale,
        z.max(initial=0), z.min(initial=0), np.abs(z).mean(),
        np.abs(z).sum()/total, np.mean(u>0), np.std(u)].astype(float).tolist()


def collect(a):
    import torch
    source=[json.loads(x) for x in open(a.source) if x.strip()]
    data={str(x['key']):x for x in json.load(open(a.data))}
    records={x['key']:x for x in map(json.loads,open(a.records))}
    done=set()
    if a.resume and Path(a.out).exists(): done={json.loads(x)['key'] for x in open(a.out) if x.strip()}
    elif Path(a.out).exists(): raise FileExistsError(f'{a.out} exists; use --resume')
    load_model=importlib.import_module('61_grad_span_proposal').load_model
    model,tok=load_model(a.model,a.dtype,a.device)
    att=SpanAttributor(model,tok,device=a.device,baseline='mean',length_norm=True,max_rows=a.batch)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,'a') as fh:
      for n,src in enumerate(source):
        key=src['key']
        if key in done: continue
        raw,rr=data[key],records[key]; pred=str(rr['parsed_answer'])
        right,wrong=str(raw['rgt_ans']),str(raw['wrg_ans']); other=wrong if pred==right else right
        item=Item.from_dict(dict(raw,pred=pred,gold=other)); item.pred,item.gold=pred,other
        prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1); prep.spans=spans
        s0=att.S0(prep); u,_=att.u_of_sets(prep,[[i] for i in range(len(spans))],S0=s0)
        row={'key':key,'group':src['group'],'correct':src['correct'],'S0':s0,'n_spans':len(spans),
          'u':u.tolist(),'span_text':[s.text for s in spans],
          'full_summary':[float(u.mean()),float(u.std()),float(u.min()),float(u.max()),
                          float(np.abs(u).mean()),float(np.abs(u).max()),float(np.quantile(np.abs(u),.9))],
          'top_features':{str(k):top_features(u,s0,k) for k in a.topk}}
        fh.write(json.dumps(row,ensure_ascii=False)+'\n'); fh.flush()
        print(f'[{n+1}/{len(source)}] {key} y={int(src["correct"])} spans={len(spans)} max={np.abs(u).max():.3f}',flush=True)


def train(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows=[json.loads(x) for x in open(a.out) if x.strip()]; source={x['key']:x for x in map(json.loads,open(a.source))}
    y=np.array([int(x['correct']) for x in rows]); groups=np.array([x['group'] for x in rows]); B=np.array([[x['S0'],abs(x['S0'])] for x in rows]); F=np.array([x['full_summary'] for x in rows]); R=np.array([source[x['key']]['response_features'] for x in rows])
    sets={'likelihood':B,'random16_response':R,'full_distribution':F}
    for k in a.topk: sets[f'oracle_top{k}']=np.array([x['top_features'][str(k)] for x in rows]); sets[f'likelihood_oracle_top{k}']=np.c_[B,sets[f'oracle_top{k}']]
    cv=StratifiedGroupKFold(a.folds,shuffle=True,random_state=a.seed); report={'n':len(rows),'correct':int(y.sum())}
    for name,X in sets.items():
      est=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=5000,class_weight='balanced',random_state=a.seed)); p=cross_val_predict(est,X,y,groups=groups,cv=cv,method='predict_proba')[:,1]
      report[name]={'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}
    Path(a.report).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','train','all']); p.add_argument('--source',default='runs/69_generation_flip_n128_q16.jsonl'); p.add_argument('--data',default='../shuffled_prepend_names_question.json'); p.add_argument('--records',default='../tool_gate_correctness_names_llama31_8b/records.jsonl'); p.add_argument('--out',default='runs/70_oracle_topk_n128.jsonl'); p.add_argument('--report',default='runs/70_oracle_topk_n128_report.json'); p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--dtype',default='float32'); p.add_argument('--device',default='cuda'); p.add_argument('--batch',type=int,default=16); p.add_argument('--topk',type=int,nargs='+',default=[1,3,5,10]); p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',action='store_true'); a=p.parse_args(); set_seed(a.seed)
    if a.stage in ('collect','all'): collect(a)
    if a.stage in ('train','all'): train(a)
if __name__=='__main__': main()
