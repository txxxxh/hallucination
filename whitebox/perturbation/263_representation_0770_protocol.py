#!/usr/bin/env python3
"""Reproduce the leakage-free representation protocol that yielded Scientist 0.770."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'; MODEL='NousResearch/Meta-Llama-3.1-8B-Instruct'
LAYER=14; SEEDS=(42,43,44)
def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def rows(ds):
    if ds=='scientist': return importlib.import_module('100_collect_multilayer_trajectory')._scientist_rows('known')
    if ds=='trivia': return [dict(key=x['key'],group=x['key'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'127_triviaqa_balanced_n1000.jsonl')]
    if ds=='gsm8k': return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['question'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl')]
    return [dict(key=x['key'],group=x['group'],correct=int(x['correct']),context=x['context'],question=x['question'],pred=x['generation'],raw=x) for x in read(RUNS/'166_drop1000/drop_balanced_n1000.jsonl')]
def user_text(ds,r):
    if ds=='scientist': return r['raw']['prompt']
    if ds=='trivia': return f"Answer using the context. Output only the short answer.\n\nContext:\n{r['context']}\n\nQuestion: {r['question']}"
    if ds=='drop': return f"Read the passage and answer the question. Return only the shortest direct answer, with no explanation.\n\nPassage:\n{r['context']}\n\nQuestion: {r['question']}"
    return 'Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n'+r['question']
def collect(a):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=a.out/a.dataset/'features';out.mkdir(parents=True,exist_ok=True);rs=rows(a.dataset)
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval()
    for i,r in enumerate(rs):
        f=out/(r['key']+'.npz')
        if a.resume and f.exists(): continue
        p=tok.apply_chat_template([{'role':'user','content':user_text(a.dataset,r)}],tokenize=False,add_generation_prompt=True)
        pi=tok.encode(p,add_special_tokens=False); ai=tok.encode(' '+str(r['pred']),add_special_tokens=False); ids=torch.tensor([pi+ai],device=model.device)
        with torch.inference_mode(): z=model(ids,output_hidden_states=True,use_cache=False).hidden_states[LAYER][0,len(pi):]
        np.savez_compressed(f,key=r['key'],group=r['group'],correct=r['correct'],last=z[-1].float().cpu().numpy().astype(np.float16),mean=z.mean(0).float().cpu().numpy().astype(np.float16))
        if (i+1)%25==0 or i+1==len(rs): print(a.dataset,i+1,'/',len(rs),flush=True)
def evaluate(a):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    fs=sorted((a.out/a.dataset/'features').glob('*.npz')); last=[];mean=[];y=[];g=[]
    for f in fs:
        z=np.load(f);last.append(z['last'].astype(np.float32));mean.append(z['mean'].astype(np.float32));y.append(1-int(z['correct']));g.append(str(z['group']))
    last=np.stack(last);mean=np.stack(mean);y=np.asarray(y);g=np.asarray(g);per=[];preds=[]
    for seed in SEEDS:
        pred=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for tr,te in cv.split(last,y,g):
            A=[];B=[]
            for x in (last,mean):
                sc=StandardScaler().fit(x[tr]);u=sc.transform(x[tr]);v=sc.transform(x[te]);pc=PCA(8,whiten=True,svd_solver='randomized',random_state=seed).fit(u);A.append(pc.transform(u));B.append(pc.transform(v))
            clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(np.concatenate(A,1),y[tr]);pred[te]=clf.predict_proba(np.concatenate(B,1))[:,1]
        preds.append(pred);per.append({'auroc':float(roc_auc_score(y,pred)),'auprc':float(average_precision_score(y,pred))})
    report={'dataset':a.dataset,'protocol':'fixed layer14 last+mean; fold-local scaler and PCA8 per block; LR C=.03; grouped 3x5 OOF','n':len(y),'errors':int(y.sum()),'groups':len(set(g)),'per_seed':per,'mean':{k:float(np.mean([q[k] for q in per])) for k in ('auroc','auprc')},'ensemble':{'auroc':float(roc_auc_score(y,np.mean(preds,0))),'auprc':float(average_precision_score(y,np.mean(preds,0)))}}
    p=a.out/a.dataset/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate']);p.add_argument('dataset',choices=['scientist','trivia','gsm8k','drop']);p.add_argument('--resume',action='store_true');p.add_argument('--out',type=Path,default=RUNS/'263_representation_0770_protocol');a=p.parse_args();(collect if a.stage=='collect' else evaluate)(a)
if __name__=='__main__':main()
