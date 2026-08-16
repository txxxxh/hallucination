#!/usr/bin/env python3
"""Collect current127 main-task features for ScientistQA rows outside known-1084."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from spanattr.core import Item,SpanAttributor,set_seed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs'
DATA=ROOT/'shuffled_prepend_names_question.json';RECORDS=ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl'
PROBES=RUNS/'77_closedbook_fact_probe_results.jsonl';CACHE=RUNS/'135_scientist_full_current127'

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 mod=importlib.import_module('125_collect_current_three_benchmarks');data={str(x['key']):x for x in json.load(DATA.open())};records={x['key']:x for x in map(json.loads,RECORDS.open())};probes={x['key']:x for x in map(json.loads,PROBES.open())}
 known={k for k,x in probes.items() if x['n_discriminative_facts']>=1 and x['binary_accuracy']>.5 and x['pairwise_owner_accuracy']>.5};rows=[records[k] for k in records if k not in known and records[k].get('parse_valid',True)];CACHE.mkdir(parents=True,exist_ok=True);set_seed(42)
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 for num,r in enumerate(rows[:a.limit or None],1):
  fp=CACHE/f"{r['key']}.npz"
  if fp.exists() and a.resume:continue
  raw=data[r['key']];pred=str(r['parsed_answer']);right=str(raw['rgt_ans']);wrong=str(raw['wrg_ans']);other=wrong if pred==right else right;item=Item.from_dict(dict(raw,pred=pred,gold=other));item.pred,item.gold=pred,other;prep=att.prepare(item);ss,cc=mod.spans(att,prep);p1,o1=mod.scan(att,prep,ss);u=(p1[0]-p1[1:])-(o1[0]-o1[1:]);top=int(np.argmax(np.abs(u)));ids=np.argsort(-np.abs(u))[:min(5,len(u))];ph,oh,h14=mod.selected_hidden(att,prep,ids)
  ca,cb=cc[top];deleted=re.sub(r'[ \t]+',' ',item.context[:ca]+item.context[cb:]);deleted=re.sub(r'\s+([,.;:!?])',r'\1',deleted).strip();raw2=dict(raw);raw2['prompt']=deleted;item2=Item.from_dict(dict(raw2,pred=pred,gold=other));item2.pred,item2.gold=pred,other;prep2=att.prepare(item2);ss2,_=mod.spans(att,prep2);p2,o2=mod.scan(att,prep2,ss2);u2=(p2[0]-p2[1:])-(o2[0]-o2[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
  np.savez_compressed(fp,key=np.asarray(r['key']),correct=np.asarray(int(r['correct'])),stage1_pred=np.r_[p1[0],p1[1:][ids]],stage1_other=np.r_[o1[0],o1[1:][ids]],stage2_pred=np.r_[p2[0],p2[1:][ids2]],stage2_other=np.r_[o2[0],o2[1:][ids2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=h14.astype(np.float16));print(f'[{num}/{len(rows)}] {r["key"]}',flush=True)
if __name__=='__main__':main()
