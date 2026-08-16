#!/usr/bin/env python3
"""Compatibility fetcher for the namespaced HotpotQA Hub repository."""
import argparse,importlib,json
from pathlib import Path
def main():
 from datasets import load_dataset
 mod=importlib.import_module('130_prepare_hotpotqa');p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1200);p.add_argument('--max-context-chars',type=int,default=3600);p.add_argument('--items',type=Path,default=Path('runs/130_hotpotqa_items_n1200.jsonl'));a=p.parse_args();ds=load_dataset('hotpotqa/hotpot_qa','distractor',split='validation',streaming=True);a.items.parent.mkdir(parents=True,exist_ok=True);count=0
 with a.items.open('w') as out:
  for row in ds:
   context,used,n_support=mod.compact_context(row,a.max_context_chars);rec={'key':str(row['id']),'question':row['question'],'context':context,'answer':row['answer'],'aliases':mod.answer_aliases(row['answer']),'level':row.get('level','unknown'),'type':row.get('type','unknown'),'supporting_titles':list(dict.fromkeys(row['supporting_facts']['title'])),'context_titles':used,'n_supporting_titles':n_support};out.write(json.dumps(rec,ensure_ascii=False)+'\n');count+=1
   if count>=a.n:break
 print(json.dumps({'fetched':count,'out':str(a.items)},indent=2))
if __name__=='__main__':main()
