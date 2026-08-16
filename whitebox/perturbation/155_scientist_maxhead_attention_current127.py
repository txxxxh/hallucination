#!/usr/bin/env python3
"""High-fidelity max-head attention variant of the pruned current127 run."""
import argparse,importlib
from pathlib import Path
import numpy as np
m=importlib.import_module('152_scientist_attention_pruned_current127')
m.CACHE=m.RUNS/'155_scientist_maxhead_attention_current127';m.OUT=m.RUNS/'155_scientist_maxhead_attention_current127_report.json'
def maxhead(att,prep):
 import torch
 P=len(prep.prompt_ids);maps=[]
 for ans in(prep.pred_variant_ids[0],prep.gold_variant_ids[0]):
  ids=torch.cat([prep.prompt_ids,ans]).unsqueeze(0)
  with torch.inference_mode():out=att.model(input_ids=ids,output_attentions=True,use_cache=False)
  # average answer-query positions and layers, retain heads; max head per token
  x=torch.stack([A[0,:,P-1:P+len(ans)-1,:P].float().mean(1)for A in out.attentions]).mean(0).cpu().numpy();maps.append(x);del out
 return maps[0].max(0)
m.contrastive_token_attention=maxhead
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate','all']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=64);p.add_argument('--blocks',type=int,default=12);p.add_argument('--keep',type=int,default=7);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.stage in('collect','all'):m.collect(a)
 if a.stage in('evaluate','all'):m.evaluate()
if __name__=='__main__':main()
