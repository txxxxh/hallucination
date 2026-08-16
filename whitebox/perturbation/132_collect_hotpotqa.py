#!/usr/bin/env python3
"""Collect the frozen current127 features for a HotpotQA manifest."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
from spanattr.core import Item,Span,SpanAttributor,set_seed
RUNS=Path(__file__).resolve().parent/'runs'

def spans(att,prep):
 words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",prep.item.context,flags=re.UNICODE));enc=att.tok(prep.item.context,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping']
 if ids and isinstance(ids[0],list):ids=ids[0]
 if off and isinstance(off[0],list) and off[0] and isinstance(off[0][0],list):off=off[0]
 if list(ids)!=prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist():raise RuntimeError('offset mismatch')
 ss=[];cc=[]
 for wi in range(0,len(words),2):
  a=words[wi].start();b=words[min(wi+1,len(words)-1)].end();cov=[j for j,(x,y) in enumerate(off) if y>a and x<b]
  if cov:ss.append(Span(len(ss),prep.ctx_start+cov[0],prep.ctx_start+cov[-1]+1,prep.item.context[a:b]));cc.append((a,b))
 prep.spans=ss;return ss,cc

def scan(att,prep,ss):
 import torch
 z=torch.zeros(prep.prompt_ids.shape[0],device=att.device);A=torch.stack([z,*[att.alpha_from_spans(prep,[i]) for i in range(len(ss))]]);p,o=att.class_scores_batched(prep,A);return p.numpy(),o.numpy()

def selected_hidden(att,prep,ids):
 import torch
 z=torch.zeros(prep.prompt_ids.shape[0],device=att.device);A=torch.stack([z,*[att.alpha_from_spans(prep,[int(i)]) for i in ids]]);outs=[[],[]]
 for start in range(0,len(A),att.max_rows):
  a=A[start:start+att.max_rows];pe=att._embeds(prep,a)
  for ci,ans in enumerate((prep.pred_variant_ids[0],prep.gold_variant_ids[0])):
   ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(a),-1,-1);seq=torch.cat([pe,ae.to(pe.dtype)],1);mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
   with torch.inference_mode():out=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
   outs[ci].append(out.hidden_states[16][:,pe.shape[1]+len(ans)-1].float().cpu())
   if ci==0 and start==0:layer14=out.hidden_states[14][0,pe.shape[1]+len(ans)-1].float().cpu().numpy()
   del out,seq
  del pe
 return torch.cat(outs[0]).numpy(),torch.cat(outs[1]).numpy(),layer14

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=RUNS/'131_hotpotqa_balanced_n200.jsonl');p.add_argument('--out-dir',type=Path,default=RUNS/'132_hotpotqa_current127');p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=32);p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=0);a=p.parse_args();set_seed(42)
 rows=[json.loads(x) for x in a.manifest.open() if x.strip()];a.out_dir.mkdir(parents=True,exist_ok=True);model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 for num,r in enumerate(rows[:a.limit or None],1):
  fp=a.out_dir/f"{r['key']}.npz"
  if fp.exists() and a.resume:continue
  prep=att.prepare(Item(r['key'],r['context'],r['question'],r['other_answer'],r['generation']));ss,cc=spans(att,prep);p1,o1=scan(att,prep,ss);u=(p1[0]-p1[1:])-(o1[0]-o1[1:]);top=int(np.argmax(np.abs(u)));ids=np.argsort(-np.abs(u))[:min(5,len(u))];ph,oh,h14=selected_hidden(att,prep,ids)
  ca,cb=cc[top];deleted=re.sub(r'[ \t]+',' ',r['context'][:ca]+r['context'][cb:]);deleted=re.sub(r'\s+([,.;:!?])',r'\1',deleted).strip();prep2=att.prepare(Item(r['key']+'_d',deleted,r['question'],r['other_answer'],r['generation']));ss2,_=spans(att,prep2);p2,o2=scan(att,prep2,ss2);u2=(p2[0]-p2[1:])-(o2[0]-o2[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
  np.savez_compressed(fp,key=np.asarray(r['key']),group=np.asarray(r['key']),correct=np.asarray(int(r['correct'])),level=np.asarray(r['level']),qtype=np.asarray(r['type']),generation_words=np.asarray(r['generation_words']),other_words=np.asarray(r['other_words']),deleted_text=np.asarray(ss[top].text),stage1_pred=np.r_[p1[0],p1[1:][ids]],stage1_other=np.r_[o1[0],o1[1:][ids]],stage2_pred=np.r_[p2[0],p2[1:][ids2]],stage2_other=np.r_[o2[0],o2[1:][ids2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=h14.astype(np.float16));print(f'[{num}/{len(rows)}] {r["key"]}',flush=True)
if __name__=='__main__':main()
