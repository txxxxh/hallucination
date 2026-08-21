#!/usr/bin/env python3
"""Collect the compact two-stage physical-deletion features on three benchmarks."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from spanattr.core import Item, Span, SpanAttributor, set_seed
HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"

def jobs(ds,n,trivia_manifest):
 if ds=="trivia":
  rows=[json.loads(x) for x in trivia_manifest.open() if x.strip()]
  return [(r["key"],r["key"],int(r["correct"]),r["context"],r["question"],r["generation"],r["other_answer"]) for r in rows]
 if ds=="halueval":
  rows=[json.loads(x) for x in open("/home/tong56/other_bench/qa_data (2).json") if x.strip()][:n]; out=[]
  for i,r in enumerate(rows):
   g=f"hq{i:05d}"; out += [(g+"_right",g,1,r["knowledge"],r["question"],r["right_answer"],r["hallucinated_answer"]),(g+"_hall",g,0,r["knowledge"],r["question"],r["hallucinated_answer"],r["right_answer"])]
  return out
 rows=json.load(open("/home/tong56/whitebox/question_and_result.json"))[:n]; out=[]
 for i,r in enumerate(rows):
  right=r["answer"]-1; q=f"Option1: {r['options'][0]}\nOption2: {r['options'][1]}\nWhich option is correct?"; g=f"rl{i:05d}"
  for oi in (0,1): out.append((f"{g}_{oi+1}",g,int(oi==right),r["question"],q,r["options"][oi],r["options"][1-oi]))
 return out

def spans(att,prep):
 words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",prep.item.context,flags=re.UNICODE)); enc=att.tok(prep.item.context,add_special_tokens=False,return_offsets_mapping=True); ids=enc["input_ids"]; off=enc["offset_mapping"]
 if ids and isinstance(ids[0],list):ids=ids[0]
 if off and isinstance(off[0],list) and off[0] and isinstance(off[0][0],list):off=off[0]
 if list(ids)!=prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist():raise RuntimeError("offset mismatch")
 ss=[];cc=[]
 for wi in range(0,len(words),2):
  a=words[wi].start();b=words[min(wi+1,len(words)-1)].end(); cov=[j for j,(x,y) in enumerate(off) if y>a and x<b]
  if cov:ss.append(Span(len(ss),prep.ctx_start+cov[0],prep.ctx_start+cov[-1]+1,prep.item.context[a:b]));cc.append((a,b))
 prep.spans=ss;return ss,cc

def scan(att,prep,ss):
 import torch
 z=torch.zeros(prep.prompt_ids.shape[0],device=att.device); A=torch.stack([z,*[att.alpha_from_spans(prep,[i]) for i in range(len(ss))]]);p,o=att.class_scores_batched(prep,A);return p.numpy(),o.numpy()

def selected_hidden(att,prep,ids,layer14_pooling="last"):
 import torch
 z=torch.zeros(prep.prompt_ids.shape[0],device=att.device);A=torch.stack([z,*[att.alpha_from_spans(prep,[int(i)]) for i in ids]]); outs=[[],[]]
 for start in range(0,len(A),att.max_rows):
  a=A[start:start+att.max_rows];pe=att._embeds(prep,a)
  for ci,ans in enumerate((prep.pred_variant_ids[0],prep.gold_variant_ids[0])):
   ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(a),-1,-1);seq=torch.cat([pe,ae.to(pe.dtype)],1);mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
   with torch.inference_mode():out=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
   h16=out.hidden_states[16][:,pe.shape[1]+len(ans)-1].float().cpu();outs[ci].append(h16)
   if ci==0 and start==0:
    answer_hidden=out.hidden_states[14][0,pe.shape[1]:pe.shape[1]+len(ans)].float()
    if layer14_pooling=="mean": layer14=answer_hidden.mean(0).cpu().numpy()
    elif layer14_pooling=="last": layer14=answer_hidden[-1].cpu().numpy()
    else: raise ValueError(f"unknown layer14 pooling: {layer14_pooling}")
   del out,seq
  del pe
 return torch.cat(outs[0]).numpy(),torch.cat(outs[1]).numpy(),layer14

def main():
 p=argparse.ArgumentParser();p.add_argument("dataset",choices=["trivia","halueval","reallife"]);p.add_argument("--questions",type=int,default=0);p.add_argument("--trivia-manifest",type=Path,default=RUNS/"98_triviaqa_balanced_n238.jsonl");p.add_argument("--out-dir",type=Path);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--batch",type=int,default=32);p.add_argument("--resume",action="store_true");p.add_argument("--limit",type=int,default=0);a=p.parse_args();set_seed(42)
 n=a.questions or (128 if a.dataset=="halueval" else 200); rows=jobs(a.dataset,n,a.trivia_manifest);outdir=a.out_dir or RUNS/f"125_{a.dataset}_current127";outdir.mkdir(parents=True,exist_ok=True);model,tok=importlib.import_module("61_grad_span_proposal").load_model(a.model,"bfloat16","cuda");att=SpanAttributor(model,tok,device="cuda",baseline="mean",length_norm=True,max_rows=a.batch)
 for num,(key,group,label,ctx,q,pred,other) in enumerate(rows[:a.limit or None],1):
  fp=outdir/f"{key}.npz"
  if fp.exists() and a.resume:continue
  item=Item(key,ctx,q,other,pred);prep=att.prepare(item);ss,cc=spans(att,prep);p1,o1=scan(att,prep,ss);u=(p1[0]-p1[1:])-(o1[0]-o1[1:]);top=int(np.argmax(np.abs(u)));ids=np.argsort(-np.abs(u))[:min(5,len(u))];ph,oh,h14=selected_hidden(att,prep,ids)
  ca,cb=cc[top];deleted=re.sub(r"[ \t]+"," ",ctx[:ca]+ctx[cb:]);deleted=re.sub(r"\s+([,.;:!?])",r"\1",deleted).strip();prep2=att.prepare(Item(key+"_d",deleted,q,other,pred));ss2,_=spans(att,prep2);p2,o2=scan(att,prep2,ss2);u2=(p2[0]-p2[1:])-(o2[0]-o2[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
  np.savez_compressed(fp,key=np.asarray(key),group=np.asarray(group),correct=np.asarray(label),deleted_text=np.asarray(ss[top].text),stage1_pred=np.r_[p1[0],p1[1:][ids]],stage1_other=np.r_[o1[0],o1[1:][ids]],stage2_pred=np.r_[p2[0],p2[1:][ids2]],stage2_other=np.r_[o2[0],o2[1:][ids2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=h14.astype(np.float16));print(f"[{num}/{len(rows)}] {key}",flush=True)
if __name__=="__main__":main()
