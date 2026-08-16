#!/usr/bin/env python3
"""Two-stage search: disjoint spans, physical top-1 deletion, then rerank."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from spanattr.core import Item, Span, SpanAttributor, set_seed

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"

def disjoint_word_spans(att,prep):
 words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",prep.item.context,flags=re.UNICODE))
 enc=att.tok(prep.item.context,add_special_tokens=False,return_offsets_mapping=True)
 ids=enc["input_ids"]; off=enc["offset_mapping"]
 if ids and isinstance(ids[0],list): ids=ids[0]
 if off and isinstance(off[0],list) and off[0] and isinstance(off[0][0],list): off=off[0]
 if list(ids)!=prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist(): raise RuntimeError("token offset mismatch")
 out=[]; chars=[]
 for wi in range(0,len(words),2):
  a=words[wi].start(); b=words[min(wi+1,len(words)-1)].end()
  covered=[ti for ti,(x,y) in enumerate(off) if y>a and x<b]
  if covered:
   out.append(Span(len(out),prep.ctx_start+covered[0],prep.ctx_start+covered[-1]+1,prep.item.context[a:b])); chars.append((a,b))
 prep.spans=out
 return out,chars

def hidden(att,prep,alphas,ans,layer=16):
 import torch
 chunks=[]
 for start in range(0,len(alphas),att.max_rows):
  a=alphas[start:start+att.max_rows]; pe=att._embeds(prep,a); B=pe.shape[0]
  ae=att.emb_layer(ans).detach().unsqueeze(0).expand(B,-1,-1); seq=torch.cat([pe,ae.to(pe.dtype)],1)
  mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
  with torch.inference_mode(): out=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
  chunks.append(out.hidden_states[layer][:,pe.shape[1]+len(ans)-1].float().cpu()); del out,seq,pe
 return torch.cat(chunks).numpy()

def scores(att,prep,spans):
 import torch
 zero=torch.zeros(prep.prompt_ids.shape[0],device=att.device)
 A=torch.stack([zero,*[att.alpha_from_spans(prep,[i]) for i in range(len(spans))]])
 p,o=att.class_scores_batched(prep,A)
 return p.numpy(),o.numpy()

def selected_hidden(att,prep,ids):
 import torch
 zero=torch.zeros(prep.prompt_ids.shape[0],device=att.device)
 A=torch.stack([zero,*[att.alpha_from_spans(prep,[int(i)]) for i in ids]])
 return hidden(att,prep,A,prep.pred_variant_ids[0]),hidden(att,prep,A,prep.gold_variant_ids[0])

def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=RUNS/"88_known_gt05_n1084.jsonl")
 p.add_argument("--data",type=Path,default=HERE.parent/"shuffled_prepend_names_question.json")
 p.add_argument("--records",type=Path,default=HERE.parent/"tool_gate_correctness_names_llama31_8b"/"records.jsonl")
 p.add_argument("--out-dir",type=Path,default=RUNS/"120_physical_delete_rerank")
 p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); p.add_argument("--device",default="cuda"); p.add_argument("--dtype",default="bfloat16")
 p.add_argument("--batch",type=int,default=16); p.add_argument("--topk",type=int,default=5); p.add_argument("--limit",type=int,default=0); p.add_argument("--resume",action="store_true"); p.add_argument("--seed",type=int,default=42)
 a=p.parse_args(); set_seed(a.seed)
 source=[json.loads(x) for x in a.source.open() if x.strip()]; data={str(x["key"]):x for x in json.load(a.data.open())}; records={x["key"]:x for x in map(json.loads,a.records.open())}; a.out_dir.mkdir(parents=True,exist_ok=True)
 model,tok=importlib.import_module("61_grad_span_proposal").load_model(a.model,a.dtype,a.device); att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=a.batch)
 for n,src in enumerate(source[:a.limit or None],1):
  key=src["key"]; target=a.out_dir/f"{key}.npz"
  if target.exists() and a.resume: continue
  raw=data[key]; rec=records[key]; pred=str(rec["parsed_answer"]); right=str(raw["rgt_ans"]); wrong=str(raw["wrg_ans"]); other=wrong if pred==right else right
  item=Item.from_dict(dict(raw,pred=pred,gold=other)); item.pred,item.gold=pred,other; prep=att.prepare(item); spans,chars=disjoint_word_spans(att,prep)
  p1,o1=scores(att,prep,spans); pu1=p1[0]-p1[1:]; ou1=o1[0]-o1[1:]; mu1=pu1-ou1; top1=int(np.argmax(np.abs(mu1))); ids1=np.argsort(-np.abs(mu1))[:min(a.topk,len(mu1))]
  ca,cb=chars[top1]; deleted=(item.context[:ca]+item.context[cb:]); deleted=re.sub(r"[ \t]+"," ",deleted); deleted=re.sub(r"\s+([,.;:!?])",r"\1",deleted).strip()
  raw2=dict(raw); raw2["context"]=deleted; raw2["prompt"]=deleted; item2=Item.from_dict(dict(raw2,pred=pred,gold=other)); item2.pred,item2.gold=pred,other; prep2=att.prepare(item2); spans2,chars2=disjoint_word_spans(att,prep2)
  p2,o2=scores(att,prep2,spans2); pu2=p2[0]-p2[1:]; ou2=o2[0]-o2[1:]; mu2=pu2-ou2; ids2=np.argsort(-np.abs(mu2))[:min(a.topk,len(mu2))]
  np.savez_compressed(target,key=np.asarray(key),deleted_text=np.asarray(spans[top1].text),deleted_context=np.asarray(deleted),stage1_text=np.asarray([spans[i].text for i in ids1]),stage2_text=np.asarray([spans2[i].text for i in ids2]),stage1_pred_scores=np.r_[p1[0],p1[1:][ids1]],stage1_other_scores=np.r_[o1[0],o1[1:][ids1]],stage2_pred_scores=np.r_[p2[0],p2[1:][ids2]],stage2_other_scores=np.r_[o2[0],o2[1:][ids2]])
  print(f"[{n}/{len(source)}] {key} delete={spans[top1].text!r}",flush=True)

if __name__=="__main__": main()
