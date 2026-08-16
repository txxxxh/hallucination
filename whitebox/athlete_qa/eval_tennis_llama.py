#!/usr/bin/env python3
"""Evaluate TennisQA probes and names-only questions with local Llama."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

def norm(s):
    s=s.casefold(); s=re.sub(r"[^\w\s]"," ",s); return " ".join(s.split())

def match_name(text, correct, wrong):
    t,c,w=norm(text),norm(correct),norm(wrong)
    hc,hw=c in t,w in t
    if hc and not hw:return "correct"
    if hw and not hc:return "wrong"
    if t==c:return "correct"
    if t==w:return "wrong"
    return "unmatched"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="pilot_v1/primary_questions.jsonl")
    p.add_argument("--out",default="pilot_v1/llama_eval"); p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--probe-batch",type=int,default=64); p.add_argument("--name-batch",type=int,default=16)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in open(a.data) if x.strip()]
    import torch
    # Disable the PyTorch native Triton bmm override; use standard eager CUDA.
    try:
        torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError):
        pass
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    yes=tok.encode("Yes",add_special_tokens=False); no=tok.encode("No",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1: raise ValueError((yes,no))
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={"":0},
        low_cpu_mem_usage=True,attn_implementation="sdpa",local_files_only=True).eval()
    flat=[]
    for i,x in enumerate(rows):
        for j,q in enumerate(x["probes"]): flat.append((i,j,q))
    pres=[]
    instruction="Answer from your own factual knowledge, without external sources. Answer exactly Yes or No.\nQuestion: "
    for st in range(0,len(flat),a.probe_batch):
        b=flat[st:st+a.probe_batch]
        texts=[tok.apply_chat_template([{"role":"user","content":instruction+q["question"]}],tokenize=False,add_generation_prompt=True) for _,_,q in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        with torch.inference_mode(): logits=model(**enc,use_cache=False).logits[:,-1,[no[0],yes[0]]].float()
        py=torch.softmax(logits,-1)[:,1].cpu().numpy()
        for prob,(i,j,q) in zip(py,b): pres.append((i,j,{**q,"p_yes":float(prob),"pred_yes":bool(prob>=.5),"correct":bool((prob>=.5)==bool(q["correct_answer"]))}))
        print(f"probes {min(st+len(b),len(flat))}/{len(flat)}",flush=True)
    by=[[] for _ in rows]
    for i,j,q in pres:by[i].append(q)
    generations=[]
    for st in range(0,len(rows),a.name_batch):
        b=rows[st:st+a.name_batch]
        texts=[tok.apply_chat_template([{"role":"user","content":x["prepend_names_prompt"]+"\nOutput only the person's name."}],tokenize=False,add_generation_prompt=True) for x in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        with torch.inference_mode(): gen=model.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
        prompt_len=enc.input_ids.shape[1]
        outs=tok.batch_decode(gen[:,prompt_len:],skip_special_tokens=True)
        generations.extend(outs); print(f"names {min(st+len(b),len(rows))}/{len(rows)}",flush=True)
    results=[]
    for x,probes,g in zip(rows,by,generations):
        probes=sorted(probes,key=lambda z:z["correct_answer"],reverse=True)
        ncorrect=sum(q["correct"] for q in probes); state=("knows_both" if ncorrect==2 else "knows_one" if ncorrect==1 else "knows_neither")
        outcome=match_name(g,x["correct_answer"],x["wrong_answer"])
        results.append({"id":x["id"],"correct_answer":x["correct_answer"],"wrong_answer":x["wrong_answer"],
            "probe_state":state,"n_probes_correct":ncorrect,"probes":probes,"generation":g,
            "name_outcome":outcome,"name_correct":outcome=="correct"})
    with open(out/"results.jsonl","w") as f:
        for x in results:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    summary={"model":a.model,"n_items":len(results),"n_probes":2*len(results),
        "probe_accuracy":float(np.mean([q["correct"] for x in results for q in x["probes"]])),
        "probe_positive_accuracy":float(np.mean([next(q for q in x["probes"] if q["correct_answer"]==1)["correct"] for x in results])),
        "probe_negative_accuracy":float(np.mean([next(q for q in x["probes"] if q["correct_answer"]==0)["correct"] for x in results])),
        "knowledge_states":{s:sum(x["probe_state"]==s for x in results) for s in ("knows_both","knows_one","knows_neither")},
        "names_accuracy":float(np.mean([x["name_correct"] for x in results])),
        "names_correct":sum(x["name_correct"] for x in results),
        "names_unmatched":sum(x["name_outcome"]=="unmatched" for x in results),"by_probe_state":{}}
    for s in ("knows_both","knows_one","knows_neither"):
        z=[x for x in results if x["probe_state"]==s]
        summary["by_probe_state"][s]={"n":len(z),"names_correct":sum(x["name_correct"] for x in z),
            "names_accuracy":float(np.mean([x["name_correct"] for x in z])) if z else None}
    json.dump(summary,open(out/"summary.json","w"),indent=2,ensure_ascii=False); print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
