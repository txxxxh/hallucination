#!/usr/bin/env python3
import json,re
from pathlib import Path

def norm(s):
    return " ".join(re.sub(r"[^\w\s]"," ",s.casefold()).split())

def outcome(text,correct,wrong):
    t,c,w=norm(text),norm(correct),norm(wrong); hc,hw=c in t,w in t
    return "correct" if hc and not hw else "wrong" if hw and not hc else "unmatched"

def main():
    import torch
    try: torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError,ImportError): pass
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root=Path("pilot_v1"); items={x["id"]:x for x in map(json.loads,open(root/"primary_questions.jsonl"))}
    prior=[x for x in map(json.loads,open(root/"llama_eval/results.jsonl")) if x["probe_state"]=="knows_both"]
    rows=[items[x["id"]] for x in prior]; model_id="NousResearch/Meta-Llama-3.1-8B-Instruct"
    tok=AutoTokenizer.from_pretrained(model_id,use_fast=True,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True,attn_implementation="sdpa",local_files_only=True).eval()
    outs=[]
    for st in range(0,len(rows),8):
        b=rows[st:st+8]; texts=[tok.apply_chat_template([{"role":"user","content":x["prepend_profiles_prompt"]+"\nOutput only the person's name."}],tokenize=False,add_generation_prompt=True) for x in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        with torch.inference_mode(): gen=model.generate(**enc,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
        dec=tok.batch_decode(gen[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
        for x,g in zip(b,dec): outs.append({"id":x["id"],"correct_answer":x["correct_answer"],"wrong_answer":x["wrong_answer"],"generation":g,"outcome":outcome(g,x["correct_answer"],x["wrong_answer"])})
    with open(root/"llama_eval/profiles_known_both_results.jsonl","w") as f:
        for x in outs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    s={"n":len(outs),"correct":sum(x["outcome"]=="correct" for x in outs),"wrong":sum(x["outcome"]=="wrong" for x in outs),"unmatched":sum(x["outcome"]=="unmatched" for x in outs)}; s["accuracy"]=s["correct"]/s["n"]
    json.dump(s,open(root/"llama_eval/profiles_known_both_summary.json","w"),indent=2); print(json.dumps(s,indent=2))
if __name__=="__main__":main()
