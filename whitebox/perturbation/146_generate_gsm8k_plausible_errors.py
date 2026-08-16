#!/usr/bin/env python3
"""Augment real Llama GSM8K errors to 1500 with Llama-simulated reasoning errors."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FINAL_RE = re.compile(r"<answer>\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*</answer>", re.I)

def norm(x: str) -> str: return x.replace(",", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", type=Path, default=RUNS/"140_gsm8k_natural/regraded_generations_full_train.jsonl")
    ap.add_argument("--real", type=Path, default=RUNS/"140_gsm8k_natural/natural_errors_curated_full_train.jsonl")
    ap.add_argument("--out", type=Path, default=RUNS/"146_gsm8k_plausible_errors_n1500.jsonl")
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    a = ap.parse_args()
    real = [json.loads(x) for x in a.real.open() if x.strip()]
    have = {x["id"] for x in real}; need = a.target-len(real)
    allrows = [json.loads(x) for x in a.all.open() if x.strip()]
    pool = [x for x in allrows if x["id"] not in have and x.get("gold_final") and x.get("gold_solution")]
    pool.sort(key=lambda x: hashlib.sha1((x["id"]+"plausible-v1").encode()).hexdigest())

    existing = []
    tmp = a.out.with_suffix(".generated.jsonl")
    if tmp.exists(): existing = [json.loads(x) for x in tmp.open() if x.strip()]
    done = {x["id"] for x in existing}; candidates = [x for x in pool if x["id"] not in done]

    import torch
    try: torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError): pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        device_map={"":0}, low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True).eval()
    instruction = ("Simulate a plausible but incorrect student solution. Make exactly one locally plausible "
        "reasoning mistake, then consistently follow it. Prefer semantic mistakes (wrong operation, omitted "
        "step, confused rate/base, unit conversion) over arbitrary +/-1 changes. The final answer MUST differ "
        "from the provided correct answer. Be concise. End exactly with <answer>NUMBER</answer>.\n\n")
    idx=0
    while len(existing) < need and idx < len(candidates):
        batch=candidates[idx:idx+a.batch]; idx += len(batch)
        texts=[]
        for x in batch:
            content=instruction+"Problem:\n"+x["question"]+"\n\nCorrect solution (use only to design the mistake):\n"+x["gold_solution"]
            texts.append(tok.apply_chat_template([{"role":"user","content":content}], tokenize=False,
                                                  add_generation_prompt=True))
        enc=tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out=model.generate(**enc, max_new_tokens=192, do_sample=True, temperature=.8, top_p=.95,
                               pad_token_id=tok.eos_token_id)
        for x, ids, n in zip(batch, out, enc.attention_mask.sum(1)):
            response=tok.decode(ids[-(len(ids)-enc.input_ids.shape[1]):], skip_special_tokens=True).strip()
            m=FINAL_RE.search(response); wrong=norm(m.group(1)) if m else None; gold=norm(str(x["gold_final"]))
            if wrong and wrong != gold:
                existing.append({**x, "predicted_final":wrong, "model_response":response,
                    "correct":False, "error_source":"llama_simulated_single_reasoning_error"})
        with tmp.open("w") as f:
            for x in existing: f.write(json.dumps(x,ensure_ascii=False)+"\n")
        print(f"accepted {len(existing)}/{need}; scanned {idx}/{len(pool)}", flush=True)
    if len(existing) < need: raise RuntimeError(f"only made {len(existing)} of {need}")
    combined=real+existing[:need]
    with a.out.open("w") as f:
        for x in combined: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print(json.dumps({"n":len(combined),"real":len(real),"simulated":need},indent=2))

if __name__ == "__main__": main()
