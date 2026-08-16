#!/usr/bin/env python3
"""Build GSM8K two-choice items using naturally generated wrong answers."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--errors", type=Path, default=RUNS / "140_gsm8k_natural/natural_errors_curated_full_train.jsonl")
    p.add_argument("--out", type=Path, default=RUNS / "143_gsm8k_natural_mc.jsonl")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.errors.open() if x.strip()]
    items = []
    for x in rows:
        gold, wrong = x["gold_final"], x["predicted_final"]
        flip = int(hashlib.sha1((x["id"] + str(a.seed)).encode()).hexdigest(), 16) % 2
        order = [gold, wrong] if flip == 0 else [wrong, gold]
        prompt = ("Choose one of the following two options as the answer to the math problem below. "
                  "Output only 1 or 2.\n1. " + order[0] + "\n2. " + order[1] +
                  "\nQuestion:\n" + x["question"])
        items.append({"key": x["id"], "group": x["id"], "question": x["question"],
                      "gold_answer": gold, "natural_wrong_answer": wrong,
                      "candidate_order": order, "correct_position": 1 + order.index(gold),
                      "prepend_names_prompt": prompt,
                      "natural_wrong_reasoning": x["model_response"]})

    import torch
    try: torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError): pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    one, two = tok.encode("1", add_special_tokens=False), tok.encode("2", add_special_tokens=False)
    if len(one) != 1 or len(two) != 1: raise RuntimeError((one, two))
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval()
    for st in range(0, len(items), a.batch):
        batch = items[st:st + a.batch]
        texts = [tok.apply_chat_template([{"role":"user", "content":x["prepend_names_prompt"]}],
                 tokenize=False, add_generation_prompt=True) for x in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits[:, -1, [one[0], two[0]]].float()
        probs = torch.softmax(logits, -1).cpu().numpy()
        for x, prob in zip(batch, probs):
            choice = 1 + int(prob[1] > prob[0]); chosen = x["candidate_order"][choice - 1]
            other = x["candidate_order"][2 - choice]
            x.update({"choice": choice, "p_choice1": float(prob[0]), "p_choice2": float(prob[1]),
                      "generation": chosen, "other_answer": other,
                      "correct": choice == x["correct_position"]})
        print(f"{min(st + len(batch), len(items))}/{len(items)}", flush=True)
    with a.out.open("w") as f:
        for x in items: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary = {"n": len(items), "correct": sum(x["correct"] for x in items),
               "incorrect": sum(not x["correct"] for x in items),
               "accuracy": float(np.mean([x["correct"] for x in items])),
               "gold_position1": sum(x["correct_position"] == 1 for x in items),
               "choice1": sum(x["choice"] == 1 for x in items)}
    a.out.with_name(a.out.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
