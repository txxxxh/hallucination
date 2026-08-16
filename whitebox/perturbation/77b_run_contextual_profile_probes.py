#!/usr/bin/env python3
"""Run atomic Yes/No probes with both source profiles supplied as context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "runs/76_closedbook_fact_probe_manifest.jsonl"
DEFAULT_DATA = ROOT.parent / "shuffled_prepend_profiles_question.json"
DEFAULT_OUTPUT = ROOT / "runs/77b_contextual_profile_probe_results.jsonl"
DEFAULT_SUMMARY = ROOT / "runs/77b_contextual_profile_probe_summary.json"


def profile_context(prompt: str) -> str:
    marker = ("\nChoose exactly one profile from the two, and output the name of the person "
              "as the answer to the following question:\n")
    return prompt.split(marker, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(x) for x in open(args.manifest) if x.strip()]
    source = {str(x["key"]): profile_context(x["prompt"]) for x in json.load(open(args.data))}
    flat = []
    for i, item in enumerate(items):
        context = source[str(item["key"])]
        for j, probe in enumerate(item["probes"]):
            question = probe["prompt"].split("Question: ", 1)[1]
            prompt = (f"{context}\n\nUsing only the two profiles above, answer the factual question. "
                      f"Answer exactly Yes or No.\nQuestion: {question}")
            flat.append((i, j, probe, prompt))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Expected single Yes/No tokens, got {yes_ids=} {no_ids=}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).eval()

    probabilities = np.empty(len(flat), np.float32)
    for start in range(0, len(flat), args.batch_size):
        batch = flat[start:start + args.batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        ) for _, _, _, prompt in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=8192,
                        add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits[:, -1, [no_ids[0], yes_ids[0]]].float()
            probabilities[start:start + len(batch)] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        if start == 0 or (start // args.batch_size + 1) % 25 == 0:
            print(f"scored {start + len(batch)}/{len(flat)}", flush=True)

    by_item = [[] for _ in items]
    for probability, (i, _, probe, _) in zip(probabilities, flat):
        by_item[i].append({**probe, "p_yes": float(probability),
                           "pred_yes": bool(probability >= .5),
                           "correct": bool((probability >= .5) == probe["gold_yes"])})

    output = []
    for item, probes in zip(items, by_item):
        n_facts = item["n_discriminative_facts"]
        binary = float(np.mean([p["correct"] for p in probes])) if probes else None
        pair_correct = []
        for fact_id in range(n_facts):
            pair = [p for p in probes if p["probe_id"].split("::")[1] == f"f{fact_id}"]
            owner = next(p for p in pair if p["gold_yes"])
            other = next(p for p in pair if not p["gold_yes"])
            pair_correct.append(owner["p_yes"] > other["p_yes"])
        pair_acc = float(np.mean(pair_correct)) if pair_correct else None
        output.append({
            "key": item["key"], "right_qid": item["right_qid"],
            "right_answer": item["right_answer"], "wrong_answer": item["wrong_answer"],
            "n_discriminative_facts": n_facts, "n_binary_probes": len(probes),
            "binary_accuracy": binary, "pairwise_owner_accuracy": pair_acc,
            "known": bool(n_facts >= 1 and binary >= .75 and pair_acc >= .75),
            "probes": probes,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        for row in output:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    eligible = [r for r in output if r["n_discriminative_facts"] >= 1]
    summary = {
        "protocol": "atomic Yes/No probes conditioned on both full profiles",
        "n_items": len(output), "n_eligible": len(eligible),
        "micro_binary_accuracy": float(np.mean([p["correct"] for r in output for p in r["probes"]])),
        "mean_item_binary_accuracy": float(np.mean([r["binary_accuracy"] for r in eligible])),
        "mean_pairwise_owner_accuracy": float(np.mean([r["pairwise_owner_accuracy"] for r in eligible])),
        "threshold_sensitivity": {}, "model": args.model,
    }
    for threshold in (.5, .75, 1.0):
        summary["threshold_sensitivity"][str(threshold)] = {
            "strict_gt": sum(r["binary_accuracy"] > threshold and r["pairwise_owner_accuracy"] > threshold for r in eligible),
            "ge": sum(r["binary_accuracy"] >= threshold and r["pairwise_owner_accuracy"] >= threshold for r in eligible),
        }
    json.dump(summary, open(args.summary, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
