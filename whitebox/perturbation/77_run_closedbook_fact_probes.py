#!/usr/bin/env python3
"""Run Llama closed-book Yes/No factual probes and create known/unknown split."""

import argparse
import json
from pathlib import Path

import numpy as np

MANIFEST = Path("/home/tong56/whitebox/perturbation/runs/76_closedbook_fact_probe_manifest.jsonl")
OUTPUT = Path("/home/tong56/whitebox/perturbation/runs/77_closedbook_fact_probe_results.jsonl")
SUMMARY = Path("/home/tong56/whitebox/perturbation/runs/77_closedbook_fact_probe_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(x) for x in open(MANIFEST) if x.strip()]
    flat = [(i, j, p) for i, item in enumerate(items) for j, p in enumerate(item["probes"])]
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
            [{"role": "user", "content": p["prompt"]}], tokenize=False,
            add_generation_prompt=True,
        ) for _, _, p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits[:, -1, [no_ids[0], yes_ids[0]]].float()
            probabilities[start:start + len(batch)] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        if start == 0 or (start // args.batch_size + 1) % 25 == 0:
            print(f"scored {start + len(batch)}/{len(flat)}", flush=True)

    by_item = [[] for _ in items]
    for probability, (i, j, probe) in zip(probabilities, flat):
        by_item[i].append({**probe, "p_yes": float(probability),
                           "pred_yes": bool(probability >= .5),
                           "correct": bool((probability >= .5) == probe["gold_yes"])})

    output_rows = []
    for item, probes in zip(items, by_item):
        n_facts = item["n_discriminative_facts"]
        binary_accuracy = float(np.mean([p["correct"] for p in probes])) if probes else None
        pair_correct = []
        for fact_id in range(n_facts):
            pair = [p for p in probes if p["probe_id"].split("::")[1] == f"f{fact_id}"]
            owner = next(p for p in pair if p["gold_yes"])
            other = next(p for p in pair if not p["gold_yes"])
            pair_correct.append(owner["p_yes"] > other["p_yes"])
        pair_accuracy = float(np.mean(pair_correct)) if pair_correct else None
        known = bool(n_facts >= 1 and binary_accuracy >= .75 and pair_accuracy >= .75)
        output_rows.append({
            "key": item["key"], "right_qid": item["right_qid"],
            "right_answer": item["right_answer"], "wrong_answer": item["wrong_answer"],
            "n_discriminative_facts": n_facts, "n_binary_probes": len(probes),
            "binary_accuracy": binary_accuracy, "pairwise_owner_accuracy": pair_accuracy,
            "known": known, "probes": probes,
        })
    with open(OUTPUT, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eligible = [r for r in output_rows if r["n_discriminative_facts"] >= 1]
    summary = {
        "n_items": len(output_rows), "n_eligible": len(eligible),
        "n_known": sum(r["known"] for r in output_rows),
        "n_unknown_or_uncovered": sum(not r["known"] for r in output_rows),
        "known_rule": "n_facts>=1 and binary_accuracy>=0.75 and pairwise_owner_accuracy>=0.75",
        "micro_binary_accuracy": float(np.mean([p["correct"] for r in output_rows for p in r["probes"]])),
        "mean_item_binary_accuracy": float(np.mean([r["binary_accuracy"] for r in eligible])),
        "mean_pairwise_owner_accuracy": float(np.mean([r["pairwise_owner_accuracy"] for r in eligible])),
        "threshold_sensitivity": {},
        "model": args.model,
    }
    for threshold in (.5, .75, 1.0):
        summary["threshold_sensitivity"][str(threshold)] = sum(
            r["n_discriminative_facts"] >= 1
            and r["binary_accuracy"] >= threshold
            and r["pairwise_owner_accuracy"] >= threshold
            for r in output_rows
        )
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
