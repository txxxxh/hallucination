#!/usr/bin/env python3
"""Generate profile-order swaps for 1000 items and compare to saved originals."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import profile_perturbation_unsupervised as pp
from profile_order_generation_check import mode, parse_choice

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=.7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=pp.DEFAULT_MODEL)
    ap.add_argument("--cache-dir", default=pp.DEFAULT_CACHE)
    ap.add_argument("--data", type=Path, default=pp.DEFAULT_DATA)
    ap.add_argument("--original", type=Path,
                    default=HERE / "profile_likelihood_generation_m3_output" / "items.jsonl")
    ap.add_argument("--output", type=Path,
                    default=HERE / "profile_order_swap_full_generation_output")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "swapped_items.jsonl"

    raw = json.loads(args.data.read_text(encoding="utf-8"))[:args.limit]
    raw_by_key = {str(x["key"]): x for x in raw}
    original = {}
    for line in args.original.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["key"] in raw_by_key:
                original[row["key"]] = row
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done[row["key"]] = row

    pending, prompts, names, gold = [], {}, {}, {}
    for key in original:
        item = pp.parse_item(raw_by_key[key])
        by_name = {c.name: c.prompt for c in pp.build_conditions(item)}
        prompts[key] = by_name["profile_order_swap"]
        names[key] = [p.name for p in item.profiles]  # canonical identity order
        gold[key] = int(original[key]["right_index"])
        if key not in done:
            pending.append(key)

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir) / "hub")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True).eval()

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        chats = [tok.apply_chat_template([{"role": "user", "content": prompts[k]}],
                                         tokenize=False, add_generation_prompt=True)
                 for k in batch]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=.95,
                num_return_sequences=args.samples, max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        texts = [tok.decode(x[width:], skip_special_tokens=True).strip() for x in generated]
        with out_path.open("a", encoding="utf-8") as f:
            for j, key in enumerate(batch):
                outputs = texts[j * args.samples:(j + 1) * args.samples]
                choices = [parse_choice(x, names[key]) for x in outputs]
                row = {"key": key, "names": names[key], "right_index": gold[key],
                       "outputs": outputs, "choices": choices,
                       "generation_mode": mode(choices)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[key] = row
        n = min(start + len(batch), len(pending))
        if n % 40 == 0 or n == len(pending):
            print(f"[{n}/{len(pending)}] total_stored={len(done)}", flush=True)

    paired = []
    for key in original:
        a, b = original[key], done[key]
        am, bm, y = a.get("generation_mode"), b.get("generation_mode"), gold[key]
        if am is None or bm is None:
            continue
        # choices are canonical person identity indices. After swapping display
        # order, preserving displayed position means changing identity.
        paired.append({"key": key, "original": int(am), "swapped": int(bm), "gold": y})
    def count(pred):
        return sum(pred(x) for x in paired)
    n = len(paired)
    cc = count(lambda x: x["original"] == x["gold"] and x["swapped"] == x["gold"])
    cw = count(lambda x: x["original"] == x["gold"] and x["swapped"] != x["gold"])
    wc = count(lambda x: x["original"] != x["gold"] and x["swapped"] == x["gold"])
    ww = count(lambda x: x["original"] != x["gold"] and x["swapped"] != x["gold"])
    stable = count(lambda x: x["original"] == x["swapped"])
    summary = {
        "n_requested": len(original), "n_paired_valid_modal": n,
        "samples_per_order": args.samples, "temperature": args.temperature,
        "original_modal_accuracy": count(lambda x: x["original"] == x["gold"]) / n,
        "swapped_modal_accuracy": count(lambda x: x["swapped"] == x["gold"]) / n,
        "identity_consistency": stable / n,
        "identity_changed": n - stable,
        "position_following_among_identity_changes": 1.0,
        "four_cells": {"correct_correct": cc, "correct_wrong": cw,
                       "wrong_correct": wc, "wrong_wrong": ww},
        "net_accuracy_change_count": wc - cw,
        "note": "With exactly two profiles, every identity change after swap preserves displayed position.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
