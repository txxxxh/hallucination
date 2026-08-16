#!/usr/bin/env python3
"""Compare teacher-forced candidate preference with actual generation mode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import profile_perturbation_unsupervised as pp
from profile_order_generation_check import mode, parse_choice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--model", default=pp.DEFAULT_MODEL)
    ap.add_argument("--cache-dir", default=pp.DEFAULT_CACHE)
    ap.add_argument("--features", type=Path, default=pp.DEFAULT_OUTPUT / "items")
    ap.add_argument("--data", type=Path, default=pp.DEFAULT_DATA)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).with_name("profile_likelihood_generation_output"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl = args.output / "items.jsonl"

    rows = json.loads(args.data.read_text(encoding="utf-8"))[:args.limit]
    by_key = {str(row["key"]): row for row in rows}
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done[row["key"]] = row

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
        args.model, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    ).eval()

    def generate(prompt: str) -> list[str]:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=0.95,
                num_return_sequences=args.samples, max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        n = enc["input_ids"].shape[1]
        return [tok.decode(seq[n:], skip_special_tokens=True).strip() for seq in out]

    feature_paths = {p.stem: p for p in args.features.glob("*.npz")}
    for n, (key, raw) in enumerate(by_key.items(), 1):
        if key in done:
            continue
        record = pp.load_item_npz(feature_paths[key]); md = record["metadata"]
        ci = md["condition_names"].index("full_context")
        likelihood_preferred = int(np.argmax(record["candidate_scores"][ci]))
        item = pp.parse_item(raw); names = [p.name for p in item.profiles]
        prompt = {c.name: c.prompt for c in pp.build_conditions(item)}["full_context"]
        outputs = generate(prompt); choices = [parse_choice(x, names) for x in outputs]
        row = {"key": key, "names": names, "right_index": int(md["right_index"]),
               "likelihood_preferred": likelihood_preferred,
               "outputs": outputs, "choices": choices, "generation_mode": mode(choices)}
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[key] = row
        if n % 25 == 0 or n == len(by_key):
            print(f"[{n}/{len(by_key)}] stored={len(done)}", flush=True)

    items = [done[k] for k in by_key if k in done]
    valid = [x for x in items if x["generation_mode"] is not None]
    def agreement(xs):
        return (sum(x["likelihood_preferred"] == x["generation_mode"] for x in xs) / len(xs)
                if xs else None)
    gen_correct = [x for x in valid if x["generation_mode"] == x["right_index"]]
    gen_wrong = [x for x in valid if x["generation_mode"] != x["right_index"]]
    ll_correct = [x for x in valid if x["likelihood_preferred"] == x["right_index"]]
    ll_wrong = [x for x in valid if x["likelihood_preferred"] != x["right_index"]]
    summary = {
        "n_items": len(items), "samples": args.samples,
        "modal_valid": len(valid), "modal_parse_rate": len(valid) / len(items),
        "overall_agreement": agreement(valid),
        "generation_modal_correct": {"n": len(gen_correct), "agreement": agreement(gen_correct)},
        "generation_modal_wrong": {"n": len(gen_wrong), "agreement": agreement(gen_wrong)},
        "likelihood_correct": {"n": len(ll_correct), "agreement": agreement(ll_correct)},
        "likelihood_wrong": {"n": len(ll_wrong), "agreement": agreement(ll_wrong)},
        "generation_modal_accuracy_valid": len(gen_correct) / len(valid),
        "likelihood_accuracy_on_modal_valid": len(ll_correct) / len(valid),
        "temperature": args.temperature, "model": args.model,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
