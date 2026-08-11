#!/usr/bin/env python3
"""Actual-generation check for profile-order effects on the 39 reversal items."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np

import profile_perturbation_unsupervised as pp


def parse_choice(text: str, names: list[str]) -> int | None:
    normalized = re.sub(r"\s+", " ", text).casefold()
    hits = [i for i, name in enumerate(names) if name.casefold() in normalized]
    if len(hits) == 1:
        return hits[0]
    # Models occasionally output only the surname. Use it only when unique.
    surnames = [re.sub(r"[^\w'-]", "", name.split()[-1]).casefold() for name in names]
    hits = [i for i, surname in enumerate(surnames) if surname and
            re.search(rf"(?<!\w){re.escape(surname)}(?!\w)", normalized)]
    return hits[0] if len(hits) == 1 else None


def mode(values: list[int | None]) -> int | None:
    valid = [x for x in values if x is not None]
    if not valid:
        return None
    counts = np.bincount(valid, minlength=2)
    return int(np.argmax(counts)) if counts[0] != counts[1] else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=pp.DEFAULT_DATA)
    ap.add_argument("--features", type=Path, default=pp.DEFAULT_OUTPUT / "items")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).with_name("profile_order_generation_output"))
    ap.add_argument("--model", default=pp.DEFAULT_MODEL)
    ap.add_argument("--cache-dir", default=pp.DEFAULT_CACHE)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir) / "hub")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    by_key = {str(row["key"]): row for row in rows}
    selected = []
    for path in sorted(args.features.glob("*.npz")):
        record = pp.load_item_npz(path); md = record["metadata"]
        ix = {name: i for i, name in enumerate(md["condition_names"])}
        right = int(md["right_index"])
        base = int(np.argmax(record["candidate_scores"][ix["question_only"]]))
        full = int(np.argmax(record["candidate_scores"][ix["full_context"]]))
        if base == right and full != right:
            selected.append(md["key"])
    if len(selected) != 39:
        raise RuntimeError(f"expected 39 reversal items, found {len(selected)}")

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

    args.output.mkdir(parents=True, exist_ok=True)
    result = []
    for n, key in enumerate(selected, 1):
        item = pp.parse_item(by_key[key])
        conditions = {c.name: c.prompt for c in pp.build_conditions(item, args.seed)}
        names = [p.name for p in item.profiles]
        original_out = generate(conditions["full_context"])
        swapped_out = generate(conditions["profile_order_swap"])
        original_choice = [parse_choice(x, names) for x in original_out]
        swapped_choice = [parse_choice(x, names) for x in swapped_out]
        result.append({
            "key": key, "names": names, "right_index": names.index(item.right_answer),
            "original": {"outputs": original_out, "choices": original_choice,
                         "mode": mode(original_choice)},
            "swapped": {"outputs": swapped_out, "choices": swapped_choice,
                        "mode": mode(swapped_choice)},
        })
        if n % 5 == 0 or n == len(selected):
            print(f"[{n}/{len(selected)}]", flush=True)

    def aggregate(items):
        sample_total = sample_valid = sample_correct = 0
        modal_valid = modal_correct = modal_identity_flip = modal_same_identity = 0
        for row in items:
            right = row["right_index"]
            for order in ("original", "swapped"):
                vals = row[order]["choices"]
                sample_total += len(vals); sample_valid += sum(x is not None for x in vals)
                sample_correct += sum(x == right for x in vals)
                if row[order]["mode"] is not None:
                    modal_valid += 1; modal_correct += row[order]["mode"] == right
            a, b = row["original"]["mode"], row["swapped"]["mode"]
            if a is not None and b is not None:
                modal_same_identity += a == b
                modal_identity_flip += a != b
        paired = sum(row["original"]["mode"] is not None and
                     row["swapped"]["mode"] is not None for row in items)
        return {
            "n_items": len(items), "n_samples_per_order": args.samples,
            "sample_parse_rate": sample_valid / sample_total,
            "sample_accuracy_including_invalid_as_wrong": sample_correct / sample_total,
            "modal_parse_rate": modal_valid / (2 * len(items)),
            "modal_accuracy_including_ties_invalid_as_wrong": modal_correct / (2 * len(items)),
            "paired_modal_items": paired,
            "modal_keeps_identity_rate": modal_same_identity / paired if paired else None,
            "modal_follows_position_identity_flip_rate": modal_identity_flip / paired if paired else None,
        }
    summary = aggregate(result)
    summary.update({"model": args.model, "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "selection": "teacher-forced question_only correct and full_context wrong"})
    (args.output / "generations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
