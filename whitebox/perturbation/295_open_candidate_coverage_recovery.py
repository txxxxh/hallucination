#!/usr/bin/env python3
"""Recover open-ended competitor coverage without gold-label leakage.

Filters lexical containment aliases from experiment 294, then asks the same
model for a genuinely different plausible alternative only for missing rows.
Gold answers and correctness are never included in generation prompts or used
to choose candidates.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


base = importlib.import_module("294_open_ended_perturbation")


def equivalent(dataset: str, left: str, right: str) -> bool:
    a, b = base.cluster_key(dataset, left), base.cluster_key(dataset, right)
    if not a or not b:
        return False
    if a == b:
        return True
    if dataset == "trivia":
        aa, bb = f" {a} ", f" {b} "
        return aa in bb or bb in aa
    return False


def choose(dataset: str, pred: str, samples: list[str]) -> dict | None:
    valid = [answer for answer in samples
             if base.cluster_key(dataset, answer)
             and not equivalent(dataset, pred, answer)]
    return base.choose_competitor(dataset, pred, valid)


def prompt(dataset: str, row: dict) -> str:
    pred = row["generation"]
    if dataset == "gsm8k":
        return ("Produce a plausible alternative solution with a final numeric "
                "answer different from the proposed solution. Do not evaluate the "
                "proposal. End exactly with #### <number>.\n\nProblem:\n" +
                row["question"] + "\n\nProposed solution:\n" + pred)
    return ("Return one plausible short alternative answer. It must refer to a "
            "different entity or value from the proposed answer; do not merely add "
            "an explanation, title, first name, surname, or attribution. Output only "
            "the alternative.\n\nContext:\n" + row["context"] +
            "\n\nQuestion: " + row["question"] +
            "\nProposed answer: " + pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("gsm8k", "trivia"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--first-pass", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--max-new-tokens", type=int, default=96)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = base.read_jsonl(a.source)[:a.limit or None]
    first = {row["key"]: row for row in base.read_jsonl(a.first_pass)}
    accepted = {}
    rejected_alias = 0
    for key, row in first.items():
        if equivalent(a.dataset, row["pred"], row["other"]):
            rejected_alias += 1
        else:
            accepted[key] = row
    pending = [row for row in source if row["key"] not in accepted]

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True,
                                        local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval()
    recovered = 0
    for number, row in enumerate(pending, 1):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt(a.dataset, row)}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            ids = model.generate(
                **enc, do_sample=True, temperature=a.temperature,
                top_p=a.top_p, num_return_sequences=a.samples,
                max_new_tokens=a.max_new_tokens,
                pad_token_id=tok.eos_token_id)
        samples = tok.batch_decode(ids[:, enc.input_ids.shape[1]:],
                                   skip_special_tokens=True)
        competitor = choose(a.dataset, row["generation"], samples)
        if competitor is not None:
            result = base.output_row(
                a.dataset, row, competitor, samples, a.model,
                {"samples": a.samples, "temperature": a.temperature,
                 "top_p": a.top_p, "max_new_tokens": a.max_new_tokens})
            result["other_source"] = "label_blind_prompted_alternative_recovery"
            accepted[row["key"]] = result
            recovered += 1
        print(f"[{number}/{len(pending)}] {row['key']} "
              f"{'recovered' if competitor else 'missing'}", flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as handle:
        for row in source:
            if row["key"] in accepted:
                handle.write(json.dumps(accepted[row["key"]],
                                        ensure_ascii=False) + "\n")
    report = {"source": len(source), "first_pass": len(first),
              "rejected_containment_alias": rejected_alias,
              "pending": len(pending), "recovered": recovered,
              "final": len(accepted),
              "coverage": len(accepted) / len(source) if source else 0.0}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
