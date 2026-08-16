#!/usr/bin/env python3
"""Run closed-book knowledge probes and names-only QA on multidomain_v5."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DOMAINS = ("athlete", "musician", "building")


def norm(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def aliases(name: str) -> set[str]:
    values = {norm(name)}
    values.add(norm(re.sub(r"\s*\([^)]*\)\s*$", "", name)))
    values.add(norm(re.sub(r",\s*[^,]+$", "", name)))
    return {x for x in values if x}


def match_name(text: str, correct: str, wrong: str) -> str:
    target = norm(text)
    ca, wa = aliases(correct), aliases(wrong)
    ca -= wa
    wa -= aliases(correct)
    hit_c = any(x == target or x in target for x in ca)
    hit_w = any(x == target or x in target for x in wa)
    if hit_c and not hit_w:
        return "correct"
    if hit_w and not hit_c:
        return "wrong"
    return "unmatched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="multidomain_v5")
    parser.add_argument("--out", default="multidomain_v5/llama_eval")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--probe-batch", type=int, default=64)
    parser.add_argument("--name-batch", type=int, default=16)
    args = parser.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in DOMAINS:
        rows.extend(json.loads(x) for x in (root / domain / "primary_questions.jsonl").open() if x.strip())

    import torch
    try:
        torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError):
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError((yes, no))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()

    flat = [(i, j, q) for i, row in enumerate(rows) for j, q in enumerate(row["probes"])]
    by = [[] for _ in rows]
    instruction = "Answer from your own factual knowledge, without external sources. Answer exactly Yes or No.\nQuestion: "
    for start in range(0, len(flat), args.probe_batch):
        batch = flat[start:start + args.probe_batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": instruction + q["question"]}], tokenize=False, add_generation_prompt=True) for _, _, q in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits[:, -1, [no[0], yes[0]]].float()
        probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        for prob, (i, j, q) in zip(probs, batch):
            by[i].append((j, {**q, "p_yes": float(prob), "pred_yes": bool(prob >= .5),
                              "correct": bool((prob >= .5) == bool(q["correct_answer"]))}))
        print(f"probes {min(start + len(batch), len(flat))}/{len(flat)}", flush=True)

    generations = []
    for start in range(0, len(rows), args.name_batch):
        batch = rows[start:start + args.name_batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": row["prepend_names_prompt"] + "\nOutput only one candidate name."}], tokenize=False, add_generation_prompt=True) for row in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        generations.extend(tok.batch_decode(generated[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
        print(f"names {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    results = []
    for row, indexed_probes, generation in zip(rows, by, generations):
        probes = [q for _, q in sorted(indexed_probes)]
        n_correct = sum(q["correct"] for q in probes)
        state = "knows_both" if n_correct == 2 else "knows_one" if n_correct == 1 else "knows_neither"
        outcome = match_name(generation, row["correct_answer"], row["wrong_answer"])
        results.append({"id": row["id"], "domain": row["domain"], "field": row["decisive_relation"]["field"],
                        "correct_answer": row["correct_answer"], "wrong_answer": row["wrong_answer"],
                        "probe_state": state, "n_probes_correct": n_correct, "probes": probes,
                        "generation": generation, "name_outcome": outcome, "name_correct": outcome == "correct"})
    with (out / "results.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def group_stats(group):
        probe_rows = [q for row in group for q in row["probes"]]
        return {"n": len(group), "probe_accuracy": float(np.mean([q["correct"] for q in probe_rows])),
                "knowledge_states": {s: sum(r["probe_state"] == s for r in group) for s in ("knows_both", "knows_one", "knows_neither")},
                "names_accuracy": float(np.mean([r["name_correct"] for r in group])),
                "names_correct": sum(r["name_correct"] for r in group),
                "names_unmatched": sum(r["name_outcome"] == "unmatched" for r in group)}
    summary = {"model": args.model, "overall": group_stats(results),
               "by_domain": {d: group_stats([r for r in results if r["domain"] == d]) for d in DOMAINS}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
