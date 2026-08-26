#!/usr/bin/env python3
"""Build label-blind open-ended competitors for the perturbation detector.

The candidate generator never reads gold answers or correctness when selecting
``other``.  It samples the same answer prompt, clusters answers by task-specific
normalization, removes the cluster containing the greedy prediction, and uses
the largest remaining cluster as the counterfactual candidate.  The resulting
manifest is directly consumable by ``158_collect_paper4_matrix.py``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NUMBER_RE = re.compile(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_trivia(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = "".join(" " if c in string.punctuation else c for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonical_gsm8k(text: str) -> str | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        # Sampling occasionally emits only a bare final number.
        matches = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])", text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", "")).normalize()
    except InvalidOperation:
        return None
    return format(value, "f") if value.is_finite() else None


def cluster_key(dataset: str, answer: str) -> str | None:
    return canonical_gsm8k(answer) if dataset == "gsm8k" else normalize_trivia(answer)


def choose_competitor(dataset: str, pred: str, samples: list[str]) -> dict | None:
    """Choose without labels; deterministic ties favor the earlier sample."""
    pred_key = cluster_key(dataset, pred)
    clusters: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for index, answer in enumerate(samples):
        key = cluster_key(dataset, answer)
        if key and key != pred_key:
            clusters[key].append((index, answer.strip()))
    if not clusters:
        return None
    key, members = max(clusters.items(), key=lambda item: (len(item[1]), -item[1][0][0]))
    # The first sampled member avoids selecting with any downstream label/metric.
    return {"answer": members[0][1], "cluster_key": key,
            "cluster_count": len(members), "cluster_members": [x[1] for x in members]}


def answer_prompt(dataset: str, row: dict) -> str:
    if dataset == "gsm8k":
        return ("Solve the grade-school math problem step by step. End with the "
                "final numeric answer exactly as: #### <number>\n\nProblem:\n" +
                row["question"])
    return ("Answer using the context. Output only the short answer.\n\nContext:\n" +
            row["context"] + "\n\nQuestion: " + row["question"])


def output_row(dataset: str, row: dict, competitor: dict, samples: list[str],
               model: str, decoding: dict) -> dict:
    # correct/gold fields are copied only for evaluation; none are passed to
    # choose_competitor.  Keep this construction after selection to make that
    # separation obvious and auditable.
    if dataset == "gsm8k":
        context = row["question"]
        question = "Provide the complete solution to this math problem."
        pred = row["generation"]
    else:
        context, question, pred = row["context"], row["question"], row["generation"]
    return {
        "key": row["key"], "group": row.get("group", row["key"]),
        "correct": int(row["correct"]), "context": context,
        "question": question, "pred": pred, "other": competitor["answer"],
        "prompt_mode": False, "model": model, "dataset": dataset,
        "other_source": "label_blind_sampling_second_semantic_cluster",
        "pred_cluster_key": cluster_key(dataset, pred),
        "other_cluster_key": competitor["cluster_key"],
        "other_cluster_count": competitor["cluster_count"],
        "sample_cluster_keys": [cluster_key(dataset, x) for x in samples],
        "samples": samples, "candidate_decoding": decoding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("gsm8k", "trivia"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=.9)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.temperature <= 0:
        parser.error("open competitor generation requires --samples >= 1 and temperature > 0")
    defaults = {
        "gsm8k": RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl",
        "trivia": RUNS / "127_triviaqa_balanced_n1000.jsonl",
    }
    args.source = args.source or defaults[args.dataset]
    args.output = args.output or RUNS / f"294_{args.dataset}_open_candidates.jsonl"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); torch.manual_seed(args.seed)
    rows = read_jsonl(args.source)[:args.limit or None]
    done = {r["key"] for r in read_jsonl(args.output)} if args.resume and args.output.exists() else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True).eval()
    decoding = {"samples": args.samples, "temperature": args.temperature,
                "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                "seed": args.seed}
    mode = "a" if args.resume and args.output.exists() else "w"
    kept = skipped = 0
    with args.output.open(mode) as handle:
        for number, row in enumerate(rows, 1):
            if row["key"] in done:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": answer_prompt(args.dataset, row)}],
                tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, num_return_sequences=args.samples,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:],
                                           skip_special_tokens=True)
            competitor = choose_competitor(args.dataset, row["generation"], texts)
            if competitor is None:
                skipped += 1
                print(f"[{number}/{len(rows)}] {row['key']} no distinct cluster", flush=True)
                continue
            handle.write(json.dumps(output_row(args.dataset, row, competitor, texts,
                                               args.model, decoding), ensure_ascii=False) + "\n")
            handle.flush(); kept += 1
            print(f"[{number}/{len(rows)}] {row['key']} other={competitor['cluster_key']!r}", flush=True)
    print(json.dumps({"source_rows": len(rows), "new_kept": kept,
                      "new_skipped_no_competitor": skipped,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
