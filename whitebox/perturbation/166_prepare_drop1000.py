#!/usr/bin/env python3
"""Download DROP, generate natural short answers, and build a balanced 1000 pool."""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_DIR = RUNS / "166_drop1000"


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = "".join(" " if c in string.punctuation else c for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def prepare(args):
    from datasets import load_dataset

    dataset = load_dataset("ucinlp/drop", split="validation")
    rows = []
    for item in dataset:
        passage = str(item["passage"]).strip()
        if not passage or len(passage) > args.max_passage_chars:
            continue
        raw_answers = [str(x).strip() for x in item["answers_spans"]["spans"] if str(x).strip()]
        normalized = [normalize(x) for x in raw_answers if normalize(x)]
        if not normalized:
            continue
        counts = Counter(normalized)
        gold, count = counts.most_common(1)[0]
        if count / len(normalized) < args.min_consensus:
            continue
        representative = next(x for x in raw_answers if normalize(x) == gold)
        rows.append({
            "key": str(item["query_id"]), "group": str(item["section_id"]),
            "context": passage, "question": str(item["question"]).strip(),
            "gold_answer": representative, "normalized_gold": gold,
            "annotation_count": len(normalized), "consensus_count": count,
            "consensus_fraction": count / len(normalized),
        })
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit or None]
    with args.items.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"downloaded": len(dataset), "prepared": len(rows),
                      "items": str(args.items)}, indent=2))


def completed(path):
    if not path.exists():
        return {}
    return {row["key"]: row for row in
            (json.loads(line) for line in path.open() if line.strip())}


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError):
        pass
    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    done = completed(args.generations) if args.resume else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval()
    mode = "a" if args.resume and args.generations.exists() else "w"
    pending = [x for x in rows if x["key"] not in done]
    with args.generations.open(mode) as output:
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            prompts = []
            for row in batch:
                content = (
                    "Read the passage and answer the question. Return only the shortest "
                    "direct answer, with no explanation.\n\nPassage:\n" + row["context"] +
                    "\n\nQuestion: " + row["question"]
                )
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True))
            encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_input_tokens,
                                add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            prompt_length = encoded.input_ids.shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_length:],
                                           skip_special_tokens=True)
            for row, text in zip(batch, texts):
                answer = text.strip().split("\n")[0].strip()
                prediction = normalize(answer)
                record = {
                    **row, "generation": answer,
                    "normalized_generation": prediction,
                    "correct": prediction == row["normalized_gold"],
                    "generation_words": len(answer.split()),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                done[row["key"]] = record
            n_correct = sum(x["correct"] for x in done.values())
            n_wrong = len(done) - n_correct
            print(f"completed={len(done)}/{len(rows)} correct={n_correct} wrong={n_wrong}",
                  flush=True)
            if n_correct >= args.target_per_class and n_wrong >= args.target_per_class:
                break


def balance(args):
    rows = list(completed(args.generations).values())
    correct = [x for x in rows if x["correct"]]
    wrong = [x for x in rows if not x["correct"]]
    if min(len(correct), len(wrong)) < args.target_per_class:
        raise RuntimeError(f"need {args.target_per_class}/class; have {len(correct)}/{len(wrong)}")
    rng = random.Random(args.seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    selected = correct[:args.target_per_class] + wrong[:args.target_per_class]
    rng.shuffle(selected)
    with args.manifest.open("w") as output:
        for row in selected:
            record = {
                "key": row["key"], "group": row["group"],
                "correct": int(row["correct"]), "context": row["context"],
                "question": row["question"], "generation": row["generation"],
                "other_answer": row["gold_answer"],
                "generation_words": row["generation_words"],
                "consensus_fraction": row["consensus_fraction"],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "generated": len(rows), "generated_correct": len(correct),
        "generated_incorrect": len(wrong), "balanced": len(selected),
        "groups": len(set(x["group"] for x in selected)),
        "mean_words_correct": sum(x["generation_words"] for x in correct[:args.target_per_class]) / args.target_per_class,
        "mean_words_incorrect": sum(x["generation_words"] for x in wrong[:args.target_per_class]) / args.target_per_class,
        "manifest": str(args.manifest),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "generate", "balance", "all"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-passage-chars", type=int, default=3000)
    parser.add_argument("--min-consensus", type=float, default=.60)
    parser.add_argument("--target-per-class", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.items = args.out_dir / "items.jsonl"
    args.generations = args.out_dir / "generations.jsonl"
    args.manifest = args.out_dir / "drop_balanced_n1000.jsonl"
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("generate", "all"):
        generate(args)
    if args.stage in ("balance", "all"):
        balance(args)


if __name__ == "__main__":
    main()
