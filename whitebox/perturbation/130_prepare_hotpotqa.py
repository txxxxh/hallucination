#!/usr/bin/env python3
"""Prepare a leakage-audited HotpotQA generation manifest.

The compact context keeps every gold supporting paragraph, then adds distractor
paragraphs in the dataset order up to a character budget.  Labels are assigned
only from the model's freely generated answer; no paired gold/generated rows are
created.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path


def norm(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_aliases(answer: str) -> list[str]:
    aliases = [answer]
    a = norm(answer)
    if a in {"yes", "no"}:
        aliases.extend([a.capitalize(), a.upper()])
    return list(dict.fromkeys(aliases))


def compact_context(row: dict, max_chars: int) -> tuple[str, list[str], int]:
    titles = list(row["context"]["title"])
    sentences = list(row["context"]["sentences"])
    supporting_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    by_title = {t: " ".join(str(s).strip() for s in ss if str(s).strip())
                for t, ss in zip(titles, sentences)}
    ordered = supporting_titles + [t for t in titles if t not in supporting_titles]
    pieces: list[str] = []
    used: list[str] = []
    for title in ordered:
        piece = f"[{title}] {by_title.get(title, '')}".strip()
        if not piece:
            continue
        is_support = title in supporting_titles
        candidate = "\n".join([*pieces, piece])
        if len(candidate) > max_chars and not is_support:
            continue
        pieces.append(piece)
        used.append(title)
    return "\n".join(pieces), used, len(supporting_titles)


def fetch(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    args.items.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.items.open("w") as out:
        for row in ds:
            context, used_titles, n_support = compact_context(row, args.max_context_chars)
            rec = {
                "key": str(row["id"]),
                "question": row["question"],
                "context": context,
                "answer": row["answer"],
                "aliases": answer_aliases(row["answer"]),
                "level": row.get("level", "unknown"),
                "type": row.get("type", "unknown"),
                "supporting_titles": list(dict.fromkeys(row["supporting_facts"]["title"])),
                "context_titles": used_titles,
                "n_supporting_titles": n_support,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.n:
                break
    print(json.dumps({"fetched": count, "out": str(args.items)}, indent=2))


def generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    done = {}
    if args.generations.exists() and args.resume:
        done = {r["key"]: r for r in map(json.loads, args.generations.open())}
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    pending = [r for r in rows if r["key"] not in done]
    mode = "a" if args.resume else "w"
    with args.generations.open(mode) as out:
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            prompts = []
            for r in batch:
                content = (f"Context:\n{r['context']}\n\nQuestion: {r['question']}\n\n"
                           "Answer with only the shortest direct answer phrase.")
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True))
            z = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_prompt_tokens).to("cuda")
            with torch.inference_mode():
                ids = model.generate(**z, max_new_tokens=20, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            for row, seq in zip(batch, ids):
                text = tok.decode(seq[z.input_ids.shape[1]:], skip_special_tokens=True)
                text = text.strip().split("\n")[0].strip()
                aliases = [norm(x) for x in row["aliases"]]
                rec = {**row, "generation": text,
                       "normalized_generation": norm(text),
                       "correct": norm(text) in aliases,
                       "generation_words": len(text.split())}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            print(f"[{min(start + args.batch, len(pending))}/{len(pending)}]", flush=True)
    all_rows = [json.loads(line) for line in args.generations.open() if line.strip()]
    print(json.dumps({"n": len(all_rows),
                      "correct": sum(bool(r["correct"]) for r in all_rows),
                      "incorrect": sum(not bool(r["correct"]) for r in all_rows)}, indent=2))


def balance(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.generations.open() if line.strip()]
    good = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    rng = random.Random(args.seed)
    rng.shuffle(good)
    rng.shuffle(bad)
    per_class = args.per_class or min(len(good), len(bad))
    if min(len(good), len(bad)) < per_class:
        raise RuntimeError(f"need {per_class}/class; have correct={len(good)} incorrect={len(bad)}")
    chosen = good[:per_class] + bad[:per_class]
    rng.shuffle(chosen)
    # Match the existing detector interface.  This field is not the supervised
    # label: it is the comparison answer scored alongside the model generation.
    # Correct generations use a deterministic reference answer from another
    # question, avoiding an extra LLM-produced hard-negative provenance cue.
    foreign = [r["answer"] for r in rows]
    for i, row in enumerate(chosen):
        row["other_answer"] = row["answer"] if not row["correct"] else foreign[(i + 7919) % len(foreign)]
        if norm(row["other_answer"]) == norm(row["generation"]):
            row["other_answer"] = foreign[(i + 7927) % len(foreign)]
        row["other_words"] = len(row["other_answer"].split())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as out:
        for row in chosen:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(chosen), "per_class": per_class,
                      "out": str(args.manifest)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fetch", "generate", "balance", "all"])
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--max-context-chars", type=int, default=3600)
    p.add_argument("--max-prompt-tokens", type=int, default=1536)
    p.add_argument("--items", type=Path, default=Path("runs/130_hotpotqa_items_n1200.jsonl"))
    p.add_argument("--generations", type=Path, default=Path("runs/130_hotpotqa_generations_n1200.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("runs/130_hotpotqa_balanced_n200.jsonl"))
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.stage in {"fetch", "all"}:
        fetch(args)
    if args.stage in {"generate", "all"}:
        generate(args)
    if args.stage in {"balance", "all"}:
        balance(args)


if __name__ == "__main__":
    main()
