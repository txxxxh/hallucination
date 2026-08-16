#!/usr/bin/env python3
"""Generate free-form GSM8K solutions and retain naturally wrong answers.

Unlike ``136_build_eval_gsm8k_mc.py``, this script does not construct a
distractor or ask the model to choose between candidates.  The model writes a
full solution, and correctness is determined only by comparing the generated
final number with GSM8K's gold final number.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT.parent / "other_bench/GSM8K"
DEFAULT_OUT = HERE / "runs/140_gsm8k_natural"
FINAL_RE = re.compile(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def canonical_number(text: str) -> str | None:
    matches = FINAL_RE.findall(text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    value = value.normalize()
    return format(value, "f")


def load_rows(split: str, seed: int, limit: int) -> list[dict]:
    import pandas as pd

    filename = ("train-00000-of-00001.parquet" if split == "train" else
                "test-00000-of-00001 (2).parquet")
    frame = pd.read_parquet(SOURCE / filename)
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    if limit:
        indices = indices[:limit]
    rows = []
    for index in indices:
        item = frame.iloc[index]
        gold = canonical_number(str(item.answer))
        if gold is None:
            raise ValueError(f"cannot parse gold answer at {split}:{index}")
        rows.append({
            "id": f"gsm8k_{split}_{index:05d}",
            "source_index": index,
            "split": split,
            "question": str(item.question).strip(),
            "gold_solution": str(item.answer),
            "gold_final": gold,
        })
    return rows


def read_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed[row["id"]] = row
    return completed


def prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem. Show your reasoning "
        "step by step. End your response with the final numeric answer in "
        "exactly this format: #### <number>\n\nProblem:\n" + question
    )


def generate(args: argparse.Namespace, rows: list[dict], all_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
    except (AttributeError, ImportError):
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval()

    completed = read_completed(all_path) if args.resume else {}
    pending = [row for row in rows if row["id"] not in completed]
    mode = "a" if args.resume and all_path.exists() else "w"
    with all_path.open(mode) as output:
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(row["question"])}],
                tokenize=False, add_generation_prompt=True) for row in batch]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                kwargs.update(temperature=args.temperature, top_p=args.top_p)
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            prompt_len = encoded.input_ids.shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_len:],
                                           skip_special_tokens=True)
            for row, text in zip(batch, texts):
                predicted = canonical_number(text)
                result = {
                    **row,
                    "generation": text.strip(),
                    "predicted_final": predicted,
                    "parse_ok": predicted is not None,
                    "correct": predicted == row["gold_final"],
                    "generation_tokens": len(tokenizer.encode(
                        text, add_special_tokens=False)),
                    "model": args.model,
                    "decoding": {
                        "temperature": args.temperature,
                        "top_p": args.top_p if args.temperature > 0 else None,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                completed[row["id"]] = result
            wrong = sum(not row["correct"] for row in completed.values())
            parsed = sum(row["parse_ok"] for row in completed.values())
            print(f"completed={len(completed)}/{len(rows)} "
                  f"parsed={parsed} natural_wrong={wrong}", flush=True)
            if args.target_wrong and wrong >= args.target_wrong:
                break


def write_derivatives(all_path: Path, out_dir: Path) -> None:
    rows = list(read_completed(all_path).values())
    errors = [row for row in rows if row["parse_ok"] and not row["correct"]]
    parse_failures = [row for row in rows if not row["parse_ok"]]
    for filename, subset in (("natural_errors.jsonl", errors),
                             ("parse_failures.jsonl", parse_failures)):
        with (out_dir / filename).open("w") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n_generated": len(rows),
        "n_correct": sum(row["correct"] for row in rows),
        "n_natural_errors": len(errors),
        "n_parse_failures": len(parse_failures),
        "accuracy_among_parsed": (
            sum(row["correct"] for row in rows) / (len(rows) - len(parse_failures))
            if len(rows) != len(parse_failures) else None),
        "all_generations": str(all_path),
        "natural_errors": str(out_dir / "natural_errors.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="maximum questions to attempt; 0 means the full split")
    parser.add_argument("--target-wrong", type=int, default=300,
                        help="stop after collecting this many natural errors; 0 disables")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.out_dir / "generations.jsonl"
    rows = load_rows(args.split, args.seed, args.limit)
    generate(args, rows, all_path)
    write_derivatives(all_path, args.out_dir)


if __name__ == "__main__":
    main()
