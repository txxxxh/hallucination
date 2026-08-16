#!/usr/bin/env python3
"""Evaluate a model on fixed, small GSM8K train/test pools without touching run 136."""
import argparse
import importlib
import json
from pathlib import Path


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-n", type=int, default=300)
    parser.add_argument("--test-n", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    base = importlib.import_module("136_build_eval_gsm8k_mc")
    for split, n, seed in (("train", args.train_n, 42), ("test", args.test_n, 43)):
        items = base.build(split, n, seed)
        results = base.evaluate(items, args.model, args.batch)
        dump(args.out / f"{split}.jsonl", items)
        dump(args.out / f"{split}_results.jsonl", results)
        print(split, {"n": len(items), "correct": sum(x["name_correct"] for x in results)})


if __name__ == "__main__":
    main()
