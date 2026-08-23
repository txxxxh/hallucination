#!/usr/bin/env python3
"""Build per-backbone copies of the frozen clean GSM8K n=942 manifest.

The responses and labels are the parse-valid, natural-error-balanced Llama
manifest that produced the audited 0.743 exact AUROC.  Only the ``model`` field
differs between copies so the strict collector can evaluate every backbone on
the identical response-detection benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = {
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = [json.loads(line) for line in args.source.open() if line.strip()]
    if len(source) != 942:
        raise RuntimeError(f"expected 942 rows, found {len(source)}")
    if sum(int(row["correct"]) for row in source) != 471:
        raise RuntimeError("source is not the frozen 471/471 balanced manifest")
    if len({row["key"] for row in source}) != len(source):
        raise RuntimeError("duplicate keys in source manifest")

    for short_name, model_id in MODELS.items():
        output = args.output_root / short_name / "gsm8k.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for row in source:
                manifest_row = {
                    "key": row["key"],
                    "group": row.get("group", row["key"]),
                    "correct": int(row["correct"]),
                    "context": row["question"],
                    "question": "Provide the complete solution to this math problem.",
                    "pred": row["generation"],
                    "other": row["reference_solution"],
                    "prompt_mode": False,
                    "model": model_id,
                }
                handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
        print(f"{output}: n=942, correct=471, error=471")


if __name__ == "__main__":
    main()
