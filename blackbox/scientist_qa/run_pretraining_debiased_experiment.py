#!/usr/bin/env python3
"""Run a paired original-vs-paraphrased ScientistQA experiment.

Example:
  OPENAI_API_KEY=... python scientist_qa/run_pretraining_debiased_experiment.py \
      --model gpt-5-mini --reasoning-effort low --repetitions 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "question" / "pretraining_debiased_first100_question.json"
DEFAULT_OUTPUT = ROOT / "benchmark_result" / "gpt" / "pretraining_debiased_first100_v2_results.json"

SYSTEM_PROMPT = """## Task: Answer the Biographical Identification Question

You will be given a short biographical paragraph describing a person.
Identify the person and provide a confidence score from 0 to 100.
You must choose exactly one of the two names supplied in the question.

Return only a JSON object with this schema:
{"answer": "full candidate name", "confidence": 0}
Do not provide explanations or any other text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--condition", choices=("both", "original", "paraphrased"), default="both")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in model output: {text!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        raise ValueError(f"Invalid response object: {value!r}")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    return {"answer": value["answer"].strip(), "confidence": confidence}


def call_model(client: Any, args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": 512,
    }
    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(**request)
            raw = response.choices[0].message.content or ""
            return parse_response(raw), raw
        except Exception:
            if attempt + 1 == args.max_retries:
                raise
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"metadata": {}, "results": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError(f"Existing output has an invalid schema: {path}")
    return value


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_calls": len(rows), "by_condition": {}}
    for condition in ("original", "paraphrased"):
        subset = [row for row in rows if row["condition"] == condition]
        if subset:
            correct = sum(bool(row["correct"]) for row in subset)
            summary["by_condition"][condition] = {
                "n": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
                "hallucinations": len(subset) - correct,
            }

    # A paired flip is defined only for calls having the same item/repetition.
    indexed = {(r["source_index"], r["repetition"], r["condition"]): r for r in rows}
    flips = {"wrong_to_right": 0, "right_to_wrong": 0, "unchanged_right": 0, "unchanged_wrong": 0}
    paired = 0
    for index, repetition, condition in list(indexed):
        if condition != "original":
            continue
        original = indexed[(index, repetition, "original")]
        revised = indexed.get((index, repetition, "paraphrased"))
        if revised is None:
            continue
        paired += 1
        pair = (bool(original["correct"]), bool(revised["correct"]))
        flips[{(False, True): "wrong_to_right", (True, False): "right_to_wrong", (True, True): "unchanged_right", (False, False): "unchanged_wrong"}[pair]] += 1
    if paired:
        summary["paired_n"] = paired
        summary["paired_outcomes"] = flips
        summary["accuracy_change"] = (
            summary["by_condition"]["paraphrased"]["accuracy"]
            - summary["by_condition"]["original"]["accuracy"]
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.repetitions < 1 or args.start < 0 or (args.limit is not None and args.limit < 1):
        raise SystemExit("repetitions and limit must be positive; start must be nonnegative")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running the experiment.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the SDK with: python -m pip install openai") from exc

    client_args: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_args["base_url"] = args.base_url
    client = OpenAI(**client_args)

    input_bytes = args.input.read_bytes()
    dataset_sha256 = hashlib.sha256(input_bytes).hexdigest()
    questions = json.loads(input_bytes)
    questions = questions[args.start : None if args.limit is None else args.start + args.limit]
    payload = load_existing(args.output)
    old_meta = payload.get("metadata", {})
    if payload["results"]:
        old_settings = (old_meta.get("model"), old_meta.get("reasoning_effort"))
        if old_settings != (args.model, args.reasoning_effort):
            raise SystemExit("Output already contains results for different model settings; choose another --output.")
        if old_meta.get("dataset_sha256") != dataset_sha256:
            raise SystemExit("Output contains results for a different dataset revision; choose another --output.")
    payload["metadata"] = {
        "input": str(args.input.resolve()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "dataset_sha256": dataset_sha256,
        "rewrite_version": questions[0].get("rewrite_version") if questions else None,
        "system_prompt": SYSTEM_PROMPT,
    }
    completed = {
        (row["source_index"], row["repetition"], row["condition"])
        for row in payload["results"]
    }

    conditions = ["original", "paraphrased"] if args.condition == "both" else [args.condition]
    jobs = [(q, rep, condition) for q in questions for rep in range(args.repetitions) for condition in conditions]
    random.Random(args.seed).shuffle(jobs)
    for number, (question, repetition, condition) in enumerate(jobs, 1):
        key = (question["source_index"], repetition, condition)
        if key in completed:
            continue
        prompt = question["original_prompt"] if condition == "original" else question["prompt"]
        parsed, raw = call_model(client, args, prompt)
        candidates = [question["rgt_ans"], question["wrg_ans"]]
        answer_key = canonical(parsed["answer"])
        matched = next((name for name in candidates if canonical(name) == answer_key), None)
        row = {
            "source_index": question["source_index"],
            "key": question["key"],
            "repetition": repetition,
            "condition": condition,
            "answer": parsed["answer"],
            "confidence": parsed["confidence"],
            "matched_candidate": matched,
            "gold_answer": question["rgt_ans"],
            "correct": matched == question["rgt_ans"],
            "raw_response": raw,
        }
        payload["results"].append(row)
        payload["summary"] = summarize(payload["results"])
        save(args.output, payload)
        print(f"[{number}/{len(jobs)}] {question['key']} {condition}: {parsed['answer']} ({'correct' if row['correct'] else 'wrong'})", flush=True)

    payload["summary"] = summarize(payload["results"])
    save(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
