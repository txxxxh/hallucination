#!/usr/bin/env python3
"""Run the multidomain closed-book probes with GPT-5.2."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://api.openai.com/v1/responses"
DOMAINS = ("athlete", "musician", "building")


def response_text(body):
    if body.get("output_text"):
        return body["output_text"]
    return "".join(
        content.get("text", "")
        for output in body.get("output", [])
        for content in output.get("content", [])
        if content.get("type") == "output_text"
    )


def parse_yes_no(text):
    match = re.match(r"\s*(yes|no)\b", text, re.I)
    return None if match is None else match.group(1).lower() == "yes"


def request_one(task, model, retries, timeout):
    domain, row, probe_index, probe = task
    prompt = (
        "Answer from your own factual knowledge, without external sources. "
        "Answer exactly Yes or No.\nQuestion: " + probe["question"]
    )
    payload = json.dumps({
        "model": model,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 16,
        "store": False,
        "input": [{"role": "user", "content": prompt}],
    }).encode()
    for attempt in range(retries + 1):
        request = Request(API, data=payload, headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            generation = response_text(body).strip()
            pred_yes = parse_yes_no(generation)
            return {
                "domain": domain, "id": row["id"], "field": row["decisive_relation"]["field"],
                "probe_index": probe_index, "question": probe["question"],
                "correct_answer": probe["correct_answer"], "generation": generation,
                "pred_yes": pred_yes,
                "correct": pred_yes is not None and pred_yes == bool(probe["correct_answer"]),
                "response_id": body.get("id"), "usage": body.get("usage", {}),
            }
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {error.code}: {detail[:500]}") from error
        except (URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(min(60, 2 ** attempt + random.random()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "multidomain_v5")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "multidomain_v5/gpt52_probe_eval")
    parser.add_argument("--model", default="gpt-5.2-2025-12-11")
    parser.add_argument("--key-file", type=Path, default=Path("/home/tong56/.openai_api_key"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY") and args.key_file.exists():
        os.environ["OPENAI_API_KEY"] = args.key_file.read_text().strip()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    tasks = []
    for domain in args.domains:
        rows = [json.loads(line) for line in open(args.root / domain / "primary_questions.jsonl")]
        for row in rows[:args.limit or None]:
            tasks.extend((domain, row, index, probe) for index, probe in enumerate(row["probes"]))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "probe_results.jsonl"
    done = {}
    if args.resume and path.exists():
        for result in map(json.loads, path.open()):
            done[(result["domain"], result["id"], result["probe_index"])] = result
    pending = [task for task in tasks if (task[0], task[1]["id"], task[2]) not in done]
    lock = threading.Lock()
    mode = "a" if args.resume and path.exists() else "w"
    with path.open(mode) as handle, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(request_one, task, args.model, args.retries, args.timeout): task for task in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            key = (result["domain"], result["id"], result["probe_index"])
            with lock:
                done[key] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"[{index}/{len(pending)}] stored={len(done)}/{len(tasks)}", flush=True)

    grouped = defaultdict(list)
    for result in done.values():
        grouped[(result["domain"], result["id"])].append(result)
    items = []
    for (domain, item_id), probes in sorted(grouped.items()):
        probes.sort(key=lambda result: result["probe_index"])
        n_correct = sum(result["correct"] for result in probes)
        state = "knows_both" if n_correct == 2 else "knows_one" if n_correct == 1 else "knows_neither"
        items.append({"domain": domain, "id": item_id, "field": probes[0]["field"],
                      "probe_state": state, "n_probes_correct": n_correct, "probes": probes})
    with (args.out / "results.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def stats(group):
        probes = [probe for item in group for probe in item["probes"]]
        return {"n_items": len(group), "n_probes": len(probes),
                "probe_accuracy": sum(probe["correct"] for probe in probes) / len(probes),
                "unparsed": sum(probe["pred_yes"] is None for probe in probes),
                "knowledge_states": {state: sum(item["probe_state"] == state for item in group)
                                     for state in ("knows_both", "knows_one", "knows_neither")}}
    usage = {key: sum(result.get("usage", {}).get(key, 0) or 0 for result in done.values())
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    summary = {"model": args.model, "calls": len(done), "overall": stats(items),
               "by_domain": {domain: stats([item for item in items if item["domain"] == domain])
                             for domain in args.domains}, "usage": usage}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
