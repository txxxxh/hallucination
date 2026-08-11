#!/usr/bin/env python3
"""Paired original/swap profile evaluation through the OpenAI Responses API."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import profile_perturbation_unsupervised as pp
from profile_order_generation_check import parse_choice

HERE = Path(__file__).resolve().parent
API_URL = "https://api.openai.com/v1/responses"


def extract_text(response):
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


def request_one(api_key, model, prompt, retries, timeout):
    payload = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
        "store": False,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        req = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                body = json.loads(res.read().decode("utf-8"))
            return extract_text(body), body.get("id"), body.get("usage", {})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            delay = min(60.0, (2 ** attempt) + random.random())
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            time.sleep(min(60.0, (2 ** attempt) + random.random()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.2-2025-12-11")
    ap.add_argument("--key-file", type=Path, default=Path("/home/tong56/.openai_api_key"))
    ap.add_argument("--data", type=Path, default=pp.DEFAULT_DATA)
    ap.add_argument("--output", type=Path,
                    default=HERE / "profile_order_gpt52_api_output")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl = args.output / "items.jsonl"
    api_key = args.key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("Empty API key file")

    raw = json.loads(args.data.read_text(encoding="utf-8"))[:args.limit]
    tasks, metadata = [], {}
    for row in raw:
        item = pp.parse_item(row)
        conditions = {c.name: c.prompt for c in pp.build_conditions(item)}
        key = str(row["key"])
        names = [p.name for p in item.profiles]
        right = names.index(item.right_answer)
        metadata[key] = {"names": names, "right_index": right}
        tasks.extend([(key, "original", conditions["full_context"]),
                      (key, "swapped", conditions["profile_order_swap"])])

    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["key"], row["order"])] = row
    pending = [x for x in tasks if (x[0], x[1]) not in done]
    lock = threading.Lock()
    completed = 0

    def work(task):
        key, order, prompt = task
        text, response_id, usage = request_one(
            api_key, args.model, prompt, args.retries, args.timeout)
        names = metadata[key]["names"]
        return {"key": key, "order": order, "names": names,
                "right_index": metadata[key]["right_index"], "output": text,
                "choice": parse_choice(text, names), "response_id": response_id,
                "usage": usage, "model": args.model}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            with lock:
                with jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[(row["key"], row["order"])] = row
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    print(f"[{completed}/{len(pending)}] stored={len(done)}/{len(tasks)}",
                          flush=True)

    pairs = []
    for key in metadata:
        a, b = done.get((key, "original")), done.get((key, "swapped"))
        if not a or not b or a["choice"] is None or b["choice"] is None:
            continue
        pairs.append((a, b))
    n = len(pairs)
    cc = sum(a["choice"] == a["right_index"] and b["choice"] == b["right_index"]
             for a, b in pairs)
    cw = sum(a["choice"] == a["right_index"] and b["choice"] != b["right_index"]
             for a, b in pairs)
    wc = sum(a["choice"] != a["right_index"] and b["choice"] == b["right_index"]
             for a, b in pairs)
    ww = sum(a["choice"] != a["right_index"] and b["choice"] != b["right_index"]
             for a, b in pairs)
    usage = {}
    for row in done.values():
        for k, v in row.get("usage", {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    summary = {
        "model": args.model, "n_requested": len(metadata), "calls_requested": len(tasks),
        "calls_completed": len(done), "paired_valid": n,
        "original_accuracy": (cc + cw) / n if n else None,
        "swapped_accuracy": (cc + wc) / n if n else None,
        "identity_consistency": sum(a["choice"] == b["choice"] for a, b in pairs) / n
                                if n else None,
        "identity_changed": sum(a["choice"] != b["choice"] for a, b in pairs),
        "four_cells": {"correct_correct": cc, "correct_wrong": cw,
                       "wrong_correct": wc, "wrong_wrong": ww},
        "net_accuracy_change_count": wc - cw, "aggregate_usage": usage,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
