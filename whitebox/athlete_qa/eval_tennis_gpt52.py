#!/usr/bin/env python3
"""Evaluate the 200 TennisQA name-choice items with the OpenAI Responses API."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, random, re, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def norm(s):
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def match_name(text, correct, wrong):
    t, c, w = norm(text), norm(correct), norm(wrong)
    hc, hw = c in t, w in t
    if hc and not hw: return "correct"
    if hw and not hc: return "wrong"
    if t == c: return "correct"
    if t == w: return "wrong"
    return "unmatched"


def response_text(body):
    if body.get("output_text"):
        return body["output_text"]
    parts = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def call(row, model, retries=8):
    payload = json.dumps({
        "model": model,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
        "input": [{"role": "user", "content": row["prepend_names_prompt"] +
                   "\nOutput only the person's name."}],
    }).encode()
    request = Request("https://api.openai.com/v1/responses", data=payload,
                      headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                               "Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=120) as response:
                body = json.load(response)
            text = response_text(body).strip()
            outcome = match_name(text, row["correct_answer"], row["wrong_answer"])
            return {"id": row["id"], "correct_answer": row["correct_answer"],
                    "wrong_answer": row["wrong_answer"], "generation": text,
                    "name_outcome": outcome, "name_correct": outcome == "correct",
                    "response_id": body.get("id"), "usage": body.get("usage", {})}
        except HTTPError as error:
            message = error.read().decode(errors="replace")
            if error.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"HTTP {error.code}: {message[:500]}") from error
        except (URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
        time.sleep(min(30, (2 ** attempt) + random.random()))
    raise RuntimeError("unreachable")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path(__file__).resolve().parent / "pilot_v1/primary_questions.jsonl")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "pilot_v1/gpt52_eval")
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    rows = [json.loads(x) for x in a.data.open() if x.strip()][:a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "results.jsonl"
    done = {}
    if a.resume and path.exists():
        done = {x["id"]: x for x in map(json.loads, path.open())}
    pending = [x for x in rows if x["id"] not in done]
    mode = "a" if a.resume and path.exists() else "w"
    with path.open(mode) as handle, concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(call, row, a.model): row for row in pending}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            done[result["id"]] = result
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{len(done)}/{len(rows)}] {result['id']} {result['name_outcome']}", flush=True)
    results = [done[x["id"]] for x in rows]
    usage = {key: sum(x.get("usage", {}).get(key, 0) or 0 for x in results)
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    summary = {"model": a.model, "n_items": len(results),
               "correct": sum(x["name_correct"] for x in results),
               "accuracy": sum(x["name_correct"] for x in results) / len(results),
               "wrong": sum(x["name_outcome"] == "wrong" for x in results),
               "unmatched": sum(x["name_outcome"] == "unmatched" for x in results),
               "usage": usage}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
