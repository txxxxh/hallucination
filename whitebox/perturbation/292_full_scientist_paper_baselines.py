#!/usr/bin/env python3
"""Paper-faithful baselines on the frozen 2,894-row Scientist population.

The canonical keys are exactly those evaluated by experiment 273 (SAPLMA).
This prevents the newer all-row helper (currently 2,902 rows) from silently
changing the population.  Method implementations and hyperparameters are
delegated unchanged to experiments 265, 266, and 284.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "292_full_scientist_paper_baselines"
CANONICAL = RUNS / "273_full_scientist_saplma_paper" / "predictions.jsonl"


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def canonical_rows(_dataset="scientist"):
    keys = {str(row["key"]) for row in read(CANONICAL)}
    rows = importlib.import_module(
        "100_collect_multilayer_trajectory")._scientist_rows("all")
    rows = [row for row in rows if str(row["key"]) in keys]
    if len(keys) != 2894 or len(rows) != 2894:
        raise RuntimeError(f"canonical alignment failed: keys={len(keys)} rows={len(rows)}")
    if len({str(row['key']) for row in rows}) != 2894:
        raise RuntimeError("duplicate canonical Scientist keys")
    return rows


def prepare_semantic():
    source = RUNS / "269_full_scientist_semantic_entropy" / "scientist" / "samples.jsonl"
    target = OUT / "semantic_entropy" / "scientist" / "samples.jsonl"
    keys = {str(row["key"]) for row in canonical_rows()}
    samples = [row for row in read(source) if str(row["key"]) in keys]
    if len(samples) != 2894 or len({str(x['key']) for x in samples}) != 2894:
        raise RuntimeError(f"semantic sample alignment failed: {len(samples)}/2894")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in samples))
    print(f"prepared {len(samples)} canonical semantic-entropy samples at {target}")


def semantic_score():
    prepare_semantic()
    method = importlib.import_module("266_semantic_entropy_paper")
    args = argparse.Namespace(dataset="scientist", out=OUT / "semantic_entropy")
    method.score(args)


def icr(stage, resume):
    method = importlib.import_module("265_icr_probe_paper")
    method.rows = canonical_rows
    args = argparse.Namespace(dataset="scientist", resume=resume, out=OUT / "icr_probe")
    (method.collect if stage == "collect" else method.evaluate)(args)


def selfcheck(stage, resume, batch, nli_batch, seed):
    method = importlib.import_module("284_selfcheckgpt_nli_paper")
    method.base.rows = canonical_rows
    args = argparse.Namespace(dataset="scientist", resume=resume, batch=batch,
                              nli_batch=nli_batch, seed=seed,
                              cache=Path("/tmp/selfcheckgpt_hf"),
                              out=OUT / "selfcheckgpt_nli")
    (method.sample if stage == "sample" else method.score)(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["audit", "semantic_score", "icr_collect",
                                         "icr_evaluate", "selfcheck_sample",
                                         "selfcheck_score"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nli-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.task == "audit":
        print(json.dumps({"canonical_rows": len(canonical_rows()),
                          "canonical_file": str(CANONICAL)}, indent=2))
    elif args.task == "semantic_score":
        semantic_score()
    elif args.task.startswith("icr_"):
        icr(args.task.removeprefix("icr_"), args.resume)
    else:
        selfcheck(args.task.removeprefix("selfcheck_"), args.resume,
                  args.batch, args.nli_batch, args.seed)


if __name__ == "__main__":
    main()
