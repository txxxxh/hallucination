#!/usr/bin/env python3
"""Run the paper-standard SelfCheckGPT-NLI pipeline on Scientist Full."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def full_rows(_dataset):
    return importlib.import_module(
        "100_collect_multilayer_trajectory"
    )._scientist_rows("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sample", "score"))
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--nli-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/selfcheckgpt_hf"))
    parser.add_argument(
        "--out", type=Path,
        default=RUNS / "316_full_scientist_p_selfchecknli",
    )
    args = parser.parse_args()
    method = importlib.import_module("284_selfcheckgpt_nli_paper")
    method.base.rows = full_rows
    run_args = argparse.Namespace(
        dataset="scientist", batch=args.batch, nli_batch=args.nli_batch,
        seed=args.seed, resume=args.resume, cache=args.cache, out=args.out,
    )
    (method.sample if args.stage == "sample" else method.score)(run_args)


if __name__ == "__main__":
    main()
