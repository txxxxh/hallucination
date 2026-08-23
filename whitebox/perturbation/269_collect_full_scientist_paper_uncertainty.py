#!/usr/bin/env python3
"""Resume the paper-standard U baselines on all parse-valid Scientist rows."""
from __future__ import annotations
import argparse, importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

def full_rows(_dataset):
    return importlib.import_module("100_collect_multilayer_trajectory")._scientist_rows("all")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["k6", "semantic_sample", "semantic_score"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260822)
    a = p.parse_args()
    if a.method == "k6":
        m = importlib.import_module("261_paper_baseline_matrix")
        m.rows = full_rows
        ns = argparse.Namespace(dataset="scientist", batch=a.batch, seed=a.seed,
            resume=True, out=RUNS/"269_full_scientist_k6")
        m.collect(ns)
    else:
        m = importlib.import_module("266_semantic_entropy_paper")
        m.rows = full_rows
        ns = argparse.Namespace(dataset="scientist", batch=a.batch, seed=a.seed,
            resume=True, out=RUNS/"269_full_scientist_semantic_entropy")
        (m.sample if a.method == "semantic_sample" else m.score)(ns)

if __name__ == "__main__": main()
