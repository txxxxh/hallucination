#!/usr/bin/env python3
"""Targeted second-round feature ablations and cross-cache verification."""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
from pathlib import Path

import numpy as np


ablation = importlib.import_module("161_current127_feature_ablation")
RUNS = Path(__file__).resolve().parent / "runs"

ablation.CURVES.update({
    "stage1_pred16": np.arange(0, 16),
    "stage1_other16": np.arange(16, 32),
    "stage1_no_stats22": np.r_[0:11, 16:27],
    "stage1_stats10": np.r_[11:16, 27:32],
    "stage1_norm12": np.r_[0, 6:11, 16, 22:27],
})


def targeted_configs():
    configs = []

    def add(name, curve="stage1_full32", hidden=(0, 1, 2, 3), hdim=8, ldim=48):
        configs.append(dict(name=name, curve=curve, hidden=list(hidden),
                            hdim=hdim, ldim=ldim))

    add("stage1_allh_l48")
    for curve in ("stage1_core12", "stage1_pred16", "stage1_other16",
                  "stage1_no_stats22", "stage1_stats10", "stage1_norm12",
                  "none0"):
        add("curve_" + curve, curve=curve)
    for size in range(5):
        for subset in itertools.combinations(range(4), size):
            label = "".join(map(str, subset)) or "none"
            add("stage1_hidden_" + label, hidden=subset,
                hdim=8 if subset else 0)
    for hdim in (4, 6, 8):
        for ldim in (36, 40, 44, 48):
            if hdim == 8 and ldim == 48:
                continue
            add(f"stage1_h{hdim}_l{ldim}", hdim=hdim, ldim=ldim)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=RUNS / "159_scientist_classgrad_sentence_current127")
    parser.add_argument("--output", type=Path,
                        default=RUNS / "162_current127_targeted_ablation.json")
    args = parser.parse_args()
    ablation.configs = targeted_configs
    report = ablation.run(args.cache)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "results": report["results"][:15]}, indent=2))


if __name__ == "__main__":
    main()
