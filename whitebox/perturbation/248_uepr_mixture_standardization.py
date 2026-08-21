#!/usr/bin/env python3
"""Standardize detector balanced accuracy to a common U/R composition."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/247_uepr_conditioned_detector_audit/items.jsonl"
OUT = HERE / "runs/248_uepr_mixture_standardization"
AXES = ("U", "R")
BENCHMARKS = ("scientist", "trivia", "gsm8k")
METHODS = ("exact", "attention")


def metrics(rows, method, reference):
    observed, standardized, rates, composition = [], [], {}, {}
    for y in (0, 1):
        z = [r for r in rows if r["error"] == y]
        hit = np.array([(r[method] >= .5) == bool(y) for r in z], float)
        observed.append(float(hit.mean()))
        rates[str(y)] = {}; composition[str(y)] = {}
        value = 0.0
        for axis in AXES:
            mask = np.array([r["dominant_axis"] == axis for r in z])
            rate = float(hit[mask].mean())
            rates[str(y)][axis] = rate
            composition[str(y)][axis] = float(mask.mean())
            value += reference[str(y)][axis] * rate
        standardized.append(value)
    obs, std = float(np.mean(observed)), float(np.mean(standardized))
    return {"observed_balanced_accuracy": obs,
            "pooled_composition_standardized_balanced_accuracy": std,
            "composition_contribution": obs - std,
            "class_axis_accuracy": rates, "class_axis_composition": composition}


def main():
    rows = [json.loads(x) for x in SOURCE.open()]
    reference = {str(y): {axis: (sum(r["error"] == y and r["dominant_axis"] == axis for r in rows) /
                                sum(r["error"] == y for r in rows)) for axis in AXES}
                 for y in (0, 1)}
    report = {"protocol": ("post-stratification of threshold-0.5 balanced accuracy; "
                           "common reference is pooled U/R-dominant composition separately within "
                           "correct and error classes; 5000 stratified item bootstraps"),
              "reference_composition": reference, "results": {}}
    rng = np.random.default_rng(20260820)
    for benchmark in BENCHMARKS:
        z = [r for r in rows if r["benchmark"] == benchmark]
        report["results"][benchmark] = {}
        by_class = {y: [r for r in z if r["error"] == y] for y in (0, 1)}
        for method in METHODS:
            result = metrics(z, method, reference)
            boot = []
            for _ in range(5000):
                sample = []
                for y in (0, 1):
                    source = by_class[y]
                    sample += [source[i] for i in rng.integers(0, len(source), len(source))]
                try:
                    boot.append(metrics(sample, method, reference)["composition_contribution"])
                except (ValueError, ZeroDivisionError):
                    pass
            result["composition_contribution_ci95"] = [float(x) for x in np.quantile(boot, [.025, .975])]
            report["results"][benchmark][method] = result
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
