#!/usr/bin/env python3
"""Retrospective item-level U/R-conditioned audit of Llama detectors.

This intentionally uses only axes with near-complete, ID-aligned coverage on
Scientist, TriviaQA, and GSM8K.  It does not treat E/P selected subsets as
benchmark prevalence estimates.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FEATURES = RUNS / "paper4_self_matrix_v2/features/llama"
OUT = RUNS / "247_uepr_conditioned_detector_audit"
BASE = importlib.import_module("159_evaluate_paper4_matrix")
SEEDS = (42, 43, 44)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def axis_maps(dataset):
    if dataset == "scientist":
        rows = read_jsonl(RUNS / "226_four_axis_taxonomy_audit/items.jsonl")
        return {r["key"]: (r["u_score"], r["r_score"], r["error"]) for r in rows}
    if dataset == "trivia":
        u = {r["key"]: r for r in read_jsonl(RUNS / "232_trivia_u_split_confirmation/samples.jsonl")}
        rr = {r["key"]: r for r in read_jsonl(RUNS / "238_trivia_question_end_r_confirmation/items.jsonl")}
        return {k: (u[k]["u_score"], rr[k]["r_score"], u[k]["greedy_error"])
                for k in u.keys() & rr.keys()}
    u = {r["key"]: r for r in read_jsonl(RUNS / "236_gsm8k_u_split_confirmation/items.jsonl")}
    rr = {r["key"]: r for r in read_jsonl(RUNS / "233_gsm8k_question_end_r_confirmation/items.jsonl")}
    return {k: (u[k]["u_score"], rr[k]["r_score"], u[k]["greedy_error"])
            for k in u.keys() & rr.keys()}


def oof(data, dataset):
    y, groups = data["y"], data["groups"]
    probabilities = []
    for seed in SEEDS:
        p = np.zeros(len(y), float)
        if dataset == "trivia":
            splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(data["scalar"], y)
        else:
            splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(data["scalar"], y, groups)
        for tr, te in splits:
            a, b = BASE.transform_fold(BASE.subset(data, tr), BASE.subset(data, te), seed)
            m = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                   solver="liblinear", random_state=seed).fit(a, y[tr])
            p[te] = m.predict_proba(b)[:, 1]
        probabilities.append(p)
    return np.mean(probabilities, axis=0)


def auc(y_error, score):
    return float(roc_auc_score(y_error, score)) if len(np.unique(y_error)) == 2 else None


def percentile(x):
    return (rankdata(x, method="average") - .5) / len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"protocol": ("Llama; exact/attention mean of seeds 42/43/44 grouped 5-fold OOF; "
                           "item-ID join to independently produced U/R confirmation artifacts; "
                           "within-benchmark percentile axes; E/P excluded because coverage is selected"),
              "benchmarks": {}}
    pooled = []
    for dataset in ("scientist", "trivia", "gsm8k"):
        axes = axis_maps(dataset)
        methods = {}
        base_data = None
        for method in ("exact", "attention"):
            data = BASE.load_directory(FEATURES / dataset / method)
            if base_data is None:
                base_data = data
            methods[method] = dict(zip(data["keys"], 1.0 - oof(data, dataset)))
        rows = []
        labels_by_key = dict(zip(base_data["keys"], 1 - base_data["y"]))
        for key in sorted(axes.keys() & methods["exact"].keys() & methods["attention"].keys()):
            u, r, error = axes[key]
            if int(error) != int(labels_by_key[key]):
                continue
            rows.append({"key": key, "error": int(error), "u": float(u), "r": float(r),
                         "exact": float(methods["exact"][key]),
                         "attention": float(methods["attention"][key])})
        u = np.array([x["u"] for x in rows]); r = np.array([x["r"] for x in rows])
        y = np.array([x["error"] for x in rows]); up, rp = percentile(u), percentile(r)
        dominant = np.where(up >= rp, "U", "R")
        rec = {"n": len(rows), "errors": int(y.sum()),
               "coverage_vs_detector": len(rows) / len(base_data["y"]),
               "axis_error_auroc": {"U": auc(y, u), "R": auc(y, r)},
               "detector_auroc": {}, "high_axis_error_concentration": {},
               "dominant_axis_counts_among_errors": {}, "conditional_detector_auroc": {},
               "axis_detector_loss_spearman": {}}
        for axis, values in (("U", up), ("R", rp)):
            high = values >= .7
            rec["high_axis_error_concentration"][axis] = {
                "high_n": int(high.sum()), "high_error_rate": float(y[high].mean()),
                "share_of_errors_high": float(high[y == 1].sum() / y.sum())}
        for method in ("exact", "attention"):
            score = np.array([x[method] for x in rows])
            rec["detector_auroc"][method] = auc(y, score)
            loss = -(y*np.log(np.clip(score,1e-7,1-1e-7)) +
                     (1-y)*np.log(np.clip(1-score,1e-7,1-1e-7)))
            rec["axis_detector_loss_spearman"][method] = {
                "U": float(spearmanr(up, loss).statistic),
                "R": float(spearmanr(rp, loss).statistic)}
            rec["conditional_detector_auroc"][method] = {
                axis: auc(y[dominant == axis], score[dominant == axis]) for axis in ("U", "R")}
        rec["dominant_axis_counts_among_errors"] = {
            axis: int(np.sum((dominant == axis) & (y == 1))) for axis in ("U", "R")}
        report["benchmarks"][dataset] = rec
        for i, row in enumerate(rows):
            pooled.append({**row, "benchmark": dataset, "u_percentile": float(up[i]),
                           "r_percentile": float(rp[i]), "dominant_axis": dominant[i]})
    with (OUT / "items.jsonl").open("w") as f:
        for row in pooled:
            f.write(json.dumps(row) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
