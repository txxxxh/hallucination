#!/usr/bin/env python3
"""Leakage-safe score stacking of P and SelfCheckGPT-NLI on Scientist Full."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


base = importlib.import_module("272_full_scientist_standard_upr_tables")
RUNS = base.RUNS
SEEDS = (42, 43, 44, 45, 46, 47)
DIMS = (None, 8, 8, 8, 8, 48)


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def metrics(y, score):
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def transform(blocks, train, test, seed):
    left, right = [], []
    for values, dim in zip(blocks, DIMS):
        scaler = StandardScaler().fit(values[train])
        x, z = scaler.transform(values[train]), scaler.transform(values[test])
        if dim is not None:
            pca = PCA(
                min(dim, len(train) - 1, x.shape[1]), whiten=True,
                svd_solver="randomized", random_state=seed,
            ).fit(x)
            x, z = pca.transform(x), pca.transform(z)
        left.append(x)
        right.append(z)
    return np.concatenate(left, axis=1), np.concatenate(right, axis=1)


def p_predict(blocks, y, train, test, seed):
    x, z = transform(blocks, train, test, seed)
    model = LogisticRegression(
        C=.03, max_iter=5000, class_weight="balanced",
        solver="liblinear", random_state=seed,
    ).fit(x, y[train])
    return model.predict_proba(z)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        default=RUNS / "316_full_scientist_p_selfchecknli" / "fusion",
    )
    args = parser.parse_args()
    rows = base.load()
    keys = [row["key"] for row in rows]
    y = np.asarray([row["error"] for row in rows])
    blocks = [np.stack([row["p_scalar"] for row in rows])]
    blocks += [np.stack([row["p_hidden"][j] for row in rows]) for j in range(4)]
    blocks += [np.stack([row["p_layer"] for row in rows])]
    nli_rows = {row["key"]: row for row in read(args.scores)}
    if set(keys) != set(nli_rows):
        common = len(set(keys) & set(nli_rows))
        raise RuntimeError(
            f"key mismatch P={len(keys)} NLI={len(nli_rows)} common={common}"
        )
    nli = np.asarray([nli_rows[key]["score"] for key in keys])
    indices = np.arange(len(y))
    reports, predictions = [], []
    for seed in SEEDS:
        dev, test = map(np.asarray, train_test_split(
            indices, test_size=.2, stratify=y, random_state=seed
        ))
        oof = np.zeros(len(dev))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for inner_train, inner_val in cv.split(dev, y[dev]):
            oof[inner_val] = p_predict(
                blocks, y, dev[inner_train], dev[inner_val], seed
            )
        p_test = p_predict(blocks, y, dev, test, seed)
        stacker = LogisticRegression(
            C=1, max_iter=2000, class_weight="balanced", random_state=seed
        ).fit(np.c_[oof, nli[dev]], y[dev])
        fused = stacker.predict_proba(np.c_[p_test, nli[test]])[:, 1]
        report = {
            "seed": seed,
            "P": metrics(y[test], p_test),
            "SelfCheckGPT_NLI": metrics(y[test], nli[test]),
            "P_plus_SelfCheckGPT_NLI": metrics(y[test], fused),
            "stack_coefficients": stacker.coef_[0].tolist(),
            "stack_intercept": float(stacker.intercept_[0]),
        }
        reports.append(report)
        predictions.extend({
            "seed": seed, "key": keys[i], "error": int(y[i]),
            "p_error": float(p), "nli_error": float(nli[i]),
            "fused_error": float(f),
        } for i, p, f in zip(test, p_test, fused))
        print(seed, report, flush=True)
    methods = ("P", "SelfCheckGPT_NLI", "P_plus_SelfCheckGPT_NLI")
    result = {
        "dataset": "scientist_full", "n": len(y),
        "protocol": "outer stratified 80/20 seeds42-47; P trained on outer dev; stacker trained on 5-fold OOF P scores within dev; untouched outer test",
        "per_seed": reports,
        "summary": {
            method: {
                metric + "_mean": float(np.mean([
                    row[method][metric] for row in reports
                ])) for metric in ("auroc", "auprc")
            } for method in methods
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
