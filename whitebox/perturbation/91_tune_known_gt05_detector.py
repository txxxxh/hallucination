#!/usr/bin/env python3
"""Grouped-OOF hyperparameter sweep for the 1,084-item hidden-delta detector.

This is a model-selection experiment on the existing data, not an independent
test estimate.  All scaling and PCA fits happen inside each CV training fold.
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent / "runs"


def load_rows(source: Path, oracle_path: Path, cache: Path, topks: list[int]):
    src = {x["key"]: x for x in map(json.loads, source.open())}
    oracle = {x["key"]: x for x in map(json.loads, oracle_path.open())}
    rows = []
    for path in sorted(glob.glob(str(cache / "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            o = oracle[key]
            all_u = np.asarray(o["u"], np.float32)
            top_u = np.asarray(z["top_u"], np.float32)
            s0 = float(o["S0"])
            states = {}
            for answer_name in ("answer_last", "answer_mean"):
                h = np.asarray(z[answer_name], np.float32)[0]
                h0, delta = h[0], h[1:] - h[0]
                for k in topks:
                    u, d = top_u[:k], delta[:k]

                    def wm(mask, weights):
                        if not mask.any():
                            return np.zeros(d.shape[1], np.float32)
                        return (d[mask] * weights[mask, None]).sum(0) / (np.abs(weights[mask]).sum() + 1e-9)

                    pos, neg = wm(u > 0, u), wm(u < 0, -u)
                    signed = (d * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)
                    absolute = (d * np.abs(u[:, None])).sum(0) / (np.abs(u).sum() + 1e-9)
                    margin = np.r_[
                        u, np.abs(u), u / (abs(s0) + 1e-6), u.max(initial=0),
                        u.min(initial=0), np.abs(u).mean(),
                        np.abs(u).sum() / (np.abs(all_u).sum() + 1e-9),
                        np.mean(all_u > 0), np.std(all_u),
                    ].astype(np.float32)
                    states[(answer_name, k)] = (margin, h0, pos, neg, signed, absolute)
            rows.append((key, str(src[key]["group"]), int(src[key]["correct"]), states))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "88_known_gt05_n1084.jsonl")
    p.add_argument("--oracle", type=Path, default=ROOT / "88_oracle_top11_known_gt05.jsonl")
    p.add_argument("--cache", type=Path, default=ROOT / "88_hidden_delta_top11_known_gt05")
    p.add_argument("--report", type=Path, default=ROOT / "91_known_gt05_hparam_search.json")
    p.add_argument("--topks", type=int, nargs="+", default=[3, 5, 8, 11])
    p.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24])
    p.add_argument("--Cs", type=float, nargs="+", default=[.003, .01, .03, .1, .3, 1., 3., 10.])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()

    rows = load_rows(args.source, args.oracle, args.cache, args.topks)
    assert len(rows) == 1084, len(rows)
    y = np.asarray([r[2] for r in rows])
    groups = np.asarray([r[1] for r in rows])
    configs = list(product(("answer_last", "answer_mean"), args.topks,
                           ("three_block", "delta_only", "signed", "absolute"),
                           args.dims, args.Cs))
    scores = {cfg: [] for cfg in configs}

    for seed in args.seeds:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        predictions = {cfg: np.zeros(len(y), np.float32) for cfg in configs}
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups), 1):
            for answer_name, k in product(("answer_last", "answer_mean"), args.topks):
                packed = [r[3][(answer_name, k)] for r in rows]
                margin = np.stack([x[0] for x in packed])
                hidden = [np.stack([x[i] for x in packed]) for i in range(1, 6)]
                ms = StandardScaler().fit(margin[tr])
                mt, mv = ms.transform(margin[tr]), ms.transform(margin[te])
                projected = []
                max_dim = max(args.dims)
                for block in hidden:
                    scaler = StandardScaler().fit(block[tr])
                    zt = scaler.transform(block[tr])
                    pc = PCA(max_dim, whiten=True, svd_solver="randomized", random_state=seed).fit(zt)
                    projected.append((pc.transform(zt), pc.transform(scaler.transform(block[te]))))
                modes = {
                    "three_block": [0, 1, 2],
                    "delta_only": [1, 2],
                    "signed": [3],
                    "absolute": [4],
                }
                for mode, dim, C in product(modes, args.dims, args.Cs):
                    ids = modes[mode]
                    xtr = np.concatenate([mt] + [projected[i][0][:, :dim] for i in ids], axis=1)
                    xte = np.concatenate([mv] + [projected[i][1][:, :dim] for i in ids], axis=1)
                    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                             solver="liblinear", random_state=seed)
                    clf.fit(xtr, y[tr])
                    cfg = (answer_name, k, mode, dim, C)
                    predictions[cfg][te] = clf.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for cfg, prob in predictions.items():
            scores[cfg].append({
                "auroc": float(roc_auc_score(y, prob)),
                "auprc": float(average_precision_score(y, prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y, prob >= .5)),
            })

    results = []
    for cfg, vals in scores.items():
        result = dict(zip(("answer", "topk", "mode", "pca", "C"), cfg))
        for metric in ("auroc", "auprc", "balanced_accuracy"):
            v = np.asarray([x[metric] for x in vals])
            result[f"mean_{metric}"] = float(v.mean())
            result[f"std_{metric}"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        result["per_seed"] = vals
        results.append(result)
    results.sort(key=lambda x: (x["mean_auroc"], x["mean_auprc"]), reverse=True)
    report = {
        "warning": "Hyperparameters selected on these OOF scores; best score is selection-biased, not an independent test estimate.",
        "n": len(rows), "groups": len(set(groups)), "seeds": args.seeds,
        "search_space": {"topks": args.topks, "dims": args.dims, "Cs": args.Cs,
                         "answers": ["answer_last", "answer_mean"],
                         "modes": ["three_block", "delta_only", "signed", "absolute"]},
        "n_configs": len(configs), "top_50": results[:50], "all_results": results,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(args.report), "n_configs": len(configs), "top_10": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
