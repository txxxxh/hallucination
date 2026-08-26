#!/usr/bin/env python3
"""Matched-split P, Aiersilan-R, and P+R evaluation on three benchmarks."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "293_multibench_p_aiersilan_fusion"
SEEDS = (42, 43, 44)
LAYER = 14


def fixed(scores, width=6):
    """Match the Scientist feature contract: base score plus five spans."""
    scores = np.asarray(scores, dtype=np.float32)
    return np.pad(scores[:width], (0, max(0, width - len(scores))))


def ch(scores):
    scores = fixed(scores)
    u = scores[0] - scores[1:]
    scale = abs(float(scores[0])) + 1e-6
    return np.r_[scores[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(scores):
    scores = fixed(scores)
    return np.r_[scores[0], scores[0] - scores[1:]]


def wd(hidden, effect):
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * effect[:, None]).sum(0) / (np.abs(effect).sum() + 1e-9)


def p_row(path, full):
    with np.load(path, allow_pickle=True) as z:
        p, o = z["stage1_pred"].astype(np.float32), z["stage1_other"].astype(np.float32)
        ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
        scalar = np.r_[ch(p), ch(o)]
        if full:
            q, r = z["stage2_pred"].astype(np.float32), z["stage2_other"].astype(np.float32)
            scalar = np.r_[scalar, ch2(q), ch2(r), p[0] - q[0], o[0] - r[0],
                           (p[0] - o[0]) - (q[0] - r[0])]
        hidden = (ph[0], wd(ph, p[0] - p[1:]), oh[0], wd(oh, o[0] - o[1:]))
        return str(z["key"].item()), int(z["correct"]), scalar, hidden, z["layer14"].astype(np.float32)


def load_p(dataset):
    specs = {
        "trivia": (RUNS / "127_trivia1000_current127", False, 1000, 48),
        "gsm8k": (RUNS / "141_gsm8k_natural_current127", True, 942, 48),
        "drop": (RUNS / "167_drop1000_exact", False, 1000, 44),
    }
    cache, full, expected, layer_dim = specs[dataset]
    rows = [p_row(path, full) for path in sorted(cache.glob("*.npz"))]
    if len(rows) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} P rows, got {len(rows)}")
    keys = [row[0] for row in rows]
    y = np.asarray([row[1] for row in rows])
    blocks = [np.stack([row[2] for row in rows])]
    blocks += [np.stack([row[3][j] for row in rows]) for j in range(4)]
    blocks += [np.stack([row[4] for row in rows])]
    return keys, y, blocks, [None, 8, 8, 8, 8, layer_dim]


def transform(blocks, train, test, dims, seed):
    left, right, used = [], [], []
    for values, dim in zip(blocks, dims):
        scaler = StandardScaler().fit(values[train])
        a, b = scaler.transform(values[train]), scaler.transform(values[test])
        actual = None if dim is None else min(dim, len(train) - 1, a.shape[1])
        if actual is not None:
            pca = PCA(actual, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pca.transform(a), pca.transform(b)
        left.append(a); right.append(b); used.append(actual)
    return np.concatenate(left, 1), np.concatenate(right, 1), used


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "balanced_accuracy": float(balanced_accuracy_score(y, score >= .5))}


def evaluate(dataset):
    keys, y, p_blocks, p_dims = load_p(dataset)
    source = OUT / "hidden_states" / f"llama3.1-8b__{dataset}.pt"
    saved = torch.load(source, map_location="cpu")
    rmap = {key: saved["hidden_states"][i, LAYER].float().numpy()
            for i, key in enumerate(saved["keys"])}
    missing = [key for key in keys if key not in rmap]
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} P keys absent from R cache")
    r_blocks = [np.stack([rmap[key] for key in keys])]
    configs = {"P": (p_blocks, p_dims), "R": (r_blocks, [48]),
               "P_plus_R": (p_blocks + r_blocks, p_dims + [48])}
    per_seed = {name: [] for name in configs}; predictions = []
    indices = np.arange(len(y))
    for seed in SEEDS:
        train_val, test = train_test_split(indices, test_size=.2, stratify=y, random_state=seed)
        train, val = train_test_split(train_val, test_size=.1/.8,
                                      stratify=y[train_val], random_state=seed)
        for name, (blocks, dims) in configs.items():
            x, z, used = transform(blocks, train, test, dims, seed)
            model = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                solver="liblinear", random_state=seed).fit(x, y[train])
            score = model.predict_proba(z)[:, 1]
            per_seed[name].append({"seed": seed, **metrics(y[test], score), "dims": used})
            predictions.extend({"dataset": dataset, "seed": seed, "method": name,
                "key": keys[i], "correct": int(y[i]), "score": float(s)}
                for i, s in zip(test, score))
    summary = {name: {metric: float(np.mean([row[metric] for row in rows]))
                       for metric in ("auroc", "auprc", "balanced_accuracy")}
               for name, rows in per_seed.items()}
    return {"dataset": dataset, "n": len(y), "correct": int(y.sum()),
            "split": "stratified 70/10/20; seeds 42,43,44",
            "classifier": "fold-local scaler/PCA; balanced LR C=.03",
            "summary": summary, "per_seed": per_seed}, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=("trivia", "gsm8k", "drop"))
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        report, predictions = evaluate(dataset)
        (OUT / f"{dataset}_report.json").write_text(json.dumps(report, indent=2) + "\n")
        with (OUT / f"{dataset}_predictions.jsonl").open("w") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
