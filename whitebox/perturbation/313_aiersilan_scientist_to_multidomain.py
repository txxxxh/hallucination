#!/usr/bin/env python3
"""Frozen Aiersilan transfer: Scientist known/full -> multidomain v6.

All preprocessing and classifier fitting is source-only.  Multidomain labels
are used only once, after prediction, to compute the reported metrics.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "313_aiersilan_scientist_to_multidomain"
SOURCE = RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"
TARGET_ROOT = HERE.parent / "athlete_qa" / "multidomain_v6_fixed500_musician_opt"
TARGET_CACHE = TARGET_ROOT / "current127_llama_known_both"
SEEDS = (42, 43, 44)
LAYER = 14


def metric(y, score):
    pred = score >= .5
    return {
        "n": int(len(y)), "correct": int(y.sum()),
        "incorrect": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "accuracy_at_0.5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
        "confusion_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def source_data():
    saved = torch.load(SOURCE, map_location="cpu")
    x = saved["hidden_states"][:, LAYER].float().numpy()
    y = saved["labels"].numpy().astype(int)  # Aiersilan convention: correct=1
    keys = np.asarray(saved["keys"])
    known_keys = {
        row["key"] for row in map(json.loads, (RUNS / "88_known_gt05_n1084.jsonl").open())
    }
    known = np.asarray([key in known_keys for key in keys])
    if len(y) != 2894:
        raise RuntimeError(f"unexpected Scientist full size: {len(y)}")
    if known.sum() == 1084:
        known_data = (x[known], y[known], keys[known])
    else:
        # Eight known rows are outside the parse-valid full set. Reuse the
        # official extractor/fill path instead of silently using only 1,076.
        split = importlib.import_module("312_aiersilan_split_only_known_full")
        split.fill_missing()
        kx, ky_error, _ = split.datasets()["known1084"]
        known_rows = importlib.import_module(
            "100_collect_multilayer_trajectory")._scientist_rows("known")
        known_data = (kx, 1 - ky_error, np.asarray([r["key"] for r in known_rows]))
    if len(known_data[1]) != 1084:
        raise RuntimeError(f"unexpected Scientist known size: {len(known_data[1])}")
    return {"scientist_known": known_data, "scientist_full": (x, y, keys)}


def target_data():
    rows = importlib.import_module("150_multidomain_v6_scientist_frozen_transfer").rows()
    values, kept = [], []
    for row in rows:
        path = TARGET_CACHE / f"{row['key']}.npz"
        if not path.exists():
            raise RuntimeError(f"missing target feature: {path}")
        with np.load(path, allow_pickle=True) as z:
            values.append(z["layer14"].astype(np.float32))
        kept.append(row)
    return np.stack(values), np.asarray([r["correct"] for r in kept]), kept


def evaluate(name, source, tx, ty, target_rows):
    sx, sy, source_keys = source
    target_keys = {r["key"] for r in target_rows}
    overlap = target_keys.intersection(source_keys.tolist())
    if overlap:
        raise RuntimeError(f"source/target key overlap: {len(overlap)}")
    seed_scores = []
    for seed in SEEDS:
        scaler = StandardScaler().fit(sx)
        a, b = scaler.transform(sx), scaler.transform(tx)
        pca = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
        a, b = pca.transform(a), pca.transform(b)
        clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=seed).fit(a, sy)
        seed_scores.append(clf.predict_proba(b)[:, 1])
    score = np.mean(seed_scores, axis=0)
    masks = {"all": np.ones(len(ty), dtype=bool)}
    for domain in ("athlete", "musician", "building"):
        masks[domain] = np.asarray([r["domain"] == domain for r in target_rows])
    subsets = {}
    for subset, mask in masks.items():
        subsets[subset] = metric(ty[mask], score[mask])
        subsets[subset]["per_seed_auroc"] = [
            float(roc_auc_score(ty[mask], s[mask])) for s in seed_scores
        ]
    predictions = [{"source": name, "key": row["key"], "domain": row["domain"],
                    "field": row["field"], "correct": int(label),
                    "prob_correct": float(prob)}
                   for row, label, prob in zip(target_rows, ty, score)]
    return {"source": name, "source_n": int(len(sy)),
            "source_correct": int(sy.sum()), "target_n": int(len(ty)),
            "source_target_key_overlap": 0, "subsets": subsets}, predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tx, ty, target_rows = target_data()
    reports = {}
    all_predictions = []
    for name, source in source_data().items():
        report, predictions = evaluate(name, source, tx, ty, target_rows)
        reports[name] = report
        all_predictions.extend(predictions)
    output = {
        "protocol": "strict task split; frozen Scientist-only training and multidomain-only testing",
        "representation": "Aiersilan generated-candidate last-token hidden state, layer 14",
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "classifier": "source-only StandardScaler + PCA48 (whitened) + balanced LR C=.03",
        "seeds": list(SEEDS),
        "target": "multidomain_v6_fixed500_musician_opt; probe_state=knows_both; unmatched excluded",
        "target_labels_used_for_fitting_or_tuning": False,
        "results": reports,
    }
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    with (OUT / "predictions.jsonl").open("w") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
