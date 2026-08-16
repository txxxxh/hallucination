#!/usr/bin/env python3
"""Ablate per-span |u| and u/|score| copies from candidate features."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent; RUNS = HERE/"runs"


def candidate_channel(score, u):
    return np.r_[score, u, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def met(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def main():
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, groups, y, margin, hidden, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    cached = {}
    for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            cached[str(z["key"].item())] = np.r_[
                candidate_channel(z["pred_scores"][0], z["pred_u"]),
                candidate_channel(z["other_scores"][0], z["other_u"])]
    separate = np.stack([cached[k] for k in keys]).astype(np.float32)
    # Original layout: signed u[0:5], abs(u)[5:10], normalized u[10:15], summaries[15:].
    margin_pruned = np.c_[margin[:, :5], margin[:, 10:]]
    variants = ("margin_original_scalar", "margin_pruned_scalar", "separate_pruned_scalar",
                "margin_original_full", "margin_pruned_full", "separate_pruned_full",
                "margin_pruned_plus_separate_full")
    scores = {name: [] for name in variants}
    for seed in (42, 43, 44):
        prediction = {name: np.zeros(len(y)) for name in variants}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(margin, y, groups), 1):
            def scale(values):
                scaler = StandardScaler().fit(values[train])
                return scaler.transform(values[train]), scaler.transform(values[test])
            mt, mv = scale(margin); mpt, mpv = scale(margin_pruned); st, sv = scale(separate)
            htr, hte = [], []
            for values in hidden:
                scaler = StandardScaler().fit(values[train]); q = scaler.transform(values[train])
                pca = PCA(12, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
                htr.append(pca.transform(q)); hte.append(pca.transform(scaler.transform(values[test])))
            values = last[:, 3]; scaler = StandardScaler().fit(values[train]); q = scaler.transform(values[train])
            pca = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
            common_train = np.concatenate([*htr, pca.transform(q)], 1)
            common_test = np.concatenate([*hte, pca.transform(scaler.transform(values[test]))], 1)
            sets = {
                "margin_original_scalar": (mt, mv), "margin_pruned_scalar": (mpt, mpv),
                "separate_pruned_scalar": (st, sv),
                "margin_original_full": (np.c_[mt, common_train], np.c_[mv, common_test]),
                "margin_pruned_full": (np.c_[mpt, common_train], np.c_[mpv, common_test]),
                "separate_pruned_full": (np.c_[st, common_train], np.c_[sv, common_test]),
                "margin_pruned_plus_separate_full":
                    (np.c_[mpt, st, common_train], np.c_[mpv, sv, common_test])}
            for name, (xtr, xte) in sets.items():
                clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                         solver="liblinear", random_state=seed).fit(xtr, y[train])
                prediction[name][test] = clf.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for name in variants: scores[name].append(met(y, prediction[name]))
    results = []
    for name, values in scores.items():
        results.append({"variant": name, **{f"mean_{k}": float(np.mean([v[k] for v in values]))
                                             for k in values[0]}, "per_seed": values})
    results.sort(key=lambda x: x["mean_auroc"], reverse=True)
    report = {"protocol": "Scientist question-grouped 3x5-fold OOF", "dimensions": {
        "margin_original": int(margin.shape[1]), "margin_pruned": int(margin_pruned.shape[1]),
        "separate_pruned": int(separate.shape[1])}, "results": results}
    path = RUNS/"115_drop_normalized_candidate_detector.json"; path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
