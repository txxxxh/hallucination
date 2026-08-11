#!/usr/bin/env python3
"""Select the frozen detector's PCA dimension using train-only group CV."""

import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = "/home/tong56/whitebox/perturbation/runs"
MANIFEST = f"{ROOT}/73_split_n700_train600_test100.json"
ORACLE = f"{ROOT}/73_oracle_top11_n700.jsonl"
CACHE = f"{ROOT}/73_hidden_delta_top11_n700"
OUT = f"{ROOT}/74_pca_dim_selection_train600.json"
DIMS = (8, 12, 16, 24, 32, 48, 64)


def load_features():
    manifest = {x["key"]: x for x in json.load(open(MANIFEST))}
    oracle = {}
    for line in open(ORACLE):
        row = json.loads(line)
        oracle[row["key"]] = row

    rows = []
    for path in sorted(glob.glob(f"{CACHE}/*.npz")):
        with np.load(path, allow_pickle=True) as data:
            key = str(data["key"].item())
            o = oracle[key]
            all_u = np.asarray(o["u"], np.float32)
            u = np.asarray(data["top_u"], np.float32)
            s0 = float(o["S0"])
            margin = np.concatenate(
                [
                    u,
                    np.abs(u),
                    u / (abs(s0) + 1e-6),
                    np.asarray(
                        [
                            u.max(initial=0),
                            u.min(initial=0),
                            np.abs(u).mean(),
                            np.abs(u).sum() / (np.abs(all_u).sum() + 1e-9),
                            np.mean(all_u > 0),
                            np.std(all_u),
                        ],
                        np.float32,
                    ),
                ]
            )
            hidden = np.asarray(data["answer_last"], np.float32)[0]
            h0 = hidden[0]
            delta = hidden[1:] - h0

            def weighted(mask, weights):
                if not mask.any():
                    return np.zeros(4096, np.float32)
                return (delta[mask] * weights[mask, None]).sum(0) / (
                    np.abs(weights[mask]).sum() + 1e-9
                )

            rows.append(
                (
                    key,
                    manifest[key]["split"],
                    manifest[key]["group"],
                    int(manifest[key]["correct"]),
                    margin,
                    h0,
                    weighted(u > 0, u),
                    weighted(u < 0, -u),
                )
            )
    assert len(rows) == 700
    return rows


def transform_fold(margin, hidden, fit_idx, eval_idx):
    margin_scaler = StandardScaler().fit(margin[fit_idx])
    fit_parts = [margin_scaler.transform(margin[fit_idx])]
    eval_parts = [margin_scaler.transform(margin[eval_idx])]
    for branch in hidden:
        scaler = StandardScaler().fit(branch[fit_idx])
        zfit = scaler.transform(branch[fit_idx])
        pca = PCA(n_components=max(DIMS), whiten=True, svd_solver="randomized", random_state=42)
        pca.fit(zfit)
        fit_parts.append(pca.transform(zfit))
        eval_parts.append(pca.transform(scaler.transform(branch[eval_idx])))
    return fit_parts, eval_parts


def join(parts, dim):
    return np.concatenate([parts[0]] + [x[:, :dim] for x in parts[1:]], axis=1)


def main():
    rows = load_features()
    split = np.asarray([x[1] for x in rows])
    groups = np.asarray([x[2] for x in rows])
    y = np.asarray([x[3] for x in rows])
    margin = np.stack([x[4] for x in rows])
    hidden = [np.stack([x[i] for x in rows]) for i in (5, 6, 7)]
    train_global = np.flatnonzero(split == "train")
    test_global = np.flatnonzero(split == "test")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = defaultdict(list)
    for fold, (fit_local, val_local) in enumerate(
        cv.split(train_global, y[train_global], groups[train_global]), 1
    ):
        fit_idx, val_idx = train_global[fit_local], train_global[val_local]
        fit_parts, val_parts = transform_fold(margin, hidden, fit_idx, val_idx)
        for dim in DIMS:
            clf = LogisticRegression(
                C=0.5, max_iter=5000, class_weight="balanced", random_state=42
            ).fit(join(fit_parts, dim), y[fit_idx])
            prob = clf.predict_proba(join(val_parts, dim))[:, 1]
            scores[dim].append(
                {
                    "fold": fold,
                    "n": len(val_idx),
                    "auroc": roc_auc_score(y[val_idx], prob),
                    "auprc": average_precision_score(y[val_idx], prob),
                    "balanced_accuracy": balanced_accuracy_score(y[val_idx], prob >= 0.5),
                }
            )
        print(f"finished fold {fold}/5", flush=True)

    summary = {}
    for dim in DIMS:
        summary[str(dim)] = {
            "final_feature_dims": 39 + 3 * dim,
            "folds": scores[dim],
            "mean_auroc": float(np.mean([x["auroc"] for x in scores[dim]])),
            "std_auroc": float(np.std([x["auroc"] for x in scores[dim]], ddof=1)),
            "mean_auprc": float(np.mean([x["auprc"] for x in scores[dim]])),
            "mean_balanced_accuracy": float(
                np.mean([x["balanced_accuracy"] for x in scores[dim]])
            ),
        }
    selected = max(DIMS, key=lambda d: (summary[str(d)]["mean_auroc"], -d))

    fit_parts, test_parts = transform_fold(margin, hidden, train_global, test_global)
    clf = LogisticRegression(
        C=0.5, max_iter=5000, class_weight="balanced", random_state=42
    ).fit(join(fit_parts, selected), y[train_global])
    prob = clf.predict_proba(join(test_parts, selected))[:, 1]
    test_result = {
        "n": len(test_global),
        "auroc": roc_auc_score(y[test_global], prob),
        "auprc": average_precision_score(y[test_global], prob),
        "balanced_accuracy_at_0.5": balanced_accuracy_score(
            y[test_global], prob >= 0.5
        ),
    }
    report = {
        "selection_protocol": "5-fold StratifiedGroupKFold on the 600 training examples only",
        "pca_solver": "randomized",
        "candidate_dims_per_hidden_branch": list(DIMS),
        "cv": summary,
        "selected_dim_per_hidden_branch": selected,
        "selected_final_feature_dims": 39 + 3 * selected,
        "heldout_test_reference": test_result,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
