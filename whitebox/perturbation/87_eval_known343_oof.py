#!/usr/bin/env python3
"""Frozen PCA8 perturbation detector on the 343 probe-perfect items."""
import glob, json
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = "/home/tong56/whitebox/perturbation/runs"
SOURCE = f"{ROOT}/79_strict_known_n770.jsonl"
ORACLE = f"{ROOT}/79_oracle_top11_known770.jsonl"
CACHE = f"{ROOT}/79_hidden_delta_top11_known770"
REPORT = f"{ROOT}/87_known343_oof_report.json"
PREDS = f"{ROOT}/87_known343_oof_predictions.jsonl"
DIM = 8


def main():
    all_src = {x["key"]: x for x in map(json.loads, open(SOURCE))}
    src = {
        key: row
        for key, row in all_src.items()
        if row["knowledge_binary_accuracy"] == 1.0
        and row["knowledge_pairwise_owner_accuracy"] == 1.0
    }
    oracle = {x["key"]: x for x in map(json.loads, open(ORACLE))}
    rows = []
    for path in sorted(glob.glob(CACHE + "/*.npz")):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            if key not in src:
                continue
            o = oracle[key]
            ua = np.asarray(o["u"], np.float32)
            u = np.asarray(z["top_u"], np.float32)
            s = float(o["S0"])
            margin = np.r_[
                u,
                np.abs(u),
                u / (abs(s) + 1e-6),
                u.max(initial=0),
                u.min(initial=0),
                np.abs(u).mean(),
                np.abs(u).sum() / (np.abs(ua).sum() + 1e-9),
                np.mean(ua > 0),
                np.std(ua),
            ]
            hidden = np.asarray(z["answer_last"], np.float32)[0]
            h0 = hidden[0]
            delta = hidden[1:] - h0

            def weighted(mask, weight):
                if not mask.any():
                    return np.zeros(4096, np.float32)
                return (delta[mask] * weight[mask, None]).sum(0) / (
                    np.abs(weight[mask]).sum() + 1e-9
                )

            rows.append(
                (
                    key,
                    src[key]["group"],
                    int(src[key]["correct"]),
                    margin,
                    h0,
                    weighted(u > 0, u),
                    weighted(u < 0, -u),
                )
            )

    assert len(rows) == 343, len(rows)
    keys = np.array([x[0] for x in rows])
    groups = np.array([x[1] for x in rows])
    y = np.array([x[2] for x in rows])
    margins = np.stack([x[3] for x in rows])
    hidden_blocks = [np.stack([x[i] for x in rows]) for i in (4, 5, 6)]

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    p_full = np.zeros(len(y))
    p_margin = np.zeros(len(y))
    folds = np.zeros(len(y), int)
    fold_rows = []
    for fold, (train, test) in enumerate(cv.split(margins, y, groups), 1):
        margin_scaler = StandardScaler().fit(margins[train])
        margin_train = margin_scaler.transform(margins[train])
        margin_test = margin_scaler.transform(margins[test])
        train_parts = [margin_train]
        test_parts = [margin_test]
        explained = []
        for block in hidden_blocks:
            scaler = StandardScaler().fit(block[train])
            z_train = scaler.transform(block[train])
            pca = PCA(
                DIM, whiten=True, svd_solver="randomized", random_state=42
            ).fit(z_train)
            train_parts.append(pca.transform(z_train))
            test_parts.append(pca.transform(scaler.transform(block[test])))
            explained.append(float(pca.explained_variance_ratio_.sum()))
        x_train = np.concatenate(train_parts, 1)
        x_test = np.concatenate(test_parts, 1)
        clf = LogisticRegression(
            C=0.5, max_iter=5000, class_weight="balanced", random_state=42
        ).fit(x_train, y[train])
        p_full[test] = clf.predict_proba(x_test)[:, 1]
        baseline = LogisticRegression(
            C=0.5, max_iter=5000, class_weight="balanced", random_state=42
        ).fit(margin_train, y[train])
        p_margin[test] = baseline.predict_proba(margin_test)[:, 1]
        folds[test] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train),
                "test_n": len(test),
                "test_correct": int(y[test].sum()),
                "groups": len(set(groups[test])),
                "auroc": float(roc_auc_score(y[test], p_full[test])),
                "auprc": float(average_precision_score(y[test], p_full[test])),
                "explained_variance": explained,
            }
        )

    def metrics(prob):
        pred = prob >= 0.5
        return {
            "auroc": float(roc_auc_score(y, prob)),
            "auprc": float(average_precision_score(y, prob)),
            "accuracy_at_0.5": float(accuracy_score(y, pred)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
            "confusion_tn_fp_fn_tp": confusion_matrix(
                y, pred, labels=[0, 1]
            ).ravel().tolist(),
        }

    by_group = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    group_names = list(by_group)
    rng = np.random.default_rng(20260811)
    aucs, auprcs, balanced, lifts = [], [], [], []
    for _ in range(5000):
        indices = np.concatenate(
            [by_group[g] for g in rng.choice(group_names, len(group_names), replace=True)]
        )
        if len(np.unique(y[indices])) < 2:
            continue
        full_auc = roc_auc_score(y[indices], p_full[indices])
        aucs.append(full_auc)
        auprcs.append(average_precision_score(y[indices], p_full[indices]))
        balanced.append(
            balanced_accuracy_score(y[indices], p_full[indices] >= 0.5)
        )
        lifts.append(full_auc - roc_auc_score(y[indices], p_margin[indices]))

    report = {
        "protocol": "probe-perfect subset only; frozen top11/layer16/PCA8/C0.5; 5-fold StratifiedGroupKFold OOF",
        "subset_rule": "knowledge_binary_accuracy == 1.0 and knowledge_pairwise_owner_accuracy == 1.0",
        "n": len(y),
        "correct": int(y.sum()),
        "incorrect": int(len(y) - y.sum()),
        "groups": len(set(groups)),
        "final_dims": 63,
        "full": metrics(p_full),
        "margin_only": metrics(p_margin),
        "folds": fold_rows,
        "group_bootstrap_95ci": {
            "auroc": np.quantile(aucs, [0.025, 0.975]).tolist(),
            "auprc": np.quantile(auprcs, [0.025, 0.975]).tolist(),
            "balanced_accuracy": np.quantile(balanced, [0.025, 0.975]).tolist(),
            "auroc_lift_over_margin": np.quantile(lifts, [0.025, 0.975]).tolist(),
        },
        "auroc_lift_over_margin_point": float(
            roc_auc_score(y, p_full) - roc_auc_score(y, p_margin)
        ),
    }
    with open(REPORT, "w") as handle:
        json.dump(report, handle, indent=2)
    with open(PREDS, "w") as handle:
        for i in range(len(y)):
            handle.write(
                json.dumps(
                    {
                        "key": str(keys[i]),
                        "group": str(groups[i]),
                        "correct": int(y[i]),
                        "fold": int(folds[i]),
                        "prob_full": float(p_full[i]),
                        "prob_margin": float(p_margin[i]),
                    }
                )
                + "\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
