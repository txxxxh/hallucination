#!/usr/bin/env python3
"""Tune a deployable P + raw uncertainty-trajectory model on the frozen pilot.

Unlike 286, this does not consume the full-population OOF U score.  Every
transform and classifier is fit on the fixed 500-item pilot, then applied once
to its disjoint complement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/281_scientist_stagewise_ur_pilot128"
MANIFEST = HERE / "runs/76_closedbook_fact_probe_manifest.jsonl"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def label_invariant_channels(u: np.ndarray) -> np.ndarray:
    """Keep standard uncertainty; make right/wrong bins permutation invariant."""
    if u.ndim != 3 or u.shape[1:] != (4, 5):
        raise ValueError(f"expected [n,4,5] uncertainty array, got {u.shape}")
    a, b, invalid = u[:, :, 2], u[:, :, 3], u[:, :, 4]
    probs = np.stack([a, b, invalid], axis=-1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)
    return np.stack(
        [u[:, :, 0], u[:, :, 1], invalid, np.maximum(a, b), np.minimum(a, b),
         np.abs(a - b), a + b, entropy], axis=-1
    )


def trajectory_features(u: np.ndarray) -> np.ndarray:
    """Turn four states x safe uncertainty channels into trajectory features."""
    u = label_invariant_channels(u)
    # State meanings: original, neutralized, deleted, deleted+neutralized.
    base_delta = u[:, 1:] - u[:, [0]]
    step_delta = np.diff(u, axis=1)
    curvature = np.diff(u, n=2, axis=1)
    channel_stats = np.concatenate(
        [u.mean(1), u.std(1), u.min(1), u.max(1), np.ptp(u, axis=1)], axis=1
    )
    # Two interaction contrasts: effect of neutralization before/after deletion,
    # and deletion before/after neutralization.
    interactions = np.concatenate(
        [(u[:, 3] - u[:, 2]) - (u[:, 1] - u[:, 0]),
         (u[:, 3] - u[:, 1]) - (u[:, 2] - u[:, 0])], axis=1
    )
    return np.concatenate(
        [u.reshape(len(u), -1), base_delta.reshape(len(u), -1),
         step_delta.reshape(len(u), -1), curvature.reshape(len(u), -1),
         channel_stats, interactions], axis=1
    )


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def candidates():
    out = {}
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        out[f"logistic_C={c:g}"] = make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                               solver="liblinear"),
        )
    for leaves in (7, 15):
        out[f"histgb_leaves={leaves}"] = make_pipeline(
            SimpleImputer(),
            HistGradientBoostingClassifier(max_iter=150, learning_rate=.04,
                                           max_leaf_nodes=leaves,
                                           min_samples_leaf=20,
                                           l2_regularization=3.0,
                                           random_state=287),
        )
    return out


def main():
    pilot_keys = {x["key"] for x in read_jsonl(RUN / "predictions.jsonl")}
    full = read_jsonl(RUN / "predictions_uniform_2894.jsonl")
    p_by_key = {x["key"]: float(x["P"]) for x in full}
    y_by_key = {x["key"]: int(x["error"]) for x in full}
    groups = {x["key"]: str(x["right_qid"]) for x in read_jsonl(MANIFEST)}

    rows = []
    for path in sorted(RUN.glob("question_*.npz")):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            if key not in p_by_key:
                continue
            rows.append((key, z["uncertainty"].astype(np.float64)))
    keys = np.asarray([x[0] for x in rows])
    u = np.stack([x[1] for x in rows])
    y = np.asarray([y_by_key[k] for k in keys])
    p = np.asarray([p_by_key[k] for k in keys])
    group = np.asarray([groups[k] for k in keys])
    train = np.asarray([k in pilot_keys for k in keys])
    test = ~train
    if train.sum() != 500 or test.sum() != 2394:
        raise RuntimeError(f"unexpected split: pilot={train.sum()}, holdout={test.sum()}")

    # P enters on a stable log-odds scale; all remaining columns are raw
    # trajectory-derived quantities and are transformed inside each fold.
    tf = trajectory_features(u)
    x = np.c_[logit(np.clip(p, 1e-6, 1 - 1e-6)), tf]
    train_idx = np.flatnonzero(train)
    cv_rows = []
    models = candidates()
    for name, model in models.items():
        seed_scores = []
        for seed in (42, 43, 44):
            pred = np.full(train.sum(), np.nan)
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            for fit_local, val_local in cv.split(x[train], y[train], group[train]):
                fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
                model.fit(x[fit_idx], y[fit_idx])
                pred[val_local] = model.predict_proba(x[val_idx])[:, 1]
            seed_scores.append(pred)
        score = np.mean(seed_scores, axis=0)
        cv_rows.append({"model": name, **metrics(y[train], score)})

    # Model choice is made exclusively from pilot grouped-CV AUROC.
    selected = max(cv_rows, key=lambda z: (z["auroc"], z["auprc"]))["model"]
    final_model = models[selected]
    final_model.fit(x[train], y[train])
    score = final_model.predict_proba(x)[:, 1]

    train_groups, test_groups = set(group[train]), set(group[test])
    report = {
        "protocol": "label-invariant four-state trajectory features (original-answer disagreement retained; right/wrong bins permutation-invariant); model selected by 3x5 grouped CV on frozen pilot n=500; refit on pilot only; evaluated once on disjoint remainder n=2394",
        "feature_count_including_P": int(x.shape[1]),
        "pilot_holdout_group_overlap": len(train_groups & test_groups),
        "pilot_cv_candidates": sorted(cv_rows, key=lambda z: z["auroc"], reverse=True),
        "selected_model": selected,
        "pilot_descriptive_refit": {"P": metrics(y[train], p[train]),
                                    "P_plus_trajectory": metrics(y[train], score[train])},
        "holdout": {"P": metrics(y[test], p[test]),
                    "P_plus_trajectory": metrics(y[test], score[test])},
    }

    rng = np.random.default_rng(287)
    test_idx = np.flatnonzero(test)
    boot = []
    for _ in range(10000):
        idx = rng.choice(test_idx, len(test_idx), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append([roc_auc_score(y[idx], score[idx]) - roc_auc_score(y[idx], p[idx]),
                     average_precision_score(y[idx], score[idx]) - average_precision_score(y[idx], p[idx])])
    boot = np.asarray(boot)
    report["holdout_paired_bootstrap_10000_delta_vs_P_ci95"] = {
        "auroc": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "auprc": np.quantile(boot[:, 1], [.025, .975]).tolist(),
    }

    (RUN / "report_pu_raw_trajectory.json").write_text(json.dumps(report, indent=2) + "\n")
    with (RUN / "predictions_pu_raw_trajectory.jsonl").open("w") as handle:
        for i, key in enumerate(keys):
            handle.write(json.dumps({"key": key, "error": int(y[i]), "split": "pilot" if train[i] else "holdout", "P": float(p[i]), "P_plus_trajectory": float(score[i])}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
