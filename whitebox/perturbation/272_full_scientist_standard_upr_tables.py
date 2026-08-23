#!/usr/bin/env python3
"""Standard U/R/P binary tables on all 2,894 parse-valid Scientist rows.

U is Farquhar et al. probability-weighted Semantic Entropy.  Its binary high
threshold is the label-free 70th percentile fitted on each outer-train fold.
R is the paper-matrix fixed layer-14 last+mean representation protocol.
P is the exact-current127 feature stack used by experiment 153, extended to
the full cache without using closed-book probes as features or routing signals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "272_full_scientist_standard_upr_tables_rightqid"
SEEDS = (42, 43, 44)


def read_jsonl(path):
    return [json.loads(x) for x in path.open() if x.strip()]


def components(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for row in rows:
        union(row["right_qid"], row["wrong_qid"])
    return np.asarray([find(row["right_qid"]) for row in rows])


def fixed(x, n=6):
    x = np.asarray(x, np.float32)
    return np.pad(x[:n], (0, max(0, n-len(x))))


def channel(x):
    x = fixed(x)
    delta = x[0] - x[1:]
    scale = abs(float(x[0])) + 1e-6
    return np.r_[x[0], delta, delta/scale, delta.max(initial=0),
                 delta.min(initial=0), np.abs(delta).mean(), delta.std(),
                 np.mean(delta > 0)]


def channel2(x):
    x = fixed(x)
    return np.r_[x[0], x[0]-x[1:]]


def weighted_hidden(hidden, scores):
    hidden = hidden.astype(np.float32)
    delta = hidden[1:] - hidden[0]
    effect = fixed(scores)[0] - fixed(scores)[1:]
    return (delta * effect[:, None]).sum(0) / (np.abs(effect).sum()+1e-9)


def perturbation_blocks(key):
    full = RUNS / "135_scientist_full_current127" / f"{key}.npz"
    if full.exists():
        with np.load(full, allow_pickle=True) as z:
            p, o = z["stage1_pred"], z["stage1_other"]
            q, r = z["stage2_pred"], z["stage2_other"]
            ph, oh = z["pred_hidden"], z["other_hidden"]
            layer = z["layer14"].astype(np.float32)
    else:
        with np.load(RUNS/"120_physical_delete_rerank"/f"{key}.npz",
                     allow_pickle=True) as z:
            p, o = z["stage1_pred_scores"], z["stage1_other_scores"]
            q, r = z["stage2_pred_scores"], z["stage2_other_scores"]
        with np.load(RUNS/"116_dual_candidate_hidden_top5"/f"{key}.npz",
                     allow_pickle=True) as z:
            ph, oh = z["pred_hidden"], z["other_hidden"]
        with np.load(RUNS/"100_scientist_trajectory_l8"/f"{key}.npz",
                     allow_pickle=True) as z:
            layer = z["mean"].astype(np.float32)[3]
    scalar = np.r_[channel(p), channel(o), channel2(q), channel2(r),
                   p[0]-q[0], o[0]-r[0],
                   (p[0]-o[0])-(q[0]-r[0])].astype(np.float32)
    hidden = [ph[0].astype(np.float32), weighted_hidden(ph, p),
              oh[0].astype(np.float32), weighted_hidden(oh, o)]
    return scalar, hidden, layer


def load():
    records = {x["key"]: x for x in read_jsonl(
        ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl")}
    manifest = {x["key"]: x for x in read_jsonl(
        RUNS/"76_closedbook_fact_probe_manifest.jsonl")}
    entropy = {x["key"]: x for x in read_jsonl(
        RUNS/"269_full_scientist_semantic_entropy"/"scientist"/"scores.jsonl")}
    rows = []
    trajectory = RUNS / "141_scientist_all_trajectory_l8"
    for fp in sorted(trajectory.glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            if (key not in records or key not in manifest or key not in entropy
                    or not records[key].get("parse_valid", True)):
                continue
            layers = z["layers"].astype(int)
            idx = int(np.flatnonzero(layers == 14)[0])
            r_last = z["last"].astype(np.float32)[idx]
            r_mean = z["mean"].astype(np.float32)[idx]
        ps, ph, pl = perturbation_blocks(key)
        rows.append({**manifest[key], "key": key,
                     "error": int(not records[key]["correct"]),
                     "entropy": float(entropy[key]["semantic_entropy"]),
                     "r_last": r_last, "r_mean": r_mean,
                     "p_scalar": ps, "p_hidden": ph, "p_layer": pl})
    if len(rows) != 2894:
        raise RuntimeError(f"aligned parse-valid rows {len(rows)}/2894")
    return rows


def transform_blocks(blocks, train, test, dims, seed):
    a, b = [], []
    used_dims = []
    for values, dim in zip(blocks, dims):
        scaler = StandardScaler().fit(values[train])
        x, z = scaler.transform(values[train]), scaler.transform(values[test])
        if dim is not None:
            used = min(dim, x.shape[0]-1, x.shape[1])
            pca = PCA(used, whiten=True, svd_solver="randomized",
                      random_state=seed).fit(x)
            x, z = pca.transform(x), pca.transform(z)
            used_dims.append(used)
        else:
            used_dims.append(None)
        a.append(x); b.append(z)
    return np.concatenate(a, 1), np.concatenate(b, 1), used_dims


def error_probability(train_x, test_x, y, train, seed):
    model = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                               solver="liblinear", random_state=seed)
    model.fit(train_x, y[train])
    return model.predict_proba(test_x)[:, list(model.classes_).index(1)]


def binary_metrics(y, pred, score=None):
    tn = int(np.sum((~pred) & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tp = int(np.sum(pred & (y == 1)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[1], zero_division=0)
    out = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
           "accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "hallucination_precision": float(precision[0]),
           "hallucination_recall": float(recall[0]),
           "hallucination_f1": float(f1[0])}
    if score is not None:
        out.update(auroc=float(roc_auc_score(y, score)),
                   auprc=float(average_precision_score(y, score)))
    return out


def combination_table(y, flags, order):
    result = {}
    for bits in np.ndindex(*(2 for _ in order)):
        # Report high first for readability.
        highs = tuple(not bool(x) for x in bits)
        take = np.ones(len(y), dtype=bool)
        labels = []
        for name, high in zip(order, highs):
            take &= flags[name] == high
            labels.append(f"{name}_{'high' if high else 'low'}")
        n = int(take.sum()); h = int(y[take].sum())
        result["__".join(labels)] = {
            "n": n, "hallucination": h, "correct": n-h,
            "hallucination_rate": None if n == 0 else float(h/n)}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    y = np.asarray([x["error"] for x in rows])
    groups = np.asarray([x["right_qid"] for x in rows])
    entropy = np.asarray([x["entropy"] for x in rows])
    r_blocks = [np.stack([x["r_last"] for x in rows]),
                np.stack([x["r_mean"] for x in rows])]
    p_blocks = [np.stack([x["p_scalar"] for x in rows])]
    p_blocks += [np.stack([x["p_hidden"][j] for x in rows]) for j in range(4)]
    p_blocks += [np.stack([x["p_layer"] for x in rows])]
    all_r, all_p, all_u = [], [], []
    fold_thresholds = []
    for seed in SEEDS:
        r_score = np.zeros(len(y)); p_score = np.zeros(len(y))
        u_high = np.zeros(len(y), dtype=bool)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(r_blocks[0], y, groups), 1):
            r_train, r_test, r_dims = transform_blocks(
                r_blocks, train, test, [8, 8], seed)
            p_train, p_test, p_dims = transform_blocks(
                p_blocks, train, test, [None, 8, 8, 8, 8, 48], seed)
            r_score[test] = error_probability(r_train, r_test, y, train, seed)
            p_score[test] = error_probability(p_train, p_test, y, train, seed)
            threshold = float(np.quantile(entropy[train], .70))
            u_high[test] = entropy[test] > threshold
            fold_thresholds.append({"seed": seed, "fold": fold,
                                    "threshold": threshold,
                                    "n_train": int(len(train)),
                                    "n_test": int(len(test)),
                                    "R_pca_dims": r_dims,
                                    "P_pca_dims": p_dims})
        all_r.append(r_score); all_p.append(p_score); all_u.append(u_high)
    r_score = np.mean(all_r, axis=0)
    p_score = np.mean(all_p, axis=0)
    # Majority vote over the three fold-specific, label-free thresholds.
    u_vote = np.mean(all_u, axis=0)
    flags = {"U": u_vote >= .5, "R": r_score >= .5, "P": p_score >= .5}
    report = {
        "protocol": {
            "population": "2894 parse-valid full Scientist rows",
            "folds": "right-person grouped 3x5 OOF, matching the Scientist representation paper-matrix protocol",
            "probe_usage": "none in features, routing, labels, or thresholds",
            "U": "Farquhar et al. probability-weighted Semantic Entropy; outer-train label-free 70th percentile; 3-seed majority vote",
            "R": "fixed layer14 last+mean; per-block fold-local scaler/PCA up to 8; LR C=.03",
            "P": "exact-current127 scalars + four hidden PCA up to 8 blocks + layer14 PCA up to 48; dimensions cap at n_train-1; all transforms fold-local; LR C=.03",
            "orientation": "all high flags and scores mean predicted hallucination",
        },
        "n": len(y), "hallucination": int(y.sum()),
        "correct": int((1-y).sum()), "components": len(set(groups)),
        "entropy_fold_thresholds": fold_thresholds,
        "single_method": {
            "U": binary_metrics(y, flags["U"], entropy),
            "R": binary_metrics(y, flags["R"], r_score),
            "P": binary_metrics(y, flags["P"], p_score),
        },
        "U_x_R": combination_table(y, flags, ("U", "R")),
        "U_x_P": combination_table(y, flags, ("U", "P")),
        "U_x_R_x_P": combination_table(y, flags, ("U", "R", "P")),
    }
    with (OUT/"predictions.jsonl").open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"key": row["key"], "error": int(y[i]),
                                     "entropy": float(entropy[i]),
                                     "u_vote": float(u_vote[i]),
                                     "r_error_probability": float(r_score[i]),
                                     "p_error_probability": float(p_score[i]),
                                     "U_high": bool(flags["U"][i]),
                                     "R_high": bool(flags["R"][i]),
                                     "P_high": bool(flags["P"][i])}) + "\n")
    (OUT/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"single_method": report["single_method"],
                      "U_x_R": report["U_x_R"],
                      "U_x_P": report["U_x_P"],
                      "U_x_R_x_P": report["U_x_R_x_P"]}, indent=2))


if __name__ == "__main__":
    main()
