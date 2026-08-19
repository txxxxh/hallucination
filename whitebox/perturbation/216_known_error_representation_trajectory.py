#!/usr/bin/env python3
"""Representation trajectory study on the full probe-known Scientist subset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CACHE = RUNS / "141_scientist_all_trajectory_l8"
OUT = RUNS / "216_known_error_representation_trajectory.json"
PREDS = RUNS / "216_known_error_representation_trajectory_predictions.jsonl"
SEEDS = (42, 43, 44)


def known(p):
    return bool(p["n_discriminative_facts"] >= 1 and
                p["binary_accuracy"] > .5 and
                p["pairwise_owner_accuracy"] > .5)


def metric(y, score):
    return {"n": int(len(y)), "errors": int(y.sum()),
            "error_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def model_predictions(blocks, logits, y, groups, mode):
    """Fold-local transforms; return mean 3x5 OOF probabilities."""
    outputs = []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(logits, y, groups):
            a, b = [], []
            if mode in ("uncertainty", "fusion"):
                sc = StandardScaler().fit(logits[train])
                a.append(sc.transform(logits[train])); b.append(sc.transform(logits[test]))
            if mode in ("representation", "fusion"):
                for values in blocks:
                    sc = StandardScaler().fit(values[train])
                    x, z = sc.transform(values[train]), sc.transform(values[test])
                    dim = min(8, x.shape[1], len(train)-1)
                    pc = PCA(dim, whiten=True, svd_solver="randomized",
                             random_state=seed).fit(x)
                    a.append(pc.transform(x)); b.append(pc.transform(z))
            clf = LogisticRegression(C=.03, max_iter=5000,
                                     class_weight="balanced", solver="liblinear",
                                     random_state=seed)
            clf.fit(np.concatenate(a, 1), y[train])
            pred[test] = clf.predict_proba(np.concatenate(b, 1))[:, 1]
        outputs.append(pred)
    return np.mean(outputs, axis=0)


def bootstrap_delta(y, a, b, draws=2000):
    rng = np.random.default_rng(20260818)
    values = []
    for _ in range(draws):
        take = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[take])) == 2:
            values.append(roc_auc_score(y[take], b[take]) -
                          roc_auc_score(y[take], a[take]))
    return {"point": float(roc_auc_score(y, b)-roc_auc_score(y, a)),
            "bootstrap_95ci": np.quantile(values, [.025, .975]).tolist()}


def main():
    probes = {x["key"]: x for x in map(json.loads,
              (RUNS / "77_closedbook_fact_probe_results.jsonl").open())}
    records = {x["key"]: x for x in map(json.loads,
               (HERE.parent / "tool_gate_correctness_names_llama31_8b" /
                "records.jsonl").open())}
    manifest = {x["key"]: x for x in map(json.loads,
                (RUNS / "76_closedbook_fact_probe_manifest.jsonl").open())}
    rows = []
    for path in sorted(CACHE.glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            if key not in records or not records[key].get("parse_valid", True):
                continue
            if not known(probes[key]):
                continue
            rows.append({"key": key, "group": manifest[key]["right_qid"],
                         "error": int(not records[key]["correct"]),
                         "layers": z["layers"].astype(int),
                         "last": z["last"].astype(np.float32),
                         "mean": z["mean"].astype(np.float32),
                         "logits": z["logits"].astype(np.float32)})
    y = np.asarray([r["error"] for r in rows])
    groups = np.asarray([r["group"] for r in rows])
    logits = np.stack([r["logits"] for r in rows])
    last = np.stack([r["last"] for r in rows])
    mean = np.stack([r["mean"] for r in rows])
    layers = rows[0]["layers"].tolist()

    # Absolute states and layer-to-layer changes are kept as separate PCA blocks.
    absolute = [last[:, i] for i in range(len(layers))] + [mean[:, i] for i in range(len(layers))]
    deltas = ([last[:, i]-last[:, i-1] for i in range(1, len(layers))] +
              [mean[:, i]-mean[:, i-1] for i in range(1, len(layers))])
    late = [last[:, i] for i in range(3, len(layers))] + [mean[:, i] for i in range(3, len(layers))]

    scores = {}
    scores["mean_token_nll"] = -logits[:, 0]
    scores["uncertainty_lr"] = model_predictions([], logits, y, groups, "uncertainty")
    for name, blocks in (("absolute_trajectory", absolute),
                         ("delta_trajectory", deltas), ("late_trajectory", late)):
        scores[name] = model_predictions(blocks, logits, y, groups, "representation")
        scores[name + "_plus_uncertainty"] = model_predictions(blocks, logits, y, groups, "fusion")

    # Layerwise error decodability: the simplest empirical trajectory diagnostic.
    layerwise = []
    for i, layer in enumerate(layers):
        p = model_predictions([last[:, i], mean[:, i]], logits, y, groups,
                              "representation")
        scores[f"layer_{layer}"] = p
        layerwise.append({"layer": int(layer), **metric(y, p)})

    report = {
        "protocol": ("Full parse-valid probe-known Scientist subset; error target; right-person "
                     "grouped 3x5 OOF; all scaling/PCA train-fold only. Certainty subsets are "
                     "post-hoc quantiles of mean-token NLL."),
        "n": len(y), "errors": int(y.sum()), "groups": len(set(groups)),
        "layers": layers,
        "overall": {k: metric(y, v) for k, v in scores.items()},
        "layerwise": layerwise,
        "incremental": {
            "uncertainty_to_absolute": bootstrap_delta(
                y, scores["uncertainty_lr"], scores["absolute_trajectory_plus_uncertainty"]),
            "uncertainty_to_delta": bootstrap_delta(
                y, scores["uncertainty_lr"], scores["delta_trajectory_plus_uncertainty"]),
            "representation_to_fusion": bootstrap_delta(
                y, scores["absolute_trajectory"], scores["absolute_trajectory_plus_uncertainty"]),
        },
        "certainty_strata": {},
    }
    nll = scores["mean_token_nll"]
    edges = np.quantile(nll, [0, .25, .5, .75, 1])
    for i, label in enumerate(("most_certain", "certain_mid", "uncertain_mid", "most_uncertain")):
        take = ((nll >= edges[i]) & (nll <= edges[i+1]) if i == 3 else
                (nll >= edges[i]) & (nll < edges[i+1]))
        report["certainty_strata"][label] = {
            "nll_range": [float(edges[i]), float(edges[i+1])],
            **{k: metric(y[take], v[take]) for k, v in scores.items()
               if len(np.unique(y[take])) == 2}}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    with PREDS.open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"key": row["key"], "error": int(y[i]),
                                     **{k: float(v[i]) for k, v in scores.items()}}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
