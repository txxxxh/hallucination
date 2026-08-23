#!/usr/bin/env python3
"""Leakage-aware U/P/R combinations on the full parse-valid Scientist set.

U: unperturbed generated-answer logit/entropy statistics.
P: input-span neutralization and physical-delete response curves.
R: unperturbed multilayer hidden-state trajectory statistics.

Evidence features are deliberately excluded.  Model/weight/order selection is
performed inside each outer candidate-identity-grouped fold.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "268_scientist_full_upr_combinations"
SEEDS = (42, 43, 44)


def read_jsonl(path):
    return [json.loads(x) for x in path.open() if x.strip()]


def fixed(x, n=6):
    x = np.asarray(x, np.float32)
    return np.pad(x[:n], (0, max(0, n-len(x))))


def channel(x):
    x = fixed(x); delta = x[0] - x[1:]; scale = abs(float(x[0])) + 1e-6
    return np.r_[x[0], delta, delta/scale, delta.max(initial=0),
                 delta.min(initial=0), np.abs(delta).mean(), delta.std(),
                 np.mean(delta > 0)]


def channel2(x):
    x = fixed(x)
    return np.r_[x[0], x[0]-x[1:]]


def p_features(a, b, c, d):
    """Same gold-free perturbation summary used by the current127 detector."""
    return np.r_[channel(a), channel(b), channel2(c), channel2(d),
                 a[0]-c[0], b[0]-d[0], (a[0]-b[0])-(c[0]-d[0])]


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for x in rows: union(x["right_qid"], x["wrong_qid"])
    return np.asarray([find(x["right_qid"]) for x in rows])


def load():
    records = {x["key"]: x for x in read_jsonl(
        ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl")}
    manifest = {x["key"]: x for x in read_jsonl(
        RUNS / "76_closedbook_fact_probe_manifest.jsonl")}
    probes = {x["key"]: x for x in read_jsonl(
        RUNS / "77_closedbook_fact_probe_results.jsonl")}
    known_keys = {x["key"] for x in probes.values() if
                  x["n_discriminative_facts"] >= 1 and
                  x["binary_accuracy"] > .5 and
                  x["pairwise_owner_accuracy"] > .5}
    rows = []
    for fp in sorted((RUNS / "141_scientist_all_trajectory_l8").glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            if key not in records or not records[key].get("parse_valid", True):
                continue
            # R uses compact per-layer distributional statistics and their
            # first differences; this remains a representation trajectory.
            ls = z["last_stats"].astype(np.float32)
            ms = z["mean_stats"].astype(np.float32)
            rfeat = np.r_[ls.ravel(), ms.ravel(), np.diff(ls, axis=0).ravel(),
                          np.diff(ms, axis=0).ravel()]
            # Independent closed-book probe confidence/separation/entropy.
            # The probes contain no Scientist profile/context, so this is
            # knowledge uncertainty rather than grounded evidence.
            probe_u = importlib.import_module(
                "134_scientist_full_knowledge_error").features(
                    probes[key], records[key]).astype(np.float32)
            ufeat = np.r_[probe_u, z["logits"].astype(np.float32)]
        pf = RUNS / "135_scientist_full_current127" / f"{key}.npz"
        if pf.exists():
            with np.load(pf, allow_pickle=True) as q:
                pfeat = p_features(q["stage1_pred"], q["stage1_other"],
                                   q["stage2_pred"], q["stage2_other"])
        else:
            pf = RUNS / "120_physical_delete_rerank" / f"{key}.npz"
            if not pf.exists():
                continue
            with np.load(pf, allow_pickle=True) as q:
                pfeat = p_features(q["stage1_pred_scores"], q["stage1_other_scores"],
                                   q["stage2_pred_scores"], q["stage2_other_scores"])
        rows.append({**manifest[key], "key": key,
                     "error": int(not records[key]["correct"]),
                     "known": int(key in known_keys), "U": ufeat,
                     "P": pfeat.astype(np.float32), "R": rfeat})
    if len(rows) != 2894:
        raise RuntimeError(f"incomplete aligned cache: {len(rows)}/2894")
    return rows


def clf(C=.03):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", solver="liblinear"))


def fit_prob(x, y, tr, te, C=.03):
    m = clf(C).fit(x[tr], y[tr])
    return m.predict_proba(x[te])[:, list(m.classes_).index(1)]


def inner_predictions(x, y, groups, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, groups): p[te] = fit_prob(x, y, tr, te)
    return p


def choose_weight(y, a, b):
    grid = np.linspace(0, 1, 21)
    scored = [(roc_auc_score(y, w*a+(1-w)*b), w) for w in grid]
    return max(scored)[1]


def metrics(y, p):
    h = p >= .5
    pr, rc, f, _ = precision_recall_fscore_support(y, h, labels=[1], zero_division=0)
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, h)),
            "precision_error_at_0.5": float(pr[0]),
            "recall_error_at_0.5": float(rc[0]),
            "f1_error_at_0.5": float(f[0])}


def bootstrap_delta(y, base, candidate, draws=2000):
    rng = np.random.default_rng(20260822); values = []
    for _ in range(draws):
        take = rng.integers(0, len(y), len(y))
        if len(np.unique(y[take])) == 2:
            values.append(roc_auc_score(y[take], candidate[take]) -
                          roc_auc_score(y[take], base[take]))
    return {"point": float(roc_auc_score(y, candidate)-roc_auc_score(y, base)),
            "ci95": np.quantile(values, [.025, .975]).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load(); y = np.asarray([x["error"] for x in rows]); known = np.asarray([x["known"] for x in rows])
    groups = components(rows)
    X = {k: np.stack([x[k] for x in rows]) for k in "UPR"}
    pred_names = ["U", "P", "R", "UP_early", "UPR_early", "UP_late_tuned",
                  "UPR_late_tuned", "U_gate_PU_soft", "U_gate_PRU_soft",
                  "U_gate_PU_hard_range", "U_gate_PRU_hard_range"]
    all_seed = []; selections = []
    for seed in SEEDS:
        out = {k: np.zeros(len(y)) for k in pred_names}; gate_out = np.zeros(len(y))
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X["U"], y, groups), 1):
            # Base and early-fusion detectors.
            for name, xx in (("U", X["U"]), ("P", X["P"]), ("R", X["R"]),
                             ("UP_early", np.c_[X["U"], X["P"]]),
                             ("UPR_early", np.c_[X["U"], X["P"], X["R"]])):
                out[name][te] = fit_prob(xx, y, tr, te)

            # Inner OOF scores choose late-fusion weights without seeing outer test.
            inn = {}
            for name in "UPR": inn[name] = inner_predictions(X[name][tr], y[tr], groups[tr], seed+100*fold)
            wup = choose_weight(y[tr], inn["U"], inn["P"])
            upr_grid = []
            for wu in np.linspace(0, 1, 11):
                for wp in np.linspace(0, 1-wu, 11):
                    wr = 1-wu-wp
                    upr_grid.append((roc_auc_score(y[tr], wu*inn["U"]+wp*inn["P"]+wr*inn["R"]), wu, wp, wr))
            _, wu, wp, wr = max(upr_grid)
            out["UP_late_tuned"][te] = wup*out["U"][te] + (1-wup)*out["P"][te]
            out["UPR_late_tuned"][te] = wu*out["U"][te] + wp*out["P"][te] + wr*out["R"][te]

            # U-only knowledge gate; experts are fitted separately by knowledge stratum.
            gate = clf().fit(X["U"][tr], known[tr])
            qte = gate.predict_proba(X["U"][te])[:, list(gate.classes_).index(1)]
            gate_out[te] = qte
            experts = {}
            for tag, feat in (("U", X["U"]), ("P", X["P"]), ("PR", np.c_[X["P"], X["R"]])):
                for kval in (0, 1):
                    sub = tr[known[tr] == kval]
                    experts[tag, kval] = clf().fit(feat[sub], y[sub]).predict_proba(feat[te])[:, 1]
            # Unknown region uses U; known region uses P or P+R.
            for suffix, kt in (("PU", "P"), ("PRU", "PR")):
                soft = qte*experts[kt, 1] + (1-qte)*experts["U", 0]
                hard = np.where(qte >= .7, experts[kt, 1],
                                np.where(qte <= .3, experts["U", 0], soft))
                out[f"U_gate_{suffix}_soft"][te] = soft
                out[f"U_gate_{suffix}_hard_range"][te] = hard
            selections.append({"seed": seed, "fold": fold, "UP_weight_U": float(wup),
                               "UPR_weights": {"U": float(wu), "P": float(wp), "R": float(wr)}})
        all_seed.append((out, gate_out))

    mean = {k: np.mean([z[0][k] for z in all_seed], axis=0) for k in pred_names}
    gate = np.mean([z[1] for z in all_seed], axis=0)
    result = {k: {"mean_probability": metrics(y, p),
                  "per_seed": [metrics(y, z[0][k]) for z in all_seed]}
              for k, p in mean.items()}
    ranking = sorted(pred_names, key=lambda k: result[k]["mean_probability"]["auroc"], reverse=True)
    best = ranking[0]
    report = {
        "protocol": "2894 parse-valid Scientist; candidate-identity connected-component grouped 3x5 OOF; fold-local transforms/models; inner grouped CV selects late-fusion weights; no evidence features",
        "n": len(y), "errors": int(y.sum()), "correct": int((1-y).sum()),
        "known": int(known.sum()), "unknown": int((1-known).sum()), "components": len(set(groups)),
        "feature_definitions": {"U": "independent closed-book probe confidence/separation/entropy statistics plus 7 unperturbed main-task logit/entropy statistics; no profile/context evidence",
                                "P": "63 scalar span-neutralization/physical-delete response features",
                                "R": "330 compact absolute and delta multilayer hidden-state trajectory statistics"},
        "knowledge_gate": metrics(known, gate), "results": result, "ranking": ranking,
        "selected_weights": selections,
        "best": best,
        "best_vs_U": bootstrap_delta(y, mean["U"], mean[best]),
        "best_vs_P": bootstrap_delta(y, mean["P"], mean[best]),
        "warnings": ["known/unknown labels come from independent closed-book probes and are used only to train the auxiliary gate/stratified experts",
                     "the 28 connected components make grouped fold composition variable; per-seed results must be inspected",
                     "this compares cached score families on one benchmark and is not external-domain validation"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "predictions.jsonl").open("w") as f:
        for i, x in enumerate(rows):
            f.write(json.dumps({"key": x["key"], "error": int(y[i]), "known": int(known[i]),
                                "u_known_gate": float(gate[i]),
                                "scores": {k: float(mean[k][i]) for k in pred_names}}) + "\n")
    print(json.dumps({"ranking": [(k, result[k]["mean_probability"]) for k in ranking],
                      "knowledge_gate": report["knowledge_gate"], "best_vs_U": report["best_vs_U"]}, indent=2))


if __name__ == "__main__": main()
