#!/usr/bin/env python3
"""Train and deploy a probe-free U/R knowledge-state head.

Closed-book probes are supervision *only* in ``train-eval``.  The feature and
``predict`` paths cannot read probe artifacts.  U consists only of seven
teacher-forced generation uncertainty statistics.  R consists only of the
question-final hidden state collected before answer generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
QUESTION_CACHE = RUNS / "147_question_only_hidden_v3"
UNCERTAINTY_CACHE = RUNS / "141_scientist_all_trajectory_l8"
PROBES = RUNS / "77_closedbook_fact_probe_results.jsonl"
MANIFEST = RUNS / "76_closedbook_fact_probe_manifest.jsonl"
RECORDS = ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl"
OUT = RUNS / "277_probe_free_ur_knowledge_head"
LAYERS = (8, 10, 12, 14, 16, 18, 20, 22)
SEEDS = (42, 43, 44)
U_FIELDS = ("mean_token_logprob", "minimum_token_logprob", "token_logprob_std",
            "mean_token_entropy", "max_token_entropy", "mean_top2_logit_margin",
            "answer_token_count")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


class DSU:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[b] = a


def group_map(manifest):
    dsu = DSU()
    for x in manifest.values(): dsu.union(x["right_qid"], x["wrong_qid"])
    return {k: dsu.find(x["right_qid"]) for k, x in manifest.items()}


def feature_keys(question_cache, uncertainty_cache):
    q = {x.stem for x in Path(question_cache).glob("*.npz")}
    u = {x.stem for x in Path(uncertainty_cache).glob("*.npz")}
    return sorted(q & u)


def load_features(keys, question_cache, uncertainty_cache):
    """Probe-blind feature loader. Do not add label/probe arguments here."""
    us, rs = [], []
    for key in keys:
        with np.load(Path(uncertainty_cache) / f"{key}.npz", allow_pickle=False) as z:
            names = set(z.files)
            if "logits" not in names:
                raise KeyError(f"{key}: missing logits")
            u = z["logits"].astype(np.float32)
            if u.shape != (7,):
                raise ValueError(f"{key}: expected seven U values, got {u.shape}")
        with np.load(Path(question_cache) / f"{key}.npz", allow_pickle=False) as z:
            names = set(z.files)
            if names != {"key", "hidden"}:
                raise RuntimeError(f"{key}: question cache whitelist violation: {sorted(names)}")
            hidden = z["hidden"].astype(np.float32)
            if hidden.ndim != 2 or max(LAYERS) >= len(hidden):
                raise ValueError(f"{key}: invalid question hidden shape {hidden.shape}")
            r = hidden[list(LAYERS)]
        us.append(u); rs.append(r)
    return np.stack(us), np.stack(rs)


def labels_from_probes(keys, probe_path):
    """The sole function allowed to inspect probe outputs."""
    probes = {str(x["key"]): x for x in read_jsonl(probe_path)}
    missing = set(keys) - probes.keys()
    if missing: raise RuntimeError(f"missing probe labels for {len(missing)} keys")
    return np.asarray([
        int(probes[k]["n_discriminative_facts"] >= 1 and
            probes[k]["binary_accuracy"] > .5 and
            probes[k]["pairwise_owner_accuracy"] > .5)
        for k in keys
    ])


def binary_model(C=.03):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", solver="liblinear"))


def fit_r_ensemble(r, y):
    models = []
    for i in range(r.shape[1]):
        model = binary_model(.03).fit(r[:, i], y)
        models.append(model)
    return models


def predict_r(models, r):
    return np.mean([m.predict_proba(r[:, i])[:, 1]
                    for i, m in enumerate(models)], axis=0)


def metrics(y, p):
    h = p >= .5
    return {"n": int(len(y)), "known": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "accuracy_at_0.5": float(accuracy_score(y, h)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, h)),
            "macro_f1_at_0.5": float(f1_score(y, h, average="macro")),
            "confusion_rows_unknown_known": confusion_matrix(y, h, labels=[0, 1]).tolist()}


def safe_subset_metrics(y, p, mask):
    yy, pp = y[mask], p[mask]
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return {"n": int(len(yy)), "known": int(yy.sum()), "auroc": None}
    return metrics(yy, pp)


def train_eval(args):
    manifest = {str(x["key"]): x for x in read_jsonl(args.manifest)}
    records = {str(x["key"]): x for x in read_jsonl(args.records)}
    groups_by_key = group_map(manifest)
    keys = [k for k in feature_keys(args.question_cache, args.uncertainty_cache)
            if k in manifest and k in records and records[k].get("parse_valid", True)]
    # Probe-derived values are created after the probe-blind feature matrix.
    u, r = load_features(keys, args.question_cache, args.uncertainty_cache)
    y = labels_from_probes(keys, args.probes)
    groups = np.asarray([groups_by_key[k] for k in keys])
    correct = np.asarray([int(records[k]["correct"]) for k in keys])
    seed_predictions = []
    for seed in SEEDS:
        pred = {name: np.zeros(len(y)) for name in ("U", "R_question", "UR_late")}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for tr, te in cv.split(u, y, groups):
            um = binary_model(.03).fit(u[tr], y[tr])
            rm = fit_r_ensemble(r[tr], y[tr])
            pu_tr = um.predict_proba(u[tr])[:, 1]
            pr_tr = predict_r(rm, r[tr])
            fusion = binary_model(.3).fit(np.c_[pu_tr, pr_tr], y[tr])
            pred["U"][te] = um.predict_proba(u[te])[:, 1]
            pred["R_question"][te] = predict_r(rm, r[te])
            pred["UR_late"][te] = fusion.predict_proba(
                np.c_[pred["U"][te], pred["R_question"][te]])[:, 1]
        seed_predictions.append(pred)
        print(seed, {k: roc_auc_score(y, v) for k, v in pred.items()}, flush=True)
    mean = {name: np.mean([x[name] for x in seed_predictions], axis=0)
            for name in seed_predictions[0]}
    results = {}
    for name, p in mean.items():
        results[name] = {"all": metrics(y, p),
                         "generated_correct_only": safe_subset_metrics(y, p, correct == 1),
                         "generated_error_only": safe_subset_metrics(y, p, correct == 0),
                         "per_seed_auroc": [float(roc_auc_score(y, x[name]))
                                            for x in seed_predictions]}
    # Frozen deployment model. Its inference object contains no probe values.
    um = binary_model(.03).fit(u, y)
    rm = fit_r_ensemble(r, y)
    fusion = binary_model(.3).fit(
        np.c_[um.predict_proba(u)[:, 1], predict_r(rm, r)], y)
    artifact = {"schema_version": 1, "layers": LAYERS, "u_fields": U_FIELDS,
                "u_model": um, "r_models": rm, "fusion_model": fusion,
                "feature_contract": {"probe_features": [],
                    "U": "seven generated-answer logit/entropy statistics",
                    "R": "question-only, pre-generation hidden states"}}
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out / "model.joblib")
    report = {"protocol": ("Probe labels only; no probe probabilities/features in X; "
                            "candidate-QID connected-component grouped 3x5 OOF; "
                            "fold-local models; R is question-only before generation"),
              "n": len(y), "known": int(y.sum()), "unknown": int((1-y).sum()),
              "groups": len(set(groups)), "feature_contract": artifact["feature_contract"],
              "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "oof_predictions.jsonl").open("w") as f:
        for i, key in enumerate(keys):
            f.write(json.dumps({"key": key, "probe_known_label": int(y[i]),
                                "generated_correct": int(correct[i]),
                                "prob_known": {k: float(v[i]) for k, v in mean.items()}}) + "\n")
    print(json.dumps(report, indent=2))


def predict(args):
    """Deployment path: deliberately has no probe/manifest/record dependency."""
    artifact = joblib.load(args.model_artifact)
    if artifact["feature_contract"].get("probe_features") != []:
        raise RuntimeError("refusing artifact containing probe features")
    global LAYERS
    LAYERS = tuple(artifact["layers"])
    keys = feature_keys(args.question_cache, args.uncertainty_cache)
    u, r = load_features(keys, args.question_cache, args.uncertainty_cache)
    pu = artifact["u_model"].predict_proba(u)[:, 1]
    pr = predict_r(artifact["r_models"], r)
    pur = artifact["fusion_model"].predict_proba(np.c_[pu, pr])[:, 1]
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w") as f:
        for key, a, b, c in zip(keys, pu, pr, pur):
            f.write(json.dumps({"key": key, "prob_known_U": float(a),
                                "prob_known_R_question": float(b),
                                "prob_known_UR": float(c)}) + "\n")
    print(json.dumps({"n": len(keys), "predictions": str(args.predictions),
                      "probe_files_read": 0}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train-eval")
    tr.add_argument("--question-cache", type=Path, default=QUESTION_CACHE)
    tr.add_argument("--uncertainty-cache", type=Path, default=UNCERTAINTY_CACHE)
    tr.add_argument("--probes", type=Path, default=PROBES)
    tr.add_argument("--manifest", type=Path, default=MANIFEST)
    tr.add_argument("--records", type=Path, default=RECORDS)
    tr.add_argument("--out", type=Path, default=OUT)
    pr = sub.add_parser("predict")
    pr.add_argument("--question-cache", type=Path, default=QUESTION_CACHE)
    pr.add_argument("--uncertainty-cache", type=Path, default=UNCERTAINTY_CACHE)
    pr.add_argument("--model-artifact", type=Path, default=OUT/"model.joblib")
    pr.add_argument("--predictions", type=Path, default=OUT/"full_predictions.jsonl")
    args = ap.parse_args()
    train_eval(args) if args.command == "train-eval" else predict(args)


if __name__ == "__main__": main()
