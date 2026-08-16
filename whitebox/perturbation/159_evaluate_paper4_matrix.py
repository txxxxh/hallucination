#!/usr/bin/env python3
"""Unified OOF and frozen-transfer evaluation for the paper4 feature matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODELS = ("llama", "mistral", "qwen", "falcon3")
METHODS = ("exact", "attention")
IN_DOMAIN = ("scientist", "trivia", "gsm8k")
EXPECTED = {"scientist": 1084, "multidomain": 477, "trivia": 1000, "gsm8k": 942}
SEEDS = (42, 43, 44)


def fixed(values, size=6):
    values = np.asarray(values, np.float32)[:size]
    return np.pad(values, (0, max(0, size - len(values))))


def channel(values):
    values = fixed(values)
    delta = values[0] - values[1:]
    scale = abs(float(values[0])) + 1e-6
    return np.r_[values[0], delta, delta / scale, delta.max(initial=0),
                 delta.min(initial=0), np.abs(delta).mean(), delta.std(),
                 np.mean(delta > 0)]


def channel2(values):
    values = fixed(values)
    return np.r_[values[0], values[0] - values[1:]]


def weighted_delta(hidden, effects):
    hidden = np.asarray(hidden, np.float32)
    effects = np.asarray(effects, np.float32)
    delta = hidden[1:1 + len(effects)] - hidden[0]
    return (delta * effects[:, None]).sum(0) / (np.abs(effects).sum() + 1e-9)


def load_directory(path: Path, expected: int):
    rows = []
    query_rows = []
    for file in sorted(path.glob("*.npz")):
        with np.load(file, allow_pickle=True) as z:
            p = z["stage1_pred"].astype(np.float32)
            o = z["stage1_other"].astype(np.float32)
            q = z["stage2_pred"].astype(np.float32)
            r = z["stage2_other"].astype(np.float32)
            ph = z["pred_hidden"].astype(np.float32)
            oh = z["other_hidden"].astype(np.float32)
            scalar = np.r_[channel(p), channel(o), channel2(q), channel2(r),
                           p[0] - q[0], o[0] - r[0],
                           (p[0] - o[0]) - (q[0] - r[0])]
            hidden = (ph[0], weighted_delta(ph, p[0] - p[1:]), oh[0],
                      weighted_delta(oh, o[0] - o[1:]))
            rows.append((str(z["key"].item()), str(z["group"].item()),
                         int(z["correct"]), scalar, hidden,
                         z["layer14"].astype(np.float32)))
            query_rows.append((int(z["stage1_candidates"]), int(z["stage1_full"]),
                               int(z["stage2_candidates"]), int(z["stage2_full"])))
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} npz files, found {len(rows)}")
    keys = np.array([x[0] for x in rows])
    groups = np.array([x[1] for x in rows])
    labels = np.array([x[2] for x in rows], dtype=int)
    scalar = np.stack([x[3] for x in rows])
    hidden = [np.stack([x[4][i] for x in rows]) for i in range(4)]
    layer = np.stack([x[5] for x in rows])
    queries = np.asarray(query_rows, dtype=int)
    return {"keys": keys, "groups": groups, "y": labels, "scalar": scalar,
            "hidden": hidden, "layer": layer, "queries": queries}


def metric(y, probability):
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def transform_fold(train, test, seed):
    train_parts, test_parts = [], []
    pairs = [(train["scalar"], test["scalar"], None)]
    pairs += [(train["hidden"][i], test["hidden"][i], 8) for i in range(4)]
    pairs += [(train["layer"], test["layer"], 48)]
    for source, target, dimension in pairs:
        scaler = StandardScaler().fit(source)
        a, b = scaler.transform(source), scaler.transform(target)
        if dimension is not None:
            dimension = min(dimension, len(a) - 1, a.shape[1])
            pca = PCA(dimension, whiten=True, svd_solver="randomized",
                      random_state=seed).fit(a)
            a, b = pca.transform(a), pca.transform(b)
        train_parts.append(a)
        test_parts.append(b)
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def subset(data, indices):
    return {"scalar": data["scalar"][indices],
            "hidden": [x[indices] for x in data["hidden"]],
            "layer": data["layer"][indices]}


def query_summary(queries, method):
    selected = queries[:, 0] + queries[:, 2]
    full = queries[:, 1] + queries[:, 3]
    # Attention method uses two candidate-conditioned forwards in each stage.
    charged = selected + (4 if method == "attention" else 0)
    return {"selected_span_queries_mean": float(selected.mean()),
            "full_enumeration_span_queries_mean": float(full.mean()),
            "charged_queries_mean": float(charged.mean()),
            "reduction_vs_full": float(1 - charged.sum() / full.sum())}


def evaluate_oof(data, dataset, method):
    y, groups = data["y"], data["groups"]
    seed_results, seed_probabilities = [], []
    for seed in SEEDS:
        probability = np.zeros(len(y), dtype=float)
        if dataset == "trivia":
            splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
            splits = splitter.split(data["scalar"], y)
        else:
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            splits = splitter.split(data["scalar"], y, groups)
        for train_index, test_index in splits:
            x_train, x_test = transform_fold(subset(data, train_index),
                                             subset(data, test_index), seed)
            classifier = LogisticRegression(C=.03, max_iter=5000,
                                            class_weight="balanced",
                                            solver="liblinear",
                                            random_state=seed)
            classifier.fit(x_train, y[train_index])
            probability[test_index] = classifier.predict_proba(x_test)[:, 1]
        seed_probabilities.append(probability)
        seed_results.append(metric(y, probability))
    fields = ("auroc", "auprc", "accuracy", "balanced_accuracy", "macro_f1")
    mean = {name: float(np.mean([x[name] for x in seed_results])) for name in fields}
    return {"n": len(y), "groups": len(set(groups)), "positive": int(y.sum()),
            "protocol": "3 seeds x 5-fold OOF; grouped except TriviaQA unique-key stratified",
            "mean": mean, "per_seed": seed_results,
            "queries": query_summary(data["queries"], method)}


def frozen_transfer(source, target, method):
    source_y, target_y = source["y"], target["y"]
    probabilities = []
    for seed in SEEDS:
        x_source, x_target = transform_fold(source, target, seed)
        classifier = LogisticRegression(C=.03, max_iter=5000,
                                        class_weight="balanced",
                                        solver="liblinear", random_state=seed)
        classifier.fit(x_source, source_y)
        probabilities.append(classifier.predict_proba(x_target)[:, 1])
    average = np.mean(probabilities, axis=0)
    subsets = {"all": metric(target_y, average)}
    for domain in sorted(set(target["groups"])):
        mask = target["groups"] == domain
        subsets[domain] = metric(target_y[mask], average[mask])
        subsets[domain]["per_seed_auroc"] = [
            float(roc_auc_score(target_y[mask], p[mask])) for p in probabilities]
    return {"protocol": "frozen transfer: scaler/PCA/LR fit only on Scientist 1084; no target fitting",
            "source_n": len(source_y), "target_n": len(target_y),
            "subsets": subsets, "queries": query_summary(target["queries"], method)}


def write_outputs(report, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for model, model_report in report["models"].items():
        for method, result in model_report.items():
            for dataset, evaluation in result["in_domain"].items():
                rows.append({"model": model, "method": method, "evaluation": dataset,
                             **evaluation["mean"],
                             "query_reduction": evaluation["queries"]["reduction_vs_full"]})
            for subset_name, values in result["scientist_to_multidomain"]["subsets"].items():
                rows.append({"model": model, "method": method,
                             "evaluation": f"scientist_to_{subset_name}",
                             **{k: values[k] for k in ("auroc", "auprc", "accuracy",
                                                       "balanced_accuracy", "macro_f1")},
                             "query_reduction": result["scientist_to_multidomain"]["queries"]["reduction_vs_full"]})
    columns = ("model", "method", "evaluation", "auroc", "auprc", "accuracy",
               "balanced_accuracy", "macro_f1", "query_reduction")
    with (output / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Paper4 unified evaluation", "", report["protocol"], "",
             "## In-domain 3x5-fold OOF", "",
             "| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["evaluation"] in IN_DOMAIN:
            lines.append(f"| {row['model']} | {row['method']} | {row['evaluation']} | "
                         f"{row['auroc']:.3f} | {row['auprc']:.3f} | "
                         f"{row['balanced_accuracy']:.3f} | {row['query_reduction']:.1%} |")
    lines += ["", "## Frozen Scientist to multidomain", "",
              "| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["evaluation"].startswith("scientist_to_"):
            target = row["evaluation"].replace("scientist_to_", "")
            lines.append(f"| {row['model']} | {row['method']} | {target} | "
                         f"{row['auroc']:.3f} | {row['auprc']:.3f} | "
                         f"{row['balanced_accuracy']:.3f} | {row['query_reduction']:.1%} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path,
                        default=Path(__file__).resolve().parent / "runs/paper4_matrix/features")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "runs/paper4_matrix/evaluation")
    args = parser.parse_args()
    report = {"protocol": "fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix",
              "seeds": list(SEEDS), "models": {}}
    for model in MODELS:
        report["models"][model] = {}
        for method in METHODS:
            print(f"[{model}/{method}] loading", flush=True)
            loaded = {dataset: load_directory(args.feature_root / model / dataset / method,
                                               EXPECTED[dataset])
                      for dataset in (*IN_DOMAIN, "multidomain")}
            in_domain = {}
            for dataset in IN_DOMAIN:
                print(f"[{model}/{method}] OOF {dataset}", flush=True)
                in_domain[dataset] = evaluate_oof(loaded[dataset], dataset, method)
            print(f"[{model}/{method}] frozen multidomain", flush=True)
            transfer = frozen_transfer(loaded["scientist"], loaded["multidomain"], method)
            report["models"][model][method] = {
                "in_domain": in_domain, "scientist_to_multidomain": transfer}
            write_outputs(report, args.output_dir)
    write_outputs(report, args.output_dir)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
