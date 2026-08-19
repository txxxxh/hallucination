#!/usr/bin/env python3
"""Frozen same-model Scientist -> multidomain-v6 evaluation for Qwen/Mistral."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paper4_eval", HERE / "159_evaluate_paper4_matrix.py")
paper4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper4)

MODELS = {
    "mistral": {"source_n": 621, "target_n": 800},
    "qwen": {"source_n": 1204, "target_n": 469},
}


def read_manifest(path):
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            result[row["key"]] = row
    return result


def evaluate(model, config, source_root, target_root, manifest_root):
    source = paper4.load_directory(source_root / model / "scientist" / "exact",
                                   config["source_n"])
    target = paper4.load_directory(target_root / model / "multidomain" / "exact",
                                   config["target_n"])
    manifest = read_manifest(manifest_root / model / "multidomain.jsonl")
    probabilities = []
    for seed in paper4.SEEDS:
        x_source, x_target = paper4.transform_fold(source, target, seed)
        classifier = LogisticRegression(C=.03, max_iter=5000,
                                        class_weight="balanced",
                                        solver="liblinear", random_state=seed)
        classifier.fit(x_source, source["y"])
        probabilities.append(classifier.predict_proba(x_target)[:, 1])
    probability = np.mean(probabilities, axis=0)
    subsets = {"all": paper4.metric(target["y"], probability)}
    for domain in sorted(set(target["groups"])):
        mask = target["groups"] == domain
        subsets[domain] = paper4.metric(target["y"][mask], probability[mask])
        subsets[domain]["per_seed_auroc"] = [
            float(roc_auc_score(target["y"][mask], p[mask])) for p in probabilities]
    fields = sorted({manifest[k].get("field", "unknown") for k in target["keys"]})
    for field in fields:
        mask = np.array([manifest[k].get("field", "unknown") == field for k in target["keys"]])
        if mask.sum() and len(set(target["y"][mask])) == 2:
            subsets[f"field:{field}"] = paper4.metric(target["y"][mask], probability[mask])
    predictions = []
    for key, domain, label, prob in zip(target["keys"], target["groups"],
                                         target["y"], probability):
        predictions.append({"key": str(key), "domain": str(domain),
                            "field": manifest[str(key)].get("field"),
                            "correct": int(label),
                            "p_correct": float(prob),
                            "predicted_correct": int(prob >= .5)})
    report = {
        "model": model,
        "protocol": ("frozen same-model transfer; scalar47 + four hidden PCA8 + "
                     "layer14 PCA48; scaler/PCA/LR fitted only on Scientist; "
                     "LR C=.03 class_weight=balanced; seeds 42/43/44 averaged; no target fitting"),
        "source_n": len(source["y"]), "target_n": len(target["y"]),
        "subsets": subsets, "queries": paper4.query_summary(target["queries"], "exact"),
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source-root", type=Path,
                        default=HERE / "runs/paper4_self_matrix_v2/features")
    parser.add_argument("--target-root", type=Path,
                        default=HERE / "runs/v6_self_transfer/features")
    parser.add_argument("--manifest-root", type=Path,
                        default=HERE / "runs/v6_self_transfer/models")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "runs/v6_self_transfer/evaluation")
    args = parser.parse_args()
    report, predictions = evaluate(args.model, MODELS[args.model], args.source_root,
                                   args.target_root, args.manifest_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / f"{args.model}_predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
