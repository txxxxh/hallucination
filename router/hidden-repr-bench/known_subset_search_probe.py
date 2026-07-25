#!/usr/bin/env python3
"""Probe whether knowledge confidence gates search within construct-known items.

This analysis is model-free: it consumes records.jsonl and hidden/*.pt produced
by tool_gate_calibration.py.

For every hidden-state layer it measures:
  1. Cross-validated AUROC for predicting SEARCH inside know_prior == "known".
  2. Direction agreement between that SEARCH probe and a global knowledge probe.

The global knowledge probe is trained with known=1. Its coefficient is negated
before comparison, so both vectors point toward "less known / more search".
Positive cosine similarity therefore means directionally consistent probes.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("known_subset_search_probe")


def load_inputs(out: Path):
    import torch

    records = [
        json.loads(line)
        for line in (out / "records.jsonl").open()
        if line.strip()
    ]
    hidden, kept = [], []
    for record in records:
        path = out / "hidden" / f"{record['qid']}.pt"
        if not path.exists():
            LOG.warning("missing hidden state for qid=%s", record["qid"])
            continue
        item = torch.load(path, map_location="cpu", weights_only=False)
        hidden.append(item["hidden"].float().numpy())
        kept.append(record)
    if not hidden:
        raise ValueError(f"no readable hidden states under {out / 'hidden'}")
    return kept, np.stack(hidden)


def make_cv(y: np.ndarray, requested_folds: int, seed: int):
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(y, minlength=2)
    folds = min(requested_folds, int(counts.min()))
    if folds < 2:
        raise ValueError(
            f"need at least 2 examples in each action class; counts={counts.tolist()}"
        )
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), folds


def cv_auc(x: np.ndarray, y: np.ndarray, folds: int, seed: int, c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv, actual_folds = make_cv(y, folds, seed)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=c,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
    )
    probability = cross_val_predict(
        estimator, x, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    return float(roc_auc_score(y, probability)), actual_folds


def raw_space_direction(x: np.ndarray, y: np.ndarray, c: float, seed: int):
    """Fit a standardized linear probe and return its raw-coordinate direction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    model = LogisticRegression(
        max_iter=2000,
        C=c,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    ).fit(z, y)
    # score = coef @ ((x - mean) / scale), hence raw coefficient = coef / scale.
    direction = model.coef_[0] / scaler.scale_
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("probe produced a zero or non-finite direction")
    return direction / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--result-file", default="known_subset_search_probe.json"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    out = Path(args.output_dir)
    records, hidden = load_inputs(out)

    y_known = np.array(
        [int(record["know_prior"] == "known") for record in records], dtype=int
    )
    y_search = np.array(
        [int(record["action"] == "search") for record in records], dtype=int
    )
    known_mask = y_known == 1
    known_search = y_search[known_mask]
    known_counts = np.bincount(known_search, minlength=2)
    if known_counts.min() < 2:
        raise ValueError(
            "known subset needs both SEARCH and non-SEARCH examples; "
            f"counts={known_counts.tolist()}"
        )

    layers = []
    for layer in range(1, hidden.shape[1]):  # exclude embedding layer
        x_all = hidden[:, layer]
        x_known = x_all[known_mask]

        auc, actual_folds = cv_auc(
            x_known, known_search, args.folds, args.seed, args.c
        )
        search_direction = raw_space_direction(
            x_known, known_search, args.c, args.seed
        )
        known_direction = raw_space_direction(
            x_all, y_known, args.c, args.seed
        )
        # known_direction points toward "known"; negate it to point "unknown".
        cosine_to_unknown = float(np.dot(search_direction, -known_direction))
        layers.append(
            {
                "layer": layer,
                "known_search_cv_auroc": round(auc, 4),
                "search_vs_unknown_direction_cosine": round(
                    cosine_to_unknown, 4
                ),
                "cv_folds": actual_folds,
            }
        )
        LOG.info(
            "layer=%d AUROC=%.4f cosine(search, unknown)=%.4f",
            layer,
            auc,
            cosine_to_unknown,
        )

    best_auc = max(layers, key=lambda row: row["known_search_cv_auroc"])
    best_joint = max(
        layers,
        key=lambda row: (
            row["known_search_cv_auroc"]
            * max(row["search_vs_unknown_direction_cosine"], 0.0)
        ),
    )
    result = {
        "n_total": len(records),
        "n_known": int(known_mask.sum()),
        "known_action_counts": {
            "non_search": int(known_counts[0]),
            "search": int(known_counts[1]),
        },
        "direction_convention": (
            "positive cosine means the within-known SEARCH direction aligns "
            "with the global UNKNOWN direction"
        ),
        "best_auroc_layer": best_auc,
        "best_positive_joint_layer": best_joint,
        "layers": layers,
    }
    result_path = out / args.result_file
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "layers"},
                     indent=2, ensure_ascii=False))
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
