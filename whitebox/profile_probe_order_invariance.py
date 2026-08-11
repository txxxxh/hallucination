#!/usr/bin/env python3
"""Test whether layer probes track answer identity or displayed profile position."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import profile_perturbation_unsupervised as pp

HERE = Path(__file__).resolve().parent


def load_keys(path):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [str(r["key"]) for r in rows if r.get("generation_mode") is not None]


def model(nc, seed):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=nc, svd_solver="randomized", random_state=seed),
        LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                           random_state=seed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=HERE / "profile_perturbation_forward_output" / "items")
    ap.add_argument("--generation-labels", type=Path,
                    default=HERE / "profile_likelihood_generation_m3_output" / "items.jsonl")
    ap.add_argument("--output", type=Path,
                    default=HERE / "profile_probe_order_invariance_output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca-components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys, y, original, swapped = [], [], [], []
    layers = None
    for key in load_keys(args.generation_labels):
        path = args.features / f"{key}.npz"
        if not path.exists():
            continue
        r = pp.load_item_npz(path)
        md = r["metadata"]
        ix = {name: i for i, name in enumerate(md["condition_names"])}
        if not md["condition_changed"][ix["profile_order_swap"]]:
            continue
        if layers is None:
            layers = [int(v) for v in md["layers"]]
        keys.append(key)
        y.append(int(md["right_index"]))
        original.append(r["hidden"][ix["full_context"]].astype(np.float32))
        swapped.append(r["hidden"][ix["profile_order_swap"]].astype(np.float32))
    y = np.asarray(y, int)
    original, swapped = np.stack(original), np.stack(swapped)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    results = []
    saved = {}

    for lp, layer in enumerate(layers):
        preds = {name: np.empty(len(y), int) for name in
                 ["orig_to_orig", "orig_to_swap", "swap_to_swap",
                  "swap_to_orig", "paired_to_orig", "paired_to_swap"]}
        for tr, te in cv.split(original, y):
            nc = min(args.pca_components, len(tr) - 2, original.shape[2])
            mo = model(nc, args.seed).fit(original[tr, lp], y[tr])
            preds["orig_to_orig"][te] = mo.predict(original[te, lp])
            preds["orig_to_swap"][te] = mo.predict(swapped[te, lp])
            ms = model(nc, args.seed).fit(swapped[tr, lp], y[tr])
            preds["swap_to_swap"][te] = ms.predict(swapped[te, lp])
            preds["swap_to_orig"][te] = ms.predict(original[te, lp])
            # Same answer-identity label for both orders; held-out split is by item.
            mp = model(min(args.pca_components, 2 * len(tr) - 2, original.shape[2]),
                       args.seed).fit(
                np.concatenate([original[tr, lp], swapped[tr, lp]]),
                np.concatenate([y[tr], y[tr]]))
            preds["paired_to_orig"][te] = mp.predict(original[te, lp])
            preds["paired_to_swap"][te] = mp.predict(swapped[te, lp])
        row = {"layer": layer}
        for name, pred in preds.items():
            row[name] = {
                "content_identity_accuracy": float(np.mean(pred == y)),
                "display_position_accuracy": float(np.mean(pred == (1 - y))),
            }
            saved[f"L{layer}_{name}"] = pred
        # Direct answer to follows-content vs follows-position for cross-order transfer.
        cross_content = (np.sum(preds["orig_to_swap"] == y)
                         + np.sum(preds["swap_to_orig"] == y))
        cross_total = 2 * len(y)
        row["cross_order_content_minus_position"] = float(
            (2 * cross_content - cross_total) / cross_total)
        row["paired_order_invariant_mean_accuracy"] = float(np.mean([
            np.mean(preds["paired_to_orig"] == y),
            np.mean(preds["paired_to_swap"] == y)]))
        results.append(row)

    out = {
        "n_items": len(y), "layers": layers,
        "label": "original answer identity index; unchanged after profile swap",
        "protocol": "5-fold item-disjoint cross-fitting; PCA+L2 logistic",
        "interpretation": (
            "On swapped inputs, prediction y follows person identity; prediction 1-y follows "
            "display position. Cross-order transfer is the discriminating statistic."),
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.output / "predictions.npz", keys=np.asarray(keys), gold=y,
                        **saved)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
