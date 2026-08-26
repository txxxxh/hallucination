#!/usr/bin/env python3
"""Validation-only tuning of P+Aiersilan under Aiersilan random splits."""
from __future__ import annotations

import importlib
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


base = importlib.import_module("272_full_scientist_standard_upr_tables")
splitter = importlib.import_module("289_p_aiersilan_official_split")
OUT = base.RUNS / "290_tune_p_aiersilan_official"
AIERSILAN = base.RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"


def transformed(values, train, val, test, max_dim, seed):
    scaler = StandardScaler().fit(values[train])
    tr = scaler.transform(values[train]); va = scaler.transform(values[val])
    te = scaler.transform(values[test])
    if max_dim is not None:
        dim = min(max_dim, len(train) - 1, tr.shape[1])
        pca = PCA(dim, whiten=True, svd_solver="randomized",
                  random_state=seed).fit(tr)
        tr, va, te = pca.transform(tr), pca.transform(va), pca.transform(te)
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32)


def metric(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [row["key"] for row in rows]
    y = np.asarray([row["error"] for row in rows])
    saved = torch.load(AIERSILAN, map_location="cpu")
    a_by_key = {key: saved["hidden_states"][i, 14].float().numpy()
                for i, key in enumerate(saved["keys"])}
    a = np.stack([a_by_key[key] for key in keys])
    p_values = [np.stack([row["p_scalar"] for row in rows])]
    p_values += [np.stack([row["p_hidden"][j] for row in rows])
                 for j in range(4)]
    p_values += [np.stack([row["p_layer"] for row in rows])]

    seed_reports = []
    predictions = []
    for seed in base.SEEDS:
        train, val, test = splitter.split_indices(y, seed)
        # Fit each expensive transform once at its maximum dimension. Smaller
        # candidates use prefixes of the same train-fitted PCA basis.
        ps = [transformed(p_values[0], train, val, test, None, seed)]
        ps += [transformed(x, train, val, test, 16, seed) for x in p_values[1:5]]
        ps += [transformed(p_values[5], train, val, test, 96, seed)]
        aa = transformed(a, train, val, test, 192, seed)

        candidates = []
        for ph in (4, 8, 16):
            for pl in (24, 48, 72, 96):
                p_tr = np.c_[ps[0][0], *[z[0][:, :ph] for z in ps[1:5]],
                             ps[5][0][:, :pl]]
                p_va = np.c_[ps[0][1], *[z[1][:, :ph] for z in ps[1:5]],
                             ps[5][1][:, :pl]]
                p_te = np.c_[ps[0][2], *[z[2][:, :ph] for z in ps[1:5]],
                             ps[5][2][:, :pl]]
                for ad in (16, 32, 48, 64, 96, 128, 192):
                    xtr = np.c_[p_tr, aa[0][:, :ad]]
                    xva = np.c_[p_va, aa[1][:, :ad]]
                    xte = np.c_[p_te, aa[2][:, :ad]]
                    for c in (.003, .01, .03, .1, .3, 1., 3.):
                        model = LogisticRegression(
                            C=c, max_iter=5000, class_weight="balanced",
                            solver="liblinear", random_state=seed).fit(xtr, y[train])
                        va_score = model.predict_proba(xva)[:, 1]
                        candidates.append((metric(y[val], va_score)["auroc"],
                                           metric(y[val], va_score)["auprc"],
                                           f"lr_ph={ph}_pl={pl}_a={ad}_C={c:g}",
                                           model, xte, xtr, xva))

        # Add a compact set of nonlinear heads on the strongest conventional
        # representation; these are also selected only by validation AUROC.
        ph, pl, ad = 8, 48, 48
        xtr = np.c_[ps[0][0], *[z[0][:, :ph] for z in ps[1:5]],
                    ps[5][0][:, :pl], aa[0][:, :ad]]
        xva = np.c_[ps[0][1], *[z[1][:, :ph] for z in ps[1:5]],
                    ps[5][1][:, :pl], aa[1][:, :ad]]
        xte = np.c_[ps[0][2], *[z[2][:, :ph] for z in ps[1:5]],
                    ps[5][2][:, :pl], aa[2][:, :ad]]
        for leaves in (5, 9, 15):
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=.04, max_leaf_nodes=leaves,
                min_samples_leaf=20, l2_regularization=3.,
                random_state=seed).fit(xtr, y[train])
            va_score = model.predict_proba(xva)[:, 1]
            candidates.append((metric(y[val], va_score)["auroc"],
                               metric(y[val], va_score)["auprc"],
                               f"hist_leaves={leaves}", model, xte, xtr, xva))

        best = max(candidates, key=lambda item: (item[0], item[1]))
        _, _, name, model, best_xte, _, _ = best
        test_score = model.predict_proba(best_xte)[:, 1]
        test_metrics = metric(y[test], test_score)
        seed_reports.append({"seed": seed, "selected": name,
                             "validation": {"auroc": best[0], "auprc": best[1]},
                             "test": test_metrics,
                             "candidates": len(candidates)})
        for idx, score in zip(test, test_score):
            predictions.append({"seed": seed, "key": keys[idx],
                                "error": int(y[idx]), "score": float(score)})
        print(f"seed={seed} selected={name} val={best[0]:.6f} "
              f"test={test_metrics['auroc']:.6f}", flush=True)

    report = {
        "protocol": "Aiersilan stratified 70/10/20 seeds 42-44; hyperparameters selected independently per seed on validation AUROC; test untouched until final selected-model evaluation",
        "search": "P hidden PCA 4/8/16; P layer PCA 24/48/72/96; Aiersilan PCA 16/32/48/64/96/128/192; LR C .003..3; three HistGB heads",
        "per_seed": seed_reports,
        "test_mean": {k: float(np.mean([r["test"][k] for r in seed_reports]))
                      for k in ("auroc", "auprc")},
        "test_std": {k: float(np.std([r["test"][k] for r in seed_reports]))
                     for k in ("auroc", "auprc")},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
