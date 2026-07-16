"""
Evaluation protocol for small-n hallucination detection.

Replaces the single train/test split (which at n=100 gives a 25-example
test set where AUROC has +/- 0.10 std across seeds) with:

  1. Repeated stratified k-fold CV -> mean AUROC +/- std over folds
  2. A permutation test -> p-value that AUROC > chance
  3. Feature-block ablation (logit / role-lap / tokenflow / lapeig)
Input: a jsonl where each record has "hallucinated": bool and a flat
"features": {name: value} dict (from TokenIndexedDetector.extract).

Usage:
  python eval_protocol.py features.jsonl
  python eval_protocol.py features.jsonl --blocks logit rho lap
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BLOCKS = {
    "logit": ("lp_", "ent_", "margin_"),
    "rho": ("rho_",),
    "rolelap": ("lapc", "laps", "lap_logratio", "lap_ans", "lapc_ans", "laps_ans"),
    "lapeig": ("lapeig",),
}


def load(path):
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    y = np.array([r["hallucinated"] for r in recs], dtype=int)
    keys = sorted(set().union(*[r["features"].keys() for r in recs]))
    X = np.array([[r["features"].get(k, 0.0) for k in keys] for r in recs])
    return X, y, keys


def select(keys, block_names):
    prefixes = tuple(p for b in block_names for p in BLOCKS[b])
    return [i for i, k in enumerate(keys) if k.startswith(prefixes)]


def cv_auroc(X, y, n_splits=5, n_repeats=10, seed=0):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l1", C=0.5,
                                           solver="liblinear", max_iter=5000,
                                           class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    aucs = []
    for tr, te in rskf.split(X, y):
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def permutation_pvalue(X, y, n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    obs = cv_auroc(X, y, n_repeats=2).mean()
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        null.append(cv_auroc(X, yp, n_repeats=1).mean())
    null = np.array(null)
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_jsonl")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help=f"subset of {list(BLOCKS)} (default: all)")
    ap.add_argument("--permutation", action="store_true")
    args = ap.parse_args()

    X, y, keys = load(args.features_jsonl)
    print(f"n={len(y)}  positives={y.sum()}  features={X.shape[1]}")

    combos = ([["logit"], ["rho"], ["rolelap"], ["lapeig"],
               ["logit", "rho"], ["logit", "rho", "rolelap"],
               list(BLOCKS)]
              if args.blocks is None else [args.blocks])

    for blocks in combos:
        cols = select(keys, blocks)
        if not cols:
            print(f"{'+'.join(blocks):28s} (no matching features)")
            continue
        aucs = cv_auroc(X[:, cols], y)
        print(f"{'+'.join(blocks):28s} AUROC {aucs.mean():.3f} +/- {aucs.std():.3f} "
              f"(5-fold x10)")

    if args.permutation:
        cols = select(keys, list(BLOCKS))
        obs, p = permutation_pvalue(X[:, cols], y)
        print(f"\npermutation test: observed AUROC={obs:.3f}, p={p:.3f}")


if __name__ == "__main__":
    main()
