#!/usr/bin/env python3
"""Compare perturbation-aware hidden-delta feature constructions."""

import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = "/home/tong56/whitebox/perturbation/runs"
MANIFEST = f"{ROOT}/73_split_n700_train600_test100.json"
ORACLE = f"{ROOT}/73_oracle_top11_n700.jsonl"
CACHE = f"{ROOT}/73_hidden_delta_top11_n700"
OUT = f"{ROOT}/75_delta_construction_cv_train600.json"
R = 8


def load():
    meta = {x["key"]: x for x in json.load(open(MANIFEST))}
    oracle = {x["key"]: x for x in map(json.loads, open(ORACLE))}
    rows = []
    for path in sorted(glob.glob(f"{CACHE}/*.npz")):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            u = np.asarray(z["top_u"], np.float32)
            ua = np.asarray(oracle[key]["u"], np.float32)
            s0 = float(oracle[key]["S0"])
            margin = np.r_[
                u,
                np.abs(u),
                u / (abs(s0) + 1e-6),
                u.max(initial=0),
                u.min(initial=0),
                np.abs(u).mean(),
                np.abs(u).sum() / (np.abs(ua).sum() + 1e-9),
                np.mean(ua > 0),
                np.std(ua),
            ].astype(np.float32)
            h = np.asarray(z["answer_last"], np.float32)[0]
            rows.append((key, meta[key], margin, h[0], h[1:] - h[0], u))
    assert len(rows) == 700
    return rows


def geometry(h0, delta):
    """Rotation-invariant geometry of an original state and its 11 interventions."""
    eps = 1e-8
    dn = np.linalg.norm(delta, axis=1)
    h0n = np.linalg.norm(h0)
    masked = h0[None] + delta
    mn = np.linalg.norm(masked, axis=1)
    cos_d_h0 = (delta @ h0) / (dn * h0n + eps)
    cos_mask_h0 = (masked @ h0) / (mn * h0n + eps)
    unit = delta / (dn[:, None] + eps)
    gram_off = (unit @ unit.T)[np.triu_indices(11, 1)]
    singular = np.linalg.svd(delta, compute_uv=False)
    energy = singular**2
    spectrum = energy / (energy.sum() + eps)
    effective_rank = np.exp(-(spectrum * np.log(spectrum + eps)).sum())
    scalars = np.asarray(
        [
            dn.mean(), dn.std(), dn.min(), dn.max(),
            dn.max() / (dn.mean() + eps),
            gram_off.mean(), gram_off.std(), gram_off.min(), gram_off.max(),
            effective_rank, spectrum[0], spectrum[:3].sum(),
        ], np.float32
    )
    return np.r_[dn / (h0n + eps), cos_d_h0, cos_mask_h0, gram_off, spectrum, scalars]


def weighted(delta, u, positive):
    mask = u > 0 if positive else u < 0
    w = u if positive else -u
    if not mask.any():
        return np.zeros(delta.shape[1], np.float32)
    return (delta[mask] * w[mask, None]).sum(0) / (np.abs(w[mask]).sum() + 1e-9)


def scale(train, val):
    s = StandardScaler().fit(train)
    return s.transform(train), s.transform(val)


def build_fold(M, H0, D, U, fit, val):
    mf, mv = scale(M[fit], M[val])
    gf, gv = scale(np.stack([geometry(H0[i], D[i]) for i in fit]),
                   np.stack([geometry(H0[i], D[i]) for i in val]))

    hs = StandardScaler().fit(H0[fit])
    hpca = PCA(R, whiten=True, svd_solver="randomized", random_state=42).fit(hs.transform(H0[fit]))
    h0f = hpca.transform(hs.transform(H0[fit]))
    h0v = hpca.transform(hs.transform(H0[val]))

    # A shared perturbation basis is fit to all individual train deltas, so rank-wise
    # coordinates are directly comparable and no label/test information enters it.
    ds = StandardScaler().fit(D[fit].reshape(-1, D.shape[-1]))
    dfit_flat = ds.transform(D[fit].reshape(-1, D.shape[-1]))
    dpca = PCA(R, whiten=True, svd_solver="randomized", random_state=42).fit(dfit_flat)
    df = dpca.transform(dfit_flat).reshape(len(fit), 11, R)
    dv = dpca.transform(ds.transform(D[val].reshape(-1, D.shape[-1]))).reshape(len(val), 11, R)

    def moments(x, u):
        rank = np.linspace(-1, 1, 11, dtype=np.float32)
        rank = rank - rank.mean()
        slope = np.einsum("nkr,k->nr", x, rank) / np.sum(rank**2)
        uc = u - u.mean(1, keepdims=True)
        xc = x - x.mean(1, keepdims=True)
        corr_u = np.einsum("nkr,nk->nr", xc, uc) / (
            np.sqrt(np.sum(xc**2, axis=1) * np.sum(uc**2, axis=1)[:, None]) + 1e-8
        )
        return np.c_[x.mean(1), x.std(1), x.min(1), x.max(1),
                     np.sqrt(np.mean(x**2, axis=1)), slope, corr_u]

    momf, momv = scale(moments(df, U[fit]), moments(dv, U[val]))

    pos = np.stack([weighted(D[i], U[i], True) for i in range(len(D))])
    neg = np.stack([weighted(D[i], U[i], False) for i in range(len(D))])
    base_parts_f, base_parts_v = [mf, h0f], [mv, h0v]
    for branch in (pos, neg):
        ss = StandardScaler().fit(branch[fit])
        pp = PCA(R, whiten=True, svd_solver="randomized", random_state=42).fit(ss.transform(branch[fit]))
        base_parts_f.append(pp.transform(ss.transform(branch[fit])))
        base_parts_v.append(pp.transform(ss.transform(branch[val])))
    basef, basev = np.c_[tuple(base_parts_f)], np.c_[tuple(base_parts_v)]
    orderedf, orderedv = np.c_[mf, h0f, df.reshape(len(fit), -1)], np.c_[mv, h0v, dv.reshape(len(val), -1)]
    momentsf, momentsv = np.c_[mf, h0f, momf], np.c_[mv, h0v, momv]
    return {
        "baseline_posneg_mean": (basef, basev),
        "baseline_plus_geometry": (np.c_[basef, gf], np.c_[basev, gv]),
        "ordered_delta": (orderedf, orderedv),
        "ordered_delta_plus_geometry": (np.c_[orderedf, gf], np.c_[orderedv, gv]),
        "delta_moments": (momentsf, momentsv),
        "delta_moments_plus_geometry": (np.c_[momentsf, gf], np.c_[momentsv, gv]),
        "geometry_only_delta": (np.c_[mf, h0f, gf], np.c_[mv, h0v, gv]),
    }


def main():
    rows = load()
    split = np.asarray([x[1]["split"] for x in rows])
    groups = np.asarray([x[1]["group"] for x in rows])
    y = np.asarray([int(x[1]["correct"]) for x in rows])
    M = np.stack([x[2] for x in rows]); H0 = np.stack([x[3] for x in rows])
    D = np.stack([x[4] for x in rows]); U = np.stack([x[5] for x in rows])
    train = np.flatnonzero(split == "train"); test = np.flatnonzero(split == "test")
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    scores = defaultdict(list)
    for fold, (a, b) in enumerate(cv.split(train, y[train], groups[train]), 1):
        fit, val = train[a], train[b]
        variants = build_fold(M, H0, D, U, fit, val)
        for name, (xf, xv) in variants.items():
            clf = LogisticRegression(C=.5, max_iter=5000, class_weight="balanced", random_state=42).fit(xf, y[fit])
            p = clf.predict_proba(xv)[:, 1]
            scores[name].append({"fold": fold, "n": len(val), "feature_dims": xf.shape[1],
                                 "auroc": roc_auc_score(y[val], p),
                                 "auprc": average_precision_score(y[val], p),
                                 "balanced_accuracy": balanced_accuracy_score(y[val], p >= .5)})
        print(f"finished fold {fold}/5", flush=True)
    summary = {name: {"feature_dims": vals[0]["feature_dims"], "folds": vals,
                      "mean_auroc": float(np.mean([v["auroc"] for v in vals])),
                      "std_auroc": float(np.std([v["auroc"] for v in vals], ddof=1)),
                      "mean_auprc": float(np.mean([v["auprc"] for v in vals])),
                      "mean_balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in vals]))}
               for name, vals in scores.items()}
    selected = max(summary, key=lambda n: summary[n]["mean_auroc"])
    variants = build_fold(M, H0, D, U, train, test)
    xf, xt = variants[selected]
    clf = LogisticRegression(C=.5, max_iter=5000, class_weight="balanced", random_state=42).fit(xf, y[train])
    p = clf.predict_proba(xt)[:, 1]
    report = {"protocol": "5-fold train-only StratifiedGroupKFold; heldout evaluated only for CV winner",
              "shared_delta_pca_dims": R, "variants": summary, "selected": selected,
              "heldout_reference": {"n": len(test), "auroc": roc_auc_score(y[test], p),
                                    "auprc": average_precision_score(y[test], p),
                                    "balanced_accuracy_at_0.5": balanced_accuracy_score(y[test], p >= .5)}}
    with open(OUT, "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)); print(OUT)


if __name__ == "__main__": main()
