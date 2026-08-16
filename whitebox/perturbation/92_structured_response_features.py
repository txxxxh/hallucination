#!/usr/bin/env python3
"""Evaluate compact per-span and spectral features on the known>0.5 set.

All learned transforms (scalers and PCA) are fit inside each grouped CV fold.
The compact spectrum is rotation invariant and computed per item without labels.
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent / "runs"


def weighted(delta, u, positive):
    mask = u > 0 if positive else u < 0
    w = u if positive else -u
    if not mask.any():
        return np.zeros(delta.shape[1], np.float32)
    return (delta[mask] * w[mask, None]).sum(0) / (np.abs(w[mask]).sum() + 1e-9)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compact_spectrum(delta, u):
    """Small rotation-invariant description of a top-5 response matrix."""
    eps = 1e-8
    norms = np.linalg.norm(delta, axis=1)
    unit = delta / (norms[:, None] + eps)
    cosines = (unit @ unit.T)[np.triu_indices(len(delta), 1)]
    singular = np.linalg.svd(delta, compute_uv=False)
    energy = singular**2
    frac = energy / (energy.sum() + eps)
    effective_rank = np.exp(-(frac * np.log(frac + eps)).sum())
    stable_rank = energy.sum() / (energy[0] + eps)
    pos = weighted(delta, u, True)
    neg = weighted(delta, u, False)
    pos_neg_cos = float(pos @ neg / (np.linalg.norm(pos) * np.linalg.norm(neg) + eps))
    return np.asarray([
        frac[0], frac[:2].sum(), frac[:3].sum(), effective_rank, stable_rank,
        singular[0] / (singular[1] + eps), singular[-1] / (singular[0] + eps),
        norms.mean(), norms.std(), norms.max() / (norms.mean() + eps),
        cosines.mean(), cosines.std(), cosines.min(), cosines.max(),
        corr(u, norms), corr(np.abs(u), norms),
        corr(rankdata(u), rankdata(norms)), pos_neg_cos,
    ], np.float32)


def load(topk):
    source = {x["key"]: x for x in map(json.loads, (ROOT / "88_known_gt05_n1084.jsonl").open())}
    oracle = {x["key"]: x for x in map(json.loads, (ROOT / "88_oracle_top11_known_gt05.jsonl").open())}
    rows = []
    for path in sorted(glob.glob(str(ROOT / "88_hidden_delta_top11_known_gt05" / "*.npz"))):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            u = np.asarray(z["top_u"], np.float32)[:topk]
            all_u = np.asarray(oracle[key]["u"], np.float32)
            s0 = float(oracle[key]["S0"])
            margin = np.r_[u, np.abs(u), u / (abs(s0) + 1e-6),
                           u.max(initial=0), u.min(initial=0), np.abs(u).mean(),
                           np.abs(u).sum() / (np.abs(all_u).sum() + 1e-9),
                           np.mean(all_u > 0), np.std(all_u)].astype(np.float32)
            h = np.asarray(z["answer_last"], np.float32)[0]
            h0, delta = h[0], h[1:topk + 1] - h[0]
            rows.append((key, str(source[key]["group"]), int(source[key]["correct"]),
                         margin, h0, weighted(delta, u, True), weighted(delta, u, False),
                         delta, u, compact_spectrum(delta, u)))
    assert len(rows) == 1084
    return rows


def scale(train, test):
    s = StandardScaler().fit(train)
    return s.transform(train), s.transform(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--baseline-pca", type=int, default=12)
    ap.add_argument("--span-pcas", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--Cs", type=float, nargs="+", default=[.01, .03, .05, .075, .1, .15, .3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", type=Path, default=ROOT / "92_structured_response_report.json")
    args = ap.parse_args()
    rows = load(args.topk)
    y = np.asarray([r[2] for r in rows]); groups = np.asarray([r[1] for r in rows])
    M = np.stack([r[3] for r in rows]); base = [np.stack([r[i] for r in rows]) for i in (4, 5, 6)]
    D = np.stack([r[7] for r in rows]); U = np.stack([r[8] for r in rows]); S = np.stack([r[9] for r in rows])
    variants = ("baseline", "plus_span", "plus_spectrum", "plus_both")
    configs = list(product(variants, args.span_pcas, args.Cs))
    collected = {c: [] for c in configs}
    for seed in args.seeds:
        pred = {c: np.zeros(len(y), np.float32) for c in configs}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(M, y, groups), 1):
            mt, mv = scale(M[tr], M[te]); btr, bte = [mt], [mv]
            for block in base:
                sc = StandardScaler().fit(block[tr]); zt = sc.transform(block[tr])
                pc = PCA(args.baseline_pca, whiten=True, svd_solver="randomized", random_state=seed).fit(zt)
                btr.append(pc.transform(zt)); bte.append(pc.transform(sc.transform(block[te])))
            base_tr, base_te = np.concatenate(btr, 1), np.concatenate(bte, 1)
            st, sv = scale(S[tr], S[te])
            # One shared basis across every individual training response.
            ds = StandardScaler().fit(D[tr].reshape(-1, D.shape[-1]))
            zflat = ds.transform(D[tr].reshape(-1, D.shape[-1]))
            pc = PCA(max(args.span_pcas), whiten=True, svd_solver="randomized", random_state=seed).fit(zflat)
            dztr = pc.transform(zflat).reshape(len(tr), args.topk, -1)
            dzte = pc.transform(ds.transform(D[te].reshape(-1, D.shape[-1]))).reshape(len(te), args.topk, -1)
            for dim in args.span_pcas:
                def moments(z, u):
                    z = z[:, :, :dim]
                    rank = np.linspace(1., -1., args.topk, dtype=np.float32); rank -= rank.mean()
                    slope = np.einsum("nkr,k->nr", z, rank) / np.sum(rank**2)
                    uc = u - u.mean(1, keepdims=True); zc = z - z.mean(1, keepdims=True)
                    cu = np.einsum("nkr,nk->nr", zc, uc) / (np.sqrt(np.sum(zc**2, 1) * np.sum(uc**2, 1)[:, None]) + 1e-8)
                    return np.c_[z.mean(1), z.std(1), z.min(1), z.max(1),
                                 np.sqrt(np.mean(z**2, 1)), slope, cu]
                raw_tr, raw_te = moments(dztr, U[tr]), moments(dzte, U[te])
                ft, fv = scale(raw_tr, raw_te)
                parts = {
                    "baseline": (base_tr, base_te),
                    "plus_span": (np.c_[base_tr, ft], np.c_[base_te, fv]),
                    "plus_spectrum": (np.c_[base_tr, st], np.c_[base_te, sv]),
                    "plus_both": (np.c_[base_tr, ft, st], np.c_[base_te, fv, sv]),
                }
                for variant in variants:
                    xtr, xte = parts[variant]
                    for C in args.Cs:
                        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                                 solver="liblinear", random_state=seed).fit(xtr, y[tr])
                        pred[(variant, dim, C)][te] = clf.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for cfg, p in pred.items():
            collected[cfg].append({"auroc": float(roc_auc_score(y, p)),
                                   "auprc": float(average_precision_score(y, p)),
                                   "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))})
    results = []
    for cfg, values in collected.items():
        row = dict(zip(("variant", "span_pca", "C"), cfg))
        for metric in ("auroc", "auprc", "balanced_accuracy"):
            v = np.asarray([x[metric] for x in values]); row[f"mean_{metric}"] = float(v.mean()); row[f"std_{metric}"] = float(v.std(ddof=1))
        row["per_seed"] = values; results.append(row)
    results.sort(key=lambda x: (x["mean_auroc"], x["mean_auprc"]), reverse=True)
    report = {"warning": "Same-data grouped-CV feature selection; use nested CV or new data for an unbiased estimate.",
              "n": len(y), "topk": args.topk, "baseline_pca": args.baseline_pca,
              "spectrum_dims": S.shape[1], "n_configs": len(configs), "results": results}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(args.out), "top_12": results[:12]}, indent=2))


if __name__ == "__main__":
    main()
