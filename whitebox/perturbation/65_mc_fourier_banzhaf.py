# -*- coding: utf-8 -*-
"""Exact and Monte-Carlo Banzhaf/Fourier interaction experiment.

For m disjoint candidate spans, evaluate u(x)=S(0)-S(x) on every Boolean
vertex (optional cache), then compare exact uniform-context Banzhaf effects to
T-sample Monte Carlo estimators. A fixed first-order gate-gradient surrogate is
used only as a control variate, never as the estimand.
"""
from __future__ import annotations

import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import (Item, Span, SpanAttributor, set_seed, spearman,
                           exhaustive_select, second_order_objective)


def load_model(name, dtype, device):
    import torch
    if os.environ.get("SPANATTR_DISABLE_NATIVE_BMM") == "1":
        from torch._native.registry import deregister_op_overrides
        deregister_op_overrides(disable_op_symbols="bmm")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dt, attn_implementation="eager").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def effects(y, chi):
    """Correctly normalized uniform Banzhaf main and pair effects."""
    n = len(y)
    main = 2.0 * np.mean(y[:, None] * chi, axis=0)
    pair = 4.0 * (chi.T @ (y[:, None] * chi)) / n
    np.fill_diagonal(pair, 0.0)
    return main, pair


def contribution_matrices(y, chi, proxy=None, proxy_main=None):
    """Per-sample main/pair contributions, optionally residualized at c=1."""
    m = chi.shape[1]
    if proxy is None:
        residual = y
        base_main = np.zeros(m)
    else:
        residual = y - proxy
        base_main = np.asarray(proxy_main)
    main = 2.0 * residual[:, None] * chi + base_main[None, :]
    pairs = list(itertools.combinations(range(m), 2))
    pair = np.stack([4.0 * residual * chi[:, i] * chi[:, j]
                     for i, j in pairs], axis=1)
    return main, pair, pairs


def crossfit_cv(A, C, gamma, rng):
    """Two-fold cross-fit c=Cov(A,C)/Var(C); returns unbiased pseudo-values."""
    n, p = A.shape
    order = rng.permutation(n)
    folds = [order[::2], order[1::2]]
    out = np.empty_like(A)
    cs = []
    for test, train in [(folds[0], folds[1]), (folds[1], folds[0])]:
        aa, cc = A[train], C[train]
        ac = aa - aa.mean(0); c0 = cc - cc.mean(0)
        cov = np.mean(ac * c0, axis=0)
        var = np.mean(c0 * c0, axis=0)
        coef = np.divide(cov, var, out=np.zeros(p), where=var > 1e-12)
        out[test] = A[test] - (C[test] - gamma[None, :]) * coef[None, :]
        cs.append(coef)
    return out, np.mean(cs, axis=0)


def bootstrap_ci(contrib, rng, draws=500):
    n = len(contrib)
    vals = np.empty((draws, contrib.shape[1]), dtype=float)
    for b in range(draws):
        ids = rng.integers(0, n, size=n)
        vals[b] = contrib[ids].mean(axis=0)
    return np.quantile(vals, .025, axis=0), np.quantile(vals, .975, axis=0)


def matrix_from_pairs(v, pairs, m):
    out = np.zeros((m, m), dtype=float)
    for x, (i, j) in zip(v, pairs):
        out[i, j] = out[j, i] = x
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--in61", required=True)
    ap.add_argument("--local62", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_rows", type=int, default=32)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50,100,200,400])
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    r61 = json.loads(open(args.in61).readline())
    local = json.loads(open(args.local62).readline()) if args.local62 else None
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.max_rows)
    item = Item(r61["item_id"], r61["context"], r61["question"],
                r61["gold"], r61["pred"],
                context_prefix=r61.get("context_prefix", ""),
                gold_variants=r61.get("gold_variants", []),
                pred_variants=r61.get("pred_variants", []))
    prep = att.prepare(item)
    cand = r61["candidates"]
    meta = [r61["spans"][i] for i in cand]
    prep.spans = [Span(i, s["start"], s["end"], s["text"])
                  for i, s in enumerate(meta)]
    m = len(prep.spans)
    n_all = 1 << m
    X = ((np.arange(n_all)[:, None] >> np.arange(m)) & 1).astype(np.int8)
    chi = 2.0 * X - 1.0
    S0 = att.S0(prep)

    t0 = time.time()
    if os.path.exists(args.cache):
        cache = np.load(args.cache)
        y = cache["u"]
        if len(y) != n_all:
            raise ValueError("cache has wrong number of vertices")
        print(f"Loaded {n_all} exact vertices from {args.cache}")
    else:
        sets = [list(np.flatnonzero(row)) for row in X]
        y, _ = att.u_of_sets(prep, sets, S0=S0)
        np.savez_compressed(args.cache, u=y, masks=X)
        print(f"Computed and cached {n_all} exact vertices in {time.time()-t0:.1f}s")

    # Fixed white-box additive surrogate from one gradient at the original prompt.
    g = att.grad_alpha(prep)
    grad_main = np.asarray([-g[s.start:s.end].sum() for s in prep.spans])
    proxy_all = X @ grad_main

    exact_main, exact_pair_m = effects(y, chi)
    pair_ids = list(itertools.combinations(range(m), 2))
    exact_pair = np.asarray([exact_pair_m[i,j] for i,j in pair_ids])
    local_pair = None
    if local is not None:
        local_pair = np.asarray([local["I"][i][j] for i,j in pair_ids])

    rng = np.random.default_rng(args.seed)
    reports = []
    for T in args.budgets:
        ids = rng.integers(0, n_all, size=T)  # iid Bernoulli vertices, with replacement
        yt, ct, xt = y[ids], chi[ids], X[ids]
        gp = xt @ grad_main
        A1, A2, pairs = contribution_matrices(yt, ct)
        CV1, CV2, _ = contribution_matrices(yt, ct, gp, grad_main)
        # Cross-fit optimal c using the fixed proxy; C expectations are known:
        # gamma_main=grad_main and gamma_pair=0.
        C1, C2, _ = contribution_matrices(gp, ct)
        CF1, c_main = crossfit_cv(A1, C1, grad_main,
                                  np.random.default_rng(args.seed + T))
        CF2, c_pair = crossfit_cv(A2, C2, np.zeros(len(pairs)),
                                  np.random.default_rng(args.seed + 2*T))
        estimators = {"mc": (A1, A2), "cv_c1": (CV1, CV2),
                      "cv_crossfit": (CF1, CF2)}
        methods = {}
        for name, (M1, M2) in estimators.items():
            e1, e2 = M1.mean(0), M2.mean(0)
            lo, hi = bootstrap_ci(M2, np.random.default_rng(args.seed+T+len(name)),
                                  draws=args.bootstrap)
            I_est = matrix_from_pairs(e2, pairs, m)
            sel = exhaustive_select(e1, I_est, min(args.k,m), cap=50000)
            methods[name] = {
                "rho_main": spearman(e1, exact_main),
                "mae_main": float(np.mean(np.abs(e1-exact_main))),
                "rho_pair": spearman(e2, exact_pair),
                "mae_pair": float(np.mean(np.abs(e2-exact_pair))),
                "pair_ci_coverage": float(np.mean((lo<=exact_pair)&(exact_pair<=hi))),
                "mean_pair_ci_width": float(np.mean(hi-lo)),
                "mean_pair_variance": float(np.mean(np.var(M2,axis=0,ddof=1))),
                "selection": sel,
                "selection_text": [prep.spans[i].text for i in sel],
                "objective": float(second_order_objective(sel,e1,I_est)),
            }
        methods["cv_crossfit"]["mean_abs_c_pair"] = float(np.mean(np.abs(c_pair)))
        methods["cv_crossfit"]["mean_abs_c_main"] = float(np.mean(np.abs(c_main)))
        reports.append({"T":T,"methods":methods})
        print(f"T={T}: " + " | ".join(
            f"{q} rhoI={v['rho_pair']:+.3f} MAE={v['mae_pair']:.3f} "
            f"CI={v['pair_ci_coverage']:.2f} sel={v['selection_text']}"
            for q,v in methods.items()))

    exact_sel = exhaustive_select(exact_main, exact_pair_m, min(args.k,m), cap=50000)
    result = {
        "item_id": r61["item_id"], "m":m, "n_vertices":n_all,
        "span_text":[s.text for s in prep.spans],
        "grad_main":[float(x) for x in grad_main],
        "exact_banzhaf_main":[float(x) for x in exact_main],
        "exact_banzhaf_pair":[[float(x) for x in row] for row in exact_pair_m],
        "exact_selection":exact_sel,
        "exact_selection_text":[prep.spans[i].text for i in exact_sel],
        "rho_exact_banzhaf_vs_local_fd": (spearman(exact_pair,local_pair)
                                            if local_pair is not None else None),
        "mae_exact_banzhaf_vs_local_fd": (float(np.mean(np.abs(exact_pair-local_pair)))
                                            if local_pair is not None else None),
        "reports":reports,
        "runtime_sec":time.time()-t0,
    }
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    with open(args.out,"w") as f: json.dump(result,f)
    print("Exact Banzhaf selection:",result["exact_selection_text"])
    print("Banzhaf/local rho:",result["rho_exact_banzhaf_vs_local_fd"])
    print("Wrote",args.out)


if __name__ == "__main__":
    main()
