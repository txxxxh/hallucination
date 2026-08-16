#!/usr/bin/env python3
"""Exact-vs-low-query pilot for pairwise span interactions.

The script first obtains an exact finite-difference interaction matrix for a
moderate, token-disjoint candidate set.  It then evaluates low-query recovery
methods against that same matrix.  CUR and hard-impute only expose the entries
that their stated query budget would have measured; oracle truncated SVD is an
unattainable upper bound on what rank-r structure alone could achieve.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, Span, SpanAttributor, nms_disjoint, set_seed


def offdiag_values(a):
    return np.asarray(a)[np.triu_indices(len(a), 1)]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def truncate_symmetric(a, rank):
    w, v = np.linalg.eigh((a + a.T) / 2)
    keep = np.argsort(-np.abs(w))[:min(rank, len(w))]
    out = (v[:, keep] * w[keep]) @ v[:, keep].T
    np.fill_diagonal(out, 0.0)
    return out


def cur_reconstruct(truth, anchors, ridge=1e-6):
    """Symmetric skeleton reconstruction from fully observed anchor columns."""
    anchors = np.asarray(sorted(set(int(x) for x in anchors)), dtype=int)
    c = truth[:, anchors]
    w = truth[np.ix_(anchors, anchors)]
    # pinv is deliberately regularized: interaction matrices are indefinite
    # and their anchor intersections can be nearly singular.
    scale = max(float(np.linalg.norm(w, 2)), 1.0)
    wp = np.linalg.pinv(w + ridge * scale * np.eye(len(w)))
    out = c @ wp @ c.T
    out = (out + out.T) / 2
    np.fill_diagonal(out, 0.0)
    return out, float(np.linalg.cond(w)) if len(w) else float("inf")


def hard_impute(truth, observed_pairs, rank, steps=200):
    """Iterative rank-r completion with exact projection onto observed pairs."""
    n = len(truth)
    mask = np.eye(n, dtype=bool)
    value = np.zeros_like(truth)
    for i, j in observed_pairs:
        mask[i, j] = mask[j, i] = True
        value[i, j] = value[j, i] = truth[i, j]
    out = value.copy()
    for _ in range(steps):
        old = out.copy()
        out = truncate_symmetric(out, rank)
        out[mask] = value[mask]
        if np.linalg.norm(out - old) <= 1e-7 * (np.linalg.norm(old) + 1e-8):
            break
    # Report a genuinely low-rank prediction, then restore queried entries so
    # downstream selection uses all information paid for by the method.
    pred = truncate_symmetric(out, rank)
    pred[mask] = value[mask]
    np.fill_diagonal(pred, 0.0)
    return pred


def metrics(pred, truth, singles):
    tri = np.triu_indices(len(truth), 1)
    p, y = pred[tri], truth[tri]
    abs_order_y = np.argsort(-np.abs(y))
    abs_order_p = np.argsort(-np.abs(p))
    top5 = set(abs_order_y[:min(5, len(y))])
    pair_utility_y, pair_utility_p = [], []
    pairs = list(zip(*tri))
    for k, (i, j) in enumerate(pairs):
        pair_utility_y.append(singles[i] + singles[j] + y[k])
        pair_utility_p.append(singles[i] + singles[j] + p[k])
    chosen = int(np.argmax(pair_utility_p))
    best = int(np.argmax(pair_utility_y))
    denom = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    return {
        "nrmse": float(np.sqrt(np.mean((p - y) ** 2)) / denom),
        "pearson": pearson(p, y),
        "top1_abs_hit": bool(abs_order_p[0] == abs_order_y[0]),
        "top5_abs_recall": float(len(set(abs_order_p[:5]) & top5) / max(len(top5), 1)),
        "best_utility_hit": bool(chosen == best),
        "best_utility_regret": float(pair_utility_y[best] - pair_utility_y[chosen]),
        "chosen_pair": [int(x) for x in pairs[chosen]],
        "true_best_pair": [int(x) for x in pairs[best]],
    }


def choose_disjoint(r, m):
    spans = [Span(idx=i, start=int(s["start"]), end=int(s["end"]), text=s["text"])
             for i, s in enumerate(r["spans"])]
    utility = np.abs([float(s["u"]) for s in r["spans"]])
    ids = nms_disjoint(utility, spans, m)
    return ids


def main():
    import importlib
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--in61", default="runs/61.jsonl")
    p.add_argument("--out", default="runs/92_lowrank_interaction_pilot.json")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--pair_budgets", type=int, nargs="+", default=[48, 96, 144])
    p.add_argument("--max_rows", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)

    rows = [json.loads(x) for x in open(a.in61) if x.strip()][:a.samples]
    loader = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = loader(a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean",
                         length_norm=True, max_rows=a.max_rows)
    results = []
    start_all = time.time()

    for sample_idx, r in enumerate(rows):
        item = Item(r["item_id"], r["context"], r["question"], r["gold"], r["pred"],
                    context_prefix=r.get("context_prefix", ""),
                    gold_variants=r.get("gold_variants", []),
                    pred_variants=r.get("pred_variants", []))
        prep = att.prepare(item)
        ids = choose_disjoint(r, a.m)
        meta = [r["spans"][i] for i in ids]
        prep.spans = [Span(idx=k, start=int(s["start"]), end=int(s["end"]), text=s["text"])
                      for k, s in enumerate(meta)]
        singles = np.asarray([float(s["u"]) for s in meta])
        pairs = list(itertools.combinations(range(len(meta)), 2))
        s0 = att.S0(prep)
        t0 = time.time()
        pair_gains, _ = att.u_of_sets(prep, [list(x) for x in pairs], S0=s0)
        truth = np.zeros((len(meta), len(meta)), dtype=float)
        for (i, j), gain in zip(pairs, pair_gains):
            truth[i, j] = truth[j, i] = float(gain - singles[i] - singles[j])
        exact_seconds = time.time() - t0

        sv = np.linalg.svd(truth, compute_uv=False)
        energy = np.cumsum(sv ** 2) / max(float(np.sum(sv ** 2)), 1e-12)
        methods = {"zero_interaction": {"queries": 0, **metrics(np.zeros_like(truth), truth, singles)}}
        for rank in a.ranks:
            oracle = truncate_symmetric(truth, rank)
            methods[f"oracle_svd_r{rank}"] = {"queries": len(pairs), "upper_bound": True,
                                               **metrics(oracle, truth, singles)}
            for selection in ("top_single", "random"):
                if selection == "top_single":
                    anchors = np.argsort(-np.abs(singles))[:min(rank, len(singles))]
                else:
                    rng_anchor = np.random.default_rng(a.seed + 100003 * sample_idx + rank)
                    anchors = rng_anchor.choice(len(singles), min(rank, len(singles)), replace=False)
                pred, cond = cur_reconstruct(truth, anchors)
                q = sum(1 for i, j in pairs if i in set(anchors) or j in set(anchors))
                methods[f"cur_{selection}_r{rank}"] = {
                    "queries": q, "anchors": [int(x) for x in anchors], "anchor_cond": cond,
                    **metrics(pred, truth, singles),
                }

        rng = np.random.default_rng(a.seed + 700001 * sample_idx)
        order = rng.permutation(len(pairs))
        for budget in a.pair_budgets:
            obs = [pairs[int(k)] for k in order[:min(budget, len(pairs))]]
            for rank in a.ranks:
                pred = hard_impute(truth, obs, rank)
                methods[f"hard_impute_q{len(obs)}_r{rank}"] = {
                    "queries": len(obs), **metrics(pred, truth, singles),
                }

        result = {
            "item_id": r["item_id"], "m": len(meta), "n_exact_pairs": len(pairs),
            "exact_seconds": exact_seconds, "candidate_source_ids": ids,
            "span_text": [s["text"] for s in meta], "single_u": singles.tolist(),
            "spectrum": sv.tolist(),
            "energy_at_rank": {str(rank): float(energy[min(rank, len(energy)) - 1]) for rank in a.ranks},
            "methods": methods,
        }
        results.append(result)
        print(f"[{sample_idx + 1}/{len(rows)}] {r['item_id']} m={len(meta)} "
              f"pairs={len(pairs)} exact={exact_seconds:.1f}s E4={result['energy_at_rank'].get('4')}",
              flush=True)

    # Macro averages make the pilot readable without hiding per-item failures.
    names = sorted(set.intersection(*(set(x["methods"]) for x in results))) if results else []
    summary = {}
    for name in names:
        ms = [x["methods"][name] for x in results]
        summary[name] = {
            "queries": float(np.mean([x["queries"] for x in ms])),
            "nrmse": float(np.mean([x["nrmse"] for x in ms])),
            "pearson": float(np.mean([x["pearson"] for x in ms])),
            "top1_abs_hit_rate": float(np.mean([x["top1_abs_hit"] for x in ms])),
            "top5_abs_recall": float(np.mean([x["top5_abs_recall"] for x in ms])),
            "best_utility_hit_rate": float(np.mean([x["best_utility_hit"] for x in ms])),
            "best_utility_regret": float(np.mean([x["best_utility_regret"] for x in ms])),
        }
    report = {"config": vars(a), "elapsed_seconds": time.time() - start_all,
              "items": results, "summary": summary}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
