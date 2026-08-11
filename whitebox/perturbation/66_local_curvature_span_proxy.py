# -*- coding: utf-8 -*-
"""Approximate local single-span neutralization gains with shared derivatives.

For span direction v_i in token-gate space,

    u_i = S(0) - S(v_i)
        ~= -g(0)^T v_i - 1/2 v_i^T H(0) v_i.

The quadratic forms for every span are estimated simultaneously with
Hutchinson probes.  A Hessian-vector product is approximated by a central
finite difference of gradients, avoiding materializing the Hessian and the
large memory cost of second-order autograd through the language model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import Item, SpanAttributor, pearson, set_seed, spearman
from importlib import import_module


def recall_at(pred, truth, k_truth: int, k_pred: int) -> float:
    a = set(np.argsort(-np.abs(truth))[:k_truth].tolist())
    b = set(np.argsort(-np.abs(pred))[:k_pred].tolist())
    return len(a & b) / max(1, len(a))


def metrics(pred, truth):
    return {
        "spearman_signed": spearman(pred, truth),
        "pearson_signed": pearson(pred, truth),
        "spearman_abs": spearman(np.abs(pred), np.abs(truth)),
        "mae": float(np.mean(np.abs(pred - truth))),
        "recall_top5_at5": recall_at(pred, truth, 5, 5),
        "recall_top5_at10": recall_at(pred, truth, 5, 10),
        "recall_top5_at20": recall_at(pred, truth, 5, 20),
    }


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--in61", required=True)
    ap.add_argument("--out", default="runs/66_local_curvature.json")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    rec = json.loads(open(args.in61).readline())
    load_model = import_module("61_grad_span_proposal").load_model
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=1)
    item = Item.from_dict(rec)
    prep = att.prepare(item)
    spans = att.build_word_spans(prep, widths=(2, 3), stride=1)

    # Preserve the exact span order/coverage whose measured labels are in 61.
    by_key = {(s.start, s.end): s for s in spans}
    spans = [by_key[(int(s["start"]), int(s["end"]))] for s in rec["spans"]]
    for i, s in enumerate(spans):
        s.idx = i
    prep.spans = spans

    truth = np.asarray([s["u"] for s in rec["spans"]], dtype=float)
    ig = np.asarray([s["ig"] for s in rec["spans"]], dtype=float)
    P = prep.prompt_ids.shape[0]
    V = np.zeros((len(spans), P), dtype=np.float64)
    for i, s in enumerate(spans):
        V[i, s.start:s.end] = 1.0

    t0 = time.time()
    g0 = att.grad_alpha(prep).astype(np.float64)
    linear = -(V @ g0)

    rng = np.random.default_rng(args.seed)
    q_each = np.zeros((args.probes, len(spans)), dtype=np.float64)
    for r in range(args.probes):
        z = np.zeros(P, dtype=np.float32)
        z[prep.ctx_start:prep.ctx_end] = rng.choice(
            [-1.0, 1.0], size=prep.ctx_end - prep.ctx_start)
        za = torch.tensor(z, device=args.device)
        gp = att.grad_alpha(prep, args.epsilon * za).astype(np.float64)
        gm = att.grad_alpha(prep, -args.epsilon * za).astype(np.float64)
        hz = (gp - gm) / (2.0 * args.epsilon)
        # E[(v.z)(v.Hz)] = v.H.v for Rademacher z.
        q_each[r] = (V @ z.astype(np.float64)) * (V @ hz)
        print(f"probe {r + 1}/{args.probes}", flush=True)

    reports = {}
    for R in [x for x in (1, 2, 4, 8, 16, 32) if x <= args.probes]:
        curvature = q_each[:R].mean(axis=0)
        quadratic = linear - 0.5 * curvature
        reports[str(R)] = {
            "linear": metrics(linear, truth),
            "quadratic": metrics(quadratic, truth),
            "top20_quadratic": [spans[int(i)].text for i in
                                np.argsort(-np.abs(quadratic))[:20]],
        }

    true_top = np.argsort(-np.abs(truth))[:20]
    out = {
        "item_id": rec["item_id"], "n_spans": len(spans),
        "probes": args.probes, "epsilon": args.epsilon,
        "gradient_evaluations": 1 + 2 * args.probes,
        "truth_top20": [{"text": spans[int(i)].text,
                          "u": float(truth[i]), "idx": int(i)} for i in true_top],
        "ig": metrics(ig, truth),
        "reports": reports,
        "linear": linear.tolist(),
        "curvature_samples": q_each.tolist(),
        "runtime_sec": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"ig": out["ig"], "reports": reports,
                      "runtime_sec": out["runtime_sec"]}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
