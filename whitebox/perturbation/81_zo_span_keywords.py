#!/usr/bin/env python3
"""Stage 81: query-only embedding-direction search for keyword spans.

Each 2/3-word span gets a shared perturbation direction. Zeroth-order (ZO)
optimization uses forward margin evaluations only and receives exactly the
same Frobenius perturbation budget as mean-embedding neutralization.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, build_toy, nms_disjoint, set_seed, spearman

SMOKE_ITEMS = [Item("zo1", "alpha beta gamma delta epsilon zeta eta theta",
                    "which greek letter", "delta", "theta")]


def project_ball(z, radius=1.0):
    import torch
    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return z * torch.clamp(torch.as_tensor(radius, device=z.device) / norm, max=1.0)


def score_embeds(att, prep, embeds):
    import torch
    out = []
    for start in range(0, embeds.shape[0], att.max_rows):
        pe = embeds[start:start + att.max_rows]
        with torch.inference_mode():
            score = (att._class_logprob(pe, prep.pred_variant_ids)
                     - att._class_logprob(pe, prep.gold_variant_ids))
        out.append(score.detach().float().cpu())
    return torch.cat(out)


def make_basis(d, rank, device, dtype, seed):
    import torch
    rank = min(rank, d)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(d, rank, generator=gen, dtype=torch.float32)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis.to(device=device, dtype=dtype)


def embeds_for_z(prep, span, basis, z, per_token_radius):
    delta = per_token_radius * (z.to(basis.dtype) @ basis.T)
    out = prep.E.unsqueeze(0).expand(z.shape[0], -1, -1).clone()
    out[:, span.start:span.end, :] += delta[:, None, :]
    return out


def optimize_span(att, prep, span, basis, s0, *, steps, directions, mu, lr, seed):
    """Minimize S over a unit ball with antithetic forward-only queries."""
    import torch
    width = span.end - span.start
    mean_delta = prep.Ebar[span.start:span.end] - prep.E[span.start:span.end]
    fro_budget = float(mean_delta.float().norm())
    radius = fro_budget / math.sqrt(max(width, 1))
    rank = basis.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.zeros(rank, device=prep.E.device, dtype=torch.float32)
    best_z, best_s = z.clone(), float(s0)
    trace, queries = [best_s], 0

    # Equal-query non-adaptive control: optimization must beat multiple testing.
    random_queries = 2 * steps * directions
    rz = project_ball(torch.randn(random_queries, rank, generator=gen)).to(prep.E.device)
    rs = score_embeds(att, prep, embeds_for_z(prep, span, basis, rz, radius)).numpy()
    random_best_s = min(float(s0), float(rs.min()))

    for _ in range(steps):
        u = torch.randn(directions, rank, generator=gen)
        u = (u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(prep.E.device)
        zp = project_ball(z[None] + mu * u)
        zm = project_ball(z[None] - mu * u)
        candidates = torch.cat([zp, zm])
        values = score_embeds(att, prep, embeds_for_z(prep, span, basis, candidates, radius))
        queries += len(candidates)
        local = int(values.argmin())
        if float(values[local]) < best_s:
            best_s, best_z = float(values[local]), candidates[local].detach().clone()
        vp, vm = values[:directions].to(prep.E.device), values[directions:].to(prep.E.device)
        ghat = (((vp - vm) / (2 * mu))[:, None] * u).mean(0)
        z = project_ball(z - lr * ghat / ghat.norm().clamp_min(1e-12))
        z_value = float(score_embeds(
            att, prep, embeds_for_z(prep, span, basis, z[None], radius))[0])
        queries += 1
        if z_value < best_s:
            best_s, best_z = z_value, z.detach().clone()
        trace.append(best_s)
    return {"zo_u": float(s0 - best_s), "zo_score": best_s,
            "random_best_u": float(s0 - random_best_s),
            "fro_budget": fro_budget, "per_token_radius": radius,
            "queries": queries, "trace_best_score": trace,
            "z": best_z.float().cpu().tolist()}


def load_items(args):
    if args.smoke:
        return SMOKE_ITEMS
    if not args.items:
        raise SystemExit("--items is required unless --smoke is used")
    items = [Item.from_dict(x) for x in json.load(open(args.items))]
    if args.item_id:
        items = [x for x in items if x.item_id == args.item_id]
    return items[:args.limit or None]


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--items")
    p.add_argument("--out", default="runs/81_zo_span_keywords.jsonl")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 3])
    p.add_argument("--basis_rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--mu", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.35)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--item_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extra = {}
    if args.smoke:
        model, tok = build_toy()
        args.device, args.out = "cpu", "runs/81_smoke.jsonl"
        args.basis_rank, args.steps, args.directions = 8, 2, 3
        extra = dict(prefix="ctx: ", middle=" q: {question} a: ")
    else:
        loader = importlib.import_module("61_grad_span_proposal").load_model
        model, tok = loader(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.max_rows, **extra)
    items = load_items(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for item_no, item in enumerate(items):
            prep = att.prepare(item)
            spans = att.build_word_spans(prep, widths=args.widths, stride=1)
            prep.spans = spans
            s0 = att.S0(prep)
            mean_u, _ = att.u_of_sets(prep, [[i] for i in range(len(spans))], S0=s0)
            basis = make_basis(att.d, args.basis_rank, args.device, prep.E.dtype,
                               args.seed + 1009 * item_no)
            rows = []
            for i, span in enumerate(spans):
                result = optimize_span(att, prep, span, basis, s0, steps=args.steps,
                    directions=args.directions, mu=args.mu, lr=args.lr,
                    seed=args.seed + 1000003 * item_no + i)
                rows.append(dict(span.to_dict(), mean_u=float(mean_u[i]), **result))
            zo_u = np.asarray([x["zo_u"] for x in rows])
            rnd_u = np.asarray([x["random_best_u"] for x in rows])
            mean_sel = nms_disjoint(np.abs(mean_u), spans, args.topk)
            zo_sel = nms_disjoint(zo_u, spans, args.topk)
            random_sel = nms_disjoint(rnd_u, spans, args.topk)
            rec = {"item_id": item.item_id, "context": item.context,
                "question": item.question, "gold": item.gold, "pred": item.pred,
                "S0": s0, "spans": rows,
                "mean_selected": mean_sel, "zo_selected": zo_sel,
                "random_selected": random_sel,
                "mean_keywords": [spans[i].text for i in mean_sel],
                "zo_keywords": [spans[i].text for i in zo_sel],
                "random_keywords": [spans[i].text for i in random_sel],
                "rho_mean_zo": spearman(mean_u, zo_u),
                "config": {k: v for k, v in vars(args).items() if k != "items"}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{item_no+1}/{len(items)}] {item.item_id} rho={rec['rho_mean_zo']:+.3f}\n"
                  f"  mean: {rec['mean_keywords']}\n  ZO:   {rec['zo_keywords']}", flush=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
