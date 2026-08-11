# -*- coding: utf-8 -*-
"""
61_grad_span_proposal.py  --  Stage 1: fine-grained span proposal.

For each item:
  1. build 2/3-token sliding-window spans over the context region
  2. first-order gate gradient   u_hat_i = -sum_{t in span_i} dS/dalpha_t |_0
  3. integrated gradients        IG_i    (completeness: sum_t IG_t = u(all))
  4. MEASURED single-span gain   u_i     = S(0) - S(1_i)          <- ground truth
  5. attention baseline          att_i   (the incumbent top-k method)
  6. sigma_null from position/length-matched random spans
  7. NMS -> m token-disjoint candidates handed to stage 2

The headline number this script exists to produce is the CALIBRATION rho
between the cheap first-order proxies (u_hat, IG) and the measured u.
If rho is already very high, the whole second-order machinery in 62_/63_ is
over-engineering and should be cut to a limitation paragraph.

Usage
  python 61_grad_span_proposal.py --smoke
  python 61_grad_span_proposal.py --items data/items_example.json \
      --model meta-llama/Llama-3.1-8B-Instruct --out runs/61.jsonl --m 12
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import (Item, SpanAttributor, set_seed, spearman, pearson,
                           bootstrap_ci, nms_disjoint, build_toy)


def load_model(name: str, dtype: str, device: str):
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
        name, torch_dtype=dt,
        attn_implementation="eager",     # required for output_attentions
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


SMOKE_ITEMS = [
    Item("s1", "alpha beta gamma delta epsilon zeta eta theta iota kappa",
         "which greek letter", "delta", "kappa"),
    Item("s2", "red green blue cyan magenta yellow black white grey",
         "which colour", "cyan", "yellow"),
]


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=str, default=None)
    ap.add_argument("--out", type=str, default="runs/61_grad_span_proposal.jsonl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--baseline", type=str, default="mean", choices=["mean", "unk", "zero"])
    ap.add_argument("--widths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--span_unit", choices=["tokens", "words"], default="words",
                    help="interpret --widths as token or raw-text word counts")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--m", type=int, default=12, help="candidates kept after NMS")
    ap.add_argument("--null_draws", type=int, default=24)
    ap.add_argument("--coarse_words", type=int, default=12,
                    help="non-overlapping words per hierarchical coarse source")
    ap.add_argument("--coarse_queries", type=int, default=32,
                    help="random coarse masks used by the sparse linear surrogate")
    ap.add_argument("--coarse_keep", type=int, default=5,
                    help="coarse sources whose local word windows are measured exactly")
    ap.add_argument("--max_rows", type=int, default=16)
    ap.add_argument("--length_norm", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--item_id", type=str, default=None,
                    help="run exactly one item id/key")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    extra = {}
    if args.smoke:
        model, tok = build_toy()
        args.device, args.out = "cpu", "runs/61_smoke.jsonl"
        args.ig_steps, args.m, args.null_draws = 8, 5, 6
        items = SMOKE_ITEMS
        extra = dict(prefix="ctx: ", middle=" q: {question} a: ")
    else:
        assert args.items, "--items is required (or use --smoke)"
        model, tok = load_model(args.model, args.dtype, args.device)
        items = [Item.from_dict(d) for d in json.load(open(args.items))]
        if args.item_id is not None:
            items = [x for x in items if x.item_id == args.item_id]
            if not items:
                raise SystemExit(f"item_id not found: {args.item_id}")
        if args.limit:
            items = items[:args.limit]

    att = SpanAttributor(model, tok, device=args.device, baseline=args.baseline,
                         length_norm=bool(args.length_norm),
                         max_rows=args.max_rows, **extra)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w")
    per_item = {"rho_uhat": [], "rho_ig": [], "rho_att": [],
                "r_uhat": [], "r_ig": [], "completeness": [], "snr": [],
                "topk_null_ratio": [], "topk_min_null_ratio": []}
    pooled = {"u_hat": [], "ig": [], "u": [], "att": []}
    t0 = time.time()

    for n, item in enumerate(items):
        prep = att.prepare(item)
        hierarchy = None
        if args.span_unit == "words":
            all_spans = att.build_word_spans(prep, widths=tuple(args.widths),
                                             stride=args.stride)
            coarse = att.build_word_spans(
                prep, widths=(args.coarse_words,), stride=args.coarse_words)
            # ContextCite-style sparse linear surrogate over disjoint sources.
            # The expensive exact LOO is then restricted to windows inside the
            # highest-scoring coarse sources (AttriBoT-style hierarchy).
            if len(coarse) >= 2 and args.coarse_queries > 0:
                from sklearn.linear_model import LassoCV
                rng = np.random.default_rng(args.seed + 1009 * n)
                seen, rows = set(), []
                target = min(args.coarse_queries, 2 ** len(coarse) - 1)
                while len(rows) < target:
                    z = tuple((rng.random(len(coarse)) < 0.5).astype(int).tolist())
                    if not any(z) or z in seen:
                        continue
                    seen.add(z); rows.append(z)
                Xc = np.asarray(rows, dtype=float)
                saved = prep.spans
                prep.spans = coarse
                yc, _ = att.u_of_sets(
                    prep, [list(np.flatnonzero(z)) for z in Xc])
                prep.spans = saved
                cv = min(5, max(2, len(yc) // 4))
                surrogate = LassoCV(cv=cv, random_state=args.seed,
                                    max_iter=20000).fit(Xc, yc)
                coef = surrogate.coef_
                keep = np.argsort(-np.abs(coef))[:min(args.coarse_keep,
                                                       len(coarse))]
                regions = [coarse[int(i)] for i in keep]
                # Include boundary-crossing fine windows: a meaningful phrase
                # must not disappear merely because a coarse partition cuts it.
                spans = [s for s in all_spans if any(
                    s.end > c.start and s.start < c.end for c in regions)]
                hierarchy = {
                    "coarse_words": args.coarse_words,
                    "n_coarse": len(coarse), "n_queries": len(yc),
                    "coarse_spans": [c.to_dict() for c in coarse],
                    "coarse_coef": [float(x) for x in coef],
                    "coarse_keep": [int(i) for i in keep],
                    "surrogate_r2_train": float(surrogate.score(Xc, yc)),
                    "n_fine_all": len(all_spans),
                    "n_fine_measured": len(spans),
                }
            else:
                spans = all_spans
        else:
            spans = att.build_spans(prep, widths=tuple(args.widths),
                                    stride=args.stride)
        # build_word_spans updates prep.spans on every call; after the coarse
        # pass, explicitly install the fine subset used by all measurements.
        prep.spans = spans
        if len(spans) < 3:
            print(f"[skip] {item.item_id}: context too short ({len(spans)} spans)")
            continue

        S0 = att.S0(prep)
        u_all = S0 - float(att.S(prep, att.alpha_all(prep).unsqueeze(0))[0])

        g = att.grad_alpha(prep)
        u_hat = att.u_hat_first_order(prep, spans, g=g)
        ig_tok = att.integrated_gradients(prep, steps=args.ig_steps)
        ig_span = np.array([ig_tok[s.start:s.end].sum() for s in spans])
        comp_err = abs(ig_tok.sum() - u_all) / (abs(u_all) + 1e-8)

        u_meas, _ = att.u_of_sets(prep, [[i] for i in range(len(spans))], S0=S0)

        try:
            a_tok = att.attention_scores(prep)
            att_span = np.array([a_tok[s.start:s.end].sum() for s in spans])
        except Exception as e:                       # e.g. sdpa backend
            print(f"[warn] attention unavailable: {e}")
            att_span = np.zeros(len(spans))

        if args.span_unit == "words":
            rng = np.random.default_rng(args.seed + n)
            null_ids = rng.choice(len(spans), size=args.null_draws, replace=True)
            null_u, _ = att.u_of_sets(prep, [[int(i)] for i in null_ids], S0=S0)
            sigma = (float(np.std(null_u, ddof=1))
                     if len(null_u) > 1 else 0.0)
        else:
            sigma, null_u = att.null_sigma(prep, widths=tuple(args.widths),
                                           n_draw=args.null_draws, S0=S0,
                                           seed=args.seed + n)
        snr = float(np.mean(np.abs(u_meas)) / (sigma + 1e-12))

        cand = nms_disjoint(u_meas, spans, m=args.m)
        null_abs_q95 = (float(np.quantile(np.abs(null_u), 0.95))
                        if len(null_u) else float("nan"))
        topk_abs_u_mean = (float(np.mean(np.abs(u_meas[cand])))
                           if cand else float("nan"))
        topk_abs_u_min = (float(np.min(np.abs(u_meas[cand])))
                          if cand else float("nan"))
        topk_null_ratio = topk_abs_u_mean / (null_abs_q95 + 1e-12)
        topk_min_null_ratio = topk_abs_u_min / (null_abs_q95 + 1e-12)
        # coverage: how much of the total destroyable margin the candidates carry
        cov = float(np.abs(u_meas[cand]).sum() / (abs(u_all) + 1e-8))

        rec = {
            "item_id": item.item_id, "pred": item.pred, "gold": item.gold,
            # context/question are stored verbatim because 62_ must rebuild a
            # byte-identical prompt -- span offsets are meaningless otherwise.
            "context": item.context, "question": item.question,
            "context_prefix": item.context_prefix,
            "gold_variants": item.gold_variants, "pred_variants": item.pred_variants,
            "P": int(prep.prompt_ids.shape[0]),
            "ctx_start": prep.ctx_start, "ctx_end": prep.ctx_end,
            "S0": S0, "u_all": u_all, "sum_ig": float(ig_tok.sum()),
            "completeness_rel_err": comp_err,
            "sigma_null": sigma, "snr": snr, "coverage_cand": cov,
            "null_abs_q95": null_abs_q95,
            "topk_abs_u_mean": topk_abs_u_mean,
            "topk_abs_u_min": topk_abs_u_min,
            "topk_null_ratio": topk_null_ratio,
            "topk_min_null_ratio": topk_min_null_ratio,
            "hierarchy": hierarchy,
            "rho_uhat_u": spearman(u_hat, u_meas),
            "rho_ig_u": spearman(ig_span, u_meas),
            "rho_att_u": spearman(att_span, u_meas),
            "r_uhat_u": pearson(u_hat, u_meas),
            "r_ig_u": pearson(ig_span, u_meas),
            "candidates": cand,
            "spans": [dict(s.to_dict(), u_hat=float(u_hat[i]), ig=float(ig_span[i]),
                           u=float(u_meas[i]), att=float(att_span[i]),
                           sig=bool(abs(u_meas[i]) > 2 * sigma))
                      for i, s in enumerate(spans)],
            "config": {k: v for k, v in vars(args).items() if k != "items"},
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

        for k, v in [("rho_uhat", rec["rho_uhat_u"]), ("rho_ig", rec["rho_ig_u"]),
                     ("rho_att", rec["rho_att_u"]), ("r_uhat", rec["r_uhat_u"]),
                     ("r_ig", rec["r_ig_u"]), ("completeness", comp_err), ("snr", snr),
                     ("topk_null_ratio", topk_null_ratio),
                     ("topk_min_null_ratio", topk_min_null_ratio)]:
            if np.isfinite(v):
                per_item[k].append(v)
        pooled["u_hat"] += list(u_hat); pooled["ig"] += list(ig_span)
        pooled["u"] += list(u_meas);    pooled["att"] += list(att_span)

        print(f"[{n+1}/{len(items)}] {item.item_id}: spans={len(spans)} "
              f"u_all={u_all:+.3f} IGerr={comp_err:.3f} "
              f"topk/null95={topk_null_ratio:.2f} kth/null95={topk_min_null_ratio:.2f} "
              f"rho(uhat)={rec['rho_uhat_u']:.3f} rho(IG)={rec['rho_ig_u']:.3f} "
              f"rho(att)={rec['rho_att_u']:.3f}")

    fh.close()

    print("\n" + "=" * 70)
    print("STAGE-1 CALIBRATION REPORT")
    print("=" * 70)

    def line(name, vals):
        if not vals:
            print(f"  {name:<26} n/a")
            return
        lo, hi = bootstrap_ci(vals)
        print(f"  {name:<26} mean={np.mean(vals):+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  n={len(vals)}")

    print("Per-item Spearman vs MEASURED single-span gain u:")
    line("first-order u_hat", per_item["rho_uhat"])
    line("integrated gradients", per_item["rho_ig"])
    line("attention (incumbent)", per_item["rho_att"])
    print("Per-item Pearson:")
    line("first-order u_hat", per_item["r_uhat"])
    line("integrated gradients", per_item["r_ig"])
    print("Diagnostics:")
    line("IG completeness rel err", per_item["completeness"])
    line("top-k mean |u| / null q95", per_item["topk_null_ratio"])
    line("k-th |u| / null q95", per_item["topk_min_null_ratio"])
    up = np.asarray(pooled["u"], dtype=float)
    frac_pos = float((up > 0).mean()) if len(up) else float("nan")
    print("Sign structure of measured gains (precondition for reading I in 62_):")
    print(f"  frac(u_i > 0) = {frac_pos:.3f}   "
          f"[u_i<0 means the span supports GOLD; the redundancy/synergy narrative")
    print("   is only valid on the u_i>0 subset -- see core.py PRECONDITION]")

    print("\nPooled across all items/spans:")
    print(f"  rho(u_hat,u)={spearman(pooled['u_hat'], pooled['u']):+.3f}   "
          f"rho(IG,u)={spearman(pooled['ig'], pooled['u']):+.3f}   "
          f"rho(att,u)={spearman(pooled['att'], pooled['u']):+.3f}")

    mr = float(np.mean(per_item["rho_ig"])) if per_item["rho_ig"] else float("nan")
    print("\nDECISION RULE for whether stage 2 is warranted:")
    if np.isfinite(mr) and mr > 0.90:
        print(f"  IG rho={mr:.3f} > 0.90 -> first order already explains the effect.")
        print("  Second-order machinery is likely over-engineering; report as limitation.")
    elif np.isfinite(mr):
        print(f"  IG rho={mr:.3f} <= 0.90 -> substantial unexplained variance.")
        print("  Proceed to 62_interaction_matrix.py.")
    if (per_item["topk_null_ratio"]
            and np.mean(per_item["topk_null_ratio"]) < 1.0):
        print("  WARNING: even the mean top-k |u| does not clear the random-span")
        print("  95th percentile. Do not proceed to pairwise interactions yet.")
    elif per_item["topk_min_null_ratio"]:
        ratio = np.mean(per_item["topk_min_null_ratio"])
        if ratio < 1.0:
            print(f"  Top-k mean clears null, but the k-th candidate ratio is {ratio:.2f}.")
            print("  Proceed with fewer candidates or treat the tail as exploratory.")
        else:
            print(f"  All top-k candidates clear the null q95 on average "
                  f"(k-th ratio={ratio:.2f}); proceeding to Stage 2 is warranted.")
    print(f"\nWrote {args.out}   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
