# -*- coding: utf-8 -*-
"""
63_subset_select.py  --  Stage 3: head-to-head selection + two-tier validation.

Compares selection strategies at fixed budget k over the SAME candidate pool:

    attention_topk   the incumbent method (attention mass, no intervention)
    first_order      top-k by measured single-span gain u_i (ignores I)
    second_order     argmax of  sum u_i + sum_{i<j} I_ij   (exhaustive if affordable)
    greedy           greedy on the same objective
    random_matched   POSITION- AND LENGTH-MATCHED random spans  <- required control

Two-tier measurement, with strictly separated roles:

    TIER 1 (cheap, dense, teacher-forced): u(S) = S(0) - S(1_S).
        Used for SEARCH only. A biased search heuristic costs statistical power,
        not validity, because every selected set is then re-checked at Tier 2.

    TIER 2 (expensive, sparse, generation): sampled generation under the
        intervention; reports P(pred) drop and P(gold) rise.
        ALL OUTWARD CLAIMS REST ON TIER 2.

The script also reports the Tier-1 / Tier-2 Spearman rho with a bootstrap CI.
That number decides the framing: high rho licenses teacher-forced attribution as
a cheap proxy; low rho demotes Tier 1 to a pure search heuristic and every
claim retreats to Tier 2. Both outcomes are publishable, but they are different
papers, so the number must be reported either way.

Usage
  python 63_subset_select.py --smoke
  python 63_subset_select.py --in62 runs/62.jsonl --in61 runs/61.jsonl \
      --out runs/63.jsonl --k 3 --n_gen 20
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import (Item, Span, SpanAttributor, set_seed, build_toy,
                           spearman, bootstrap_ci, topk_first_order,
                           greedy_select, exhaustive_select, second_order_objective,
                           redundancy_clusters)


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
        name, torch_dtype=dt, attn_implementation="eager").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--in62", type=str, default="runs/62_interaction_matrix.jsonl")
    ap.add_argument("--in61", type=str, default="runs/61_grad_span_proposal.jsonl")
    ap.add_argument("--out", type=str, default="runs/63_subset_select.jsonl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument("--max_rows", type=int, default=16)
    ap.add_argument("--length_norm", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_gen", type=int, default=20, help="Tier-2 samples; 0 disables")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--exh_cap", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    extra = {}
    if args.smoke:
        model, tok = build_toy()
        args.device = "cpu"
        args.in62, args.in61 = "runs/62_smoke.jsonl", "runs/61_smoke.jsonl"
        args.out, args.k, args.n_gen, args.max_new_tokens = "runs/63_smoke.jsonl", 2, 3, 4
        extra = dict(prefix="ctx: ", middle=" q: {question} a: ")
    else:
        model, tok = load_model(args.model, args.dtype, args.device)

    att = SpanAttributor(model, tok, device=args.device, baseline=args.baseline,
                         length_norm=bool(args.length_norm),
                         max_rows=args.max_rows, **extra)

    r61 = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.in61)}
    recs = [json.loads(l) for l in open(args.in62)]
    if args.limit:
        recs = recs[:args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w")

    STRATS = ["attention_topk", "first_order", "second_order", "greedy", "random_matched"]
    t1 = {s: [] for s in STRATS}     # tier-1 u(S)
    t2p = {s: [] for s in STRATS}    # tier-2 P(pred) drop
    t2g = {s: [] for s in STRATS}    # tier-2 P(gold) rise
    tier_pairs = []
    t0 = time.time()

    for n, r in enumerate(recs):
        base = r61[r["item_id"]]
        item = Item(r["item_id"], base["context"], base["question"],
                    base["gold"], base["pred"],
                    context_prefix=base.get("context_prefix", ""),
                    gold_variants=base.get("gold_variants", []),
                    pred_variants=base.get("pred_variants", []))
        prep = att.prepare(item)
        assert prep.prompt_ids.shape[0] == base["P"], "prompt length drift"

        prep.spans = [Span(idx=k, start=s["start"], end=s["end"], text=s["text"])
                      for k, s in enumerate(r["spans"])]
        m = len(prep.spans)
        u = np.array(r["u"], dtype=float)
        I = np.array(r["I"], dtype=float)
        k = min(args.k, m)
        S0 = float(r["S0"])

        # attention score for the SAME candidate spans
        att_c = np.array([base["spans"][j]["att"]
                          for j in base["candidates"][:m]], dtype=float)

        sel = {
            "attention_topk": topk_first_order(att_c, k),
            "first_order":    topk_first_order(u, k),
            "second_order":   exhaustive_select(u, I, k, cap=args.exh_cap),
            "greedy":         greedy_select(u, I, k),
        }
        # position/length-matched random control -> appended as extra spans
        rnd = att.random_matched_set(prep, [prep.spans[i] for i in sel["first_order"]],
                                     seed=args.seed + n)
        off = len(prep.spans)
        for j, sp in enumerate(rnd):
            prep.spans.append(Span(idx=off + j, start=sp.start, end=sp.end, text=""))
        sel["random_matched"] = [off + j for j in range(len(rnd))]

        # ---- Tier 1 ----
        u_sets, _ = att.u_of_sets(prep, [sel[s] for s in STRATS], S0=S0)
        row = {"item_id": r["item_id"], "k": k, "m": m,
               "selection": {s: sel[s] for s in STRATS},
               "span_text": {s: [prep.spans[i].text for i in sel[s]] for s in STRATS},
               "tier1_u": {s: float(u_sets[i]) for i, s in enumerate(STRATS)},
               "obj2": {s: second_order_objective([x for x in sel[s] if x < m], u, I)
                        for s in STRATS}}

        # ---- Tier 2 ----
        if args.n_gen > 0:
            base_gen = att.generate_under(prep, [], n=args.n_gen,
                                          temperature=args.temperature,
                                          max_new_tokens=args.max_new_tokens,
                                          seed=args.seed)
            p_pred0 = att.match_rate(base_gen, [item.pred] + item.pred_variants)
            p_gold0 = att.match_rate(base_gen, [item.gold] + item.gold_variants)
            row["tier2_base"] = {"p_pred": p_pred0, "p_gold": p_gold0}
            row["tier2"] = {}
            for s in STRATS:
                g = att.generate_under(prep, sel[s], n=args.n_gen,
                                       temperature=args.temperature,
                                       max_new_tokens=args.max_new_tokens,
                                       seed=args.seed)
                pp = att.match_rate(g, [item.pred] + item.pred_variants)
                pg = att.match_rate(g, [item.gold] + item.gold_variants)
                row["tier2"][s] = {"p_pred": pp, "p_gold": pg,
                                   "drop_pred": p_pred0 - pp, "rise_gold": pg - p_gold0,
                                   "sample": g[:2]}
                t2p[s].append(p_pred0 - pp)
                t2g[s].append(pg - p_gold0)
                tier_pairs.append((float(u_sets[STRATS.index(s)]), p_pred0 - pp))

        for i, s in enumerate(STRATS):
            t1[s].append(float(u_sets[i]))

        red = redundancy_clusters(I, tau=float(r.get("tau", 0.0)))
        row["redundancy_clusters"] = red
        row["n_effective_units"] = len(red)
        fh.write(json.dumps(row) + "\n"); fh.flush()

        print(f"[{n+1}/{len(recs)}] {r['item_id']}: " +
              "  ".join(f"{s.split('_')[0]}={u_sets[i]:+.3f}" for i, s in enumerate(STRATS)))

    fh.close()

    print("\n" + "=" * 78)
    print(f"STAGE-3 HEAD-TO-HEAD  (k={args.k})")
    print("=" * 78)
    print(f"{'strategy':<18}{'tier1 u(S)':>22}{'tier2 drop P(pred)':>26}{'tier2 rise P(gold)':>22}")
    for s in STRATS:
        def fmt(v):
            if not v:
                return f"{'n/a':>20}"
            lo, hi = bootstrap_ci(v)
            return f"{np.mean(v):+.3f} [{lo:+.3f},{hi:+.3f}]".rjust(20)
        print(f"{s:<18}{fmt(t1[s]):>22}{fmt(t2p[s]):>26}{fmt(t2g[s]):>22}")

    # paired deltas against the two controls that matter
    def paired(a, b, label):
        if not t1[a] or not t1[b]:
            return
        d = np.array(t1[a]) - np.array(t1[b])
        lo, hi = bootstrap_ci(d)
        star = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {label:<44} mean={d.mean():+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]  {star}")

    print("\nPaired Tier-1 contrasts:")
    paired("first_order", "attention_topk", "first_order  -  attention (incumbent)")
    paired("second_order", "first_order", "second_order -  first_order (value of I)")
    paired("first_order", "random_matched", "first_order  -  position-matched null")
    paired("second_order", "random_matched", "second_order -  position-matched null")

    if tier_pairs:
        a = [x[0] for x in tier_pairs]; b = [x[1] for x in tier_pairs]
        rho = spearman(a, b)
        print(f"\nTIER-1 / TIER-2 AGREEMENT: Spearman rho = {rho:+.3f}  (n={len(a)})")
        if np.isfinite(rho) and rho >= 0.6:
            print("  -> teacher-forced margin is a defensible cheap proxy for generation;")
            print("     Tier-1 attribution can be reported directly, with Tier 2 as backing.")
        else:
            print("  -> WEAK agreement. Demote Tier 1 to a search heuristic only and make")
            print("     every headline claim on Tier-2 generation numbers.")
    print(f"\nWrote {args.out}   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
