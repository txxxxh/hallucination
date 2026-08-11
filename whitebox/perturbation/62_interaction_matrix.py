# -*- coding: utf-8 -*-
"""
62_interaction_matrix.py  --  Stage 2: pairwise interaction matrix.

Reads the candidate spans chosen by 61_. Random joint masks train a
ProxySPEX-style gradient-boosted-tree surrogate; the strongest predicted pair
interactions are then verified by exact finite differences:

    u_i  = u({i})
    u_ij = u({i,j})
    I_ij = u_ij - u_i - u_j

Sign convention (see spanattr/core.py):
    I_ij < 0  -> REDUNDANT   (substitutes; aggregate at cluster level)
    I_ij > 0  -> SYNERGISTIC (a genuine multi-token unit that top-k saliency
                              structurally cannot find)

Cost: m singles + C(m,2) pairs + 1 baseline forward pass. m=12 -> 79 rows,
batched at --max_rows per forward.

Because forward passes are all that is required, this script runs unchanged on
any causal LM -- which is what makes the Llama/DeepSeek contrast clean.

Noise threshold: estimate an empirical null distribution of I directly from
random, disjoint, width-matched span pairs. tau is the 95th percentile of the
absolute median-centred null interactions.

Usage
  python 62_interaction_matrix.py --smoke
  python 62_interaction_matrix.py --in runs/61.jsonl --out runs/62.jsonl \
      --model meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import (Item, Span, SpanAttributor, set_seed, build_toy,
                           interaction_from_gains, redundancy_clusters,
                           synergy_pairs, leading_coalition, bootstrap_ci)


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
    ap.add_argument("--in", dest="inp", type=str, default="runs/61_grad_span_proposal.jsonl")
    ap.add_argument("--out", type=str, default="runs/62_interaction_matrix.jsonl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument("--max_rows", type=int, default=16)
    ap.add_argument("--length_norm", type=int, default=1)
    ap.add_argument("--null_pairs", type=int, default=24,
                    help="deprecated legacy option")
    ap.add_argument("--proxy_queries", type=int, default=48,
                    help="random joint masks used to fit the GBT surrogate")
    ap.add_argument("--exact_pairs", type=int, default=16,
                    help="strongest proxy interactions verified exactly")
    ap.add_argument("--m_cap", type=int, default=0,
                    help="optionally keep only the first N Stage-1 NMS candidates")
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
        args.inp, args.out = "runs/61_smoke.jsonl", "runs/62_smoke.jsonl"
        extra = dict(prefix="ctx: ", middle=" q: {question} a: ")
    else:
        model, tok = load_model(args.model, args.dtype, args.device)

    att = SpanAttributor(model, tok, device=args.device, baseline=args.baseline,
                         length_norm=bool(args.length_norm),
                         max_rows=args.max_rows, **extra)

    recs = [json.loads(l) for l in open(args.inp)]
    if args.limit:
        recs = recs[:args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w")

    agg = {"n_red": [], "n_syn": [], "frac_sig": [], "I_absmax": [],
           "syn_share": [], "n_clusters": []}
    t0 = time.time()

    for n, r in enumerate(recs):
        item = Item(r["item_id"], r.get("context", ""), r.get("question", ""),
                    r["gold"], r["pred"],
                    context_prefix=r.get("context_prefix", ""),
                    gold_variants=r.get("gold_variants", []),
                    pred_variants=r.get("pred_variants", []))
        if not item.context:
            raise SystemExit(
                "record lacks 'context'; rerun 61_ with the patched writer or pass "
                "--items through. (61_ stores span offsets, which are only valid "
                "against the identical prompt.)")

        prep = att.prepare(item)
        assert prep.prompt_ids.shape[0] == r["P"], (
            f"prompt length changed ({prep.prompt_ids.shape[0]} vs {r['P']}); "
            "span offsets from 61_ would be invalid")

        candidate_ids = r["candidates"][:args.m_cap] if args.m_cap else r["candidates"]
        cand_meta = [r["spans"][i] for i in candidate_ids]
        prep.spans = [Span(idx=k, start=s["start"], end=s["end"], text=s["text"])
                      for k, s in enumerate(cand_meta)]
        m = len(prep.spans)
        if m < 2:
            print(f"[skip] {r['item_id']}: only {m} candidate(s)")
            continue

        S0 = att.S0(prep)
        from sklearn.ensemble import GradientBoostingRegressor
        pairs = list(itertools.combinations(range(m), 2))
        # Singles were already measured exactly in Stage 1: reuse those labels.
        u = np.asarray([float(s["u"]) for s in cand_meta], dtype=float)
        X_single = np.eye(m, dtype=float)
        rng = np.random.default_rng(args.seed + 1009 * n)
        seen, rows = set(), []
        max_unique = 2 ** m - m - 2
        target = min(args.proxy_queries, max_unique)
        while len(rows) < target:
            p_mask = float(rng.uniform(0.2, 0.8))
            z = tuple((rng.random(m) < p_mask).astype(int).tolist())
            if sum(z) < 2 or sum(z) == m or z in seen:
                continue
            seen.add(z); rows.append(z)
        X_rand = np.asarray(rows, dtype=float)
        y_rand, _ = att.u_of_sets(
            prep, [list(np.flatnonzero(z)) for z in X_rand], S0=S0)

        # A held-out residual supplies a conservative proxy uncertainty floor.
        order_q = rng.permutation(len(X_rand))
        n_test = max(4, len(X_rand) // 5)
        te, tr = order_q[:n_test], order_q[n_test:]
        probe = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.04,
            loss="huber", random_state=args.seed).fit(
                np.vstack([X_single, X_rand[tr]]),
                np.concatenate([u, y_rand[tr]]))
        resid = y_rand[te] - probe.predict(X_rand[te])
        tau = float(np.quantile(np.abs(resid), 0.95)) if len(resid) else 0.0
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

        proxy = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.04,
            loss="huber", random_state=args.seed).fit(
                np.vstack([X_single, X_rand]), np.concatenate([u, y_rand]))
        X_pair = np.zeros((len(pairs), m), dtype=float)
        for k_pair, (a, b) in enumerate(pairs):
            X_pair[k_pair, [a, b]] = 1.0
        u_pair_proxy = proxy.predict(X_pair)
        I_proxy = interaction_from_gains(
            u, {p: u_pair_proxy[k] for k, p in enumerate(pairs)})

        verify_order = np.argsort(-np.abs([I_proxy[a, b] for a, b in pairs]))
        verify_ids = verify_order[:min(args.exact_pairs, len(pairs))]
        verify_pairs = [pairs[int(k)] for k in verify_ids]
        u_verify, _ = att.u_of_sets(prep, [list(p) for p in verify_pairs], S0=S0)
        I = I_proxy.copy()
        for p, up in zip(verify_pairs, u_verify):
            a, b = p
            I[a, b] = I[b, a] = float(up - u[a] - u[b])
        null_I = [float(x) for x in resid]
        null_center = float(np.median(resid)) if len(resid) else 0.0
        red = redundancy_clusters(I, tau=tau)
        syn = synergy_pairs(I, tau=tau)
        coal = leading_coalition(I, thresh=0.3)

        off = I[~np.eye(m, dtype=bool)]
        n_sig = int((np.abs(off) > tau).sum() / 2)
        frac_sig = n_sig / max(len(pairs), 1)
        syn_share = (float(np.sum(off[off > tau])) /
                     (float(np.sum(np.abs(off[np.abs(off) > tau]))) + 1e-12))

        w, V = np.linalg.eigh((I + I.T) / 2)
        spec = {"eigvals": [float(x) for x in w],
                "top_eigvec": [float(x) for x in V[:, int(np.argmax(np.abs(w)))]]}

        rec = {
            "item_id": r["item_id"], "S0": S0, "sigma_null_I": sigma, "tau": tau,
            "null_I_center": null_center, "null_I": null_I,
            "proxy": {"kind": "GradientBoostingRegressor",
                      "n_random_queries": len(X_rand),
                      "n_exact_pairs": len(verify_pairs),
                      "exact_pair_indices": [int(k) for k in verify_ids],
                      "heldout_residual_q95": tau,
                      "heldout_r2": float(probe.score(X_rand[te], y_rand[te]))},
            "I_proxy": [[float(x) for x in row] for row in I_proxy],
            "m": m, "u": [float(x) for x in u],
            "I": [[float(x) for x in row] for row in I],
            "spans": [s.to_dict() for s in prep.spans],
            "redundancy_clusters": red, "synergy_pairs": syn,
            "leading_coalition": coal, "spectrum": spec,
            "n_pairs": len(pairs), "n_sig_pairs": n_sig, "frac_sig": frac_sig,
            "synergy_share": syn_share,
            "u_singles_sum": float(u.sum()),
            "u_joint_all": float(att.u_of_sets(prep, [list(range(m))], S0=S0)[0][0]),
        }
        fh.write(json.dumps(rec) + "\n"); fh.flush()

        agg["n_red"].append(sum(1 for c in red if len(c) > 1))
        agg["n_syn"].append(len(syn))
        agg["frac_sig"].append(frac_sig)
        agg["I_absmax"].append(float(np.abs(I).max()))
        agg["syn_share"].append(syn_share)
        agg["n_clusters"].append(len(red))

        print(f"[{n+1}/{len(recs)}] {r['item_id']}: m={m} tau={tau:.4f} "
              f"|I|max={np.abs(I).max():.4f} sig_pairs={n_sig}/{len(pairs)} "
              f"red_clusters={sum(1 for c in red if len(c)>1)} syn_pairs={len(syn)}")

    fh.close()

    print("\n" + "=" * 70)
    print("STAGE-2 INTERACTION REPORT")
    print("=" * 70)

    def line(name, vals, fmt="{:+.3f}"):
        if not vals:
            print(f"  {name:<30} n/a"); return
        lo, hi = bootstrap_ci(vals)
        print(f"  {name:<30} mean={fmt.format(np.mean(vals))}  "
              f"95%CI=[{fmt.format(lo)},{fmt.format(hi)}]  n={len(vals)}")

    line("frac of pairs above noise", agg["frac_sig"])
    line("redundancy clusters (size>1)", agg["n_red"], "{:.2f}")
    line("synergistic pairs", agg["n_syn"], "{:.2f}")
    line("synergy share of |I| mass", agg["syn_share"])
    line("max |I_ij|", agg["I_absmax"])

    fs = np.mean(agg["frac_sig"]) if agg["frac_sig"] else float("nan")
    print("\nINTERPRETATION:")
    if np.isfinite(fs) and fs < 0.05:
        print(f"  Only {fs:.1%} of pairs clear the noise floor. Interactions are")
        print("  effectively absent -> the additive/first-order model is adequate and")
        print("  63_ will reduce to top-k. Report this as a NEGATIVE RESULT, do not")
        print("  cluster noise.")
    else:
        ss = np.mean(agg["syn_share"]) if agg["syn_share"] else float("nan")
        print(f"  {fs:.1%} of pairs carry real interaction; synergy accounts for "
              f"{ss:.1%} of the mass.")
        print("  Synergy-dominant -> multi-token units exist that top-k cannot find.")
        print("  Redundancy-dominant -> aggregate at cluster level in the v8 pipeline.")
    print(f"\nWrote {args.out}   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
