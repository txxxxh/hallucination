# -*- coding: utf-8 -*-
"""Stage 2 using the official SPEX sparse Fourier/channel decoder.

This is deliberately separate from 62_interaction_matrix.py: that file contains
the earlier GBT surrogate experiment.  Here shapiq's paper implementation owns
the masking design and decoding; this adapter only exposes our span-mask value
function u(z)=S(0)-S(z) and converts the recovered Fourier representation to
the Möbius singleton/pair coefficients consumed by the unchanged Stage 3.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import (Item, Span, SpanAttributor, set_seed,
                           redundancy_clusters, synergy_pairs,
                           leading_coalition)


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
    from shapiq.approximator.sparse import SPEX

    class MoebiusSPEX(SPEX):
        """Keep the official sampler/decoder, expose its raw Möbius output.

        shapiq's public SPEX API converts Möbius coefficients to attribution
        indices. Stage 3 instead consumes the finite-difference/Möbius form,
        so this overrides only that final presentation conversion.
        """
        def _process_moebius(self, moebius_transform):
            d = {tuple(k): float(v) for k, v in moebius_transform.items()
                 if len(k) <= self.max_order}
            self._interaction_lookup = {k: i for i, k in enumerate(d)}
            return np.asarray(list(d.values()), dtype=float)

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--baseline", default="mean")
    ap.add_argument("--max_rows", type=int, default=16)
    ap.add_argument("--length_norm", type=int, default=1)
    ap.add_argument("--m_cap", type=int, default=0)
    ap.add_argument("--budget", type=int, default=264)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--decoder", choices=["soft", "hard"], default="soft")
    ap.add_argument("--coef_tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline=args.baseline,
                         length_norm=bool(args.length_norm), max_rows=args.max_rows)

    recs = [json.loads(line) for line in open(args.inp)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w")
    t0 = time.time()

    for n_item, r in enumerate(recs):
        item = Item(r["item_id"], r["context"], r["question"],
                    r["gold"], r["pred"],
                    context_prefix=r.get("context_prefix", ""),
                    gold_variants=r.get("gold_variants", []),
                    pred_variants=r.get("pred_variants", []))
        prep = att.prepare(item)
        candidate_ids = (r["candidates"][:args.m_cap]
                         if args.m_cap else r["candidates"])
        cand_meta = [r["spans"][i] for i in candidate_ids]
        prep.spans = [Span(idx=i, start=s["start"], end=s["end"],
                           text=s["text"]) for i, s in enumerate(cand_meta)]
        m = len(prep.spans)
        S0 = att.S0(prep)
        oracle_calls = 0

        def game(mask_matrix):
            nonlocal oracle_calls
            X = np.asarray(mask_matrix, dtype=bool)
            if X.ndim == 1:
                X = X[None, :]
            oracle_calls += len(X)
            sets = [list(np.flatnonzero(row)) for row in X]
            return att.u_of_sets(prep, sets, S0=S0)[0]

        spex = MoebiusSPEX(n=m, index="FBII", max_order=2,
                           top_order=False, random_state=args.seed + n_item,
                           decoder_type=args.decoder,
                           degree_parameter=args.degree)
        result = spex.approximate(args.budget, game)
        values = {tuple(k): float(result.values[i])
                  for k, i in result.interaction_lookup.items()}

        # Reuse Stage-1 exact singletons: the experiment is whether SPEX can
        # replace Stage-2 pair enumeration, not whether it can degrade known u_i.
        u = np.asarray([float(s["u"]) for s in cand_meta])
        u_spex = np.asarray([values.get((i,), 0.0) for i in range(m)])
        I = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(i + 1, m):
                I[i, j] = I[j, i] = values.get((i, j), 0.0)

        tau = args.coef_tol
        red = redundancy_clusters(I, tau=tau)
        syn = synergy_pairs(I, tau=tau)
        coal = leading_coalition(I, thresh=0.3)
        pairs = m * (m - 1) // 2
        off = I[np.triu_indices(m, 1)]
        n_sig = int(np.sum(np.abs(off) > tau))
        w, V = np.linalg.eigh((I + I.T) / 2)
        rec = {
            "item_id": r["item_id"], "S0": S0, "m": m,
            "u": [float(x) for x in u],
            "u_spex": [float(x) for x in u_spex],
            "I": [[float(x) for x in row] for row in I],
            "spans": [s.to_dict() for s in prep.spans],
            "tau": tau, "sigma_null_I": 0.0, "null_I": [],
            "null_I_center": 0.0,
            "redundancy_clusters": red, "synergy_pairs": syn,
            "leading_coalition": coal,
            "spectrum": {"eigvals": [float(x) for x in w],
                         "top_eigvec": [float(x) for x in
                                         V[:, int(np.argmax(np.abs(w)))]]},
            "n_pairs": pairs, "n_sig_pairs": n_sig,
            "frac_sig": n_sig / max(pairs, 1),
            "synergy_share": (float(np.sum(off[off > tau])) /
                              (float(np.sum(np.abs(off[np.abs(off) > tau]))) + 1e-12)),
            "u_singles_sum": float(u.sum()),
            "u_joint_all": float(sum(values.values())),
            "spex": {"implementation": "shapiq",
                     "decoder": args.decoder, "degree": args.degree,
                     "requested_budget": args.budget,
                     "used_budget": int(result.estimation_budget),
                     "oracle_rows": oracle_calls,
                     "n_recovered_moebius": len(values),
                     "singleton_mae_vs_exact": float(np.mean(np.abs(u_spex - u)))},
        }
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"[{n_item+1}/{len(recs)}] {r['item_id']}: m={m} "
              f"budget={result.estimation_budget} recovered={len(values)} "
              f"singleton_MAE={np.mean(np.abs(u_spex-u)):.4f} "
              f"max|I|={np.max(np.abs(I)):.4f}")

    fh.close()
    print(f"Wrote {args.out} ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
