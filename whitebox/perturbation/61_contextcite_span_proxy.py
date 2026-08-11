# -*- coding: utf-8 -*-
"""Paper-faithful ContextCite sampling/LASSO adapted to answer margin.

Sources are non-overlapping word/punctuation units from ContextCite's official
partitioner. Each Bernoulli mask physically deletes sources. The scalar output
is this pipeline's S=log P(pred)-log P(gold), rather than probability of a
generated sentence. Official LassoRegression then assigns source coefficients;
coefficients are summed over each 2/3-word window for comparison with exact LOO.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import Item, SpanAttributor, set_seed, spearman, nms_disjoint


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


def main():
    import torch
    from context_cite.context_partitioner import SimpleContextPartitioner
    from context_cite.solver import LassoRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--in61", required=True,
                    help="Stage-1 exact record used as metadata/gold standard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_ablations", type=int, default=32)
    ap.add_argument("--keep_prob", type=float, default=0.5)
    ap.add_argument("--lasso_alpha", type=float, default=0.01)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=1)
    records = [json.loads(x) for x in open(args.in61)]
    out = open(args.out, "w")
    t0 = time.time()

    for r in records:
        base_item = Item(r["item_id"], r["context"], r["question"],
                         r["gold"], r["pred"],
                         context_prefix=r.get("context_prefix", ""),
                         gold_variants=r.get("gold_variants", []),
                         pred_variants=r.get("pred_variants", []))
        partitioner = SimpleContextPartitioner(base_item.context, source_type="word")
        q = args.num_ablations
        masks = np.zeros((q, partitioner.num_sources), dtype=bool)
        outputs = np.zeros(q, dtype=float)
        for seed in range(q):
            rng = np.random.RandomState(args.seed + seed)
            mask = rng.choice([False, True], size=partitioner.num_sources,
                              p=[1-args.keep_prob, args.keep_prob])
            masks[seed] = mask
            context = partitioner.get_context(mask)
            item = Item(base_item.item_id, context, base_item.question,
                        base_item.gold, base_item.pred,
                        context_prefix=base_item.context_prefix,
                        gold_variants=base_item.gold_variants,
                        pred_variants=base_item.pred_variants)
            outputs[seed] = att.S0(att.prepare(item))

        coef, bias = LassoRegression(args.lasso_alpha).fit(masks, outputs, 1)

        # Map each original fine span to the ContextCite sources it contains.
        # Substring offsets are reconstructed exactly as the official partitioner.
        source_ranges, cursor = [], 0
        for source in partitioner.sources:
            start = base_item.context.find(source, cursor)
            source_ranges.append((start, start + len(source)))
            cursor = start + len(source)
        proxy = []
        for s in r["spans"]:
            text = s["text"]
            # Locate by ordered occurrence; token offsets disambiguate repeated
            # phrases poorly, so choose the occurrence whose source sequence
            # exactly reconstructs the span text after whitespace normalization.
            norm = " ".join(text.split())
            candidates = []
            for a in range(len(source_ranges)):
                for b in range(a, min(len(source_ranges), a + 8)):
                    joined = " ".join(partitioner.sources[a:b+1])
                    joined = joined.replace(" .", ".").replace(" ,", ",")
                    if " ".join(joined.split()) == norm:
                        candidates.append((a, b))
            if candidates:
                # Repeated surface forms have nearly identical intended proxy;
                # use the first occurrence, matching build_word_spans order.
                a, b = candidates[0]
                proxy.append(float(np.sum(coef[a:b+1])))
            else:
                # Robust fallback: sum sources whose text occurs in the span.
                proxy.append(float(sum(coef[i] for i, src in enumerate(partitioner.sources)
                                       if src in text)))
        proxy = np.asarray(proxy)
        exact = np.asarray([float(s["u"]) for s in r["spans"]])
        from spanattr.core import Span
        span_objs = [Span(i, s["start"], s["end"], s["text"])
                     for i, s in enumerate(r["spans"])]
        selected = nms_disjoint(proxy, span_objs, m=args.m)
        royal = [i for i, s in enumerate(r["spans"])
                 if s["text"].lower() == "the royal society"]
        rank = np.argsort(-proxy)
        royal_ranks = [int(np.where(rank == i)[0][0]) + 1 for i in royal]
        rec = {"item_id": r["item_id"], "n_sources": partitioner.num_sources,
               "num_ablations": q, "keep_prob": args.keep_prob,
               "lasso_alpha": args.lasso_alpha, "bias": float(bias),
               "train_r2": float(1 - np.sum((outputs-(masks@coef+bias))**2) /
                                 (np.sum((outputs-outputs.mean())**2)+1e-12)),
               "rho_proxy_exact": spearman(proxy, exact),
               "source_text": partitioner.sources,
               "source_coef": [float(x) for x in coef],
               "span_proxy": [float(x) for x in proxy],
               "selected": selected,
               "selected_text": [r["spans"][i]["text"] for i in selected],
               "royal_span_ids": royal, "royal_ranks": royal_ranks}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"{r['item_id']}: sources={partitioner.num_sources} q={q} "
              f"trainR2={rec['train_r2']:.3f} rho={rec['rho_proxy_exact']:.3f} "
              f"RoyalRanks={royal_ranks} selected={rec['selected_text']}")
    out.close()
    print(f"Wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
