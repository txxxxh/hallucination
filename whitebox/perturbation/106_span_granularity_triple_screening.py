#!/usr/bin/env python3
"""Active span-granularity and exact triple-screening pilot.

Part A compares sliding raw-word spans with widths 1, (2,3), and (4,5,6).
Part B takes the held-out Stage-82 active spans, computes exact active singles,
pairs, and triples, then tests whether the top-10 pairs can screen triples.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, nms_disjoint, set_seed

p94 = importlib.import_module("94_active_pair_screening_pilot")
p95 = importlib.import_module("95_active_abs_pair_screening_pilot")


def opt(att, prep, spans, basis, s0, a, seed, inits=None):
    return p95.optimize_abs(att, prep, spans, basis, s0, a.steps,
                            a.directions, a.mu, a.lr, seed,
                            repeats=a.repeats, inits=inits)


def warm_start(parts, chosen):
    """Concatenate optimized blocks, plus one-block/pair-only safe starts."""
    import torch
    full = torch.cat([parts[i]["z"] for i in chosen])
    starts = [full]
    for keep_n in (1, 2):
        if keep_n >= len(chosen):
            continue
        for keep in itertools.combinations(range(len(chosen)), keep_n):
            starts.append(torch.cat([
                parts[idx]["z"] if pos in keep else torch.zeros_like(parts[idx]["z"])
                for pos, idx in enumerate(chosen)
            ]))
    return starts


def grain_stats(att, prep, basis, s0, a, ni):
    configs = {"w1": (1,), "w2_3": (2, 3), "w4_6": (4, 5, 6)}
    result = {}
    for gi, (name, widths) in enumerate(configs.items()):
        spans = att.build_word_spans(prep, widths=widths, stride=1)
        vals = []
        signed = []
        queries = 0
        for si, span in enumerate(spans):
            z = opt(att, prep, [span], basis, s0, a,
                    a.seed + 10_000_019 * ni + 100_003 * gi + si)
            vals.append(z["u"])
            signed.append(z["signed_u"])
            queries += z["queries"]
        arr = np.asarray(vals)
        ids = nms_disjoint(arr, spans, min(a.grain_topk, len(spans)))
        selected = arr[ids] if ids else np.asarray([])
        # A sign crossing means this optimized perturbation moved S across zero.
        scores = s0 - np.asarray(signed)
        result[name] = {
            "widths": list(widths), "n_spans": len(spans), "queries": queries,
            "mean_abs_delta": float(arr.mean()),
            "median_abs_delta": float(np.median(arr)),
            "p90_abs_delta": float(np.quantile(arr, .9)),
            "max_abs_delta": float(arr.max()),
            "top_disjoint_mean": float(selected.mean()) if len(selected) else None,
            "top_disjoint_max": float(selected.max()) if len(selected) else None,
            "any_crossing": bool(np.any(s0 * scores <= 0)),
            "crossing_fraction": float(np.mean(s0 * scores <= 0)),
            "top_span_text": [spans[i].text for i in ids],
            "top_span_u": [float(arr[i]) for i in ids],
        }
    return result


def eval_candidates(triples, truth, candidates):
    true_idx = int(np.argmax(truth))
    ids = sorted(set(candidates))
    chosen = max(ids, key=lambda x: truth[x]) if ids else None
    return {
        "n_candidates": len(ids),
        "candidate_fraction": len(ids) / len(triples),
        "best_recall": bool(true_idx in ids),
        "regret": float(truth[true_idx] - truth[chosen]) if chosen is not None else None,
        "true_triple": list(triples[true_idx]),
        "selected_triple": list(triples[chosen]) if chosen is not None else None,
    }


def triple_test(row, all_spans, att, prep, basis, s0, a, ni):
    import torch
    # Re-run NMS from all retained scores; Stage 82 selected ids stop at five.
    active_scores = np.asarray([x["abs_u"] for x in row["methods"]["active"]])
    ids = nms_disjoint(active_scores, all_spans, a.m)
    spans = [all_spans[i] for i in ids]
    singles = [opt(att, prep, [sp], basis, s0, a,
                   a.seed + 1_000_003 * ni + 1009 * i)
               for i, sp in enumerate(spans)]
    pairs = list(itertools.combinations(range(len(spans)), 2))
    pair_results = []
    for pi, (i, j) in enumerate(pairs):
        pair_results.append(opt(att, prep, [spans[i], spans[j]], basis, s0, a,
                                a.seed + 7_000_001 * ni + 7919 * pi,
                                warm_start(singles, (i, j))))
    triples = list(itertools.combinations(range(len(spans)), 3))
    triple_results = []
    for ti, tri in enumerate(triples):
        # Include singleton starts and the three already optimized pair starts.
        starts = warm_start(singles, tri)
        for pair_pos in itertools.combinations(range(3), 2):
            pair = tuple(tri[p] for p in pair_pos)
            pr = pair_results[pairs.index(tuple(sorted(pair)))]
            blocks = []
            lo = 0
            pair_blocks = {}
            for idx, dim in zip(pair, pr["dims"]):
                pair_blocks[idx] = pr["z"][lo:lo + dim]
                lo += dim
            blocks.append(torch.cat([
                pair_blocks.get(idx, torch.zeros_like(singles[idx]["z"])) for idx in tri
            ]))
            starts.extend(blocks)
        triple_results.append(opt(att, prep, [spans[i] for i in tri], basis, s0, a,
                                  a.seed + 11_000_009 * ni + 104729 * ti, starts))
        print(f"  {row['item_id']} triple {ti + 1}/{len(triples)}", flush=True)

    su = np.asarray([z["u"] for z in singles])
    pu = np.asarray([z["u"] for z in pair_results])
    tu = np.asarray([z["u"] for z in triple_results])
    top_pair_ids = np.argsort(-pu)[:min(a.top_pairs, len(pairs))]
    top_pair_set = {pairs[i] for i in top_pair_ids}

    expanded = [ti for ti, tri in enumerate(triples)
                if any(tuple(sorted(p)) in top_pair_set for p in itertools.combinations(tri, 2))]
    # Strict 10-candidate version: for every retained pair, add its best singleton.
    strict_tris = set()
    for pi in top_pair_ids:
        i, j = pairs[pi]
        k = max((x for x in range(len(spans)) if x not in (i, j)), key=lambda x: su[x])
        strict_tris.add(tuple(sorted((i, j, k))))
    strict = [triples.index(t) for t in strict_tris]

    # Rank all expanded candidates by max_{retained pair in triple}(u_ij + u_k).
    proxy = {}
    for ti in expanded:
        tri = triples[ti]
        scores = []
        for p in itertools.combinations(tri, 2):
            p = tuple(sorted(p))
            if p in top_pair_set:
                k = next(x for x in tri if x not in p)
                scores.append(pu[pairs.index(p)] + su[k])
        proxy[ti] = max(scores)
    proxy_order = sorted(expanded, key=lambda x: proxy[x], reverse=True)
    screened = {str(b): eval_candidates(triples, tu, proxy_order[:min(b, len(proxy_order))])
                for b in a.triple_budgets}
    screened["strict_best_extension"] = eval_candidates(triples, tu, strict)
    screened["all_top_pair_extensions"] = eval_candidates(triples, tu, expanded)
    return {
        "m": len(spans), "span_ids": ids, "span_text": [s.text for s in spans],
        "n_pairs": len(pairs), "n_triples": len(triples),
        "single_u": su.tolist(), "pair_u": pu.tolist(), "triple_u": tu.tolist(),
        "top_pair_ids": [int(x) for x in top_pair_ids],
        "top_pairs": [list(pairs[x]) for x in top_pair_ids], "screening": screened,
    }


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--in82", default="runs/82_active_n30_r32_q4.jsonl")
    p.add_argument("--items", default="data/items_n128_generation_flip.json")
    p.add_argument("--basis", default="runs/81_q0000_active_basis.pt")
    p.add_argument("--out", default="runs/106_span_grain_triple_n5_m8.json")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16"); p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=5); p.add_argument("--offset", type=int, default=0); p.add_argument("--m", type=int, default=8)
    p.add_argument("--rank", type=int, default=32); p.add_argument("--steps", type=int, default=2)
    p.add_argument("--directions", type=int, default=4); p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--mu", type=float, default=.25); p.add_argument("--lr", type=float, default=.35)
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--triple-budgets", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--grain-topk", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=16); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); set_seed(a.seed)
    rows = [json.loads(x) for x in open(a.in82) if x.strip()][a.offset:a.offset + a.samples]
    items = {x.item_id: x for x in (Item.from_dict(d) for d in json.load(open(a.items)))}
    saved = torch.load(a.basis, map_location="cpu", weights_only=True)
    if set(saved["calibration_item_ids"]) & {x["item_id"] for x in rows}:
        raise ValueError("calibration leakage")
    loader = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = loader(a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean", length_norm=True,
                         max_rows=a.max_rows)
    basis = saved["basis"][:, :a.rank].to(a.device, dtype=att.emb_layer.weight.dtype)
    outputs = []; begin = time.time()
    for ni, row in enumerate(rows):
        prep = att.prepare(items[row["item_id"]]); s0 = att.S0(prep)
        all_spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        if [s.text for s in all_spans] != row["span_text"]:
            raise ValueError("span reconstruction drift")
        grain = grain_stats(att, prep, basis, s0, a, ni)
        triple = triple_test(row, all_spans, att, prep, basis, s0, a, ni)
        outputs.append({"item_id": row["item_id"], "S0": s0,
                        "granularity": grain, "triple": triple})
        print(f"[{ni + 1}/{len(rows)}] {row['item_id']} done", flush=True)

    grain_summary = {}
    for g in ("w1", "w2_3", "w4_6"):
        grain_summary[g] = {k: float(np.mean([x["granularity"][g][k] for x in outputs]))
                            for k in ("n_spans", "mean_abs_delta", "median_abs_delta",
                                      "p90_abs_delta", "max_abs_delta", "top_disjoint_mean",
                                      "top_disjoint_max", "crossing_fraction")}
        grain_summary[g]["sample_crossing_rate"] = float(np.mean([
            x["granularity"][g]["any_crossing"] for x in outputs]))
    screen_summary = {}
    keys = [str(x) for x in a.triple_budgets] + ["strict_best_extension", "all_top_pair_extensions"]
    for key in keys:
        vals = [x["triple"]["screening"][key] for x in outputs]
        screen_summary[key] = {
            "recall": float(np.mean([x["best_recall"] for x in vals])),
            "mean_regret": float(np.mean([x["regret"] for x in vals])),
            "mean_candidates": float(np.mean([x["n_candidates"] for x in vals])),
            "mean_candidate_fraction": float(np.mean([x["candidate_fraction"] for x in vals])),
        }
    report = {"config": vars(a), "elapsed_seconds": time.time() - begin,
              "granularity_summary": grain_summary,
              "triple_screening_summary": screen_summary, "items": outputs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"granularity": grain_summary, "triple": screen_summary}, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
