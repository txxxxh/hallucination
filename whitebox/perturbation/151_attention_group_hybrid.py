#!/usr/bin/env python3
"""Attention-prior group testing followed by local exact span search."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "61.jsonl"
ATTN = RUNS / "150_attention_cache"
CACHE = RUNS / "151_group_cache"
REPORT = RUNS / "151_attention_group_hybrid_report.json"


def block_layout(prep, n_blocks):
    edges = np.linspace(prep.ctx_start, prep.ctx_end, n_blocks + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_blocks) if edges[i] < edges[i + 1]]


def collect(args):
    import torch
    loader = importlib.import_module("61_grad_span_proposal")
    model, tokenizer = loader.load_model(args.model, "bfloat16", "cuda")
    from spanattr.core import Item, SpanAttributor
    att = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in SOURCE.open() if x.strip()]
    for number, row in enumerate(rows, 1):
        path = CACHE / f"{row['item_id']}.npz"
        if path.exists() and args.resume:
            continue
        item = Item(row["item_id"], row["context"], row["question"], row["gold"],
                    row["pred"], context_prefix=row.get("context_prefix", ""),
                    gold_variants=row.get("gold_variants", []),
                    pred_variants=row.get("pred_variants", []))
        prep = att.prepare(item)
        blocks = block_layout(prep, args.blocks)
        rng = np.random.default_rng(1709 + number)
        masks = rng.random((args.queries, len(blocks))) < .5
        # Add complementary pairs: more stable under nonlinear global shifts.
        half = args.queries // 2
        masks[half:2*half] = ~masks[:half]
        alphas = []
        for mask in masks:
            alpha = torch.zeros(len(prep.prompt_ids), device=att.device)
            for enabled, (a, b) in zip(mask, blocks):
                if enabled:
                    alpha[a:b] = 1
            alphas.append(alpha)
        baseline = att.S0(prep)
        scores = baseline - att.S_batched(prep, torch.stack(alphas)).numpy()
        np.savez_compressed(path, masks=masks.astype(np.uint8), effects=scores.astype(np.float32),
                            blocks=np.asarray(blocks, int))
        print(f"[{number}/{len(rows)}] {row['item_id']}", flush=True)


def attention_block_prior(row, blocks):
    with np.load(ATTN / f"{row['item_id']}.npz") as z:
        # span attention is already pooled. Map spans to blocks and take robust max.
        pred = z["pred"].astype(np.float32).mean((1, 2))
        gold = z["gold"].astype(np.float32).mean((1, 2))
    score = np.abs(pred - gold) + pred
    output = np.zeros(len(blocks))
    for i, (a, b) in enumerate(blocks):
        values = [score[j] for j, span in enumerate(row["spans"])
                  if span["end"] > a and span["start"] < b]
        output[i] = max(values, default=0.)
    return output


def select_blocks(masks, effects, prior, q, keep, prior_weight, alpha):
    X = masks[:q].astype(float)
    y = effects[:q]
    # Centering removes the all-mask/global margin shift; Ridge is stable for q < blocks.
    coef = Ridge(alpha=alpha, fit_intercept=True).fit(X, y).coef_
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    combined = np.abs(z(coef)) + prior_weight * z(prior)
    return np.argsort(-combined)[:keep], coef, combined


def evaluate():
    rows = [json.loads(x) for x in SOURCE.open() if x.strip()]
    configs = [(q, k, w, a) for q in (8, 12, 16) for k in (2, 3, 4, 5, 6)
               for w in (0., .25, .5, 1.) for a in (.1, 1., 10.)]
    accum = {c: [] for c in configs}
    baselines = {k: [] for k in (2, 3, 4, 5, 6)}
    for row in rows:
        with np.load(CACHE / f"{row['item_id']}.npz") as z:
            masks, effects, blocks = z["masks"], z["effects"], z["blocks"]
        prior = attention_block_prior(row, blocks)
        exact = np.abs(np.asarray([x["u"] for x in row["spans"]]))
        best = int(np.argmax(exact)); top5 = set(np.argsort(-exact)[:5])
        for keep in baselines:
            chosen = set(np.argsort(-prior)[:keep])
            ids = [i for i, s in enumerate(row["spans"])
                   if any(s["end"] > blocks[b][0] and s["start"] < blocks[b][1] for b in chosen)]
            baselines[keep].append((best in ids, len(top5.intersection(ids))/5,
                                    exact[ids].max(initial=0)/(exact[best]+1e-12), len(ids)))
        for config in configs:
            q, keep, weight, alpha = config
            selected, _, _ = select_blocks(masks, effects, prior, q, keep, weight, alpha)
            ids = [i for i, s in enumerate(row["spans"])
                   if any(s["end"] > blocks[b][0] and s["start"] < blocks[b][1] for b in selected)]
            # Fine stage exactly measures only these ids. Its selected keyword is
            # therefore the highest exact-effect span in the shortlist.
            accum[config].append((best in ids, len(top5.intersection(ids))/5,
                                  exact[ids].max(initial=0)/(exact[best]+1e-12), len(ids)))
    def summarize(values, q):
        a = np.asarray(values, float)
        fine = a[:, 3]
        full = np.asarray([len(r["spans"]) for r in rows], float)
        return {"top1_recall": float(a[:, 0].mean()), "top5_recall": float(a[:, 1].mean()),
                "best_effect_ratio": float(a[:, 2].mean()),
                "mean_fine_queries": float(fine.mean()),
                "mean_total_queries": float((fine + q + 2).mean()),
                "full_enumeration_mean_queries": float(full.mean()),
                "query_reduction": float(1 - (fine + q + 2).sum()/full.sum())}
    results = []
    for config, values in accum.items():
        q, keep, weight, alpha = config
        results.append({"queries": q, "blocks_kept": keep, "attention_weight": weight,
                        "ridge_alpha": alpha, **summarize(values, q)})
    # Select by recall first, then query reduction, without target-domain data.
    results.sort(key=lambda x: (x["top1_recall"], x["top5_recall"], x["query_reduction"]), reverse=True)
    attention_only = [{"blocks_kept": k, **summarize(v, 0)} for k, v in baselines.items()]
    report = {"n_items": len(rows), "design": "2 contrastive-attention forwards + q complementary random block masks + exact perturbation only inside selected blocks",
              "selection_note": "exploratory Scientist calibration; final configuration must be frozen before target evaluation",
              "attention_only": attention_only, "best_configs": results[:20], "all_configs": results}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "all_configs"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__": main()
