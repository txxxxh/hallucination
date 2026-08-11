#!/usr/bin/env python3
"""Stage 82: aggregate Stage-81 ZO versus mean keyword rankings."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in81", default="runs/81_zo_span_keywords.jsonl")
    p.add_argument("--out", default="runs/82_zo_mean_comparison.json")
    a = p.parse_args()
    records = [json.loads(x) for x in open(a.in81) if x.strip()]
    if not records: raise SystemExit("no Stage-81 records")
    per_item, am, az, ar = [], [], [], []
    for r in records:
        mean = np.asarray([s["mean_u"] for s in r["spans"]], float)
        zo = np.asarray([s["zo_u"] for s in r["spans"]], float)
        rnd = np.asarray([s["random_best_u"] for s in r["spans"]], float)
        ms, zs = set(r["mean_selected"]), set(r["zo_selected"]); union = ms | zs
        per_item.append({"item_id": r["item_id"], "rho_mean_zo": r["rho_mean_zo"],
            "topk_intersection": len(ms & zs),
            "topk_jaccard": len(ms & zs) / len(union) if union else 1.0,
            "mean_top_abs_gain": float(np.mean(np.abs(mean[list(ms)]))) if ms else None,
            "zo_top_gain": float(np.mean(zo[list(zs)])) if zs else None,
            "zo_minus_random_mean": float(np.mean(zo-rnd))})
        am.extend(mean); az.extend(zo); ar.extend(rnd)
    am, az, ar = map(np.asarray, (am, az, ar))
    rho = [x["rho_mean_zo"] for x in per_item if np.isfinite(x["rho_mean_zo"])]
    report = {"n_items": len(records), "n_spans": len(az),
        "mean_item_rank_rho": float(np.mean(rho)) if rho else None,
        "mean_topk_jaccard": float(np.mean([x["topk_jaccard"] for x in per_item])),
        "mean_zo_gain": float(az.mean()),
        "mean_average_baseline_abs_gain": float(np.abs(am).mean()),
        "mean_random_best_gain": float(ar.mean()),
        "fraction_zo_beats_mean_abs": float(np.mean(az > np.abs(am))),
        "fraction_zo_beats_random_best": float(np.mean(az > ar)), "per_item": per_item}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
