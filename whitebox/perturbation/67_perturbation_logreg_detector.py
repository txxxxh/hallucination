# -*- coding: utf-8 -*-
"""Collect gold-free embedding perturbation features and train a LR detector.

The correctness label is used only by ``train``.  During feature collection the
score is the generated option versus the other visible option; it never uses
which option is correct.  Random masks neutralize equal token chunks to the
same mean-embedding baseline as the span attribution pipeline.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spanattr.core import Item, SpanAttributor, set_seed
from importlib import import_module


def collect(args):
    import torch
    data = {str(x["key"]): x for x in json.load(open(args.data))}
    records = [json.loads(x) for x in open(args.records) if x.strip()]
    records = [x for x in records if x.get("parse_valid")]
    if args.limit:
        records = records[:args.limit]
    done = set()
    if args.resume and Path(args.features).exists():
        done = {json.loads(x)["key"] for x in open(args.features) if x.strip()}
    elif Path(args.features).exists():
        raise FileExistsError(f"{args.features} exists; pass --resume")

    load_model = import_module("61_grad_span_proposal").load_model
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.max_rows)
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(args.features, "a") as fh:
        for n, rr in enumerate(records):
            key = str(rr["key"])
            if key in done:
                continue
            raw = data[key]
            pred = str(rr["parsed_answer"])
            options = [str(raw["rgt_ans"]), str(raw["wrg_ans"])]
            other = options[1] if pred == options[0] else options[0]
            # Here Item.gold is merely the other visible option, not ground truth.
            d = dict(raw, pred=pred, gold=other)
            item = Item.from_dict(d)
            item.pred = pred
            item.gold = other
            prep = att.prepare(item)
            S0 = att.S0(prep)

            positions = np.arange(prep.ctx_start, prep.ctx_end)
            chunks = [x for x in np.array_split(positions, args.chunks) if len(x)]
            masks = []
            for _ in range(args.queries):
                z = rng.random(len(chunks)) < 0.5
                if not z.any():
                    z[rng.integers(len(chunks))] = True
                a = torch.zeros(prep.prompt_ids.shape[0], device=args.device)
                for j in np.flatnonzero(z):
                    a[chunks[int(j)]] = 1.0
                masks.append(a)
            A = torch.stack(masks)
            Sm = att.S_batched(prep, A).numpy().astype(float)
            all_a = att.alpha_all(prep).unsqueeze(0)
            Sall = float(att.S(prep, all_a)[0])
            delta = S0 - Sm

            base = [S0, abs(S0)]
            pert = [
                float(delta.mean()), float(delta.std()),
                float(delta.min()), float(delta.max()),
                float(np.abs(delta).mean()), float(np.abs(delta).max()),
                float(np.quantile(np.abs(delta), .5)),
                float(np.quantile(np.abs(delta), .9)),
                float((delta > 0).mean()), float((Sm < 0).mean()),
                float(Sm.mean()), float(Sm.std()), float(Sm.min()),
                float(Sall), float(S0 - Sall),
            ]
            out = {
                "key": key, "group": str(raw.get("rgt_ans_qid", key)),
                "correct": bool(rr["correct"]), "pred": pred,
                "other_option": other, "n_context_tokens": len(positions),
                "base_features": base, "perturb_features": pert,
                "raw_mask_scores": Sm.tolist(),
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{n+1}/{len(records)}] {key} correct={int(rr['correct'])} "
                  f"S0={S0:+.3f} d_abs={np.abs(delta).mean():.3f}", flush=True)


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, roc_auc_score)
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [json.loads(x) for x in open(args.features) if x.strip()]
    y = np.asarray([int(x["correct"]) for x in rows])
    groups = np.asarray([x["group"] for x in rows])
    Xb = np.asarray([x["base_features"] for x in rows], dtype=float)
    Xp = np.asarray([x["perturb_features"] for x in rows], dtype=float)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed)
    result = {"n": len(y), "n_correct": int(y.sum()), "folds": args.folds,
              "label": "greedy generation correctness"}
    for name, X in [("likelihood_only", Xb), ("perturbation_only", Xp),
                    ("combined", np.concatenate([Xb, Xp], axis=1))]:
        est = make_pipeline(StandardScaler(), LogisticRegression(
            C=args.C, max_iter=5000, class_weight="balanced",
            random_state=args.seed))
        p = cross_val_predict(est, X, y, groups=groups, cv=cv,
                              method="predict_proba", n_jobs=1)[:, 1]
        pred = p >= .5
        result[name] = {
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "train", "all"])
    ap.add_argument("--data", default="../shuffled_prepend_names_question.json")
    ap.add_argument("--records", default="../tool_gate_correctness_names_llama31_8b/records.jsonl")
    ap.add_argument("--features", default="runs/67_detection_features.jsonl")
    ap.add_argument("--report", default="runs/67_detection_logreg_report.json")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
