#!/usr/bin/env python3
"""Compare static and perturbation-induced hidden-state correctness features.

The expensive collection stage reuses the oracle top-k span ranking saved by
70_oracle_topk_detector.py.  Labels are copied for later evaluation but are
never used to select spans or construct features.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from spanattr.core import Item, SpanAttributor, set_seed


def _teacher_forced_hidden(att, prep, alphas, layers):
    """Return answer-last/mean and span-position hidden states for each alpha."""
    import torch

    ans = prep.pred_variant_ids[0]
    outputs = []
    for start in range(0, len(alphas), att.max_rows):
        alpha = alphas[start:start + att.max_rows]
        pe = att._embeds(prep, alpha)
        batch = pe.shape[0]
        ae = att.emb_layer(ans).detach().unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([pe, ae.to(pe.dtype)], dim=1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=att.device)
        with torch.inference_mode():
            out = att.model(inputs_embeds=seq, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
        selected = []
        prompt_len = pe.shape[1]
        for layer in layers:
            h = out.hidden_states[layer].float()
            answer = h[:, prompt_len:prompt_len + len(ans)]
            selected.append((answer[:, -1], answer.mean(dim=1), h[:, :prompt_len]))
        outputs.append(selected)
        del out, seq, pe

    merged = []
    for li in range(len(layers)):
        last = torch.cat([x[li][0] for x in outputs]).cpu().numpy()
        mean = torch.cat([x[li][1] for x in outputs]).cpu().numpy()
        prompt = torch.cat([x[li][2] for x in outputs]).cpu().numpy()
        merged.append((last, mean, prompt))
    return merged


def collect(args):
    import torch

    oracle = {x["key"]: x for x in map(json.loads, open(args.oracle))}
    source = [json.loads(x) for x in open(args.source) if x.strip()]
    data = {str(x["key"]): x for x in json.load(open(args.data))}
    records = {x["key"]: x for x in map(json.loads, open(args.records))}
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    load_model = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.batch)

    for number, src in enumerate(source[:args.limit or None], 1):
        key = src["key"]
        target = cache / f"{key}.npz"
        if target.exists() and args.resume:
            continue
        old = oracle[key]
        raw, rr = data[key], records[key]
        pred = str(rr["parsed_answer"])
        right, wrong = str(raw["rgt_ans"]), str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other))
        item.pred, item.gold = pred, other
        prep = att.prepare(item)
        spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        u = np.asarray(old["u"], dtype=np.float32)
        if len(u) != len(spans):
            raise ValueError(f"{key}: cached {len(u)} spans, rebuilt {len(spans)}")
        top_ids = np.argsort(-np.abs(u))[:args.topk]
        alphas = torch.stack([
            torch.zeros(len(prep.prompt_ids), device=args.device),
            *[att.alpha_from_spans(prep, [int(i)]) for i in top_ids],
        ])
        hidden = _teacher_forced_hidden(att, prep, alphas, args.layers)

        answer_last, answer_mean, span_before, span_after = [], [], [], []
        for last, mean, prompt in hidden:
            answer_last.append(last)
            answer_mean.append(mean)
            before, after = [], []
            for rank, span_id in enumerate(top_ids):
                span = spans[int(span_id)]
                before.append(prompt[0, span.start:span.end].mean(axis=0))
                after.append(prompt[rank + 1, span.start:span.end].mean(axis=0))
            span_before.append(before)
            span_after.append(after)
        np.savez_compressed(
            target,
            key=np.asarray(key), group=np.asarray(src["group"]),
            correct=np.asarray(int(src["correct"])), layers=np.asarray(args.layers),
            top_ids=top_ids, top_u=u[top_ids],
            answer_last=np.asarray(answer_last, dtype=np.float16),
            answer_mean=np.asarray(answer_mean, dtype=np.float16),
            span_before=np.asarray(span_before, dtype=np.float16),
            span_after=np.asarray(span_after, dtype=np.float16),
        )
        print(f"[{number}/{min(len(source), args.limit or len(source))}] {key}", flush=True)


def _aggregate(z, weights):
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.abs(weights) / (np.abs(weights).sum() + 1e-8)
    return np.einsum("k,kd->d", weights, np.asarray(z, dtype=np.float32))


def train(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    files = sorted(Path(args.cache_dir).glob("*.npz"))
    rows = [np.load(path) for path in files]
    if not rows:
        raise FileNotFoundError(f"no npz files in {args.cache_dir}")
    y = np.asarray([int(r["correct"]) for r in rows])
    groups = np.asarray([str(r["group"]) for r in rows])
    cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
    report = {"n": len(rows), "topk": args.topk, "pca": args.pca, "layers": {}}

    for layer_pos, layer in enumerate(rows[0]["layers"].tolist()):
        feature_sets = {name: [] for name in (
            "answer_static", "span_static", "answer_delta",
            "span_delta", "answer_static_delta", "all_hidden")}
        for r in rows:
            w = r["top_u"].astype(np.float32)
            answer = r["answer_mean"][layer_pos].astype(np.float32)
            answer0 = answer[0]
            answer_delta = _aggregate(answer[1:] - answer0, w)
            span0 = _aggregate(r["span_before"][layer_pos], w)
            span_delta = _aggregate(
                r["span_after"][layer_pos].astype(np.float32)
                - r["span_before"][layer_pos].astype(np.float32), w)
            feature_sets["answer_static"].append(answer0)
            feature_sets["span_static"].append(span0)
            feature_sets["answer_delta"].append(answer_delta)
            feature_sets["span_delta"].append(span_delta)
            feature_sets["answer_static_delta"].append(np.r_[answer0, answer_delta])
            feature_sets["all_hidden"].append(np.r_[answer0, span0, answer_delta, span_delta])

        layer_report = {}
        for name, values in feature_sets.items():
            X = np.asarray(values, dtype=np.float32)
            components = min(args.pca, len(X) - 1, X.shape[1])
            estimator = make_pipeline(
                StandardScaler(), PCA(n_components=components, whiten=True,
                                      random_state=args.seed),
                LogisticRegression(C=args.C, max_iter=5000,
                                   class_weight="balanced", random_state=args.seed),
            )
            p = cross_val_predict(estimator, X, y, groups=groups, cv=cv,
                                  method="predict_proba")[:, 1]
            layer_report[name] = {
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
            }
        report["layers"][str(layer)] = layer_report
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("collect", "train", "all"))
    p.add_argument("--oracle", default="runs/70_oracle_topk_n128.jsonl")
    p.add_argument("--source", default="runs/69_generation_flip_n128_q16.jsonl")
    p.add_argument("--data", default="../shuffled_prepend_names_question.json")
    p.add_argument("--records", default="../tool_gate_correctness_names_llama31_8b/records.jsonl")
    p.add_argument("--cache-dir", default="runs/71_hidden_delta_top11")
    p.add_argument("--report", default="runs/71_hidden_delta_top11_report.json")
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--topk", type=int, default=11)
    p.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pca", type=int, default=16)
    p.add_argument("--C", type=float, default=.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
