#!/usr/bin/env python3
"""Fast keyword localization from contrastive attention, calibrated against exact LOO spans."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "61.jsonl"
CACHE = RUNS / "150_attention_cache"
REPORT = RUNS / "150_fast_attention_keyword_report.json"


def attention_map(model, prep, answer_ids):
    import torch
    ids = torch.cat([prep.prompt_ids, answer_ids]).unsqueeze(0)
    with torch.inference_mode():
        output = model(input_ids=ids, output_attentions=True, use_cache=False)
    prompt_len = prep.prompt_ids.shape[0]
    layers = []
    for tensor in output.attentions:
        # [heads, answer queries, prompt keys]
        values = tensor[0, :, prompt_len-1:prompt_len+answer_ids.shape[0]-1, :prompt_len]
        layers.append(values.float().mean(1).cpu().numpy())
    return np.stack(layers)  # layer, head, prompt-token


def extract(args):
    import torch
    loader = importlib.import_module("61_grad_span_proposal")
    model, tokenizer = loader.load_model(args.model, "bfloat16", "cuda")
    from spanattr.core import Item, SpanAttributor
    attributor = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                                length_norm=True, max_rows=8)
    CACHE.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in SOURCE.open() if x.strip()]
    for index, record in enumerate(records, 1):
        path = CACHE / f"{record['item_id']}.npz"
        if path.exists() and args.resume:
            continue
        item = Item(record["item_id"], record["context"], record["question"],
                    record["gold"], record["pred"], record.get("context_prefix", ""),
                    record.get("gold_variants"), record.get("pred_variants"))
        prep = attributor.prepare(item)
        pred = attention_map(model, prep, prep.pred_variant_ids[0])
        gold = attention_map(model, prep, prep.gold_variant_ids[0])
        starts = np.asarray([x["start"] for x in record["spans"]], int)
        ends = np.asarray([x["end"] for x in record["spans"]], int)
        def pool(values):
            return np.stack([values[:, :, a:b].sum(2) for a, b in zip(starts, ends)])
        np.savez_compressed(path, pred=pool(pred).astype(np.float16),
                            gold=pool(gold).astype(np.float16))
        print(f"[{index}/{len(records)}] {record['item_id']}", flush=True)
        del pred, gold
        torch.cuda.empty_cache()


def rank_metrics(rows, scores):
    metrics = {"spearman": [], "hit1": [], "hit3": [], "hit5": [], "hit10": [],
               "top1_effect_ratio": [], "top5_effect_recall": []}
    offset = 0
    for row in rows:
        n = len(row["spans"])
        score = scores[offset:offset+n]
        effect = np.abs([x["u"] for x in row["spans"]])
        offset += n
        if np.std(score) and np.std(effect):
            from scipy.stats import spearmanr
            metrics["spearman"].append(float(spearmanr(score, effect).statistic))
        truth = int(np.argmax(effect)); order = np.argsort(-score)
        for k in (1, 3, 5, 10):
            metrics[f"hit{k}"].append(float(truth in order[:k]))
        metrics["top1_effect_ratio"].append(float(effect[order[0]] / (effect[truth] + 1e-12)))
        denom = np.sort(effect)[-5:].sum() + 1e-12
        metrics["top5_effect_recall"].append(float(effect[order[:5]].sum() / denom))
    return {key: float(np.mean(value)) for key, value in metrics.items()}


def main_features(pred, gold):
    # input: spans, layers, heads. Features retain layer patterns but avoid a 1024-D head overfit.
    diff = pred - gold
    eps = 1e-8
    blocks = [pred.mean(2), gold.mean(2), diff.mean(2), np.abs(diff).mean(2),
              pred.max(2), gold.max(2), np.abs(diff).max(2),
              diff.mean(2) / (pred.mean(2) + gold.mean(2) + eps)]
    return np.concatenate(blocks, 1).astype(np.float32)


def evaluate():
    rows = [json.loads(x) for x in SOURCE.open() if x.strip()]
    features, labels, groups = [], [], []
    heuristics = {name: [] for name in ("mean_all_pred", "last4_pred", "max_head_pred",
                                        "contrast_abs_mean", "contrast_abs_maxhead")}
    for group, row in enumerate(rows):
        with np.load(CACHE / f"{row['item_id']}.npz") as z:
            pred, gold = z["pred"].astype(np.float32), z["gold"].astype(np.float32)
        features.append(main_features(pred, gold))
        effect = np.abs([x["u"] for x in row["spans"]])
        labels.extend(np.log1p(effect / (np.median(effect) + 1e-8)))
        groups.extend([group] * len(effect))
        heuristics["mean_all_pred"].extend(pred.mean((1, 2)))
        heuristics["last4_pred"].extend(pred[:, -4:].mean((1, 2)))
        heuristics["max_head_pred"].extend(pred.mean(1).max(1))
        heuristics["contrast_abs_mean"].extend(np.abs(pred-gold).mean((1, 2)))
        heuristics["contrast_abs_maxhead"].extend(np.abs(pred-gold).mean(1).max(1))
    X, y, group_ids = np.concatenate(features), np.asarray(labels), np.asarray(groups)
    predictions = {name: np.asarray(score) for name, score in heuristics.items()}
    models = {
        "ridge_attention_layers": lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "extra_trees_attention_layers": lambda: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=.5, n_jobs=-1, random_state=42),
        "hist_attention_layers": lambda: HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, l2_regularization=10, learning_rate=.05, random_state=42),
    }
    cv = GroupKFold(5)
    for name, factory in models.items():
        score = np.zeros(len(y))
        for train, test in cv.split(X, y, group_ids):
            score[test] = factory().fit(X[train], y[train]).predict(X[test])
        predictions[name] = score
    # Existing one-backward and 32-step IG baselines, with no LOO used for ranking.
    predictions["first_order_gate_gradient"] = np.concatenate(
        [np.abs([s["u_hat"] for s in row["spans"]]) for row in rows])
    predictions["integrated_gradients_32"] = np.concatenate(
        [np.abs([s["ig"] for s in row["spans"]]) for row in rows])
    results = {name: rank_metrics(rows, score) for name, score in predictions.items()}
    report = {"n_items": len(rows), "n_spans": int(len(y)),
              "target": "rank spans by absolute exact single-span perturbation effect |u|",
              "protocol": "5-fold item-grouped OOF for learned attention rankers; exact LOO effects are labels only; attention methods require two candidate forward passes and no span perturbation enumeration",
              "methods": results}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        extract(args)
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
