#!/usr/bin/env python3
"""Strict pred-only two-stage perturbation on the legacy three benchmarks.

The rows, labels, generations, and grouping keys are identical to the former
two-answer benchmark runs.  The competing answer is deliberately never loaded
into Item/SpanAttributor and cannot affect span selection or features.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "297_multibench_pred_only"
MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
SEEDS = (42, 43, 44)
EXPECTED = {"trivia": 1000, "gsm8k": 942, "drop": 1000}


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def rows(dataset):
    if dataset == "trivia":
        raw = read_jsonl(RUNS / "127_triviaqa_balanced_n1000.jsonl")
        return [dict(key=x["key"], group=x["key"], correct=int(x["correct"]),
                     context=x["context"], question=x["question"],
                     pred=x["generation"]) for x in raw]
    if dataset == "gsm8k":
        raw = read_jsonl(RUNS / "140_gsm8k_natural" / "natural_balanced_n942.jsonl")
        return [dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
                     context=x["question"],
                     question="Provide the complete solution to this math problem.",
                     pred=x["generation"]) for x in raw]
    raw = read_jsonl(RUNS / "166_drop1000" / "drop_balanced_n1000.jsonl")
    return [dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
                 context=x["context"], question=x["question"],
                 pred=x["generation"]) for x in raw]


def prepare_pred_only(att, row, key=None, context=None):
    # gold=pred is only a teacher-forcing container requirement.  Only the
    # first score returned by class_scores_batched is ever retained or used.
    return att.prepare(Item(key or row["key"], context if context is not None else row["context"],
                            row["question"], row["pred"], row["pred"]))


def pred_scores(att, prep, spans):
    import torch
    zero = torch.zeros(len(prep.prompt_ids), device=att.device)
    alpha = torch.stack([zero, *[att.alpha_from_spans(prep, [i])
                                 for i in range(len(spans))]])
    pred, _discarded_duplicate = att.class_scores_batched(prep, alpha)
    return pred.numpy()


def pred_hidden(att, prep, ids, layer=16):
    import torch
    zero = torch.zeros(len(prep.prompt_ids), device=att.device)
    alpha = torch.stack([zero, *[att.alpha_from_spans(prep, [int(i)]) for i in ids]])
    chunks, layer14 = [], None
    answer = prep.pred_variant_ids[0]
    for start in range(0, len(alpha), att.max_rows):
        a = alpha[start:start + att.max_rows]
        pe = att._embeds(prep, a)
        ae = att.emb_layer(answer).detach().unsqueeze(0).expand(len(a), -1, -1)
        seq = __import__("torch").cat([pe, ae.to(pe.dtype)], 1)
        mask = __import__("torch").ones(seq.shape[:2], dtype=__import__("torch").long,
                                         device=att.device)
        with __import__("torch").inference_mode():
            output = att.model(inputs_embeds=seq, attention_mask=mask,
                               output_hidden_states=True, use_cache=False)
        pos = pe.shape[1] + len(answer) - 1
        chunks.append(output.hidden_states[layer][:, pos].float().cpu())
        if layer14 is None:
            layer14 = output.hidden_states[14][0, pos].float().cpu().numpy()
        del output, seq, pe
    return __import__("torch").cat(chunks).numpy(), layer14


def collect(dataset, batch, resume, limit):
    set_seed(42)
    data = rows(dataset)
    if len(data) != EXPECTED[dataset]:
        raise RuntimeError(f"{dataset}: expected {EXPECTED[dataset]} rows, got {len(data)}")
    cache = OUT / dataset
    cache.mkdir(parents=True, exist_ok=True)
    model, tokenizer = importlib.import_module("61_grad_span_proposal").load_model(
        MODEL, "bfloat16", "cuda")
    att = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                         length_norm=True, max_rows=batch)
    span_builder = importlib.import_module("125_collect_current_three_benchmarks")
    selected_rows = data[:limit or None]
    for number, row in enumerate(selected_rows, 1):
        target = cache / f'{row["key"]}.npz'
        if resume and target.exists():
            continue
        prep = prepare_pred_only(att, row)
        spans, chars = span_builder.spans(att, prep)
        p1_all = pred_scores(att, prep, spans)
        effect1 = p1_all[0] - p1_all[1:]
        ids1 = np.argsort(-np.abs(effect1))[:min(5, len(effect1))]
        hidden, layer14 = pred_hidden(att, prep, ids1)
        top = int(ids1[0])
        ca, cb = chars[top]
        deleted = re.sub(r"[ \t]+", " ", row["context"][:ca] + row["context"][cb:])
        deleted = re.sub(r"\s+([,.;:!?])", r"\1", deleted).strip()
        prep2 = prepare_pred_only(att, row, row["key"] + "_d", deleted)
        spans2, _ = span_builder.spans(att, prep2)
        p2_all = pred_scores(att, prep2, spans2)
        effect2 = p2_all[0] - p2_all[1:]
        ids2 = np.argsort(-np.abs(effect2))[:min(5, len(effect2))]
        np.savez_compressed(
            target, key=np.asarray(row["key"]), group=np.asarray(row["group"]),
            correct=np.asarray(row["correct"]), deleted_text=np.asarray(spans[top].text),
            stage1_pred=np.r_[p1_all[0], p1_all[1:][ids1]],
            stage2_pred=np.r_[p2_all[0], p2_all[1:][ids2]],
            pred_hidden=hidden.astype(np.float16), layer14=layer14.astype(np.float16),
            stage1_full=np.asarray(len(spans)), stage2_full=np.asarray(len(spans2)))
        print(f"[{dataset} {number}/{len(selected_rows)}] {row['key']} "
              f"spans={len(spans)}+{len(spans2)}", flush=True)


def ch(scores):
    effect = scores[0] - scores[1:]
    scale = abs(float(scores[0])) + 1e-6
    return np.r_[scores[0], effect, effect / scale, effect.max(initial=0),
                 effect.min(initial=0), np.abs(effect).mean(), effect.std(),
                 np.mean(effect > 0)]


def ch2(scores):
    return np.r_[scores[0], scores[0] - scores[1:]]


def wd(hidden, scores):
    effect = scores[0] - scores[1:]
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * effect[:, None]).sum(0) / (np.abs(effect).sum() + 1e-9)


def metrics(y, probability):
    return {"auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y, probability >= .5))}


def evaluate(dataset):
    data = []
    for path in sorted((OUT / dataset).glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            p = z["stage1_pred"].astype(np.float32)
            q = z["stage2_pred"].astype(np.float32)
            h = z["pred_hidden"].astype(np.float32)
            scalar = np.r_[ch(p), ch2(q), p[0] - q[0]].astype(np.float32)
            data.append((str(z["key"].item()), str(z["group"].item()),
                         int(z["correct"]), scalar, h[0], wd(h, p),
                         z["layer14"].astype(np.float32)))
    if len(data) != EXPECTED[dataset]:
        raise RuntimeError(f"{dataset}: expected {EXPECTED[dataset]} cached rows, got {len(data)}")
    keys = [x[0] for x in data]
    groups = np.asarray([x[1] for x in data])
    y = np.asarray([x[2] for x in data])
    blocks = [np.stack([x[j] for x in data]) for j in range(3, 7)]
    dims = (None, 8, 8, 48)
    seed_reports, seed_probabilities = [], []
    for seed in SEEDS:
        probability = np.zeros(len(y))
        if dataset == "trivia":
            splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(blocks[0], y)
        else:
            splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                blocks[0], y, groups)
        for fold, (train, test) in enumerate(splits, 1):
            left, right = [], []
            for values, dim in zip(blocks, dims):
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                if dim is not None:
                    pca = PCA(min(dim, len(train)-1, a.shape[1]), whiten=True,
                              svd_solver="randomized", random_state=seed).fit(a)
                    a, b = pca.transform(a), pca.transform(b)
                left.append(a); right.append(b)
            classifier = LogisticRegression(C=.03, max_iter=5000,
                class_weight="balanced", solver="liblinear", random_state=seed)
            classifier.fit(np.concatenate(left, 1), y[train])
            probability[test] = classifier.predict_proba(np.concatenate(right, 1))[:, 1]
            print(f"evaluate {dataset} seed={seed} fold={fold}/5", flush=True)
        seed_probabilities.append(probability)
        seed_reports.append({"seed": seed, **metrics(y, probability)})
    mean_probability = np.mean(seed_probabilities, axis=0)
    report = {
        "dataset": dataset, "n": len(y), "correct": int(y.sum()),
        "groups": len(set(groups)),
        "protocol": ("strict pred-only exact two-stage perturbation; legacy two-answer "
                     "benchmark rows/labels but other answer never loaded; 3x5 OOF; "
                     "grouped except TriviaQA unique-key stratified; seeds 42-44; "
                     "fold-local scaler/PCA; LR C=.03"),
        "feature_blocks": ("pred scalar [ch(stage1), ch2(stage2), delete delta] + "
                           "pred hidden original PCA8 + pred weighted displacement PCA8 + layer14 PCA48"),
        "per_seed": seed_reports,
        "mean_per_seed": {name: float(np.mean([x[name] for x in seed_reports]))
                          for name in ("auroc", "auprc", "balanced_accuracy")},
        "mean_probability": metrics(y, mean_probability),
    }
    (OUT / f"{dataset}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / f"{dataset}_predictions.jsonl").open("w") as handle:
        for i, key in enumerate(keys):
            handle.write(json.dumps({"key": key, "correct": int(y[i]),
                "probabilities": [float(x[i]) for x in seed_probabilities],
                "mean_probability": float(mean_probability[i])}) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("dataset", choices=("trivia", "gsm8k", "drop"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args.dataset, args.batch, args.resume, args.limit)
    if args.stage in ("evaluate", "all"):
        evaluate(args.dataset)


if __name__ == "__main__":
    main()
