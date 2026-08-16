#!/usr/bin/env python3
"""Collect current127 features and evaluate a Scientist-fitted frozen detector."""
from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

from spanattr.core import Item, SpanAttributor, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
DATA = ROOT / "athlete_qa" / "multidomain_v5"
EVAL = DATA / "llama_eval" / "results.jsonl"
CACHE = DATA / "current127_known_both"
REPORT = DATA / "scientist_frozen_transfer.json"
PREDS = DATA / "scientist_frozen_transfer_predictions.jsonl"
DOMAINS = ("athlete", "musician", "building")


def rows():
    questions = {}
    for domain in DOMAINS:
        path = DATA / domain / "primary_questions.jsonl"
        questions.update({x["id"]: x for x in map(json.loads, path.open())})
    results = {x["id"]: x for x in map(json.loads, EVAL.open())}
    output = []
    for key, question in questions.items():
        result = results[key]
        if result["probe_state"] != "knows_both" or result["name_outcome"] == "unmatched":
            continue
        correct = result["name_outcome"] == "correct"
        pred = question["correct_answer"] if correct else question["wrong_answer"]
        other = question["wrong_answer"] if correct else question["correct_answer"]
        output.append({"key": key, "domain": question["domain"],
                       "field": question["decisive_relation"]["field"],
                       "correct": int(correct), "prompt": question["prepend_names_prompt"],
                       "pred": pred, "other": other})
    return output


def fixed(scores, n=6):
    scores = np.asarray(scores, np.float32)
    return np.pad(scores[:n], (0, max(0, n - len(scores))))


def ch(scores):
    scores = fixed(scores)
    u = scores[0] - scores[1:]
    scale = abs(float(scores[0])) + 1e-6
    return np.r_[scores[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(scores):
    scores = fixed(scores)
    return np.r_[scores[0], scores[0] - scores[1:]]


def wd(hidden, u):
    hidden, u = np.asarray(hidden, np.float32), np.asarray(u, np.float32)
    delta = hidden[1:] - hidden[0]
    return (delta[:len(u)] * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)


def unpack(p, o, q, r, ph, oh, layer):
    scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0]-q[0], o[0]-r[0],
                   (p[0]-o[0])-(q[0]-r[0])]
    hidden = [ph[0], wd(ph, p[0]-p[1:]), oh[0], wd(oh, o[0]-o[1:])]
    return scalar, hidden, layer


def collect(args):
    mod = importlib.import_module("125_collect_current_three_benchmarks")
    CACHE.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    loader = importlib.import_module("61_grad_span_proposal")
    model, tok = loader.load_model(args.model, "bfloat16", "cuda")
    attributor = SpanAttributor(model, tok, device="cuda", baseline="mean",
                               length_norm=True, max_rows=args.batch)
    items = rows()
    for number, row in enumerate(items, 1):
        path = CACHE / f"{row['key']}.npz"
        if path.exists() and args.resume:
            continue
        item = Item.from_dict({"key": row["key"], "prompt": row["prompt"],
                               "pred": row["pred"], "gold": row["other"]})
        prepared = attributor.prepare(item)
        spans, chars = mod.spans(attributor, prepared)
        p, o = mod.scan(attributor, prepared, spans)
        u = (p[0]-p[1:])-(o[0]-o[1:])
        top = int(np.argmax(np.abs(u)))
        ids = np.argsort(-np.abs(u))[:min(5, len(u))]
        ph, oh, layer = mod.selected_hidden(attributor, prepared, ids)
        ca, cb = chars[top]
        deleted = re.sub(r"[ \t]+", " ", item.context[:ca] + item.context[cb:])
        deleted = re.sub(r"\s+([,.;:!?])", r"\1", deleted).strip()
        second_item = Item(row["key"] + "_d", deleted, item.question, row["other"],
                           row["pred"], context_prefix=item.context_prefix)
        second = attributor.prepare(second_item)
        spans2, _ = mod.spans(attributor, second)
        q, r = mod.scan(attributor, second, spans2)
        u2 = (q[0]-q[1:])-(r[0]-r[1:])
        ids2 = np.argsort(-np.abs(u2))[:min(5, len(u2))]
        np.savez_compressed(path, key=np.asarray(row["key"]),
                            stage1_pred=np.r_[p[0], p[1:][ids]],
                            stage1_other=np.r_[o[0], o[1:][ids]],
                            stage2_pred=np.r_[q[0], q[1:][ids2]],
                            stage2_other=np.r_[r[0], r[1:][ids2]],
                            pred_hidden=ph.astype(np.float16), other_hidden=oh.astype(np.float16),
                            layer14=layer.astype(np.float16))
        print(f"[{number}/{len(items)}] {row['key']}", flush=True)


def scientist_source():
    source = []
    known = [json.loads(x) for x in (RUNS / "88_known_gt05_n1084.jsonl").open() if x.strip()]
    for row in known:
        key = row["key"]
        with np.load(RUNS / "120_physical_delete_rerank" / f"{key}.npz", allow_pickle=True) as z:
            p, o, q, r = (z["stage1_pred_scores"], z["stage1_other_scores"],
                          z["stage2_pred_scores"], z["stage2_other_scores"])
        with np.load(RUNS / "116_dual_candidate_hidden_top5" / f"{key}.npz", allow_pickle=True) as z:
            ph, oh = z["pred_hidden"], z["other_hidden"]
        with np.load(RUNS / "100_scientist_trajectory_l8" / f"{key}.npz", allow_pickle=True) as z:
            layer = z["mean"].astype(np.float32)[3]
        scalar, hidden, layer = unpack(p, o, q, r, ph, oh, layer)
        source.append((int(row["correct"]), scalar, hidden, layer))
    return source


def metric(y, p):
    pred = p >= .5
    result = {"n": int(len(y)), "correct": int(y.sum()), "incorrect": int(len(y)-y.sum()),
              "accuracy_at_0.5": float(accuracy_score(y, pred)),
              "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
              "confusion_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()}
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    return result


def evaluate():
    source = scientist_source()
    target = []
    for row in rows():
        with np.load(CACHE / f"{row['key']}.npz", allow_pickle=True) as z:
            scalar, hidden, layer = unpack(z["stage1_pred"], z["stage1_other"],
                                           z["stage2_pred"], z["stage2_other"],
                                           z["pred_hidden"], z["other_hidden"], z["layer14"])
        target.append((row, scalar, hidden, layer))
    sy = np.array([x[0] for x in source])
    ty = np.array([x[0]["correct"] for x in target])
    seed_probs = []
    for seed in (42, 43, 44):
        source_parts, target_parts = [], []
        pairs = [(np.stack([x[1] for x in source]), np.stack([x[1] for x in target]), None)]
        pairs += [(np.stack([x[2][j] for x in source]), np.stack([x[2][j] for x in target]), 8) for j in range(4)]
        pairs += [(np.stack([x[3] for x in source]), np.stack([x[3] for x in target]), 48)]
        for src, tgt, dim in pairs:
            scaler = StandardScaler().fit(src)
            a, b = scaler.transform(src), scaler.transform(tgt)
            if dim is not None:
                pca = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
                a, b = pca.transform(a), pca.transform(b)
            source_parts.append(a)
            target_parts.append(b)
        clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=seed)
        clf.fit(np.concatenate(source_parts, 1), sy)
        seed_probs.append(clf.predict_proba(np.concatenate(target_parts, 1))[:, 1])
    probs = np.mean(seed_probs, axis=0)
    masks = {"all_probe_known_both": np.ones(len(target), bool)}
    for domain in DOMAINS:
        masks[domain] = np.array([x[0]["domain"] == domain for x in target])
    for field in sorted({x[0]["field"] for x in target}):
        masks[f"field:{field}"] = np.array([x[0]["field"] == field for x in target])
    report = {"protocol": "zero-shot frozen transfer; all scaler/PCA/LR fitting uses only 1084 Scientist-known rows; target restricted by independent probes to knows_both; no target labels used for fitting/tuning",
              "source_detector": "current127 scalar47 + four candidate-hidden PCA8 + layer14 PCA48; LR C=.03",
              "source_n": len(source), "target_n": len(target), "excluded_unmatched_known_both": 3,
              "ensemble_seeds": [42, 43, 44], "subsets": {}}
    for name, mask in masks.items():
        if not mask.any():
            continue
        report["subsets"][name] = metric(ty[mask], probs[mask])
        if len(np.unique(ty[mask])) == 2:
            report["subsets"][name]["per_seed_auroc"] = [float(roc_auc_score(ty[mask], p[mask])) for p in seed_probs]
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with PREDS.open("w") as handle:
        for (row, _, _, _), score in zip(target, probs):
            handle.write(json.dumps({"id": row["key"], "domain": row["domain"], "field": row["field"],
                                     "correct": bool(row["correct"]), "prob_correct": float(score)}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
