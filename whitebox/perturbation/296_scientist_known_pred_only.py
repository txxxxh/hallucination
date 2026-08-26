#!/usr/bin/env python3
"""Strict pred-only perturbation detector on Scientist-known 1,084.

Controlled against exact-current127: same rows, grouped 3x5 OOF, seeds,
scaler/PCA, LR and two-stage physical deletion.  The sole methodological
change is that span ranking, likelihood features and hidden states use only
the generated answer; no competing-answer score or state is computed.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CACHE = RUNS / "296_scientist_known_pred_only"
REPORT = RUNS / "296_scientist_known_pred_only_report.json"


def jobs():
    return importlib.import_module("152_scientist_attention_pruned_current127").jobs()


def score(att, prep, spans):
    import torch
    zero = torch.zeros(len(prep.prompt_ids), device=att.device)
    alpha = torch.stack([zero, *[att.alpha_from_spans(prep, [i])
                                 for i in range(len(spans))]])
    pred, _ = att.class_scores_batched(prep, alpha)
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
            out = att.model(inputs_embeds=seq, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
        pos = pe.shape[1] + len(answer) - 1
        chunks.append(out.hidden_states[layer][:, pos].float().cpu())
        if layer14 is None:
            layer14 = out.hidden_states[14][0, pos].float().cpu().numpy()
        del out, seq, pe
    return __import__("torch").cat(chunks).numpy(), layer14


def collect(args):
    set_seed(42); CACHE.mkdir(parents=True, exist_ok=True)
    collector = importlib.import_module("125_collect_current_three_benchmarks")
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        args.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)
    rows = jobs()
    for number, (key, group, label, prompt, pred, _other) in enumerate(rows, 1):
        target = CACHE / f"{key}.npz"
        if args.resume and target.exists():
            continue
        # Setting gold=pred satisfies the shared teacher-forcing container;
        # the second returned score is intentionally discarded everywhere.
        item = Item.from_dict({"key": key, "prompt": prompt,
                               "pred": pred, "gold": pred})
        prep = att.prepare(item)
        spans, chars = collector.spans(att, prep)
        p1_all = score(att, prep, spans)
        effect1 = p1_all[0] - p1_all[1:]
        ids1 = np.argsort(-np.abs(effect1))[:min(5, len(effect1))]
        hidden, layer14 = pred_hidden(att, prep, ids1)
        top = int(ids1[0]); ca, cb = chars[top]
        deleted = re.sub(r"[ \t]+", " ", item.context[:ca] + item.context[cb:])
        deleted = re.sub(r"\s+([,.;:!?])", r"\1", deleted).strip()
        item2 = Item(key + "_d", deleted, item.question, pred, pred,
                     context_prefix=item.context_prefix)
        prep2 = att.prepare(item2)
        spans2, _ = collector.spans(att, prep2)
        p2_all = score(att, prep2, spans2)
        effect2 = p2_all[0] - p2_all[1:]
        ids2 = np.argsort(-np.abs(effect2))[:min(5, len(effect2))]
        np.savez_compressed(
            target, key=np.asarray(key), group=np.asarray(group),
            correct=np.asarray(label), deleted_text=np.asarray(spans[top].text),
            stage1_pred=np.r_[p1_all[0], p1_all[1:][ids1]],
            stage2_pred=np.r_[p2_all[0], p2_all[1:][ids2]],
            pred_hidden=hidden.astype(np.float16), layer14=layer14.astype(np.float16),
            stage1_full=np.asarray(len(spans)), stage2_full=np.asarray(len(spans2)))
        print(f"[{number}/{len(rows)}] {key} spans={len(spans)}+{len(spans2)}", flush=True)


def ch(s):
    u = s[0] - s[1:]; scale = abs(float(s[0])) + 1e-6
    return np.r_[s[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(s):
    return np.r_[s[0], s[0] - s[1:]]


def wd(hidden, scores):
    effect = scores[0] - scores[1:]
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * effect[:, None]).sum(0) / (np.abs(effect).sum() + 1e-9)


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "balanced_accuracy": float(balanced_accuracy_score(y, score >= .5))}


def evaluate():
    rows = []
    for path in sorted(CACHE.glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            p, q = z["stage1_pred"].astype(np.float32), z["stage2_pred"].astype(np.float32)
            h = z["pred_hidden"].astype(np.float32)
            scalar = np.r_[ch(p), ch2(q), p[0] - q[0]].astype(np.float32)
            rows.append((str(z["key"].item()), str(z["group"].item()),
                         int(z["correct"]), scalar,
                         (h[0], wd(h, p)), z["layer14"].astype(np.float32)))
    if len(rows) != 1084:
        raise RuntimeError(f"expected 1084 cached rows, got {len(rows)}")
    keys = [x[0] for x in rows]
    groups = np.asarray([x[1] for x in rows]); y = np.asarray([x[2] for x in rows])
    scalar = np.stack([x[3] for x in rows])
    hidden = [np.stack([x[4][j] for x in rows]) for j in range(2)]
    layer14 = np.stack([x[5] for x in rows])
    per_seed, predictions = [], []
    for seed in (42, 43, 44):
        prob = np.zeros(len(y)); cv = StratifiedGroupKFold(5, shuffle=True,
                                                           random_state=seed)
        for fold, (train, test) in enumerate(cv.split(scalar, y, groups), 1):
            train_parts, test_parts = [], []
            for values, dim in [(scalar, None), (hidden[0], 8),
                                (hidden[1], 8), (layer14, 48)]:
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                if dim is not None:
                    pca = PCA(dim, whiten=True, svd_solver="randomized",
                              random_state=seed).fit(a)
                    a, b = pca.transform(a), pca.transform(b)
                train_parts.append(a); test_parts.append(b)
            model = LogisticRegression(C=.03, max_iter=5000,
                class_weight="balanced", solver="liblinear",
                random_state=seed).fit(np.concatenate(train_parts, 1), y[train])
            prob[test] = model.predict_proba(np.concatenate(test_parts, 1))[:, 1]
            print(f"evaluate seed={seed} fold={fold}/5", flush=True)
        per_seed.append({"seed": seed, **metrics(y, prob)})
        predictions.append(prob)
    mean_prob = np.mean(predictions, axis=0)
    report = {
        "protocol": "Scientist-known 1084; strict pred-only exact two-stage perturbation; right-person grouped 3x5 OOF; seeds 42-44; fold-local scaler/PCA; LR C=.03",
        "feature_blocks": "pred scalar [ch(stage1), ch2(stage2), delete delta] + pred hidden original PCA8 + pred weighted displacement PCA8 + layer14 PCA48",
        "explicitly_excluded": ["other scores", "other hidden", "pred-other margin",
                                "contrastive span ranking"],
        "n": len(y), "correct": int(y.sum()), "incorrect": int((1-y).sum()),
        "groups": len(set(groups)), "per_seed": per_seed,
        "mean_per_seed": {k: float(np.mean([r[k] for r in per_seed]))
                          for k in ("auroc", "auprc", "balanced_accuracy")},
        "mean_probability": metrics(y, mean_prob),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    with (CACHE / "predictions.jsonl").open("w") as handle:
        for i, key in enumerate(keys):
            handle.write(json.dumps({"key": key, "correct": int(y[i]),
                "probabilities": [float(x[i]) for x in predictions],
                "mean_probability": float(mean_prob[i])}) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("collect", "evaluate", "all"))
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=16); p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.stage in ("collect", "all"): collect(args)
    if args.stage in ("evaluate", "all"): evaluate()


if __name__ == "__main__":
    main()
