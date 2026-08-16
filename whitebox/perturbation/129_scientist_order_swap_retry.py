#!/usr/bin/env python3
"""Oracle diagnostic: retry Scientist detector mistakes after swapping option order.

The original dataset and caches are never modified.  ``prepare`` reconstructs
the current 3x5-fold OOF predictions, defines mistakes from the three-seed mean,
and writes a full copied dataset in which only those rows have option 1/2 swapped.
``collect`` recomputes the current 127-dimensional inputs for those rows.
``evaluate`` trains every fold on original-order features and substitutes swapped
features only for selected OOF test rows.

This is intentionally an oracle upper-bound experiment: selection of rows to
swap uses their labels.  It is not a deployable, leakage-free detector score.
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

from spanattr.core import Item, Span, SpanAttributor, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
SOURCE = RUNS / "88_known_gt05_n1084.jsonl"
DATA = ROOT / "shuffled_prepend_names_question.json"
RECORDS = ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl"
SWAP_DATA = RUNS / "129_scientist_mistakes_option_swapped.json"
MANIFEST = RUNS / "129_scientist_order_swap_manifest.json"
SWAP_CACHE = RUNS / "129_scientist_order_swap_current127"
REPORT = RUNS / "129_scientist_order_swap_retry_report.json"


def ch(s):
    u = s[0] - s[1:]
    z = abs(float(s[0])) + 1e-6
    return np.r_[s[0], u, u / z, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(s):
    return np.r_[s[0], s[0] - s[1:]]


def wd(h, u):
    d = h[1:].astype(np.float32) - h[0].astype(np.float32)
    return (d * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)


def metrics(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
        "errors_at_0.5": int(np.sum((p >= .5) != y)),
    }


def load_original():
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, groups, y, _, _, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    dual, new = {}, {}
    for fp in (RUNS / "116_dual_candidate_hidden_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            dual[key] = (ph[0], wd(ph, z["pred_u"].astype(np.float32)),
                         oh[0], wd(oh, z["other_u"].astype(np.float32)))
    for fp in (RUNS / "120_physical_delete_rerank").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            p, o = z["stage1_pred_scores"].astype(np.float32), z["stage1_other_scores"].astype(np.float32)
            q, r = z["stage2_pred_scores"].astype(np.float32), z["stage2_other_scores"].astype(np.float32)
            new[key] = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0]-q[0], o[0]-r[0],
                             (p[0]-o[0])-(q[0]-r[0])]
    if not all(k in dual and k in new for k in keys):
        raise RuntimeError("original feature cache is incomplete")
    scalar = np.stack([new[k] for k in keys])
    hidden = [np.stack([dual[k][j] for k in keys]) for j in range(4)]
    return keys, groups, y.astype(int), scalar, hidden, last[:, 3]


def fit_predict(keys, groups, y, scalar, hidden, layer14, swapped=None, selected=None):
    all_pred = []
    for seed in (42, 43, 44):
        pred = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for tr, te in cv.split(scalar, y, groups):
            st = StandardScaler().fit(scalar[tr])
            train_parts, test_parts = [st.transform(scalar[tr])], [st.transform(scalar[te])]
            if swapped is not None:
                use = np.array([keys[i] in selected for i in te])
                if use.any():
                    test_parts[0][use] = st.transform(np.stack([swapped[keys[i]][0] for i in te[use]]))
            for block_id, (block, dim) in enumerate([*((x, 8) for x in hidden), (layer14, 48)]):
                sc = StandardScaler().fit(block[tr]); z = sc.transform(block[tr])
                pc = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(z)
                train_parts.append(pc.transform(z)); tv = pc.transform(sc.transform(block[te]))
                if swapped is not None:
                    use = np.array([keys[i] in selected for i in te])
                    if use.any():
                        repl = np.stack([swapped[keys[i]][1 + block_id] for i in te[use]])
                        tv[use] = pc.transform(sc.transform(repl))
                test_parts.append(tv)
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed)
            clf.fit(np.concatenate(train_parts, 1), y[tr])
            pred[te] = clf.predict_proba(np.concatenate(test_parts, 1))[:, 1]
        all_pred.append(pred)
    return np.stack(all_pred)


def swap_option_order(prompt):
    pattern = re.compile(
        r"^(Choose one of the following two options as the answer to the question below:\n)"
        r"1\. ([^\n]+)\n2\. ([^\n]+)(\nQuestion:\n[\s\S]*)$")
    m = pattern.match(prompt)
    if not m:
        raise ValueError("unexpected Scientist option prompt format")
    return f"{m.group(1)}1. {m.group(3)}\n2. {m.group(2)}{m.group(4)}"


def prepare():
    keys, groups, y, scalar, hidden, layer14 = load_original()
    pred = fit_predict(keys, groups, y, scalar, hidden, layer14)
    mean_pred = pred.mean(0)
    selected = set(keys[(mean_pred >= .5) != y].tolist())
    raw = json.load(DATA.open())
    copied = []
    for row in raw:
        out = dict(row)
        if str(row["key"]) in selected:
            out["prompt"] = swap_option_order(str(row["prompt"]))
        copied.append(out)
    SWAP_DATA.write_text(json.dumps(copied, ensure_ascii=False, indent=2))
    rows = []
    for i, key in enumerate(keys):
        rows.append({"key": str(key), "group": str(groups[i]), "correct": int(y[i]),
                     "selected_for_swap": str(key) in selected,
                     "oof_probabilities": pred[:, i].tolist(), "oof_mean": float(mean_pred[i])})
    report = {"protocol": "oracle retry selection from mean of 3 grouped-OOF predictions",
              "n": len(keys), "groups": len(set(groups)), "selected_mistakes": len(selected),
              "original_mean_probability_metrics": metrics(y, mean_pred), "rows": rows}
    MANIFEST.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"wrote {SWAP_DATA}\nwrote {MANIFEST}")


def word_spans(att, prep, widths=(2, 3), stride=1):
    return att.build_word_spans(prep, widths=widths, stride=stride)


def disjoint_spans(att, prep):
    words = list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b", prep.item.context, flags=re.UNICODE))
    enc = att.tok(prep.item.context, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    if ids and isinstance(ids[0], list): ids = ids[0]
    if off and isinstance(off[0], list) and off[0] and isinstance(off[0][0], list): off = off[0]
    if list(ids) != prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist():
        raise RuntimeError("token offset mismatch")
    spans, chars = [], []
    for wi in range(0, len(words), 2):
        a, b = words[wi].start(), words[min(wi + 1, len(words) - 1)].end()
        covered = [ti for ti, (x, z) in enumerate(off) if z > a and x < b]
        if covered:
            spans.append(Span(len(spans), prep.ctx_start + covered[0],
                              prep.ctx_start + covered[-1] + 1, prep.item.context[a:b]))
            chars.append((a, b))
    prep.spans = spans
    return spans, chars


def scan(att, prep, spans):
    import torch
    zero = torch.zeros(prep.prompt_ids.shape[0], device=att.device)
    alphas = torch.stack([zero, *[att.alpha_from_spans(prep, [i]) for i in range(len(spans))]])
    p, o = att.class_scores_batched(prep, alphas)
    return p.numpy(), o.numpy()


def selected_scores(att, prep, ids):
    import torch
    zero = torch.zeros(prep.prompt_ids.shape[0], device=att.device)
    alphas = torch.stack([zero, *[att.alpha_from_spans(prep, [int(i)]) for i in ids]])
    p, o = att.class_scores_batched(prep, alphas)
    return p.numpy(), o.numpy()


def hidden(att, prep, spans, ids):
    import torch
    zero = torch.zeros(prep.prompt_ids.shape[0], device=att.device)
    alphas = torch.stack([zero, *[att.alpha_from_spans(prep, [int(i)]) for i in ids]])
    outputs = [[], []]
    layer14 = None
    for start in range(0, len(alphas), att.max_rows):
        a = alphas[start:start + att.max_rows]; pe = att._embeds(prep, a)
        for ci, ans in enumerate((prep.pred_variant_ids[0], prep.gold_variant_ids[0])):
            ae = att.emb_layer(ans).detach().unsqueeze(0).expand(len(a), -1, -1)
            seq = torch.cat([pe, ae.to(pe.dtype)], 1)
            mask = torch.ones(seq.shape[:2], dtype=torch.long, device=att.device)
            with torch.inference_mode():
                out = att.model(inputs_embeds=seq, attention_mask=mask,
                                output_hidden_states=True, use_cache=False)
            outputs[ci].append(out.hidden_states[16][:, pe.shape[1] + len(ans) - 1].float().cpu())
            if ci == 0 and start == 0:
                layer14 = out.hidden_states[14][0, pe.shape[1] + len(ans) - 1].float().cpu().numpy()
            del out, seq
        del pe
    return torch.cat(outputs[0]).numpy(), torch.cat(outputs[1]).numpy(), layer14


def collect(args):
    set_seed(42)
    manifest = json.load(MANIFEST.open())
    todo = [x for x in manifest["rows"] if x["selected_for_swap"]]
    data = {str(x["key"]): x for x in json.load(SWAP_DATA.open())}
    records = {x["key"]: x for x in map(json.loads, RECORDS.open())}
    oracle = {x["key"]: x for x in map(json.loads, (RUNS / "88_oracle_top11_known_gt05.jsonl").open())}
    SWAP_CACHE.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        args.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean", length_norm=True,
                         max_rows=args.batch)
    for number, src in enumerate(todo, 1):
        key = src["key"]; target = SWAP_CACHE / f"{key}.npz"
        if target.exists() and args.resume:
            continue
        raw, rec = data[key], records[key]
        pred, right, wrong = str(rec["parsed_answer"]), str(raw["rgt_ans"]), str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other)); item.pred, item.gold = pred, other
        prep = att.prepare(item)
        # Current physical-deletion scalar branch: disjoint two-word spans.
        ds, chars = disjoint_spans(att, prep); p1, o1 = scan(att, prep, ds)
        u = (p1[0]-p1[1:])-(o1[0]-o1[1:]); top = int(np.argmax(np.abs(u)))
        ids1 = np.argsort(-np.abs(u))[:min(5, len(u))]
        ca, cb = chars[top]
        deleted = re.sub(r"[ \t]+", " ", item.context[:ca] + item.context[cb:])
        deleted = re.sub(r"\s+([,.;:!?])", r"\1", deleted).strip()
        raw2 = dict(raw); raw2["prompt"] = deleted
        item2 = Item.from_dict(dict(raw2, pred=pred, gold=other)); item2.pred, item2.gold = pred, other
        prep2 = att.prepare(item2); ds2, _ = disjoint_spans(att, prep2); p2, o2 = scan(att, prep2, ds2)
        u2 = (p2[0]-p2[1:])-(o2[0]-o2[1:]); ids2 = np.argsort(-np.abs(u2))[:min(5, len(u2))]
        scalar = np.r_[ch(np.r_[p1[0], p1[1:][ids1]]), ch(np.r_[o1[0], o1[1:][ids1]]),
                       ch2(np.r_[p2[0], p2[1:][ids2]]), ch2(np.r_[o2[0], o2[1:][ids2]]),
                       p1[0]-p2[0], o1[0]-o2[0], (p1[0]-o1[0])-(p2[0]-o2[0])]
        # Match the original hidden branch: overlapping spans and frozen oracle top ids.
        os = word_spans(att, prep)
        top_ids = np.argsort(-np.abs(np.asarray(oracle[key]["u"])))[:5]
        op, oo = selected_scores(att, prep, top_ids)
        ph, oh, layer14 = hidden(att, prep, os, top_ids)
        pred_u = op[0] - op[1:]; other_u = oo[0] - oo[1:]
        np.savez_compressed(target, key=np.asarray(key), scalar=scalar,
                            h0_pred=ph[0].astype(np.float16), hwd_pred=wd(ph, pred_u).astype(np.float16),
                            h0_other=oh[0].astype(np.float16), hwd_other=wd(oh, other_u).astype(np.float16),
                            layer14=layer14.astype(np.float16))
        print(f"[{number}/{len(todo)}] {key}", flush=True)


def evaluate():
    keys, groups, y, scalar, hidden, layer14 = load_original()
    manifest = json.load(MANIFEST.open()); selected = {x["key"] for x in manifest["rows"] if x["selected_for_swap"]}
    swapped = {}
    for fp in SWAP_CACHE.glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            swapped[str(z["key"].item())] = (z["scalar"].astype(np.float32), z["h0_pred"].astype(np.float32),
                                               z["hwd_pred"].astype(np.float32), z["h0_other"].astype(np.float32),
                                               z["hwd_other"].astype(np.float32), z["layer14"].astype(np.float32))
    missing = selected - set(swapped)
    if missing: raise RuntimeError(f"missing {len(missing)} swapped caches")
    original = fit_predict(keys, groups, y, scalar, hidden, layer14)
    retry = fit_predict(keys, groups, y, scalar, hidden, layer14, swapped, selected)
    result = {"warning": "oracle diagnostic: labels selected which rows receive order swap",
              "selection": "mistakes from mean of three original grouped-OOF probabilities at threshold 0.5",
              "selected_rows": len(selected), "n": len(y),
              "original_per_seed": [metrics(y, p) for p in original],
              "retry_per_seed": [metrics(y, p) for p in retry],
              "original_mean": metrics(y, original.mean(0)),
              "retry_mean": metrics(y, retry.mean(0))}
    REPORT.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["prepare", "collect", "evaluate", "all"])
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=24); p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.stage in ("prepare", "all"): prepare()
    if args.stage in ("collect", "all"): collect(args)
    if args.stage in ("evaluate", "all"): evaluate()


if __name__ == "__main__":
    main()
