#!/usr/bin/env python3
"""Collect the exact feature schema used by the Scientist current127 detector on TennisQA."""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def disjoint_spans(att, prep):
    words = list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b", prep.item.context, flags=re.UNICODE))
    enc = att.tok(prep.item.context, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    if list(ids) != prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist():
        raise RuntimeError("token offset mismatch")
    spans, chars = [], []
    for wi in range(0, len(words), 2):
        a, b = words[wi].start(), words[min(wi + 1, len(words) - 1)].end()
        covered = [j for j, (x, y) in enumerate(offsets) if y > a and x < b]
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
    pred, other = att.class_scores_batched(prep, alphas)
    return pred.numpy(), other.numpy()


def selected_hidden(att, prep, ids):
    import torch
    zero = torch.zeros(prep.prompt_ids.shape[0], device=att.device)
    alphas = torch.stack([zero, *[att.alpha_from_spans(prep, [int(i)]) for i in ids]])
    collected = [[], []]
    layer14 = None
    for start in range(0, len(alphas), att.max_rows):
        alpha = alphas[start:start + att.max_rows]
        pe = att._embeds(prep, alpha)
        for ci, ans in enumerate((prep.pred_variant_ids[0], prep.gold_variant_ids[0])):
            ae = att.emb_layer(ans).detach().unsqueeze(0).expand(len(alpha), -1, -1)
            seq = torch.cat([pe, ae.to(pe.dtype)], 1)
            mask = torch.ones(seq.shape[:2], dtype=torch.long, device=att.device)
            with torch.inference_mode():
                out = att.model(inputs_embeds=seq, attention_mask=mask,
                                output_hidden_states=True, use_cache=False)
            pos = pe.shape[1] + len(ans) - 1
            collected[ci].append(out.hidden_states[16][:, pos].float().cpu())
            if ci == 0 and start == 0:
                layer14 = out.hidden_states[14][0, pos].float().cpu().numpy()
            del out, seq
        del pe
    return (torch.cat(collected[0]).numpy(), torch.cat(collected[1]).numpy(), layer14)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE.parent / "athlete_qa/pilot_v1/primary_questions.jsonl")
    p.add_argument("--results", type=Path, default=HERE.parent / "athlete_qa/pilot_v1/llama_eval/results.jsonl")
    p.add_argument("--out-dir", type=Path, default=RUNS / "134_tennis_current127")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    set_seed(42)
    data = {x["id"]: x for x in map(json.loads, a.data.open())}
    results = [json.loads(x) for x in a.results.open() if x.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    for number, result in enumerate(results[:a.limit or None], 1):
        key = result["id"]
        target = a.out_dir / f"{key}.npz"
        if target.exists() and a.resume:
            continue
        raw = data[key]
        pred = result["generation"].strip()
        if result["name_outcome"] not in {"correct", "wrong"}:
            raise RuntimeError(f"{key}: unmatched generation {pred!r}")
        # Use the canonical candidate string, not incidental generated punctuation.
        pred = raw["correct_answer"] if result["name_correct"] else raw["wrong_answer"]
        other = raw["wrong_answer"] if result["name_correct"] else raw["correct_answer"]
        item = Item.from_dict({"id": key, "prompt": raw["prepend_names_prompt"],
                               "rgt_ans": other, "pred": pred})
        item.pred, item.gold = pred, other
        prep = att.prepare(item)

        # Scientist hidden block: overlapping 2/3-word spans, top-5 by margin effect.
        overlap = att.build_word_spans(prep, widths=(2, 3), stride=1)
        po, oo = scan(att, prep, overlap)
        margin_u = (po[0] - po[1:]) - (oo[0] - oo[1:])
        overlap_ids = np.argsort(-np.abs(margin_u))[:min(5, len(margin_u))]
        pred_h, other_h, layer14 = selected_hidden(att, prep, overlap_ids)
        pred_u = po[0] - po[1:][overlap_ids]
        other_u = oo[0] - oo[1:][overlap_ids]

        # Scientist scalar block: disjoint spans, physical top-1 deletion, rerank.
        stage1, chars = disjoint_spans(att, prep)
        p1, o1 = scan(att, prep, stage1)
        u1 = (p1[0] - p1[1:]) - (o1[0] - o1[1:])
        top1 = int(np.argmax(np.abs(u1)))
        ids1 = np.argsort(-np.abs(u1))[:min(5, len(u1))]
        ca, cb = chars[top1]
        deleted = re.sub(r"[ \t]+", " ", item.context[:ca] + item.context[cb:])
        deleted = re.sub(r"\s+([,.;:!?])", r"\1", deleted).strip()
        item2 = Item(key + "_deleted", deleted, item.question, other, pred)
        prep2 = att.prepare(item2)
        stage2, _ = disjoint_spans(att, prep2)
        p2, o2 = scan(att, prep2, stage2)
        u2 = (p2[0] - p2[1:]) - (o2[0] - o2[1:])
        ids2 = np.argsort(-np.abs(u2))[:min(5, len(u2))]

        np.savez_compressed(
            target, key=np.asarray(key), correct=np.asarray(int(result["name_correct"])),
            probe_state=np.asarray(result["probe_state"]), pred=np.asarray(pred), other=np.asarray(other),
            overlap_text=np.asarray([overlap[i].text for i in overlap_ids]),
            deleted_text=np.asarray(stage1[top1].text),
            stage1_pred=np.r_[p1[0], p1[1:][ids1]], stage1_other=np.r_[o1[0], o1[1:][ids1]],
            stage2_pred=np.r_[p2[0], p2[1:][ids2]], stage2_other=np.r_[o2[0], o2[1:][ids2]],
            pred_u=pred_u.astype(np.float32), other_u=other_u.astype(np.float32),
            pred_hidden=pred_h.astype(np.float16), other_hidden=other_h.astype(np.float16),
            layer14=layer14.astype(np.float16))
        print(f"[{number}/{len(results)}] {key} correct={int(result['name_correct'])} probe={result['probe_state']}", flush=True)


if __name__ == "__main__":
    main()
