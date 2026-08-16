#!/usr/bin/env python3
"""Collect true candidate-specific multi-layer hidden states at alpha 0 and top-5 endpoints."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def hidden_for(att, prep, alphas, answer_ids, layers):
    import torch
    chunks = []
    for start in range(0, len(alphas), att.max_rows):
        alpha = alphas[start:start + att.max_rows]
        pe = att._embeds(prep, alpha)
        batch = pe.shape[0]
        ae = att.emb_layer(answer_ids).detach().unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([pe, ae.to(pe.dtype)], 1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=att.device)
        with torch.inference_mode():
            out = att.model(inputs_embeds=seq, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
        chunks.append(torch.stack([
            out.hidden_states[layer][:, pe.shape[1] + len(answer_ids) - 1]
            for layer in layers], 1).float().cpu())
        del out, seq, pe
    return torch.cat(chunks).numpy()


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=RUNS/"88_known_gt05_n1084.jsonl")
    p.add_argument("--oracle", type=Path, default=RUNS/"88_oracle_top11_known_gt05.jsonl")
    p.add_argument("--data", type=Path, default=HERE.parent/"shuffled_prepend_names_question.json")
    p.add_argument("--records", type=Path, default=HERE.parent/"tool_gate_correctness_names_llama31_8b"/"records.jsonl")
    p.add_argument("--out-dir", type=Path, default=RUNS/"118_dual_candidate_multilayer_top5")
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--layers", type=int, nargs="+", default=[10, 14, 18, 22, 26, 30])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); set_seed(a.seed)
    source = [json.loads(x) for x in a.source.open() if x.strip()]
    oracle = {x["key"]: x for x in map(json.loads, a.oracle.open())}
    data = {str(x["key"]): x for x in json.load(a.data.open())}
    records = {x["key"]: x for x in map(json.loads, a.records.open())}
    scores = {}
    for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            scores[str(z["key"].item())] = (z["pred_u"].astype(np.float32),
                                              z["other_u"].astype(np.float32))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean",
                         length_norm=True, max_rows=a.batch)
    todo = source[:a.limit or None]
    for number, src in enumerate(todo, 1):
        key = src["key"]; target = a.out_dir/f"{key}.npz"
        if target.exists() and a.resume:
            continue
        raw, record = data[key], records[key]
        pred = str(record["parsed_answer"]); right = str(raw["rgt_ans"]); wrong = str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other)); item.pred, item.gold = pred, other
        prep = att.prepare(item)
        spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        old_u = np.asarray(oracle[key]["u"], np.float32)
        top_ids = np.argsort(-np.abs(old_u))[:a.topk]
        zero = torch.zeros(prep.prompt_ids.shape[0], device=a.device)
        alphas = torch.stack([zero, *[att.alpha_from_spans(prep, [int(i)]) for i in top_ids]])
        pred_h = hidden_for(att, prep, alphas, prep.pred_variant_ids[0], a.layers)
        other_h = hidden_for(att, prep, alphas, prep.gold_variant_ids[0], a.layers)
        pred_u, other_u = scores[key]
        np.savez_compressed(target, key=np.asarray(key), top_ids=top_ids,
                            layers=np.asarray(a.layers, np.int16),
                            pred_u=pred_u, other_u=other_u,
                            pred_hidden=pred_h.astype(np.float16),
                            other_hidden=other_h.astype(np.float16))
        print(f"[{number}/{len(todo)}] {key}", flush=True)


if __name__ == "__main__":
    main()
