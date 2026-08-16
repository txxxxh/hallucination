#!/usr/bin/env python3
"""Layerwise residual-stream perturbation on the Scientist-known pool.

For every existing non-overlapping two-word context span, replace its decoder
block output by the mean output of the other context tokens in the same row.
Token count and positions are unchanged.  The prediction and alternative
answer teacher-forced scores are recorded at several intervention layers.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_CACHE = RUNS / "170_scientist_hidden_intervention"
scientist = importlib.import_module("152_scientist_attention_pruned_current127")
span_helpers = importlib.import_module("125_collect_current_three_benchmarks")


def decoder_layers(model):
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError("expected a Hugging Face Llama/Mistral-style model.model.layers")
    return layers


def replace_span_hook(ctx_start, ctx_end, spans, row_span_ids):
    """Return a hook that patches each non-control row at one decoder layer."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for row, span_id in enumerate(row_span_ids):
            if span_id < 0:  # the first row is the unperturbed control
                continue
            span = spans[span_id]
            keep = torch.ones(ctx_end - ctx_start, dtype=torch.bool,
                              device=hidden.device)
            keep[span.start - ctx_start:span.end - ctx_start] = False
            source = hidden[row, ctx_start:ctx_end][keep]
            if source.numel() == 0:
                continue
            baseline = source.mean(0)
            patched[row, span.start:span.end] = baseline.to(patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched
    return hook


def variant_logprob(att, prep, answer_ids, layer, span_ids, batch):
    values = []
    pe0 = prep.E
    answer_emb = att.emb_layer(answer_ids).detach()
    module = decoder_layers(att.model)[layer]
    for start in range(0, len(span_ids), batch):
        selected = span_ids[start:start + batch]
        size = len(selected)
        pe = pe0.unsqueeze(0).expand(size, -1, -1)
        ae = answer_emb.unsqueeze(0).expand(size, -1, -1)
        seq = torch.cat([pe, ae.to(pe.dtype)], 1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)
        handle = module.register_forward_hook(replace_span_hook(
            prep.ctx_start, prep.ctx_end, prep.spans, selected))
        try:
            with torch.inference_mode():
                logits = att.model(inputs_embeds=seq, attention_mask=mask,
                                   logits_to_keep=len(answer_ids) + 1,
                                   use_cache=False).logits
            logits = logits[:, :len(answer_ids)].float()
            targets = answer_ids.unsqueeze(0).expand(size, -1)
            token_lp = torch.log_softmax(logits, -1).gather(
                -1, targets.unsqueeze(-1)).squeeze(-1)
            score = token_lp.mean(-1) if att.length_norm else token_lp.sum(-1)
            values.append(score.cpu())
        finally:
            handle.remove()
        del seq, logits
    return torch.cat(values).numpy()


def class_scores(att, prep, variants, layer, span_ids, batch):
    parts = [variant_logprob(att, prep, answer, layer, span_ids, batch)
             for answer in variants]
    return torch.logsumexp(torch.from_numpy(np.stack(parts)), 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 28],
                        help="zero-based decoder block indices")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100,
                        help="0 means all 1084 Scientist-known rows")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    rows = scientist.jobs()
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loader = importlib.import_module("61_grad_span_proposal")
    model, tokenizer = loader.load_model(args.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)
    total_layers = len(decoder_layers(model))
    if any(layer < 0 or layer >= total_layers for layer in args.layers):
        raise ValueError(f"layers must be in [0, {total_layers - 1}]")
    config = {
        "dataset": "Scientist-known", "n": len(rows), "layers_zero_based": args.layers,
        "span_definition": "current127 non-overlapping two-word context blocks",
        "intervention": "replace target block at decoder output with same-row mean of other context tokens",
        "control_row": "unperturbed", "length_normalized_teacher_forcing": True,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for number, (key, group, label, prompt, pred, other) in enumerate(rows, 1):
        target = args.out_dir / f"{key}.npz"
        if args.resume and target.exists():
            continue
        item = Item.from_dict({"key": key, "prompt": prompt,
                               "pred": pred, "gold": other})
        prep = att.prepare(item)
        spans, _ = span_helpers.spans(att, prep)
        if not spans:
            raise RuntimeError(f"{key}: no perturbable spans")
        span_ids = [-1, *range(len(spans))]
        pred_scores, other_scores = [], []
        for layer in args.layers:
            pred_scores.append(class_scores(att, prep, prep.pred_variant_ids,
                                            layer, span_ids, args.batch))
            other_scores.append(class_scores(att, prep, prep.gold_variant_ids,
                                             layer, span_ids, args.batch))
        np.savez_compressed(
            target, key=np.asarray(key), group=np.asarray(group),
            correct=np.asarray(label), layers=np.asarray(args.layers),
            pred_scores=np.asarray(pred_scores, dtype=np.float32),
            other_scores=np.asarray(other_scores, dtype=np.float32),
            span_start=np.asarray([x.start for x in spans], dtype=np.int32),
            span_end=np.asarray([x.end for x in spans], dtype=np.int32),
            span_text=np.asarray([x.text for x in spans]),
        )
        print(f"[{number}/{len(rows)}] {key} spans={len(spans)} layers={args.layers}",
              flush=True)


if __name__ == "__main__":
    main()
