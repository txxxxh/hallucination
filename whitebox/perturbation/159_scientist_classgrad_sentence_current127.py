#!/usr/bin/env python3
"""Current127 with class-separated gate gradients and sentence-aware pruning."""
from __future__ import annotations

import argparse
import importlib
import json
import re

import numpy as np


m = importlib.import_module("152_scientist_attention_pruned_current127")
m.CACHE = m.RUNS / "159_scientist_classgrad_sentence_current127"
m.OUT = m.RUNS / "159_scientist_classgrad_sentence_current127_report.json"


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + 1e-8)


def class_gradients(att, prep):
    """Return d(pred), d(other), and d(pred-other) / d token-neutralization gate."""
    import torch

    alpha = torch.zeros(1, len(prep.prompt_ids), device=att.device,
                        requires_grad=True)
    pred, other = att.class_scores(prep, alpha)
    pred_grad, = torch.autograd.grad(pred.sum(), alpha, retain_graph=True)
    other_grad, = torch.autograd.grad(other.sum(), alpha)
    pred_grad = pred_grad[0].detach().float().cpu().numpy()
    other_grad = other_grad[0].detach().float().cpu().numpy()
    return pred_grad, other_grad, pred_grad - other_grad


def span_scores(spans, gradients):
    """Combine class-specific and contrastive first-order deletion signals."""
    channels = []
    for gradient in gradients:
        # The deletion gain is -sum(d score / d alpha) over the span.  Absolute
        # value is used only after each semantic-class channel stays separate.
        values = np.asarray([
            abs(float(gradient[span.start:span.end].sum())) for span in spans
        ])
        channels.append(_zscore(values))
    return np.sum(channels, axis=0)


def sentence_regions(att, prep):
    """Map punctuation/newline-delimited context sentences to prompt token ranges."""
    context = prep.item.context
    encoded = att.tok(context, add_special_tokens=False,
                      return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if (offsets and isinstance(offsets[0], list) and offsets[0]
            and isinstance(offsets[0][0], list)):
        offsets = offsets[0]
    expected = prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist()
    if list(ids) != expected:
        raise RuntimeError("sentence boundary tokenizer offset mismatch")

    # A semicolon often separates independent facts in the Scientist prompts;
    # newlines are also hard boundaries.  Keep terminal punctuation in-region.
    cuts = [0]
    for match in re.finditer(r"(?:[.!?;]+[\"')\]]*\s+)|(?:\n+)", context):
        cuts.append(match.end())
    cuts.append(len(context))

    regions = []
    for char_start, char_end in zip(cuts, cuts[1:]):
        covered = [i for i, (start, end) in enumerate(offsets)
                   if end > char_start and start < char_end]
        if covered:
            region = (prep.ctx_start + covered[0],
                      prep.ctx_start + covered[-1] + 1)
            if not regions or region != regions[-1]:
                regions.append(region)
    return regions or [(prep.ctx_start, prep.ctx_end)]


def sentence_shortlist(att, prep, spans, saliency_mass=.75,
                       max_candidate_fraction=.60, topk=3):
    """Select salient natural sentences until mass or candidate budget is met."""
    if not spans:
        return []
    scores = span_scores(spans, class_gradients(att, prep))
    # Shift the combined z-score to nonnegative saliency for mass accounting.
    saliency = scores - scores.min() + 1e-8
    regions = sentence_regions(att, prep)
    sentence_scores = []
    sentence_ids = []
    for start, end in regions:
        ids = [i for i, span in enumerate(spans)
               if span.end > start and span.start < end]
        sentence_ids.append(ids)
        values = np.sort(saliency[ids])[-max(1, min(topk, len(ids))):]
        sentence_scores.append(float(values.mean()) if len(values) else 0.0)

    order = np.argsort(-np.asarray(sentence_scores))
    total = float(np.sum(sentence_scores)) + 1e-12
    candidate_cap = max(1, int(np.ceil(len(spans) * max_candidate_fraction)))
    chosen, selected, covered_mass = [], set(), 0.0
    for sentence_id in order:
        proposed = selected.union(sentence_ids[int(sentence_id)])
        if chosen and len(proposed) > candidate_cap:
            continue
        chosen.append(int(sentence_id))
        selected = proposed
        covered_mass += sentence_scores[int(sentence_id)]
        if covered_mass / total >= saliency_mass:
            break
    if not selected:
        selected.update(sentence_ids[int(order[0])])
    return sorted(selected)


def evaluate():
    m.evaluate()
    report = json.loads(m.OUT.read_text())
    report["protocol"] = (
        "Scientist-known 1084, grouped 3x5 OOF, current127 LR unchanged; "
        "class-separated pred/other/margin gate gradients; punctuation-aware "
        "sentence pruning at 0.75 saliency mass and 0.60 candidate cap"
    )
    query_rows = []
    sentence_rows = []
    for path in m.CACHE.glob("*.npz"):
        with np.load(path) as data:
            query_rows.append([
                int(data["stage1_candidates"]), int(data["stage1_full"]),
                int(data["stage2_candidates"]), int(data["stage2_full"]),
            ])
            if "stage1_sentences" in data:
                sentence_rows.append([
                    int(data["stage1_sentences"]),
                    int(data["stage2_sentences"]),
                ])
    q = np.asarray(query_rows)
    report["queries"] = {
        "fine_perturbation_mean": float((q[:, 0] + q[:, 2]).mean()),
        "full_mean": float((q[:, 1] + q[:, 3]).mean()),
        "fine_perturbation_reduction": float(
            1 - (q[:, 0] + q[:, 2]).sum() / (q[:, 1] + q[:, 3]).sum()
        ),
        "screening_note": "two class-gradient backward graphs per stage; margin is their difference",
    }
    if sentence_rows:
        report["sentences"] = {
            "stage1_mean": float(np.asarray(sentence_rows)[:, 0].mean()),
            "stage2_mean": float(np.asarray(sentence_rows)[:, 1].mean()),
        }
    m.OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "evaluate", "all"))
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--saliency-mass", type=float, default=.75)
    parser.add_argument("--max-candidate-fraction", type=float, default=.60)
    parser.add_argument("--sentence-topk", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.saliency_mass <= 1:
        parser.error("--saliency-mass must be in (0, 1]")
    if not 0 < args.max_candidate_fraction <= 1:
        parser.error("--max-candidate-fraction must be in (0, 1]")

    def shortlist(att, prep, spans, keep=None, blocks=None):
        return sentence_shortlist(
            att, prep, spans, saliency_mass=args.saliency_mass,
            max_candidate_fraction=args.max_candidate_fraction,
            topk=args.sentence_topk,
        )

    m.shortlist = shortlist
    if args.stage in ("collect", "all"):
        # The inherited collector passes keep/blocks, but shortlist deliberately
        # ignores them in favor of sentence boundaries and adaptive saliency mass.
        m.collect(args)
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
