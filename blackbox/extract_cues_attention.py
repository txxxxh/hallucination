#!/usr/bin/env python3
"""Local-model cue extraction for RealLifeQA: attention + signed grad-x-input.

No span enumeration. One forward pass per item scores every scenario span
simultaneously:

  1. attention   : attention mass flowing from the answer-predicting position
                   (final prompt token) to each scenario token, averaged over
                   heads and aggregated over layers (last / last-quarter mean /
                   rollout). Ranks spans by model focus -> keyword candidates.
                   Sign-blind: cannot distinguish shortcut vs constraint.

  2. grad_input  : signed attribution of the answer-token logit difference
                       D = z(shortcut option token) - z(correct option token)
                   via gradient-x-input on the input embeddings. One backward
                   pass. Directional:
                       span score > 0  -> pushes toward shortcut -> shortcut key
                       span score < 0  -> supports correct       -> constraint key
                   This mirrors the (-,+,0) occlusion sign pattern, so role
                   assignment stays consistent with the signature-matrix logic.

Requires: torch, transformers (pip install torch transformers).
Model weights are downloaded from HuggingFace on first run.

Usage:
    python extract_cues_attention.py \
        --input question_remove.json \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --attn-agg rollout \
        --gold-file gold_spans.json \
        --outdir outputs/cue_extraction_attention

--gold-file (optional) is a JSON list of
    {"id": ..., "shortcut_span": "...", "constraint_span": "..."}
e.g. exported from the occlusion script's --annotate output, enabling
top-1 hit rate and token-F1 against the same gold spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cue_spans import (  # noqa: E402
    Span,
    match_span_to_gold,
    segment_scenario,
    token_f1,
)

import run_reallifeqa_pilot as pilot  # noqa: E402

SYSTEM_MESSAGE = "Answer with exactly one character: 1 or 2. Do not explain."


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------


class LocalScorer:
    """Holds a causal LM and computes per-token attention / attribution."""

    def __init__(self, model_name: str, device: Optional[str] = None,
                 dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = None
        if dtype == "auto":
            # float16 overflows in eager attention on bf16-trained models
            # (e.g. Qwen3) and produces NaN everywhere; prefer bfloat16.
            if device == "cuda" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        elif dtype:
            torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",  # required for output_attentions
        ).to(device)
        self.model.eval()

    # -- prompt construction -------------------------------------------------

    def build_text(self, user_prompt: str) -> str:
        """Chat-template the prompt; fall back to plain concatenation."""
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SYSTEM_MESSAGE}\n\n{user_prompt}\n\nAnswer:"

    def encode_with_offsets(self, text: str):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc, offsets

    def option_token_id(self, label: str) -> int:
        """Token id the model would emit for '1' or '2' as the next token."""
        for variant in (label, " " + label):
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # multi-token fallback: use the first token of the plain form
        return self.tokenizer.encode(label, add_special_tokens=False)[0]

    # -- scoring -------------------------------------------------------------

    def attention_scores(self, enc: Dict[str, Any], agg: str) -> List[float]:
        """Per-token attention received from the final (answer-predicting)
        position. agg in {'last', 'last_quarter', 'rollout'}."""
        torch = self.torch
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        # tuple of (1, heads, seq, seq), one per layer
        attns = [a[0].float().mean(dim=0) for a in out.attentions]  # (seq, seq)
        n_layers = len(attns)

        if agg == "last":
            row = attns[-1][-1]
        elif agg == "last_quarter":
            start = max(0, n_layers - max(1, n_layers // 4))
            row = torch.stack([a[-1] for a in attns[start:]]).mean(dim=0)
        elif agg == "rollout":
            seq = attns[0].shape[-1]
            eye = torch.eye(seq, device=attns[0].device)
            joint = eye
            for a in attns:
                mixed = 0.5 * a + 0.5 * eye
                mixed = mixed / mixed.sum(dim=-1, keepdim=True)
                joint = mixed @ joint
            row = joint[-1]
        else:
            raise ValueError(f"unknown attention aggregation: {agg}")
        return row.cpu().tolist()

    def grad_input_scores(
        self, enc: Dict[str, Any], shortcut_id: int, correct_id: int
    ) -> Tuple[List[float], Dict[str, float]]:
        """Signed per-token attribution of D = z_shortcut - z_correct at the
        final position, via gradient-x-input on input embeddings."""
        torch = self.torch
        embed = self.model.get_input_embeddings()
        input_ids = enc["input_ids"]
        inputs_embeds = embed(input_ids).detach().clone().requires_grad_(True)
        kwargs = {k: v for k, v in enc.items() if k != "input_ids"}
        out = self.model(inputs_embeds=inputs_embeds, **kwargs)
        logits = out.logits[0, -1]
        diff = logits[shortcut_id] - logits[correct_id]
        diff.backward()
        scores = (inputs_embeds.grad * inputs_embeds).sum(dim=-1)[0]
        info = {
            "logit_diff": float(diff.detach().float().cpu()),
            "logit_shortcut": float(logits[shortcut_id].detach().float().cpu()),
            "logit_correct": float(logits[correct_id].detach().float().cpu()),
        }
        return scores.detach().float().cpu().tolist(), info


# --------------------------------------------------------------------------
# Token -> span aggregation
# --------------------------------------------------------------------------


def spans_in_templated_text(
    prompt: str, templated: str, spans: List[Span]
) -> Optional[List[Span]]:
    """Re-anchor scenario spans (offsets in the raw prompt) inside the
    chat-templated string. Chat templates embed the user content verbatim,
    so a substring search suffices."""
    base = templated.find(prompt)
    if base == -1:
        # Template may normalize trailing whitespace; try the scenario spans
        # individually as a fallback.
        relocated = []
        for span in spans:
            pos = templated.find(span.text)
            if pos == -1:
                return None
            relocated.append(Span(span.index, span.text, pos, pos + len(span.text)))
        return relocated
    return [
        Span(s.index, s.text, base + s.start, base + s.end) for s in spans
    ]


def aggregate_span_scores(
    token_scores: List[float],
    offsets: List[Tuple[int, int]],
    spans: List[Span],
    reduce: str,
) -> List[Optional[float]]:
    """Sum/mean token scores whose character range overlaps each span."""
    results: List[Optional[float]] = []
    for span in spans:
        values = [
            score
            for score, (start, end) in zip(token_scores, offsets)
            if end > span.start and start < span.end and end > start
        ]
        if not values:
            results.append(None)
        elif reduce == "mean":
            results.append(sum(values) / len(values))
        else:
            results.append(sum(values))
    return results


# --------------------------------------------------------------------------
# Per-item pipeline
# --------------------------------------------------------------------------


def run_item(
    scorer: LocalScorer, item: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    prompt = item["benchmark_prompt"]
    correct = str(int(item["answer"]))
    shortcut = "1" if correct == "2" else "2"

    spans = segment_scenario(prompt, min_clause_words=args.min_clause_words)
    if len(spans) < 2:
        return {"id": item.get("id"), "error": "fewer than 2 candidate spans"}

    templated = scorer.build_text(prompt)
    anchored = spans_in_templated_text(prompt, templated, spans)
    if anchored is None:
        return {"id": item.get("id"),
                "error": "could not locate spans in templated text"}

    enc, offsets = scorer.encode_with_offsets(templated)

    # 1) attention: keyword localization (sign-blind)
    attn_tokens = scorer.attention_scores(enc, agg=args.attn_agg)
    attn_by_span = aggregate_span_scores(attn_tokens, offsets, anchored, reduce="mean")

    # 2) grad-x-input on logit difference: directional role assignment
    shortcut_id = scorer.option_token_id(shortcut)
    correct_id = scorer.option_token_id(correct)
    grad_tokens, logit_info = scorer.grad_input_scores(enc, shortcut_id, correct_id)
    grad_by_span = aggregate_span_scores(grad_tokens, offsets, anchored, reduce="sum")

    import math as _math
    numeric = attn_tokens + grad_tokens + list(logit_info.values())
    if any(v is not None and (_math.isnan(v) or _math.isinf(v)) for v in numeric):
        return {
            "id": item.get("id"),
            "error": (
                "NaN/inf in model outputs - dtype overflow. Rerun with "
                "--dtype bfloat16 (CUDA) or --dtype float32 (CPU/older GPUs); "
                "float16 overflows in eager attention on bf16-trained models."
            ),
        }

    span_rows = []
    for span, attn, grad in zip(spans, attn_by_span, grad_by_span):
        span_rows.append(
            {
                "span_index": span.index,
                "span_text": span.text,
                "attention": attn,
                "grad_input": grad,
            }
        )

    valid_attn = [r for r in span_rows if r["attention"] is not None]
    valid_grad = [r for r in span_rows if r["grad_input"] is not None]
    attn_ranking = sorted(valid_attn, key=lambda r: -r["attention"])

    # Role assignment from the signed attribution, with MAD-scaled abstention.
    pred_shortcut = pred_constraint = None
    tau = 0.0
    if valid_grad:
        grads = [r["grad_input"] for r in valid_grad]
        med = statistics.median(grads)
        mad = statistics.median(abs(g - med) for g in grads) if len(grads) > 2 else 0.0
        tau = args.tau_mad_mult * 1.4826 * mad
        max_row = max(valid_grad, key=lambda r: r["grad_input"])
        min_row = min(valid_grad, key=lambda r: r["grad_input"])
        if max_row["grad_input"] > tau:
            pred_shortcut = max_row       # pushes toward shortcut option
        if min_row["grad_input"] < -tau:
            pred_constraint = min_row     # supports correct option
        if (
            pred_shortcut is not None
            and pred_constraint is not None
            and pred_shortcut["span_index"] == pred_constraint["span_index"]
        ):
            if abs(max_row["grad_input"]) >= abs(min_row["grad_input"]):
                pred_constraint = None
            else:
                pred_shortcut = None

    record: Dict[str, Any] = {
        "id": item.get("id"),
        "correct_option": correct,
        "shortcut_option": shortcut,
        "n_spans": len(spans),
        "tau": tau,
        **logit_info,
        "spans": span_rows,
        "attn_top1_span": attn_ranking[0]["span_text"] if attn_ranking else None,
        "attn_ranking": [r["span_index"] for r in attn_ranking],
        "pred_shortcut_span": pred_shortcut["span_text"] if pred_shortcut else None,
        "pred_shortcut_score": pred_shortcut["grad_input"] if pred_shortcut else None,
        "pred_constraint_span": (
            pred_constraint["span_text"] if pred_constraint else None
        ),
        "pred_constraint_score": (
            pred_constraint["grad_input"] if pred_constraint else None
        ),
    }
    return record


# --------------------------------------------------------------------------
# Gold evaluation + reporting
# --------------------------------------------------------------------------


def load_gold(path: Optional[str]) -> Dict[Any, Dict[str, Optional[str]]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry.get("id"): entry for entry in data if isinstance(entry, dict)}


def evaluate_against_gold(
    record: Dict[str, Any],
    gold: Dict[str, Optional[str]],
    prompt: str,
    min_clause_words: int,
) -> None:
    spans = segment_scenario(prompt, min_clause_words=min_clause_words)
    by_text = {s.text: s.index for s in spans}
    for role in ("shortcut", "constraint"):
        gold_text = gold.get(f"{role}_span")
        record[f"gold_{role}_span"] = gold_text
        pred_text = record.get(f"pred_{role}_span")
        gold_idx = match_span_to_gold(spans, gold_text)
        pred_idx = by_text.get(pred_text) if pred_text else None
        record[f"{role}_hit"] = (
            gold_idx is not None and pred_idx is not None and gold_idx == pred_idx
        )
        record[f"{role}_f1"] = token_f1(pred_text, gold_text)
    # Keyword-only metric for the attention arm: does the attention top-1 span
    # match *either* gold key? (localization without role assignment)
    gold_indices = {
        match_span_to_gold(spans, gold.get("shortcut_span")),
        match_span_to_gold(spans, gold.get("constraint_span")),
    } - {None}
    top1 = record.get("attn_ranking", [None])
    record["attn_keyword_hit"] = bool(top1) and top1[0] in gold_indices


def write_outputs(outdir: Path, records: List[Dict[str, Any]], has_gold: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "cue_extraction_attention.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = [
        "id", "n_spans", "logit_diff",
        "attn_top1_span",
        "pred_shortcut_span", "pred_shortcut_score",
        "pred_constraint_span", "pred_constraint_score",
    ]
    if has_gold:
        fields += [
            "gold_shortcut_span", "gold_constraint_span",
            "attn_keyword_hit",
            "shortcut_hit", "shortcut_f1",
            "constraint_hit", "constraint_f1",
        ]
    with (outdir / "cue_extraction_attention.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    ok = [r for r in records if "error" not in r]
    lines = [
        "# Attention / grad-x-input cue extraction summary",
        "",
        f"Items processed: {len(records)} (valid: {len(ok)})",
    ]
    if has_gold and ok:
        kw = [r for r in ok if "attn_keyword_hit" in r]
        if kw:
            hits = sum(bool(r["attn_keyword_hit"]) for r in kw)
            lines.append(
                f"Attention keyword localization (top-1 hits either key): "
                f"{hits}/{len(kw)} ({hits / len(kw):.0%})"
            )
        for role in ("shortcut", "constraint"):
            with_gold = [r for r in ok if r.get(f"gold_{role}_span")]
            if with_gold:
                hits = sum(bool(r.get(f"{role}_hit")) for r in with_gold)
                mean_f1 = statistics.mean(
                    r.get(f"{role}_f1", 0.0) for r in with_gold
                )
                lines.append(
                    f"Grad-x-input {role} top-1 hit rate: {hits}/{len(with_gold)} "
                    f"({hits / len(with_gold):.0%}); mean token F1: {mean_f1:.2f}"
                )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="question_remove.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--attn-agg", choices=("last", "last_quarter", "rollout"),
        default="last_quarter",
    )
    parser.add_argument("--tau-mad-mult", type=float, default=1.5)
    parser.add_argument("--min-clause-words", type=int, default=12)
    parser.add_argument("--gold-file", default=None,
                        help="JSON list of {id, shortcut_span, constraint_span}")
    parser.add_argument("--outdir", default="outputs/cue_extraction_attention")
    args = parser.parse_args()

    data = pilot.load_data(args.input)
    if args.limit is not None and args.limit >= 0:
        data = data[: args.limit]
    gold_by_id = load_gold(args.gold_file)

    scorer = LocalScorer(args.model, device=args.device, dtype=args.dtype)
    print(f"Loaded {args.model} on {scorer.device}", file=sys.stderr)

    records: List[Dict[str, Any]] = []
    for run_index, raw_item in enumerate(data):
        item_id = (
            raw_item.get("id", run_index) if isinstance(raw_item, dict) else run_index
        )
        try:
            item = pilot._validate_item(raw_item, run_index)
            record = run_item(scorer, item, args)
            gold = gold_by_id.get(item.get("id"))
            if gold and "error" not in record:
                evaluate_against_gold(
                    record, gold, item["benchmark_prompt"], args.min_clause_words
                )
            records.append(record)
        except Exception as exc:
            records.append({"id": item_id, "error": str(exc)})
        print(
            f"[{run_index + 1}/{len(data)}] item {item_id} done",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(0.01)

    write_outputs(Path(args.outdir), records, has_gold=bool(gold_by_id))
    print(f"Wrote outputs to {args.outdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
