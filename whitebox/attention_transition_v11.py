#!/usr/bin/env python3
"""
Open-ended attention-behavior transition hallucination detector v11.

Workflow:
  generate -> score every atomic span by answer-to-span attention
  -> select top-k attention spans
  -> intervene on each selected span
  -> keep the original-answer confidence-drop behavior signal
  -> re-measure attention over every surviving selected span
  -> construct behavior-weighted attention-transition matrices
  -> use row-level transition features for span-role learning
  -> use matrix/SVD features plus predicted shortcut evidence for
     item-level hallucination detection with logistic regression.

References are used only for training pseudo-role construction and final
correctness evaluation. No reference-derived quantity is used as a test-time
feature. The transition matrix is computed from teacher-forced attention to the
model's own original generated answer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CACHE_SCHEMA_VERSION = "openended_v11_attention_transition_v2"
ROLE_NAMES = ["constraint", "shortcut", "irrelevant"]
ROLE_TO_ID = {x: i for i, x in enumerate(ROLE_NAMES)}
OPERATORS = ("delete", "neutralize", "mask")
ATTENTION_GROUPS = ("early", "middle", "late")
# Selected using only the prior 2,000-item training models: rank each item-level
# transition feature by its mean absolute standardized coefficient across the
# v8-role and transition-role variants, then retain the top 16.  The held-out
# test labels were not used for this selection.
TRANSITION_ITEM_FEATURE_NAMES = (
    "early::mask::mean_abs_entry",
    "early::neutralize::mean_abs_entry",
    "early::operator_median::positive_mass",
    "middle::neutralize::positive_mass",
    "early::neutralize::positive_mass",
    "middle::mask::mean_abs_entry",
    "middle::mask::frobenius_norm",
    "early::neutralize::frobenius_norm",
    "late::neutralize::mean_abs_entry",
    "early::mask::largest_singular_value",
    "late::operator_median::singular_entropy",
    "late::neutralize::positive_mass",
    "early::operator_median::largest_singular_value",
    "middle::neutralize::mean_abs_entry",
    "late::operator_median::largest_singular_value",
    "middle::operator_median::positive_mass",
)
FEATURE_SETS = {
    "structure_only": ("structural",),
    "behavior_only": ("behavior",),
    "attention_only": ("attention",),
    "transition_only": ("transition",),
    "behavior_attention": ("behavior", "attention"),
    "attention_behavior": ("behavior", "attention"),
    "behavior_transition": ("behavior", "transition"),
    "attention_transition": ("attention", "transition"),
    "attention_behavior_transition": ("behavior", "attention", "transition"),
    # Kept for backward-compatible ablations.
    "gradient_only": ("gradient",),
    "spectral_only": ("spectral",),
    "behavior_gradient": ("behavior", "gradient"),
    "behavior_spectral": ("behavior", "spectral"),
    "whitebox_combined": ("attention", "gradient", "spectral"),
    "all_combined": ("structural", "behavior", "attention", "gradient", "spectral", "transition"),
}
ITEM_MODES = {
    "role_only",
    "transition_only",
    "transition_plus_role",
}
DEFAULT_SYSTEM = (
    "Answer accurately and concisely. Do not invent unsupported details. "
    "Give only the answer unless a brief explanation is necessary."
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_CLAUSE_SPLIT = re.compile(
    r",\s+(?=(?:but|and|so|because|while|although|unless|only|before|after|until|whereas)\b)",
    re.I,
)
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
_NEG_RE = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|can't|won't|isn't|aren't|didn't|doesn't|don't)\b",
    re.I,
)
_REFUSAL_RE = re.compile(
    r"\b(?:i do not know|i don't know|cannot answer|can't answer|insufficient information|not enough information|unable to determine|cannot determine|no way to know)\b",
    re.I,
)


@dataclass
class Span:
    index: int
    text: str
    start: int
    end: int


@dataclass
class Example:
    item_id: str
    source_text: str
    question: str
    context: str
    references: list[str]
    raw_index: int


@dataclass
class PromptEncoding:
    prompt_text: str
    prompt_ids: torch.Tensor
    offsets: list[tuple[int, int]]
    source_start: int


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        f.flush()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ------------------------------ dataset -----------------------------------

def get_nested(row: dict[str, Any], field: Optional[str]) -> Any:
    if not field:
        return None
    cur: Any = row
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_present(row: dict[str, Any], fields: Sequence[str]) -> Any:
    for f in fields:
        v = get_nested(row, f)
        if v is not None:
            return v
    return None


def flatten_refs(value: Any) -> list[str]:
    out: list[str] = []
    def visit(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            if x.strip(): out.append(x.strip())
        elif isinstance(x, (int, float, np.integer, np.floating)):
            out.append(str(x))
        elif isinstance(x, (list, tuple, set, np.ndarray)):
            for y in x: visit(y)
        elif isinstance(x, dict):
            keys = ("aliases", "correct_answers", "answers", "answer", "text", "value", "normalized_value")
            found = False
            for k in keys:
                if k in x:
                    visit(x[k]); found = True
            if not found:
                for y in x.values(): visit(y)
    visit(value)
    result, seen = [], set()
    for s in out:
        key = s.lower().strip()
        if key and key not in seen:
            result.append(s.strip()); seen.add(key)
    return result


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.hf_dataset:
        from datasets import load_dataset
        if args.hf_subset:
            ds = load_dataset(args.hf_dataset, args.hf_subset, split=args.hf_split)
        else:
            ds = load_dataset(args.hf_dataset, split=args.hf_split)
        rows = [dict(x) for x in ds]
    else:
        path = Path(args.input)
        if not path.exists(): raise FileNotFoundError(path)
        s = path.suffix.lower()
        if s in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif s == ".json":
            raw_text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                # Several benchmark releases use JSONL content with a .json suffix.
                rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
            else:
                if isinstance(data, list): rows = data
                elif isinstance(data, dict):
                    rows = next((data[k] for k in ("data", "items", "examples", "questions") if isinstance(data.get(k), list)), None)
                    if rows is None: raise ValueError("No list-valued data/items/examples/questions key.")
                else: raise ValueError("Unsupported JSON root.")
        elif s in {".parquet", ".pq"}:
            rows = pd.read_parquet(path).to_dict("records")
        elif s in {".csv", ".tsv"}:
            rows = pd.read_csv(path, sep="\t" if s == ".tsv" else ",").to_dict("records")
        else: raise ValueError(f"Unsupported suffix: {s}")
    if args.max_samples > 0: rows = rows[:args.max_samples]
    return rows


def build_examples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[Example]:
    examples: list[Example] = []
    for i, row in enumerate(rows):
        q = get_nested(row, args.question_field) if args.question_field else None
        if q is None: q = first_present(row, ("question", "query", "prompt", "input", "instruction", "problem"))
        c = get_nested(row, args.context_field) if args.context_field else None
        if c is None: c = first_present(row, ("knowledge", "context", "passage", "story", "article", "document", "evidence"))
        a = get_nested(row, args.answers_field) if args.answers_field else None
        if a is None: a = first_present(row, ("right_answer", "correct_answers", "answers", "answer.aliases", "answer", "reference_answer", "gold_answer", "target", "output"))
        refs = flatten_refs(a)
        if q is None or not refs:
            warnings.warn(f"Skipping row {i}: missing question or references")
            continue
        q, c = str(q).strip(), "" if c is None else str(c).strip()
        if args.prompt_field:
            source = str(get_nested(row, args.prompt_field) or "").strip()
        else:
            source = f"Context:\n{c}\n\nQuestion:\n{q}" if c else f"Question:\n{q}"
        iid = get_nested(row, args.id_field) if args.id_field else None
        if iid is None: iid = first_present(row, ("id", "key", "question_id", "qid"))
        examples.append(Example(str(iid) if iid is not None else f"item_{i:06d}", source, q, c, refs, i))
    if not examples: raise ValueError("No usable examples.")
    return examples


# -------------------------- answer evaluation -----------------------------

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def token_f1(a: str, b: str) -> float:
    x, y = normalize_answer(a).split(), normalize_answer(b).split()
    if not x and not y: return 1.0
    if not x or not y: return 0.0
    cx, cy = {}, {}
    for t in x: cx[t] = cx.get(t, 0) + 1
    for t in y: cy[t] = cy.get(t, 0) + 1
    common = sum(min(n, cy.get(t, 0)) for t, n in cx.items())
    if common == 0: return 0.0
    p, r = common / len(x), common / len(y)
    return 2 * p * r / (p + r)


def max_ref_f1(pred: str, refs: Sequence[str]) -> float:
    return max((token_f1(pred, r) for r in refs), default=0.0)


def exact_or_contained(pred: str, ref: str) -> bool:
    p, r = normalize_answer(pred), normalize_answer(ref)
    if not p or not r: return False
    return p == r or (len(r.split()) >= 2 and r in p) or (len(p.split()) >= 2 and p in r)


def last_number(text: str) -> Optional[str]:
    m = _NUMBER_RE.findall(text.replace(",", ""))
    if not m: return None
    try:
        v = float(m[-1]); return str(int(v)) if v.is_integer() else f"{v:.12g}"
    except ValueError: return m[-1]


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def polarity_sig(text: str) -> tuple[int, int]:
    n = normalize_answer(text)
    yes = int(bool(re.search(r"\b(?:yes|true|correct|does|will|is)\b", n)))
    no = int(bool(_NEG_RE.search(n) or re.search(r"\b(?:no|false|incorrect)\b", n)))
    return yes, no


def entity_set(text: str) -> set[str]:
    return set(re.findall(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)*\b", text))


class CorrectnessEvaluator:
    def __init__(self, mode: str, threshold: float, engine: Optional["WhiteboxEngine"] = None):
        self.mode, self.threshold, self.engine = mode, threshold, engine
    def evaluate(self, pred: str, refs: Sequence[str], question: str) -> Optional[bool]:
        if is_refusal(pred): return None
        if self.mode == "llm_judge":
            if self.engine is None: raise RuntimeError("llm_judge needs engine")
            return self.engine.judge_answer(question, refs, pred)
        if self.mode == "numeric":
            p = last_number(pred); rs = {last_number(r) for r in refs}; rs.discard(None)
            return p is not None and p in rs
        exact = any(exact_or_contained(pred, r) for r in refs)
        if self.mode == "exact": return exact
        f1 = max_ref_f1(pred, refs)
        if self.mode == "token_f1": return f1 >= self.threshold
        return exact or f1 >= self.threshold


# -------------------------- spans/interventions ---------------------------

def segment_atomic(source: str, min_clause_words: int, min_span_words: int) -> list[Span]:
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT.split(source):
        if not sentence: continue
        st = source.find(sentence, cursor)
        if st < 0: continue
        en, cursor = st + len(sentence), st + len(sentence)
        clean = sentence.strip()
        if not clean: continue
        cst = source.find(clean, st, en + 1); cen = cst + len(clean)
        if len(_WORD_RE.findall(clean)) > min_clause_words:
            sub = 0
            for clause in _CLAUSE_SPLIT.split(clean):
                if not clause.strip(): continue
                a = clean.find(clause, sub)
                if a < 0: continue
                b, sub = a + len(clause), a + len(clause)
                stripped = clause.strip(); off = clause.find(stripped)
                pieces.append((cst + a + off, cst + a + off + len(stripped)))
        else: pieces.append((cst, cen))
    if not pieces and source.strip():
        s = source.find(source.strip()); pieces = [(s, s + len(source.strip()))]
    merged: list[tuple[int, int]] = []
    for a, b in pieces:
        if merged and len(_WORD_RE.findall(source[a:b])) < min_span_words:
            merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    out = []
    for i, (a, b) in enumerate(merged):
        text = source[a:b].strip()
        if text:
            real = source.find(text, a, b + 1); out.append(Span(i, text, real, real + len(text)))
    return out


def select_topk_attention_spans(
    spans: list[Span],
    score_rows: dict[int, dict[str, float]],
    maximum: int,
) -> tuple[list[Span], dict[int, int]]:
    """Select only the top-k answer-to-span attention spans.

    No random, low-attention, or control span is added. Span indices are kept
    unchanged so that their token mappings and cache explanations remain stable.
    """
    if not spans:
        return [], {}

    ranked = sorted(
        spans,
        key=lambda span: (
            float(score_rows.get(span.index, {}).get("score", float("-inf"))),
            float(score_rows.get(span.index, {}).get("peak", float("-inf"))),
            -span.start,
        ),
        reverse=True,
    )
    rank_by_index = {span.index: rank + 1 for rank, span in enumerate(ranked)}

    if maximum > 0:
        ranked = ranked[:maximum]

    # Restore source order for deterministic intervention/reconstruction while
    # retaining the attention rank in rank_by_index.
    selected = sorted(ranked, key=lambda span: span.start)
    return selected, rank_by_index


def intervene_aligned(
    source: str,
    spans: Sequence[Span],
    target: Span,
    op: str,
    mask: str,
    neutral: str,
) -> tuple[str, list[Span]]:
    """Apply one intervention and realign every selected span.

    The v8 deletion helper normalized whitespace after deletion, which made the
    offsets of all later spans ambiguous. v11 intentionally performs a direct
    character replacement so every surviving original span has an exact offset
    in the modified prompt.

    For delete, the target span is absent from the returned list. For mask and
    neutralize, it is replaced in-place and keeps the same span index.
    """
    if op not in OPERATORS:
        raise ValueError(f"Unknown intervention operator: {op}")
    replacement = "" if op == "delete" else (neutral if op == "neutralize" else mask)
    modified = source[:target.start] + replacement + source[target.end:]
    shift = len(replacement) - (target.end - target.start)

    aligned: list[Span] = []
    for span in spans:
        if span.index == target.index:
            if op != "delete":
                aligned.append(
                    Span(span.index, replacement, target.start, target.start + len(replacement))
                )
            continue
        if span.end <= target.start:
            aligned.append(Span(span.index, span.text, span.start, span.end))
        elif span.start >= target.end:
            aligned.append(
                Span(span.index, span.text, span.start + shift, span.end + shift)
            )
        else:
            raise RuntimeError(
                "Selected spans overlap; transition alignment requires non-overlapping spans."
            )
    return modified, aligned


def normalize_distribution(values: np.ndarray, exclude: Optional[int] = None) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    if exclude is not None and 0 <= exclude < len(x):
        x[exclude] = 0.0
    total = float(x.sum())
    if total <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x / total).astype(np.float32)


def attention_transition_vector(
    base: Sequence[float],
    intervened: Sequence[float],
    target_position: int,
    operator: str,
) -> np.ndarray:
    """Return the post-minus-pre attention redistribution over selected spans.

    For deletion, the target coordinate is excluded from both distributions and
    the remaining coordinates are renormalized. This prevents the trivial fact
    that a deleted span receives zero attention from becoming a detector feature.
    """
    exclude = target_position if operator == "delete" else None
    before = normalize_distribution(np.asarray(base), exclude=exclude)
    after = normalize_distribution(np.asarray(intervened), exclude=exclude)
    delta = after - before
    if exclude is not None:
        delta[exclude] = 0.0
    return np.nan_to_num(delta.astype(np.float32))

def structural_features(span: Span, source: str) -> np.ndarray:
    sc, sw = max(len(source), 1), max(len(_WORD_RE.findall(source)), 1)
    w = _WORD_RE.findall(span.text)
    vals = [
        span.start/sc, span.end/sc, (span.end-span.start)/sc, len(w)/sw,
        len(w), len(span.text), span.index, bool(_NEG_RE.search(span.text)),
        bool(_NUMBER_RE.search(span.text)), bool(re.search(r"\b[A-Z][\w'-]*\b", span.text)),
        "?" in span.text, ":" in span.text,
        span.text.lower().startswith(("question:", "context:", "evidence:")),
    ]
    return np.asarray(vals, np.float32)


# ---------------------------- model engine --------------------------------

def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name.lower()]


class WhiteboxEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args; self.device = torch.device(args.device)
        # This PyTorch build routes outer-product bmm through a just-in-time
        # Triton extension. The runtime image has no Python development headers,
        # and its user cache is network-backed, so use the standard ATen bmm.
        native_registry = getattr(getattr(torch, "_native", None), "registry", None)
        if native_registry is not None:
            native_registry.deregister_op_overrides(disable_op_symbols="bmm")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        if not self.tokenizer.is_fast: raise RuntimeError("Fast tokenizer required")
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        kw = dict(torch_dtype=dtype_from_name(args.dtype), trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True)
        if args.attn_implementation: kw["attn_implementation"] = args.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **kw).to(self.device)
        self.model.eval(); self.model.config.use_cache = False
        self.layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        if self.layers <= 0: raise RuntimeError("Cannot infer layer count")
        self.attention_dim = 9 * self.layers
        self.gradient_dim = 4 * (self.layers + 1)
        self.spectral_dim = (3 * args.lap_topk + 9) * self.layers

    def prompt_text(self, source: str, system: Optional[str] = None) -> str:
        messages = [{"role":"system", "content":system or self.args.system_prompt},
                    {"role":"user", "content":source + "\n\n" + self.args.answer_instruction}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

    def encode_prompt(self, source: str, system: Optional[str] = None) -> PromptEncoding:
        # Locate the source via explicit boundaries instead of searching for
        # the complete source string in the rendered chat prompt.
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        marker_start = f"<|source_start_{digest}|>"
        marker_end = f"<|source_end_{digest}|>"
        counter = 0
        while marker_start in source or marker_end in source:
            counter += 1
            marker_start = f"<|source_start_{digest}_{counter}|>"
            marker_end = f"<|source_end_{digest}_{counter}|>"

        marked = self.prompt_text(marker_start + source + marker_end, system)
        start_marker_pos = marked.find(marker_start)
        end_marker_pos = marked.find(marker_end, start_marker_pos + len(marker_start))
        if start_marker_pos < 0 or end_marker_pos < 0:
            raise RuntimeError("Chat template did not preserve source boundary markers")

        rendered_source = marked[start_marker_pos + len(marker_start):end_marker_pos]
        if rendered_source != source:
            raise RuntimeError("Chat template modified source content between boundary markers")

        text = marked[:start_marker_pos] + source + marked[end_marker_pos + len(marker_end):]
        expected = self.prompt_text(source, system)
        if text != expected:
            # Some chat templates trim leading/trailing whitespace from message
            # content. The markers prevent that trimming in `marked`, so after
            # an edge-span deletion, removing them need not reproduce the real
            # unmarked render exactly. Locate the trimmed source in that render
            # and compensate for removed leading whitespace in span offsets.
            rendered_source = source.strip()
            if not rendered_source:
                raise RuntimeError("Source became empty after chat-template trimming")
            rendered_pos = expected.find(rendered_source)
            if rendered_pos < 0 or expected.find(rendered_source, rendered_pos + 1) >= 0:
                raise RuntimeError(
                    "Could not uniquely locate trimmed source in chat template output"
                )
            leading_trim = len(source) - len(source.lstrip())
            text = expected
            start = rendered_pos - leading_trim
        else:
            start = start_marker_pos
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        if len(ids) > self.args.max_input_tokens: raise ValueError(f"Prompt {len(ids)} > max {self.args.max_input_tokens}")
        return PromptEncoding(text, ids, [(int(a), int(b)) for a,b in enc["offset_mapping"]], start)

    def span_tokens(self, prompt: PromptEncoding, span: Span) -> list[int]:
        a, b = prompt.source_start + span.start, prompt.source_start + span.end
        return [i for i,(s,e) in enumerate(prompt.offsets) if e > s and e > a and s < b]

    def generate(self, source: str, max_new: Optional[int] = None, system: Optional[str] = None) -> tuple[str, torch.Tensor]:
        p = self.encode_prompt(source, system)
        ids = p.prompt_ids.unsqueeze(0).to(self.device); mask = torch.ones_like(ids)
        kw = dict(max_new_tokens=max_new or self.args.max_new_tokens, do_sample=self.args.temperature > 0,
                  pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        if self.args.temperature > 0: kw.update(temperature=self.args.temperature, top_p=self.args.top_p)
        with torch.inference_mode(): out = self.model.generate(input_ids=ids, attention_mask=mask, **kw)
        ans = out[0, ids.shape[1]:].detach().cpu()
        if self.tokenizer.eos_token_id in ans.tolist(): ans = ans[:ans.tolist().index(self.tokenizer.eos_token_id)]
        text = self.tokenizer.decode(ans, skip_special_tokens=True).strip()
        if ans.numel() == 0:
            ans = torch.tensor(self.tokenizer("I do not know.", add_special_tokens=False)["input_ids"])
            text = "I do not know."
        return text, ans

    def judge_answer(self, question: str, refs: Sequence[str], pred: str) -> Optional[bool]:
        source = ("Return exactly CORRECT, INCORRECT, or REFUSAL.\n"
                  f"Question: {question}\nReferences: {json.dumps(list(refs), ensure_ascii=False)}\nCandidate: {pred}")
        old = self.args.temperature
        try:
            self.args.temperature = 0.0
            verdict, _ = self.generate(source, 5, "You are a strict semantic answer evaluator.")
        finally: self.args.temperature = old
        u = verdict.upper()
        if "REFUSAL" in u: return None
        if "INCORRECT" in u: return False
        return "CORRECT" in u

    def _attention_selection_layer_indices(self) -> list[int]:
        mode = self.args.attention_selection_layers
        if mode == "all":
            return list(range(self.layers))
        if mode == "last_half":
            return list(range(max(0, self.layers // 2), self.layers))
        if mode == "last_n":
            n = max(1, min(int(self.args.attention_selection_last_n), self.layers))
            return list(range(self.layers - n, self.layers))
        # Default: last quarter.
        n = max(1, self.layers // 4)
        return list(range(self.layers - n, self.layers))

    @staticmethod
    def _zscore_within_item(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(values)
        return (values - float(values.mean())) / std

    def score_spans_by_answer_attention(
        self,
        source: str,
        answer_ids_cpu: torch.Tensor,
        spans: Sequence[Span],
    ) -> dict[int, dict[str, float]]:
        """Score every candidate span before intervention selection.

        The model is teacher-forced on its original generated answer. For each
        selected transformer layer and all heads, we aggregate attention from
        every answer token to the input tokens belonging to the span.

        Metrics:
          mass    = total answer-to-span attention (length sensitive)
          density = mass / number of span tokens (length normalized)
          peak    = strongest answer-token/span-token link

        The default hybrid score is a within-item weighted combination of the
        z-scored density and peak. It therefore ranks spans by attention without
        reverting to span length.
        """
        if not spans:
            return {}

        prompt = self.encode_prompt(source)
        prompt_ids = prompt.prompt_ids.to(self.device)
        answer_ids = answer_ids_cpu.to(self.device)
        full_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        prompt_len = int(prompt_ids.numel())
        answer_indices = list(
            range(prompt_len, prompt_len + int(answer_ids.numel()))
        )
        token_map = {
            span.index: self.span_tokens(prompt, span)
            for span in spans
        }

        with torch.inference_mode():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        if outputs.attentions is None:
            raise RuntimeError(
                "The model did not return attentions for top-k span selection. "
                "Use --attn-implementation eager."
            )

        layer_indices = self._attention_selection_layer_indices()
        raw_rows: list[dict[str, float]] = []
        for span in spans:
            token_indices = token_map[span.index]
            if not token_indices or not answer_indices:
                raw_rows.append(
                    {
                        "span_index": float(span.index),
                        "mass": 0.0,
                        "density": 0.0,
                        "peak": 0.0,
                        "token_count": float(len(token_indices)),
                    }
                )
                continue

            layer_mass: list[float] = []
            layer_density: list[float] = []
            layer_peak: list[float] = []
            for layer_idx in layer_indices:
                attention = outputs.attentions[layer_idx][0].detach().float()
                # [heads, answer_tokens, span_tokens]
                sub = attention[:, answer_indices, :][:, :, token_indices]
                per_head_mass = sub.sum(dim=-1).mean(dim=-1)
                per_head_density = per_head_mass / max(len(token_indices), 1)
                per_head_peak = sub.amax(dim=(-1, -2))
                layer_mass.append(float(per_head_mass.mean().cpu()))
                layer_density.append(float(per_head_density.mean().cpu()))
                layer_peak.append(float(per_head_peak.mean().cpu()))

            raw_rows.append(
                {
                    "span_index": float(span.index),
                    "mass": float(np.mean(layer_mass)),
                    "density": float(np.mean(layer_density)),
                    "peak": float(np.mean(layer_peak)),
                    "token_count": float(len(token_indices)),
                }
            )

        masses = np.asarray([row["mass"] for row in raw_rows], dtype=np.float64)
        densities = np.asarray([row["density"] for row in raw_rows], dtype=np.float64)
        peaks = np.asarray([row["peak"] for row in raw_rows], dtype=np.float64)
        score_mode = self.args.attention_selection_score

        if score_mode == "mass":
            scores = masses
        elif score_mode == "density":
            scores = densities
        elif score_mode == "peak":
            scores = peaks
        else:
            density_weight = float(
                np.clip(self.args.attention_selection_density_weight, 0.0, 1.0)
            )
            scores = (
                density_weight * self._zscore_within_item(densities)
                + (1.0 - density_weight) * self._zscore_within_item(peaks)
            )

        result: dict[int, dict[str, float]] = {}
        for row, score in zip(raw_rows, scores):
            span_index = int(row.pop("span_index"))
            result[span_index] = {
                **row,
                "score": float(score),
            }

        del outputs, full_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def attention_features(self, attentions: Sequence[torch.Tensor], span: list[int], answer: list[int]) -> np.ndarray:
        if not span or not answer: return np.zeros(self.attention_dim, np.float32)
        vals = []
        for la in attentions:
            a = la[0].detach().float(); sub = a[:, answer, :][:,:,span]
            mass = sub.sum(-1).mean(-1); density = mass/max(len(span),1); peak = sub.amax((-1,-2))
            for m in (mass,density,peak): vals += [float(m.mean()), float(m.max()), float(m.std(unbiased=False))]
        return np.nan_to_num(np.asarray(vals,np.float32))

    def gradient_features(self, hidden: Sequence[torch.Tensor], span: list[int]) -> np.ndarray:
        if not span: return np.zeros(self.gradient_dim,np.float32)
        vals=[]
        for h in hidden:
            if h.grad is None: vals += [0.,0.,0.,0.]; continue
            x=h[0,span].detach().float(); g=h.grad[0,span].detach().float()
            gn=torch.linalg.vector_norm(g,dim=-1); ga=(g*x).abs().mean(-1)
            vals += [float(gn.mean()),float(gn.max()),float(ga.mean()),float(ga.max())]
        return np.nan_to_num(np.asarray(vals,np.float32))

    def spectral_features(self, attentions: Sequence[torch.Tensor], span: list[int], answer: list[int], prompt_len: int) -> np.ndarray:
        k=self.args.lap_topk; per=3*k+9
        if not span or not answer: return np.zeros(per*self.layers,np.float32)
        feats=[]
        for la in attentions:
            a=la[0].detach().float()
            anchors=list(range(max(0,prompt_len-self.args.spectral_anchor_tokens),prompt_len))
            nodes=sorted(set(span+answer+anchors))
            if len(nodes)>self.args.spectral_max_nodes:
                mandatory=sorted(set(span+answer)); other=[x for x in nodes if x not in set(mandatory)]
                room=max(0,self.args.spectral_max_nodes-len(mandatory))
                other=[other[int(round(x))] for x in np.linspace(0,len(other)-1,max(room,1))[:room]] if room and other else []
                nodes=sorted(set(mandatory+other))
            loc={x:i for i,x in enumerate(nodes)}; ls=[loc[x] for x in span if x in loc]; laa=[loc[x] for x in answer if x in loc]
            if len(nodes)<2 or not ls or not laa: feats += [0.]*per; continue
            sub=a[:,nodes,:][:,:,nodes]; w=.5*(sub+sub.transpose(-1,-2))
            eye=torch.eye(len(nodes),device=w.device,dtype=w.dtype).unsqueeze(0); w=w*(1-eye)
            deg=w.sum(-1).clamp_min(1e-6); inv=deg.rsqrt(); lap=eye-inv.unsqueeze(-1)*w*inv.unsqueeze(-2); lap=.5*(lap+lap.transpose(-1,-2))
            try: ev,evec=torch.linalg.eigh(lap)
            except RuntimeError: ev,evec=torch.linalg.eigh(lap+1e-5*eye)
            top=min(k,ev.shape[-1]); tv=ev[:,-top:].flip(-1)
            if top<k: tv=torch.cat([tv,torch.zeros(tv.shape[0],k-top,device=tv.device)],-1)
            feats += tv.mean(0).cpu().tolist()+tv.max(0).values.cpu().tolist()+tv.std(0,unbiased=False).cpu().tolist()
            vec=evec[:,:,-top:]; energy=vec[:,ls,:].pow(2).mean((1,2)); coupling=w[:,laa,:][:,:,ls].mean((1,2)); degree=deg[:,ls].mean(1)
            for m in (energy,coupling,degree): feats += [float(m.mean()),float(m.max()),float(m.std(unbiased=False))]
        return np.nan_to_num(np.asarray(feats,np.float32))

    def _transition_layer_groups(self) -> dict[str, list[int]]:
        """Split model layers into early/middle/late groups."""
        indices = np.array_split(np.arange(self.layers), 3)
        return {
            name: [int(x) for x in group.tolist()]
            for name, group in zip(ATTENTION_GROUPS, indices)
        }

    def attention_profile(
        self,
        attentions: Sequence[torch.Tensor],
        token_map: dict[int, list[int]],
        answer: list[int],
        span_order: Sequence[int],
    ) -> dict[str, np.ndarray]:
        """Compute normalized answer-to-span attention distributions.

        Each group value is a vector ordered by ``span_order``. For a surviving
        span we average answer-to-span attention density across all answer tokens,
        heads, and layers in the group. Missing spans (the deletion target) are
        represented by zero and are handled by ``attention_transition_vector``.
        """
        result: dict[str, np.ndarray] = {}
        if not answer:
            return {
                name: np.zeros(len(span_order), dtype=np.float32)
                for name in ATTENTION_GROUPS
            }
        for group_name, layer_indices in self._transition_layer_groups().items():
            raw: list[float] = []
            for span_index in span_order:
                toks = token_map.get(int(span_index), [])
                if not toks:
                    raw.append(0.0)
                    continue
                layer_values: list[float] = []
                for layer_index in layer_indices:
                    attn = attentions[layer_index][0].detach().float()
                    sub = attn[:, answer, :][:, :, toks]
                    # Density avoids making longer spans automatically larger.
                    density = sub.sum(dim=-1).mean(dim=-1) / max(len(toks), 1)
                    layer_values.append(float(density.mean().cpu()))
                raw.append(float(np.mean(layer_values)) if layer_values else 0.0)
            result[group_name] = normalize_distribution(np.asarray(raw, dtype=np.float32))
        return result

    def analyze(
        self,
        source: str,
        answer_ids_cpu: torch.Tensor,
        spans: Sequence[Span],
        grad: bool,
        attn: bool,
        spectral: bool,
        profile_span_order: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
        p = self.encode_prompt(source)
        prompt_ids = p.prompt_ids.to(self.device)
        ans = answer_ids_cpu.to(self.device)
        full = torch.cat([prompt_ids, ans]).unsqueeze(0)
        mask = torch.ones_like(full)
        plen = len(prompt_ids)
        answer_idx = list(range(plen, plen + len(ans)))
        token_map = {span.index: self.span_tokens(p, span) for span in spans}
        self.model.zero_grad(set_to_none=True)
        need_attentions = bool(attn or spectral or profile_span_order is not None)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(
                input_ids=full,
                attention_mask=mask,
                output_attentions=need_attentions,
                output_hidden_states=grad,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[:, plen - 1: plen + len(ans) - 1, :]
            targets = ans.unsqueeze(0)
            lp = F.log_softmax(logits.float(), -1)
            selected = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            seq = selected.mean()
            probs = lp.exp()
            entropy = -(probs * lp).sum(-1).mean()
            if grad:
                for hidden_state in out.hidden_states:
                    hidden_state.retain_grad()
                (-seq).backward()

        span_features: dict[int, dict[str, np.ndarray]] = {}
        for span in spans:
            feats: dict[str, np.ndarray] = {}
            toks = token_map[span.index]
            if attn:
                feats["attention"] = self.attention_features(out.attentions, toks, answer_idx)
            if grad:
                feats["gradient"] = self.gradient_features(out.hidden_states, toks)
            if spectral:
                feats["spectral"] = self.spectral_features(
                    out.attentions, toks, answer_idx, plen
                )
            span_features[span.index] = feats

        profile = None
        if profile_span_order is not None:
            if out.attentions is None:
                raise RuntimeError("Attention profile requested but attentions are unavailable")
            profile = self.attention_profile(
                out.attentions,
                token_map,
                answer_idx,
                profile_span_order,
            )

        result = {
            "sequence_logprob": float(seq.detach().cpu()),
            "mean_token_entropy": float(entropy.detach().cpu()),
            "span_features": span_features,
            "attention_profile": profile,
            "prompt_tokens": plen,
            "answer_tokens": len(ans),
        }
        del out, full, mask
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

# -------------------------- extraction ------------------------------------

def change_metrics(
    original: str,
    regenerated: str,
    base: dict[str, Any],
    inter: dict[str, Any],
) -> dict[str, float]:
    sim = token_f1(original, regenerated)
    return {
        # Positive means the intervention reduced support for the original answer.
        "support_delta": float(base["sequence_logprob"]) - float(inter["sequence_logprob"]),
        "semantic_similarity": sim,
        "answer_changed": float(sim < 0.80),
        "polarity_changed": float(polarity_sig(original) != polarity_sig(regenerated)),
        "entity_changed": float(entity_set(original) != entity_set(regenerated))
            if entity_set(original) or entity_set(regenerated) else 0.0,
        "number_changed": float(
            set(_NUMBER_RE.findall(original.replace(',', ''))) !=
            set(_NUMBER_RE.findall(regenerated.replace(',', '')))
        ) if _NUMBER_RE.search(original + regenerated) else 0.0,
        "intervened_original_answer_logprob": float(inter["sequence_logprob"]),
        "entropy_delta": float(inter["mean_token_entropy"]) - float(base["mean_token_entropy"]),
    }


def process_item(
    ex: Example,
    engine: WhiteboxEngine,
    evaluator: CorrectnessEvaluator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_spans = segment_atomic(
        ex.source_text,
        args.min_clause_words,
        args.min_span_words,
    )
    if not all_spans:
        raise ValueError("No spans")

    original, answer_ids = engine.generate(ex.source_text)
    correct = evaluator.evaluate(original, ex.references, ex.question)

    attention_selection = engine.score_spans_by_answer_attention(
        ex.source_text,
        answer_ids,
        all_spans,
    )
    spans, attention_rank = select_topk_attention_spans(
        all_spans,
        attention_selection,
        args.max_intervention_spans,
    )
    if not spans:
        raise ValueError("No spans survived top-k attention selection")

    span_order = [span.index for span in spans]
    position_by_index = {span_index: pos for pos, span_index in enumerate(span_order)}
    base = engine.analyze(
        ex.source_text,
        answer_ids,
        spans,
        args.compute_gradient_features,
        True,
        args.compute_spectral_features,
        profile_span_order=span_order,
    )
    if base["attention_profile"] is None:
        raise RuntimeError("Base attention profile was not computed")

    span_rows: list[dict[str, Any]] = []
    for span in spans:
        base_features = base["span_features"][span.index]
        operator_rows: list[dict[str, Any]] = []
        for operator in OPERATORS:
            modified, aligned_spans = intervene_aligned(
                ex.source_text,
                spans,
                span,
                operator,
                args.mask_text,
                args.neutral_text,
            )
            regenerated, _ = engine.generate(modified)
            regenerated_correct = evaluator.evaluate(
                regenerated,
                ex.references,
                ex.question,
            )
            analysis = engine.analyze(
                modified,
                answer_ids,
                aligned_spans,
                grad=False,
                attn=False,
                spectral=False,
                profile_span_order=span_order,
            )
            metrics = change_metrics(original, regenerated, base, analysis)
            target_position = position_by_index[span.index]
            transition = {
                group: attention_transition_vector(
                    base["attention_profile"][group],
                    analysis["attention_profile"][group],
                    target_position,
                    operator,
                )
                for group in ATTENTION_GROUPS
            }
            metrics.update(
                operator=operator,
                regenerated_answer=regenerated,
                regenerated_correct=regenerated_correct,
                attention_transition=transition,
                intervened_attention_profile=analysis["attention_profile"],
            )
            operator_rows.append(metrics)

        family = {
            "structural": structural_features(span, ex.source_text),
            "attention": base_features["attention"],
            "gradient": base_features.get(
                "gradient", np.zeros(engine.gradient_dim, np.float32)
            ),
            "spectral": base_features.get(
                "spectral", np.zeros(engine.spectral_dim, np.float32)
            ),
        }
        selection_row = attention_selection[span.index]
        span_rows.append({
            "span_uid": f"{ex.item_id}::span::{span.index}",
            "span_index": span.index,
            "span_position": position_by_index[span.index],
            "span_text": span.text,
            "span_start": span.start,
            "span_end": span.end,
            "attention_selection_rank": int(attention_rank[span.index]),
            "attention_selection_score": float(selection_row["score"]),
            "attention_selection_mass": float(selection_row["mass"]),
            "attention_selection_density": float(selection_row["density"]),
            "attention_selection_peak": float(selection_row["peak"]),
            "attention_selection_token_count": int(selection_row["token_count"]),
            "family_features": family,
            "operators": operator_rows,
        })

    return {
        "item_id": ex.item_id,
        "raw_index": ex.raw_index,
        "source_text": ex.source_text,
        "question": ex.question,
        "references": ex.references,
        "generated_answer": original,
        "original_correct": correct,
        "hallucination_label": None if correct is None else int(not correct),
        "base_sequence_logprob": base["sequence_logprob"],
        "base_mean_token_entropy": base["mean_token_entropy"],
        "base_attention_profile": base["attention_profile"],
        "span_order": span_order,
        "prompt_tokens": base["prompt_tokens"],
        "answer_tokens": base["answer_tokens"],
        "n_candidate_spans": len(all_spans),
        "n_selected_spans": len(spans),
        "spans": span_rows,
        "error": None,
    }


def extraction_cache_signature(args: argparse.Namespace) -> str:
    fields = {
        "schema": CACHE_SCHEMA_VERSION,
        "model": args.model,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "system_prompt": args.system_prompt,
        "answer_instruction": args.answer_instruction,
        "correctness_mode": args.correctness_mode,
        "token_f1_threshold": args.token_f1_threshold,
        "min_clause_words": args.min_clause_words,
        "min_span_words": args.min_span_words,
        "max_intervention_spans": args.max_intervention_spans,
        "attention_selection_layers": args.attention_selection_layers,
        "attention_selection_last_n": args.attention_selection_last_n,
        "attention_selection_score": args.attention_selection_score,
        "attention_selection_density_weight": args.attention_selection_density_weight,
        "mask_text": args.mask_text,
        "neutral_text": args.neutral_text,
        "compute_gradient_features": args.compute_gradient_features,
        "compute_spectral_features": args.compute_spectral_features,
        "lap_topk": args.lap_topk,
        "spectral_anchor_tokens": args.spectral_anchor_tokens,
        "spectral_max_nodes": args.spectral_max_nodes,
        "transition_attention_groups": ATTENTION_GROUPS,
    }
    return stable_hash(json.dumps(fields, sort_keys=True, ensure_ascii=False))


def extract_all(
    examples: Sequence[Example],
    engine: WhiteboxEngine,
    evaluator: CorrectnessEvaluator,
    args: argparse.Namespace,
    outdir: Path,
) -> list[dict[str, Any]]:
    cache = outdir / "item_cache"
    cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    cache_signature = extraction_cache_signature(args)
    for index, example in enumerate(tqdm(examples, desc="Extracting")):
        path = cache / f"{index:06d}_{stable_hash(example.item_id)}_{cache_signature}.pt"
        if path.exists() and not args.overwrite_cache:
            try:
                cached = torch_load(path)
                if cached.get("error"):
                    warnings.warn(
                        f"Retrying cached error for {example.item_id}: "
                        f"{cached['error']}"
                    )
                else:
                    records.append(cached)
                    continue
            except Exception as exc:
                warnings.warn(f"Bad cache {path}: {exc}")
        try:
            record = process_item(example, engine, evaluator, args)
        except Exception as exc:
            traceback.print_exc()
            record = {
                "item_id": example.item_id,
                "raw_index": example.raw_index,
                "source_text": example.source_text,
                "question": example.question,
                "references": example.references,
                "generated_answer": "",
                "original_correct": None,
                "hallucination_label": None,
                "spans": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_torch_save(record, path)
        records.append(record)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


# ---------------- behavior / labels / role models -------------------------

def estimate_scale(records: Sequence[dict[str,Any]], train_ids: set[str], minimum: float) -> float:
    x=[]
    for item in records:
        if item["item_id"] in train_ids:
            for span in item["spans"]: x += [abs(float(r["support_delta"])) for r in span["operators"]]
    if not x: return minimum
    med=float(np.median(x)); mad=float(np.median(np.abs(np.asarray(x)-med)))
    return max(minimum,med+1.4826*mad,1e-4)


def behavior_vector(span: dict[str,Any], scale: float) -> np.ndarray:
    by={r["operator"]:r for r in span["operators"]}; vals=[]; ds=[]; sims=[]; changes=[]
    for op in OPERATORS:
        r=by[op]; d=float(r["support_delta"]); sim=float(r["semantic_similarity"]); ch=float(r["answer_changed"])
        vals += [d,math.tanh(d/scale),sim,ch,float(r["polarity_changed"]),float(r["entity_changed"]),float(r["number_changed"]),float(r["entropy_delta"])]
        ds.append(d); sims.append(sim); changes.append(ch)
    ds=np.asarray(ds); signs=np.sign(ds[np.abs(ds)>1e-8]); agreement=abs(float(signs.mean())) if len(signs) else 0.
    vals += [float(np.median(ds)),float(np.std(ds)),float(np.max(np.abs(ds))),agreement,float(np.mean(changes)),float(np.mean(sims))]
    return np.nan_to_num(np.asarray(vals,np.float32))


def usage(span: dict[str,Any], scale: float) -> float:
    ds=np.asarray([float(r["support_delta"]) for r in span["operators"]]); ch=np.mean([r["answer_changed"] for r in span["operators"]])
    contradiction=np.mean([max(r["polarity_changed"],r["entity_changed"],r["number_changed"]) for r in span["operators"]])
    return float(np.clip(np.median(np.abs(np.tanh(ds/scale)))+.35*ch+.20*contradiction,0,1))


def pseudo_role(item: dict[str,Any], span: dict[str,Any], scale: float) -> tuple[Optional[str],float,str]:
    orig=item["original_correct"]
    if orig is None: return None,0.,"original_refusal"
    rows=span["operators"]; cs=[r["regenerated_correct"] for r in rows if r["regenerated_correct"] is not None]
    ds=np.asarray([float(r["support_delta"]) for r in rows]); sims=np.asarray([float(r["semantic_similarity"]) for r in rows]); change=float(np.mean([r["answer_changed"] for r in rows]))
    med=float(np.median(ds)); medabs=float(np.median(np.abs(ds))); sim=float(np.median(sims))
    if not orig and any(x is True for x in cs): return "shortcut",min(1.,.85+.15*sum(x is True for x in cs)/max(len(cs),1)),"wrong_to_correct"
    if orig and any(x is False for x in cs): return "constraint",min(1.,.85+.15*sum(x is False for x in cs)/max(len(cs),1)),"correct_to_wrong"
    if medabs<=.25*scale and sim>=.90 and change<=1/3: return "irrelevant",.75,"stable_low_effect"
    if med>=.50*scale and change>=1/3: return ("constraint",.70,"supports_correct") if orig else ("shortcut",.70,"supports_wrong")
    if med<=-.50*scale and change>=1/3: return ("shortcut",.55,"suppresses_correct") if orig else ("constraint",.55,"suppresses_wrong")
    return None,0.,"ambiguous"


def safe_entropy(values: np.ndarray) -> float:
    x = np.abs(np.nan_to_num(np.asarray(values, dtype=np.float64)))
    total = float(x.sum())
    if total <= 1e-12:
        return 0.0
    p = x / total
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def matrix_svd_features(
    matrix: np.ndarray,
    singular_topk: int,
) -> tuple[np.ndarray, list[str]]:
    """Compact global features for one behavior-weighted transition matrix.

    Retain the five statistic families with the strongest mean absolute
    standardized coefficients in the 2,000-item training run: mean absolute
    entry, spectral entropy, Frobenius norm, positive mass, and largest
    singular value. Negative mass is omitted because normalized attention
    transition rows have matching positive and negative mass up to numerical
    error. The weaker or redundant concentration/local-peak statistics are
    omitted. ``singular_topk`` remains in the signature for CLI/cache
    compatibility, but the compact representation no longer expands with it.
    """
    g = np.nan_to_num(np.asarray(matrix, dtype=np.float64))
    if g.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {g.shape}")
    try:
        singular = np.linalg.svd(g, compute_uv=False)
    except np.linalg.LinAlgError:
        singular = np.zeros(min(g.shape), dtype=np.float64)
    singular_sum = float(singular.sum())
    singular_prob = singular / singular_sum if singular_sum > 1e-12 else np.zeros_like(singular)
    spectral_entropy = float(
        -(singular_prob[singular_prob > 0] * np.log(singular_prob[singular_prob > 0])).sum()
    ) if singular_sum > 1e-12 else 0.0
    positive_mass = float(np.clip(g, 0.0, None).sum())
    values = [
        float(np.linalg.norm(g, ord="fro")),
        float(singular[0]) if len(singular) else 0.0,
        spectral_entropy,
        positive_mass,
        float(np.mean(np.abs(g))) if g.size else 0.0,
    ]
    names = [
        "frobenius_norm",
        "largest_singular_value",
        "singular_entropy",
        "positive_mass",
        "mean_abs_entry",
    ]
    return np.nan_to_num(np.asarray(values, np.float32)), names


def row_features(row: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.nan_to_num(np.asarray(row, dtype=np.float64))
    absolute = np.abs(x)
    values = [
        float(absolute.sum()),
        float(np.linalg.norm(x)),
        float(np.clip(x, 0.0, None).sum()),
        float(np.clip(-x, 0.0, None).sum()),
        float(absolute.max()) if x.size else 0.0,
        float(absolute.mean()) if x.size else 0.0,
        safe_entropy(x),
    ]
    names = [
        "row_l1",
        "row_l2",
        "row_positive_mass",
        "row_negative_mass",
        "row_max_abs",
        "row_mean_abs",
        "row_entropy",
    ]
    return np.asarray(values, np.float32), names


def item_behavior_summary(item: dict[str, Any], scale: float) -> tuple[np.ndarray, list[str]]:
    deltas = np.asarray([
        float(operator["support_delta"])
        for span in item["spans"]
        for operator in span["operators"]
    ], dtype=np.float64)
    changed = np.asarray([
        float(operator["answer_changed"])
        for span in item["spans"]
        for operator in span["operators"]
    ], dtype=np.float64)
    normalized = np.tanh(deltas / max(scale, 1e-8)) if deltas.size else np.zeros(0)
    values = [
        float(np.mean(deltas)) if deltas.size else 0.0,
        float(np.median(deltas)) if deltas.size else 0.0,
        float(np.std(deltas)) if deltas.size else 0.0,
        float(np.max(deltas)) if deltas.size else 0.0,
        float(np.min(deltas)) if deltas.size else 0.0,
        float(np.max(np.abs(deltas))) if deltas.size else 0.0,
        float(np.mean(normalized)) if normalized.size else 0.0,
        float(np.max(normalized)) if normalized.size else 0.0,
        float(np.mean(deltas > 0)) if deltas.size else 0.0,
        float(np.mean(changed)) if changed.size else 0.0,
    ]
    names = [
        "support_delta_mean",
        "support_delta_median",
        "support_delta_std",
        "support_delta_max",
        "support_delta_min",
        "support_delta_max_abs",
        "normalized_support_mean",
        "normalized_support_max",
        "positive_support_fraction",
        "answer_change_rate",
    ]
    return np.asarray(values, np.float32), names


def item_static_attention_summary(item: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    fields = [
        "attention_selection_score",
        "attention_selection_mass",
        "attention_selection_density",
        "attention_selection_peak",
    ]
    values: list[float] = []
    names: list[str] = []
    for field in fields:
        arr = np.asarray([float(span.get(field, 0.0)) for span in item["spans"]])
        sorted_values = np.sort(arr)[::-1]
        gap = float(sorted_values[0] - sorted_values[1]) if len(sorted_values) > 1 else (
            float(sorted_values[0]) if len(sorted_values) else 0.0
        )
        values.extend([
            float(arr.mean()) if arr.size else 0.0,
            float(arr.std()) if arr.size else 0.0,
            float(arr.max()) if arr.size else 0.0,
            gap,
        ])
        names.extend([
            f"{field}_mean",
            f"{field}_std",
            f"{field}_max",
            f"{field}_top_gap",
        ])
    return np.asarray(values, np.float32), names


def build_transition_features(
    item: dict[str, Any],
    scale: float,
    singular_topk: int,
) -> None:
    """Build global matrix features and row-level span features in-place.

    For operator d and attention group g:
        T[d,g][i,j] = post_attention[j] - pre_attention[j]
        G[d,g]      = diag(tanh(support_delta_i / scale)) @ T[d,g]

    We also compute an operator-median matrix for each layer group. Global SVD
    features are attached to the item; row statistics and absolute leading-left-
    singular-vector loadings are attached to the corresponding span.
    """
    spans = item["spans"]
    n = len(spans)
    row_feature_parts: list[list[np.ndarray]] = [[] for _ in range(n)]
    row_feature_names: list[str] = []
    global_parts: list[np.ndarray] = []
    global_names: list[str] = []
    matrices: dict[str, Any] = {}

    for group in ATTENTION_GROUPS:
        group_matrices: list[np.ndarray] = []
        for operator in OPERATORS:
            transition = np.zeros((n, n), dtype=np.float32)
            weights = np.zeros(n, dtype=np.float32)
            for row_position, span in enumerate(spans):
                by_operator = {row["operator"]: row for row in span["operators"]}
                row = by_operator[operator]
                transition[row_position] = np.asarray(
                    row["attention_transition"][group], dtype=np.float32
                )
                weights[row_position] = float(
                    np.tanh(float(row["support_delta"]) / max(scale, 1e-8))
                )
            weighted = weights[:, None] * transition
            group_matrices.append(weighted)
            matrix_key = f"{group}::{operator}"
            matrices[matrix_key] = weighted
            feats, names = matrix_svd_features(weighted, singular_topk)
            qualified_names = [f"{matrix_key}::{name}" for name in names]
            keep = [
                index for index, name in enumerate(qualified_names)
                if name in TRANSITION_ITEM_FEATURE_NAMES
            ]
            if keep:
                global_parts.append(feats[keep])
                global_names.extend([qualified_names[index] for index in keep])
            for row_position in range(n):
                row_values, row_names = row_features(weighted[row_position])
                row_feature_parts[row_position].append(row_values)
                if not row_feature_names:
                    row_feature_names.extend(
                        [f"{matrix_key}::{name}" for name in row_names]
                    )
                elif row_position == 0:
                    row_feature_names.extend(
                        [f"{matrix_key}::{name}" for name in row_names]
                    )

        aggregate = np.median(np.stack(group_matrices, axis=0), axis=0)
        aggregate_key = f"{group}::operator_median"
        matrices[aggregate_key] = aggregate
        feats, names = matrix_svd_features(aggregate, singular_topk)
        qualified_names = [f"{aggregate_key}::{name}" for name in names]
        keep = [
            index for index, name in enumerate(qualified_names)
            if name in TRANSITION_ITEM_FEATURE_NAMES
        ]
        if keep:
            global_parts.append(feats[keep])
            global_names.extend([qualified_names[index] for index in keep])

        try:
            left, _, _ = np.linalg.svd(aggregate, full_matrices=False)
            loading = np.abs(left[:, 0]) if left.size else np.zeros(n)
        except np.linalg.LinAlgError:
            loading = np.zeros(n)
        for row_position in range(n):
            row_values, row_names = row_features(aggregate[row_position])
            row_feature_parts[row_position].append(row_values)
            row_feature_parts[row_position].append(
                np.asarray([float(loading[row_position])], dtype=np.float32)
            )
            if row_position == 0:
                row_feature_names.extend(
                    [f"{aggregate_key}::{name}" for name in row_names]
                )
                row_feature_names.append(f"{aggregate_key}::leading_left_loading_abs")

    item["transition_features"] = np.concatenate(global_parts).astype(np.float32)
    item["transition_feature_names"] = global_names
    item["transition_matrices"] = matrices
    behavior_values, behavior_names = item_behavior_summary(item, scale)
    attention_values, attention_names = item_static_attention_summary(item)
    item["item_behavior_features"] = behavior_values
    item["item_behavior_feature_names"] = behavior_names
    item["item_static_attention_features"] = attention_values
    item["item_static_attention_feature_names"] = attention_names

    for row_position, span in enumerate(spans):
        transition_vector = np.concatenate(row_feature_parts[row_position]).astype(np.float32)
        span["family_features"]["transition"] = transition_vector
        span["transition_feature_names"] = row_feature_names


def attach_derived(
    records: Sequence[dict[str, Any]],
    scale: float,
    train_ids: set[str],
    singular_topk: int,
) -> dict[str, int]:
    counts = {name: 0 for name in ROLE_NAMES}
    counts["ambiguous"] = 0
    for item in records:
        build_transition_features(item, scale, singular_topk)
        for span in item["spans"]:
            span["family_features"]["behavior"] = behavior_vector(span, scale)
            span["usage"] = usage(span, scale)
            role, reliability, reason = pseudo_role(item, span, scale) \
                if item["item_id"] in train_ids else (None, 0.0, "test_unlabeled")
            span["pseudo_role"] = role
            span["role_reliability"] = reliability
            span["pseudo_role_reason"] = reason
            counts[role if role is not None else "ambiguous"] += 1
    return counts

def span_vector(span: dict[str,Any], feature_set: str) -> np.ndarray:
    return np.nan_to_num(np.concatenate([np.asarray(span["family_features"][f],np.float32).ravel() for f in FEATURE_SETS[feature_set]]),nan=0.,posinf=1e6,neginf=-1e6)


def spans_for(records: dict[str,dict[str,Any]], ids: Iterable[str]) -> list[dict[str,Any]]:
    return [s for iid in ids for s in records[iid]["spans"]]


def labeled_arrays(records: dict[str,dict[str,Any]], ids: Iterable[str], feature_set: str):
    ss=[s for s in spans_for(records,ids) if s["pseudo_role"] is not None]
    if not ss: raise RuntimeError("No pseudo-labeled spans")
    X=np.stack([span_vector(s,feature_set) for s in ss]); y=np.asarray([ROLE_TO_ID[s["pseudo_role"]] for s in ss]); w=np.asarray([s["role_reliability"] for s in ss])
    if len(np.unique(y))<2: raise RuntimeError("Need at least two role classes")
    return X,y,w,ss


def fit_role(X: np.ndarray,y: np.ndarray,w: np.ndarray,pca_dim: int,seed: int) -> dict[str,Any]:
    scaler=StandardScaler(); xs=scaler.fit_transform(X); n=min(pca_dim,xs.shape[1],max(1,xs.shape[0]-1))
    pca=PCA(n_components=n,svd_solver="randomized",random_state=seed) if xs.shape[1]>n and n>=2 else None
    xm=pca.fit_transform(xs) if pca is not None else xs
    clf=LogisticRegression(max_iter=3000,class_weight="balanced",solver="lbfgs",random_state=seed).fit(xm,y,sample_weight=w)
    return {"scaler":scaler,"pca":pca,"classifier":clf}


def predict_role(model: dict[str,Any],X: np.ndarray) -> np.ndarray:
    x=model["scaler"].transform(X); x=model["pca"].transform(x) if model["pca"] is not None else x
    part=model["classifier"].predict_proba(x); full=np.zeros((len(X),3))
    for j,c in enumerate(model["classifier"].classes_): full[:,int(c)]=part[:,j]
    z=full.sum(1,keepdims=True); empty=z[:,0]<=0; full[empty]=1/3; full[~empty]/=z[~empty]
    return full


def role_metrics(y: Sequence[int], probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    prediction = probabilities.argmax(1)
    output: dict[str, Any] = {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "class_counts": {
            ROLE_NAMES[index]: int((y == index).sum()) for index in range(3)
        },
        "per_role": {},
    }
    aucs: list[float] = []
    auprcs: list[float] = []
    for index, role_name in enumerate(ROLE_NAMES):
        binary = (y == index).astype(int)
        if len(np.unique(binary)) < 2:
            auroc = None
            auprc = None
        else:
            auroc = float(roc_auc_score(binary, probabilities[:, index]))
            auprc = float(average_precision_score(binary, probabilities[:, index]))
            aucs.append(auroc)
            auprcs.append(auprc)
        output["per_role"][role_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": float(
                precision_score(binary, prediction == index, zero_division=0)
            ),
            "recall": float(
                recall_score(binary, prediction == index, zero_division=0)
            ),
            "f1": float(f1_score(binary, prediction == index, zero_division=0)),
        }
    output["macro_ovr_auroc"] = float(np.mean(aucs)) if aucs else None
    output["macro_ovr_auprc"] = float(np.mean(auprcs)) if auprcs else None
    return output


def oof_roles(
    records: dict[str, dict[str, Any]],
    train_ids: list[str],
    labels: np.ndarray,
    feature_set: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    minority = int(np.bincount(labels).min())
    folds = min(args.cv_folds, minority)
    if folds < 2:
        raise RuntimeError("Need at least two items in each class for role OOF prediction")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=args.seed)
    ids = np.asarray(train_ids)
    all_probabilities: dict[str, np.ndarray] = {}
    true_roles: list[int] = []
    predicted_probabilities: list[np.ndarray] = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(ids, labels)):
        X, y, weights, _ = labeled_arrays(
            records,
            ids[train_index].tolist(),
            feature_set,
        )
        model = fit_role(X, y, weights, args.pca_dim, args.seed + fold)
        validation_spans = spans_for(records, ids[validation_index].tolist())
        validation_X = np.stack([
            span_vector(span, feature_set) for span in validation_spans
        ])
        probabilities = predict_role(model, validation_X)
        for span, probability in zip(validation_spans, probabilities):
            all_probabilities[span["span_uid"]] = probability
            if span["pseudo_role"] is not None:
                true_roles.append(ROLE_TO_ID[span["pseudo_role"]])
                predicted_probabilities.append(probability)
    if not predicted_probabilities:
        raise RuntimeError("No pseudo-labeled validation spans were available for role OOF metrics")
    return all_probabilities, role_metrics(
        true_roles,
        np.stack(predicted_probabilities),
    )


# ---------------- item models ---------------------------------------------

def aggregate_role_evidence(
    item: dict[str, Any],
    probabilities: dict[str, np.ndarray],
    topk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    shortcut_contributions: list[float] = []
    constraint_contributions: list[float] = []
    usages: list[float] = []
    details: list[dict[str, Any]] = []
    for span in item["spans"]:
        probability = probabilities[span["span_uid"]]
        span_usage = float(span["usage"])
        shortcut = span_usage * float(probability[ROLE_TO_ID["shortcut"]])
        constraint = span_usage * float(probability[ROLE_TO_ID["constraint"]])
        usages.append(span_usage)
        shortcut_contributions.append(shortcut)
        constraint_contributions.append(constraint)
        details.append({
            "span_uid": span["span_uid"],
            "span_index": span["span_index"],
            "span_text": span["span_text"],
            "usage": span_usage,
            "role_probabilities": {
                ROLE_NAMES[index]: float(probability[index]) for index in range(3)
            },
            "shortcut_contribution": shortcut,
            "constraint_contribution": constraint,
            "transition_features": span.get("family_features", {}).get("transition"),
            "operators": span.get("operators", []),
        })
    if not usages:
        vector = np.zeros(8, dtype=np.float32)
    else:
        denominator = float(np.sum(usages)) + 1e-8
        shortcut_sorted = sorted(shortcut_contributions, reverse=True)
        constraint_sorted = sorted(constraint_contributions, reverse=True)
        k = min(max(1, topk), len(shortcut_sorted))
        shortcut_probs = [
            float(probabilities[span["span_uid"]][ROLE_TO_ID["shortcut"]])
            for span in item["spans"]
        ]
        vector = np.asarray([
            np.sum(shortcut_contributions) / denominator,
            np.sum(constraint_contributions) / denominator,
            max(shortcut_contributions),
            np.mean(shortcut_sorted[:k]),
            max(shortcut_probs),
            np.sum(shortcut_probs),
            max(shortcut_contributions) - max(constraint_contributions),
            np.mean(shortcut_sorted[:k]) - np.mean(constraint_sorted[:k]),
        ], dtype=np.float32)
    return vector, {
        "shortcut_evidence": float(vector[0]),
        "constraint_evidence": float(vector[1]),
        "max_shortcut_contribution": float(vector[2]),
        "topk_shortcut_contribution": float(vector[3]),
        "max_shortcut_probability": float(vector[4]),
        "sum_shortcut_probability": float(vector[5]),
        "shortcut_minus_constraint_top1": float(vector[6]),
        "shortcut_minus_constraint_topk": float(vector[7]),
        "spans": details,
    }


ROLE_EVIDENCE_NAMES = [
    "shortcut_evidence_mean",
    "constraint_evidence_mean",
    "max_shortcut_contribution",
    "topk_shortcut_contribution",
    "max_shortcut_probability",
    "sum_shortcut_probability",
    "shortcut_minus_constraint_top1",
    "shortcut_minus_constraint_topk",
]


def item_vector(
    item: dict[str, Any],
    role_probabilities: Optional[dict[str, np.ndarray]],
    item_mode: str,
    topk: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if item_mode not in ITEM_MODES:
        raise ValueError(f"Unknown item mode: {item_mode}")
    parts: list[np.ndarray] = []
    names: list[str] = []
    details: dict[str, Any] = {}

    if item_mode in {"transition_only", "transition_plus_role"}:
        parts.extend([
            np.asarray(item["item_behavior_features"], dtype=np.float32),
            np.asarray(item["item_static_attention_features"], dtype=np.float32),
            np.asarray(item["transition_features"], dtype=np.float32),
        ])
        names.extend(item["item_behavior_feature_names"])
        names.extend(item["item_static_attention_feature_names"])
        names.extend(item["transition_feature_names"])

    if item_mode in {"role_only", "transition_plus_role"}:
        if role_probabilities is None:
            raise RuntimeError(f"{item_mode} requires role probabilities")
        role_vector, role_details = aggregate_role_evidence(
            item, role_probabilities, topk
        )
        parts.append(role_vector)
        names.extend(ROLE_EVIDENCE_NAMES)
        details.update(role_details)
    else:
        details.update({
            "shortcut_evidence": 0.0,
            "constraint_evidence": 0.0,
            "max_shortcut_contribution": 0.0,
            "topk_shortcut_contribution": 0.0,
            "max_shortcut_probability": 0.0,
            "sum_shortcut_probability": 0.0,
            "shortcut_minus_constraint_top1": 0.0,
            "shortcut_minus_constraint_topk": 0.0,
            "spans": [],
        })

    if not parts:
        raise RuntimeError("No item features were constructed")
    return np.nan_to_num(np.concatenate(parts).astype(np.float32)), names, details


def fit_binary_logistic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    regularization_c: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        C=regularization_c,
        random_state=seed,
    )
    classifier.fit(transformed, y)
    return {"scaler": scaler, "classifier": classifier}


def predict_binary_logistic(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    transformed = model["scaler"].transform(X)
    return model["classifier"].predict_proba(transformed)[:, 1]


def item_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    minority = int(np.bincount(y).min())
    folds = min(args.cv_folds, minority)
    if folds < 2:
        raise RuntimeError("Need at least two examples in each item class for OOF prediction")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=args.seed)
    predictions = np.zeros(len(y), dtype=np.float64)
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y)):
        model = fit_binary_logistic(
            X[train_index],
            y[train_index],
            args.seed + fold,
            args.item_logistic_c,
        )
        predictions[validation_index] = predict_binary_logistic(
            model, X[validation_index]
        )
    return predictions


def threshold_f1(y: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    output = {
        "n": len(y),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    output["auroc"] = float(roc_auc_score(y, probabilities)) \
        if len(np.unique(y)) > 1 else None
    output["auprc"] = float(average_precision_score(y, probabilities)) \
        if len(np.unique(y)) > 1 else None
    return output


def logistic_parameters(
    model: dict[str, Any],
    feature_names: Sequence[str],
    topn: int = 20,
) -> dict[str, Any]:
    coefficients = model["classifier"].coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:topn]
    return {
        "intercept": float(model["classifier"].intercept_[0]),
        "regularization_c": float(model["classifier"].C),
        "top_absolute_standardized_coefficients": [
            {
                "feature": feature_names[int(index)],
                "coefficient": float(coefficients[int(index)]),
            }
            for index in order
        ],
    }


# ---------------- evaluation / outputs ------------------------------------

def explain_stats(
    y: np.ndarray,
    shortcut: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    positive = shortcut[y == 1]
    negative = shortcut[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return {"available": False}
    rng = np.random.default_rng(seed)
    difference = float(positive.mean() - negative.mean())

    def bootstrap(values: np.ndarray) -> list[float]:
        estimates = [
            rng.choice(values, len(values), replace=True).mean() for _ in range(draws)
        ]
        return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]

    difference_bootstrap = [
        rng.choice(positive, len(positive), replace=True).mean()
        - rng.choice(negative, len(negative), replace=True).mean()
        for _ in range(draws)
    ]
    permutation_count = 0
    for _ in range(draws):
        permutation = rng.permutation(y)
        permuted_difference = (
            shortcut[permutation == 1].mean() - shortcut[permutation == 0].mean()
        )
        permutation_count += abs(permuted_difference) >= abs(difference)
    pooled = (
        (len(positive) - 1) * positive.var(ddof=1)
        + (len(negative) - 1) * negative.var(ddof=1)
    ) / max(len(positive) + len(negative) - 2, 1)
    cohens_d = difference / math.sqrt(max(float(pooled), 1e-12))
    order = np.argsort(shortcut)
    dose = []
    for quartile, group in enumerate(np.array_split(order, 4), 1):
        dose.append({
            "quartile": quartile,
            "n": len(group),
            "shortcut_evidence_min": float(shortcut[group].min()),
            "shortcut_evidence_max": float(shortcut[group].max()),
            "hallucination_rate": float(y[group].mean()),
        })
    return {
        "available": True,
        "shortcut_evidence_by_outcome": {
            "hallucination": {
                "n": len(positive),
                "mean": float(positive.mean()),
                "median": float(np.median(positive)),
                "std": float(positive.std()),
                "bootstrap_mean_95_ci": bootstrap(positive),
            },
            "correct": {
                "n": len(negative),
                "mean": float(negative.mean()),
                "median": float(np.median(negative)),
                "std": float(negative.std()),
                "bootstrap_mean_95_ci": bootstrap(negative),
            },
            "mean_difference_hallucination_minus_correct": difference,
            "difference_bootstrap_95_ci": [
                float(np.quantile(difference_bootstrap, 0.025)),
                float(np.quantile(difference_bootstrap, 0.975)),
            ],
            "label_permutation_p_value": float(
                (permutation_count + 1) / (draws + 1)
            ),
            "cohens_d": float(cohens_d),
            "shortcut_evidence_auroc": float(roc_auc_score(y, shortcut)),
            "shortcut_evidence_auprc": float(average_precision_score(y, shortcut)),
        },
        "shortcut_evidence_dose_response": dose,
    }


def evaluate_configuration(
    configuration_name: str,
    span_feature_set: str,
    item_mode: str,
    records: dict[str, dict[str, Any]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    needs_role = item_mode in {"role_only", "transition_plus_role"}
    role_oof_metrics = None
    role_model = None
    oof_role_probabilities: Optional[dict[str, np.ndarray]] = None
    test_role_probabilities: Optional[dict[str, np.ndarray]] = None

    if needs_role:
        oof_role_probabilities, role_oof_metrics = oof_roles(
            records, train_ids, y_train, span_feature_set, args
        )
        role_X, role_y, role_w, _ = labeled_arrays(
            records, train_ids, span_feature_set
        )
        role_model = fit_role(
            role_X, role_y, role_w, args.pca_dim, args.seed
        )
        test_spans = spans_for(records, test_ids)
        test_X = np.stack([
            span_vector(span, span_feature_set) for span in test_spans
        ])
        test_probabilities = predict_role(role_model, test_X)
        test_role_probabilities = {
            span["span_uid"]: probability
            for span, probability in zip(test_spans, test_probabilities)
        }
    else:
        role_X = np.zeros((0, 0), dtype=np.float32)

    train_vectors: list[np.ndarray] = []
    train_details: dict[str, Any] = {}
    feature_names: Optional[list[str]] = None
    for item_id in train_ids:
        vector, names, details = item_vector(
            records[item_id],
            oof_role_probabilities,
            item_mode,
            args.item_top_k,
        )
        train_vectors.append(vector)
        train_details[item_id] = details
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Item feature names changed across training items")
    X_train = np.stack(train_vectors)
    assert feature_names is not None

    train_oof_probability = item_oof_predictions(X_train, y_train, args)
    threshold = threshold_f1(y_train, train_oof_probability)
    item_model = fit_binary_logistic(
        X_train,
        y_train,
        args.seed,
        args.item_logistic_c,
    )
    train_full_probability = predict_binary_logistic(item_model, X_train)

    test_vectors: list[np.ndarray] = []
    test_details: dict[str, Any] = {}
    for item_id in test_ids:
        vector, names, details = item_vector(
            records[item_id],
            test_role_probabilities,
            item_mode,
            args.item_top_k,
        )
        if names != feature_names:
            raise RuntimeError("Item feature names changed between train and test")
        test_vectors.append(vector)
        test_details[item_id] = details
    X_test = np.stack(test_vectors)
    test_probability = predict_binary_logistic(item_model, X_test)

    pca = role_model["pca"] if role_model is not None else None
    return {
        "configuration": configuration_name,
        "span_feature_set": span_feature_set,
        "span_feature_families": list(FEATURE_SETS[span_feature_set]),
        "item_mode": item_mode,
        "item_feature_count": int(X_train.shape[1]),
        "item_feature_names": feature_names,
        "role_head": None if role_model is None else {
            "n_input_features": int(role_X.shape[1]),
            "n_training_spans": int(len(role_X)),
            "classifier_classes": [
                int(value) for value in role_model["classifier"].classes_
            ],
            "pca_dim": None if pca is None else int(pca.n_components_),
            "pca_explained_variance_ratio_sum": None if pca is None else float(
                pca.explained_variance_ratio_.sum()
            ),
        },
        "span_role_train_oof": role_oof_metrics,
        "selected_threshold_from_train_oof": threshold,
        "item_logistic": logistic_parameters(item_model, feature_names),
        "item_metrics": {
            "train_oof": binary_metrics(y_train, train_oof_probability, threshold),
            "train_full": binary_metrics(y_train, train_full_probability, threshold),
            "test": binary_metrics(y_test, test_probability, threshold),
        },
        "_artifacts": {
            "role_model": role_model,
            "item_model": item_model,
            "threshold": threshold,
            "feature_names": feature_names,
            "span_feature_set": span_feature_set,
            "item_mode": item_mode,
        },
        "_predictions": {
            "test_probability": test_probability,
            "test_details": test_details,
        },
    }


def parse_configuration(specification: str) -> tuple[str, str, str]:
    """Parse NAME:SPAN_FEATURE_SET:ITEM_MODE."""
    parts = [part.strip() for part in specification.split(":")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Each configuration must be NAME:SPAN_FEATURE_SET:ITEM_MODE"
        )
    name, span_feature_set, item_mode = parts
    if span_feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown span feature set: {span_feature_set}")
    if item_mode not in ITEM_MODES:
        raise ValueError(f"Unknown item mode: {item_mode}")
    return name, span_feature_set, item_mode


def base_row(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "item_id", "raw_index", "question", "generated_answer", "references",
        "original_correct", "hallucination_label", "base_sequence_logprob",
        "base_mean_token_entropy", "prompt_tokens", "answer_tokens",
        "n_candidate_spans", "n_selected_spans", "error",
    )
    return {key: item.get(key) for key in keys} | {
        "n_spans": len(item.get("spans", [])),
        "base_attention_profile": item.get("base_attention_profile"),
        "transition_feature_names": item.get("transition_feature_names"),
        "transition_features": item.get("transition_features"),
    }


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "run_config.json", vars(args))

    rows = load_rows(args)
    examples = build_examples(rows, args)
    engine = WhiteboxEngine(args)
    evaluator = CorrectnessEvaluator(
        args.correctness_mode,
        args.token_f1_threshold,
        engine,
    )
    records = extract_all(examples, engine, evaluator, args, outdir)
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    failed = [item for item in records if item.get("error")]
    valid = [
        item for item in records
        if not item.get("error")
        and item.get("hallucination_label") is not None
        and item.get("spans")
    ]
    if len(valid) < 20:
        raise RuntimeError(f"Only {len(valid)} valid items")
    labels = np.asarray([item["hallucination_label"] for item in valid], dtype=int)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Need correct and hallucinated outputs")

    ids = np.asarray([item["item_id"] for item in valid])
    train_array, test_array, y_train, y_test = train_test_split(
        ids,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    train_ids = train_array.tolist()
    test_ids = test_array.tolist()
    train_set = set(train_ids)
    scale = estimate_scale(valid, train_set, args.minimum_support_scale)
    role_counts = attach_derived(
        valid,
        scale,
        train_set,
        args.transition_singular_topk,
    )
    by_id = {item["item_id"]: item for item in valid}

    base_path = outdir / "base_open_features.jsonl"
    intervention_path = outdir / "intervention_open_features.jsonl"
    matrix_path = outdir / "transition_matrices.jsonl"
    for path in (base_path, intervention_path, matrix_path):
        if path.exists():
            path.unlink()
    for item in records:
        append_jsonl(base_path, base_row(item))
        if item.get("error"):
            continue
        append_jsonl(matrix_path, {
            "item_id": item["item_id"],
            "span_order": item.get("span_order"),
            "transition_feature_names": item.get("transition_feature_names"),
            "transition_features": item.get("transition_features"),
            "transition_matrices": item.get("transition_matrices"),
        })
        for span in item.get("spans", []):
            append_jsonl(intervention_path, {
                "item_id": item["item_id"],
                "span_uid": span["span_uid"],
                "span_index": span["span_index"],
                "span_position": span.get("span_position"),
                "span_text": span["span_text"],
                "attention_selection_rank": span.get("attention_selection_rank"),
                "attention_selection_score": span.get("attention_selection_score"),
                "attention_selection_mass": span.get("attention_selection_mass"),
                "attention_selection_density": span.get("attention_selection_density"),
                "attention_selection_peak": span.get("attention_selection_peak"),
                "attention_selection_token_count": span.get("attention_selection_token_count"),
                "usage": span.get("usage"),
                "pseudo_role": span.get("pseudo_role"),
                "role_reliability": span.get("role_reliability"),
                "pseudo_role_reason": span.get("pseudo_role_reason"),
                "transition_feature_names": span.get("transition_feature_names"),
                "transition_features": span.get("family_features", {}).get("transition"),
                "operators": span["operators"],
            })

    configurations = [
        parse_configuration(specification)
        for specification in args.configurations.split(",")
        if specification.strip()
    ]
    if not configurations:
        raise ValueError("No configurations requested")
    configuration_names = [name for name, _, _ in configurations]
    if args.primary_configuration not in configuration_names:
        raise ValueError(
            f"Primary configuration {args.primary_configuration!r} is not in --configurations"
        )

    results: dict[str, Any] = {}
    bundle_models: dict[str, Any] = {}
    for name, span_feature_set, item_mode in configurations:
        print(f"\n=== {name} ===", flush=True)
        result = evaluate_configuration(
            name,
            span_feature_set,
            item_mode,
            by_id,
            train_ids,
            test_ids,
            y_train,
            y_test,
            args,
        )
        bundle_models[name] = result.pop("_artifacts")
        results[name] = result

    primary_private = results[args.primary_configuration].pop("_predictions")
    for name, result in results.items():
        if name != args.primary_configuration:
            result.pop("_predictions", None)

    test_probability = np.asarray(primary_private["test_probability"])
    test_details = primary_private["test_details"]
    threshold = float(
        results[args.primary_configuration]["selected_threshold_from_train_oof"]
    )
    prediction_path = outdir / "predictions.jsonl"
    if prediction_path.exists():
        prediction_path.unlink()
    for item_id, label, probability in zip(test_ids, y_test, test_probability):
        item = by_id[item_id]
        append_jsonl(prediction_path, {
            "item_id": item_id,
            "question": item["question"],
            "generated_answer": item["generated_answer"],
            "references": item["references"],
            "hallucination_label": int(label),
            "hallucination_probability": float(probability),
            "predicted_hallucination": bool(probability >= threshold),
            "threshold": threshold,
            "configuration": args.primary_configuration,
            **test_details[item_id],
        })

    shortcut = np.asarray([
        test_details[item_id].get("shortcut_evidence", 0.0)
        for item_id in test_ids
    ])
    item_ranking = sorted(
        configuration_names,
        key=lambda name: results[name]["item_metrics"]["test"]["auroc"]
            if results[name]["item_metrics"]["test"]["auroc"] is not None else -1,
        reverse=True,
    )
    bundle_path = outdir / "openended_v11_bundle.joblib"
    summary = {
        "method": "open-ended attention-behavior transition detector v11",
        "model": args.model,
        "data": args.input or args.hf_dataset,
        "n_input": len(rows),
        "n_examples": len(examples),
        "n_extracted": len(records),
        "n_failed": len(failed),
        "n_refusal_or_unlabeled": len(records) - len(failed) - len(valid),
        "n_valid": len(valid),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "support_scale_from_train": scale,
        "interventions_used_for_prediction": list(OPERATORS),
        "attention_transition": {
            "groups": list(ATTENTION_GROUPS),
            "definition": "T[i,j]=post_attention_to_span_j-pre_attention_to_span_j",
            "delete_alignment": "exclude deleted coordinate and renormalize survivors",
            "behavior_weight": "tanh(support_delta_i / train_support_scale)",
            "weighted_matrix": "G=diag(behavior_weight)@T",
            "global_features": [
                "frobenius_norm",
                "largest_singular_value",
                "singular_entropy",
                "positive_mass",
                "mean_abs_entry",
            ],
            "global_feature_selection": (
                "top 16 individual features by mean absolute standardized "
                "coefficient across both shortcut variants in the prior "
                "2,000-item training run; held-out labels were not used"
            ),
            "selected_global_feature_names": list(
                TRANSITION_ITEM_FEATURE_NAMES
            ),
            "global_features_per_matrix": 5,
            "global_transition_feature_count": len(
                TRANSITION_ITEM_FEATURE_NAMES
            ),
            "span_features": "weighted row norms/mass/entropy plus leading-left loading",
            "legacy_singular_topk_argument": args.transition_singular_topk,
        },
        "span_selection": {
            "method": "top_k_answer_to_span_attention",
            "max_intervention_spans": args.max_intervention_spans,
            "control_spans_used": False,
            "layers": args.attention_selection_layers,
            "last_n": args.attention_selection_last_n
                if args.attention_selection_layers == "last_n" else None,
            "score": args.attention_selection_score,
            "hybrid_density_weight": args.attention_selection_density_weight
                if args.attention_selection_score == "hybrid" else None,
        },
        "test_prediction_uses_reference_features": False,
        "references_used_for_train_pseudo_roles": True,
        "references_used_for_final_evaluation": True,
        "primary_configuration": args.primary_configuration,
        "configurations_evaluated": configuration_names,
        "configuration_comparison": results,
        "configuration_item_test_auroc_ranking": item_ranking,
        "role_pseudo_label_counts": role_counts,
        "primary_metrics": results[args.primary_configuration]["item_metrics"],
        "primary_item_logistic": results[args.primary_configuration]["item_logistic"],
        "primary_selected_threshold_from_train_oof": threshold,
        "shortcut_explanatory_statistics_primary": explain_stats(
            y_test, shortcut, args.seed, args.bootstrap_draws
        ),
        "files": {
            "base_open_features": str(base_path),
            "intervention_open_features": str(intervention_path),
            "transition_matrices": str(matrix_path),
            "predictions": str(prediction_path),
            "model_bundle": str(bundle_path),
            "item_cache": str(outdir / "item_cache"),
        },
        "failed_items": [
            {"item_id": item["item_id"], "error": item["error"]}
            for item in failed
        ],
        "method_notes": {
            "open_ended_generation": True,
            "teacher_forcing_target": "original generated answer",
            "behavior_support_definition": "mean token logP(original answer|base)-mean token logP(original answer|intervention)",
            "transition_classifier": "standardized logistic regression",
            "shortcut_auxiliary_path": "predicted span-role probabilities are pooled into item features",
            "reference_leakage": "references are never detector inputs at test time",
            "attention_causality_warning": "the intervention is causal; attention transition is an internal-state observation, not itself proof of causality",
            "proposal_bias_warning": "top-k attention span proposal favors attention-based feature families",
        },
    }
    joblib.dump({
        "args": vars(args),
        "support_scale": scale,
        "configuration_models": bundle_models,
        "role_names": ROLE_NAMES,
        "feature_sets": FEATURE_SETS,
        "attention_groups": ATTENTION_GROUPS,
    }, bundle_path, compress=3)
    write_json(outdir / "summary.json", summary)
    print(
        "\nPrimary test metrics:\n"
        + json.dumps(summary["primary_metrics"]["test"], indent=2)
    )
    print(f"Outputs: {outdir}")


# ---------------- CLI ------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input")
    p.add_argument("--hf-dataset")
    p.add_argument("--hf-subset")
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--question-field")
    p.add_argument("--context-field")
    p.add_argument("--answers-field")
    p.add_argument("--prompt-field")
    p.add_argument("--id-field")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.25)

    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    p.add_argument("--answer-instruction", default="Provide a concise final answer.")

    p.add_argument(
        "--correctness-mode",
        choices=("hybrid", "exact", "token_f1", "numeric", "llm_judge"),
        default="hybrid",
    )
    p.add_argument("--token-f1-threshold", type=float, default=0.8)
    p.add_argument("--min-clause-words", type=int, default=12)
    p.add_argument("--min-span-words", type=int, default=2)
    p.add_argument(
        "--max-intervention-spans",
        type=int,
        default=4,
        help="Top-k answer-attention spans to intervene; 0 means all spans",
    )
    p.add_argument(
        "--attention-selection-layers",
        choices=("all", "last_half", "last_quarter", "last_n"),
        default="last_quarter",
    )
    p.add_argument("--attention-selection-last-n", type=int, default=4)
    p.add_argument(
        "--attention-selection-score",
        choices=("hybrid", "density", "mass", "peak"),
        default="hybrid",
    )
    p.add_argument("--attention-selection-density-weight", type=float, default=0.7)
    p.add_argument("--mask-text", default="[MASKED INFORMATION]")
    p.add_argument("--neutral-text", default="This detail is unspecified.")
    p.add_argument("--minimum-support-scale", type=float, default=0.05)

    # v11 defaults to attention+behavior. Gradient/spectral are optional ablations.
    p.add_argument(
        "--compute-gradient-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--compute-spectral-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--lap-topk", type=int, default=10)
    p.add_argument("--spectral-anchor-tokens", type=int, default=8)
    p.add_argument("--spectral-max-nodes", type=int, default=64)
    p.add_argument(
        "--transition-singular-topk",
        type=int,
        default=5,
        help="Number of singular values retained per transition matrix",
    )

    p.add_argument(
        "--configurations",
        default=(
            "v8_attention_behavior:behavior_attention:role_only,"
            "v11_transition_direct:behavior_attention:transition_only,"
            "v11_transition_v8shortcut:behavior_attention:transition_plus_role,"
            "v11_transition_shortcut:attention_behavior_transition:transition_plus_role"
        ),
        help="Comma-separated NAME:SPAN_FEATURE_SET:ITEM_MODE specifications",
    )
    p.add_argument(
        "--primary-configuration",
        default="v11_transition_shortcut",
    )
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--item-top-k", type=int, default=3)
    p.add_argument("--item-logistic-c", type=float, default=1.0)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--overwrite-cache", action="store_true")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    if bool(args.input) == bool(args.hf_dataset):
        p.error("Provide exactly one of --input or --hf-dataset")
    if not 0.0 <= args.attention_selection_density_weight <= 1.0:
        p.error("--attention-selection-density-weight must be in [0, 1]")
    if args.transition_singular_topk < 1:
        p.error("--transition-singular-topk must be at least 1")
    if args.item_logistic_c <= 0:
        p.error("--item-logistic-c must be positive")
    run(args)


if __name__ == "__main__":
    main()
