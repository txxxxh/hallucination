#!/usr/bin/env python3
"""
KeyShift v10: HaluEval open-answer semantic counterfactuals and causal head tracing.

Major changes from the forced-choice v9 design
------------------------------------------------
1. Native HaluEval QA format:
       knowledge, question, right_answer, hallucinated_answer
   Aliases used by common mirrors are also accepted.
2. The model receives an open QA prompt with no visible answer options.
3. The primary observable is the length-normalized teacher-forced sequence
   margin
       mean log P(right_answer | prompt)
       - mean log P(hallucinated_answer | prompt).
4. Shortcut/constraint spans can come from frozen v7 predictions. If they are
   absent, an optional sequence-margin occlusion fallback localizes them.
5. PDP paraphrases are validated and ranked using sequence-level prior and
   full-context margins.
6. Internal causal validation uses sequence-level activation patching over all
   answer-predicting positions, gradient x activation-difference pre-screening,
   repeated cross-fitting, layer-matched random controls, and stability counts.
7. Internal analysis defaults to all detected errors rather than selecting only
   strong PDP responders. A responder-only analysis remains available.

Example
-------
python keyshift_halueval_open_v10.py \
  --input data/qa_data.json \
  --detector-predictions other_bench/HaluEval/v7_llama_3000/predictions.jsonl \
  --model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --output-dir other_bench/HaluEval/keyshift_v10_open \
  --stage all \
  --editor-backend local \
  --paraphrase-candidates 10 \
  --causal-folds 4 \
  --causal-repeats 3 \
  --resume

For a fast smoke test:
python keyshift_halueval_open_v10.py \
  --input data/qa_data.json --output-dir /tmp/v10_smoke \
  --max-items 20 --stage semantic --editor-backend local

Requirements
------------
pip install "transformers>=4.44" torch accelerate numpy scipy
Optional remote editor:
pip install openai
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

# Triton kernels compiled by recent PyTorch releases can fail with EIO when
# their cache lives on the workspace's network filesystem. Keep this
# disposable compiler cache on node-local storage unless explicitly set.
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/keyshift_triton_{os.getuid()}")

import torch
import torch.nn.functional as F

# This PyTorch build routes RoPE's outer-product bmm through a Triton override.
# The runtime lacks Python development headers, so compilation cannot succeed;
# use the standard CUDA bmm kernel instead.
try:
    from torch._native.registry import deregister_op_overrides
    deregister_op_overrides(disable_op_symbols="bmm")
except (ImportError, AttributeError):
    pass
from scipy.stats import spearmanr


EPS = 1e-9
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:;\s+|\s+[—–-]\s+|,\s+(?=(?:but|yet|however|although|though|while|whereas|because|since)\b)|\s+(?=(?:but|yet|however|although|though|whereas)\b))",
    re.I,
)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?\b")
NEGATION_RE = re.compile(r"\b(?:no|not|never|none|neither|without|cannot|can't|won't|isn't|aren't|doesn't|don't|didn't)\b", re.I)
MODAL_RE = re.compile(r"\b(?:must|shall|required|need(?:s|ed)? to|should|may|might|can|cannot|only|before|after|unless|until)\b", re.I)

PARAPHRASE_REVIEW_FIELDS = (
    "semantic_equivalent",
    "entities_preserved",
    "quantities_preserved",
    "negation_modality_preserved",
    "temporal_spatial_relation_preserved",
    "correct_answer_preserved",
    "no_new_constraint",
    "no_answer_leak",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HaluItem:
    idx: int
    item_id: str
    knowledge: str
    question: str
    right_answer: str
    hallucinated_answer: str
    split: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Span:
    span_id: int
    container: str  # knowledge or question
    start: int
    end: int
    text: str


@dataclass
class SequenceScore:
    total_logprob: float
    mean_logprob: float
    token_count: int
    token_logprobs: list[float]


@dataclass
class PairScore:
    right: SequenceScore
    hallucinated: SequenceScore
    correct_margin: float
    total_margin: float
    predicted_pair: str
    is_pair_correct: bool


@dataclass
class EncodedSequence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    answer_len: int
    answer_token_ids: list[int]
    decision_positions: list[int]


@dataclass
class HeadStatePackage:
    states: dict[int, torch.Tensor]
    decision_positions: list[int]


# ---------------------------------------------------------------------------
# JSON, caching, and generic helpers
# ---------------------------------------------------------------------------


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL line {line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object")
                out.append(obj)
        return out

    with p.open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "items", "records", "questions", "qa_data"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError("input must be a JSON list, JSONL, or a wrapper containing a list")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_jsonl_by_id(path: Path, id_keys: Sequence[str] = ("item_id", "id", "idx", "item_index")) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for rec in read_records(path):
        key = None
        for name in id_keys:
            if name in rec:
                key = str(rec[name])
                break
        if key is not None:
            out[key] = rec
    return out


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JsonCache:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                warnings.warn(f"Could not parse cache {path}; starting a new cache")
                self.data = {}
        else:
            self.data = {}

    def get(self, namespace: str, payload: Any) -> Any | None:
        return self.data.get(f"{namespace}:{stable_hash(payload)}")

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        self.data[f"{namespace}:{stable_hash(payload)}"] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, default=json_default), encoding="utf-8")
        tmp.replace(self.path)


def mean_or_none(xs: Sequence[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None


def bootstrap_mean_ci(values: Sequence[float], draws: int, seed: int) -> list[float] | None:
    arr = np.asarray([x for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def bootstrap_paired_ci(a: Sequence[float], b: Sequence[float], draws: int, seed: int) -> list[float] | None:
    if len(a) != len(b) or not a:
        return None
    return bootstrap_mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), draws, seed)


def token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text)}


def token_f1(a: str, b: str) -> float:
    ta = [m.group(0).lower() for m in WORD_RE.finditer(a)]
    tb = [m.group(0).lower() for m in WORD_RE.finditer(b)]
    if not ta or not tb:
        return float(ta == tb)
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


def edit_ratio(a: str, b: str) -> float:
    # Dependency-free normalized Levenshtein distance.
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(text[start:i + 1])
                    if not isinstance(obj, dict):
                        raise ValueError("parsed JSON is not an object")
                    return obj
    raise ValueError("unterminated JSON object")


def normalize_space(text: str) -> str:
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# HaluEval format and prompt construction
# ---------------------------------------------------------------------------


def pick_first(item: Mapping[str, Any], names: Sequence[str], required: bool = True) -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    if required:
        raise KeyError(f"none of the required fields are present: {names}")
    return None


def stringify_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for entry in value:
            if isinstance(entry, str):
                parts.append(entry.strip())
            elif isinstance(entry, Mapping):
                parts.append(" ".join(str(v) for v in entry.values()))
            else:
                parts.append(str(entry))
        return "\n".join(x for x in parts if x).strip()
    if isinstance(value, Mapping):
        return "\n".join(f"{k}: {v}" for k, v in value.items()).strip()
    return str(value).strip()


def normalize_halueval_item(raw: dict[str, Any], idx: int) -> HaluItem:
    knowledge = stringify_field(pick_first(raw, ("knowledge", "context", "passage", "document")))
    question = stringify_field(pick_first(raw, ("question", "query")))
    right = stringify_field(pick_first(raw, ("right_answer", "answer", "correct_answer", "reference_answer")))
    hall = stringify_field(pick_first(raw, ("hallucinated_answer", "hallucination", "wrong_answer", "incorrect_answer")))
    if not knowledge or not question or not right or not hall:
        raise ValueError("knowledge, question, right answer, and hallucinated answer must be non-empty")
    item_id = str(pick_first(raw, ("id", "item_id", "idx", "index"), required=False) or idx)
    split = pick_first(raw, ("split", "set", "partition"), required=False)
    return HaluItem(
        idx=idx,
        item_id=item_id,
        knowledge=knowledge,
        question=question,
        right_answer=right,
        hallucinated_answer=hall,
        split=str(split) if split is not None else None,
        raw=raw,
    )


def open_qa_user_prompt(knowledge: str, question: str) -> str:
    return (
        "Use the provided knowledge to answer the question. "
        "Give a concise direct answer and do not mention answer candidates.\n\n"
        f"Knowledge:\n{knowledge.strip()}\n\n"
        f"Question:\n{question.strip()}"
    )


def prior_probe_user_prompt(span_text: str, question: str) -> str:
    return (
        "Answer the question using only the single statement below. "
        "Do not use outside context.\n\n"
        f"Statement:\n{span_text.strip()}\n\n"
        f"Question:\n{question.strip()}"
    )


def replace_container_span(item: HaluItem, span: Span, replacement: str) -> tuple[str, str]:
    if span.container == "knowledge":
        return item.knowledge[:span.start] + replacement + item.knowledge[span.end:], item.question
    if span.container == "question":
        return item.knowledge, item.question[:span.start] + replacement + item.question[span.end:]
    raise ValueError(f"unknown span container {span.container!r}")


def delete_container_span(item: HaluItem, span: Span) -> tuple[str, str]:
    knowledge, question = replace_container_span(item, span, "")
    return normalize_space(knowledge), normalize_space(question)


# ---------------------------------------------------------------------------
# Span proposal and detector span import
# ---------------------------------------------------------------------------


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def structural_spans(text: str, container: str, mode: str, min_words: int) -> list[Span]:
    base: list[tuple[int, int]] = []
    for m in SENTENCE_RE.finditer(text):
        s, e = _trim(text, m.start(), m.end())
        if s >= e:
            continue
        if mode == "sentence":
            base.append((s, e))
            continue
        sentence = text[s:e]
        cursor = 0
        pieces: list[tuple[int, int]] = []
        for boundary in CLAUSE_BOUNDARY_RE.finditer(sentence):
            a, b = _trim(sentence, cursor, boundary.start())
            if a < b:
                pieces.append((s + a, s + b))
            cursor = boundary.end()
        a, b = _trim(sentence, cursor, len(sentence))
        if a < b:
            pieces.append((s + a, s + b))
        useful = [p for p in pieces if len(WORD_RE.findall(text[p[0]:p[1]])) >= min_words]
        base.extend(useful if len(useful) >= 2 else [(s, e)])

    spans: list[Span] = []
    for start, end in base:
        candidate = text[start:end].strip()
        if len(WORD_RE.findall(candidate)) < min_words:
            continue
        anchored = text.find(candidate, start, end + 1)
        spans.append(Span(len(spans), container, anchored, anchored + len(candidate), candidate))
    return spans


def propose_item_spans(item: HaluItem, mode: str, min_words: int, include_question: bool) -> list[Span]:
    spans = structural_spans(item.knowledge, "knowledge", mode, min_words)
    if include_question:
        qspans = structural_spans(item.question, "question", mode, min_words)
        offset = len(spans)
        spans.extend(Span(offset + s.span_id, s.container, s.start, s.end, s.text) for s in qspans)
    return spans


def locate_exact_or_fuzzy(item: HaluItem, text: str, min_f1: float = 0.65) -> Span | None:
    needle = normalize_space(str(text))
    for container, source in (("knowledge", item.knowledge), ("question", item.question)):
        pos = source.find(needle)
        if pos >= 0:
            return Span(0, container, pos, pos + len(needle), needle)

    candidates = propose_item_spans(item, mode="clause", min_words=2, include_question=True)
    if not candidates:
        return None
    best = max(candidates, key=lambda s: token_f1(s.text, needle))
    return Span(0, best.container, best.start, best.end, best.text) if token_f1(best.text, needle) >= min_f1 else None


def _deep_get(obj: Any, paths: Sequence[Sequence[str]]) -> Any | None:
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def detector_info(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {
            "predicted_hallucination": False,
            "hallucination_probability": None,
            "shortcut_text": None,
            "constraint_text": None,
            "shortcut_source": None,
            "constraint_source": None,
        }
    predicted = _deep_get(record, [
        ("predicted_hallucination",),
        ("prediction", "predicted_hallucination"),
        ("detector", "predicted_hallucination"),
    ])
    prob = _deep_get(record, [
        ("hallucination_probability",),
        ("prediction", "hallucination_probability"),
        ("detector", "hallucination_probability"),
    ])
    shortcut_text = _deep_get(record, [
        ("predicted_shortcut", "text"),
        ("explanation", "predicted_shortcut", "text"),
        ("shortcut", "text"),
        ("predicted_shortcut_span",),
        ("shortcut_span",),
    ])
    constraint_text = _deep_get(record, [
        ("predicted_constraint", "text"),
        ("explanation", "predicted_constraint", "text"),
        ("constraint", "text"),
        ("predicted_constraint_span",),
        ("constraint_span",),
    ])
    # Open-ended v8 stores per-span role contributions rather than resolved
    # predicted_shortcut/predicted_constraint objects. Restrict imported spans
    # to context/knowledge (not question) and choose distinct role maxima.
    spans = record.get("spans")
    if isinstance(spans, list) and (not shortcut_text or not constraint_text):
        candidates: list[dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, Mapping):
                continue
            text = str(span.get("span_text", "")).strip()
            if text.lower().startswith("question:"):
                continue
            if text.lower().startswith("context:"):
                text = text.split(":", 1)[1].strip()
            if not text:
                continue
            candidates.append({
                "text": text,
                "shortcut": float(span.get("shortcut_contribution", 0.0)),
                "constraint": float(span.get("constraint_contribution", 0.0)),
            })
        if candidates:
            shortcut_row = max(candidates, key=lambda row: row["shortcut"])
            constraint_pool = [row for row in candidates if row is not shortcut_row]
            constraint_row = max(constraint_pool, key=lambda row: row["constraint"], default=None)
            if not shortcut_text:
                shortcut_text = shortcut_row["text"]
            if not constraint_text and constraint_row is not None:
                constraint_text = constraint_row["text"]
    return {
        "predicted_hallucination": bool(predicted),
        "hallucination_probability": float(prob) if prob is not None else None,
        "shortcut_text": str(shortcut_text) if shortcut_text else None,
        "constraint_text": str(constraint_text) if constraint_text else None,
        "shortcut_source": _deep_get(record, [("shortcut_source",), ("detector", "shortcut_source")]),
        "constraint_source": _deep_get(record, [("constraint_source",), ("detector", "constraint_source")]),
    }


# ---------------------------------------------------------------------------
# Open-answer sequence scorer
# ---------------------------------------------------------------------------


class OpenAnswerScorer:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype_name: str,
        trust_remote_code: bool,
        max_length: int,
        answer_prefix: str,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'transformers is required; install with: pip install "transformers>=4.44" accelerate'
            ) from exc
        self.device = torch.device(device)
        self.dtype = self._resolve_dtype(dtype_name)
        self.max_length = max_length
        self.answer_prefix = answer_prefix
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self._find_layers()
        self.num_heads = int(getattr(self.model.config, "num_attention_heads"))
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size is not divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        table = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if name not in table:
            raise ValueError(f"unsupported dtype {name!r}")
        return table[name]

    def _find_layers(self) -> Sequence[torch.nn.Module]:
        candidates = (
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        )
        for path in candidates:
            cur: Any = self.model
            ok = True
            for attr in path:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok and isinstance(cur, (torch.nn.ModuleList, list, tuple)):
                return cur
        raise ValueError("could not locate decoder layers for this architecture")

    def o_proj(self, layer_idx: int) -> torch.nn.Module:
        layer = self.layers[layer_idx]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            raise ValueError(f"layer {layer_idx} has no self-attention module")
        for name in ("o_proj", "out_proj", "dense"):
            if hasattr(attn, name):
                return getattr(attn, name)
        raise ValueError(f"layer {layer_idx} has no supported attention output projection")

    def chat_prompt(self, user_prompt: str, system_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "Answer from the supplied knowledge. Be concise and factual."},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {messages[0]['content']}\nUser: {user_prompt}\nAssistant:"

    def encode(self, user_prompt: str, answer: str) -> EncodedSequence:
        rendered = self.chat_prompt(user_prompt)
        prefix_text = rendered + self.answer_prefix
        prefix_ids = self.tok.encode(prefix_text, add_special_tokens=False)
        full_ids = self.tok.encode(prefix_text + answer.strip(), add_special_tokens=False)
        if full_ids[: len(prefix_ids)] == prefix_ids:
            answer_ids = full_ids[len(prefix_ids):]
            ids = full_ids
        else:
            # Defensive fallback for unusual tokenizers whose boundary encoding
            # is not prefix-stable. A whitespace answer_prefix normally avoids it.
            answer_ids = self.tok.encode(answer.strip(), add_special_tokens=False)
            ids = prefix_ids + answer_ids
        if not answer_ids:
            raise ValueError("answer tokenized to an empty sequence")
        if len(ids) > self.max_length:
            # Preserve the answer and the end of the prompt, where the question is.
            keep_prefix = self.max_length - len(answer_ids)
            if keep_prefix < 16:
                raise ValueError("answer is too long for max_length")
            prefix_ids = prefix_ids[-keep_prefix:]
            ids = prefix_ids + answer_ids
        prompt_len = len(prefix_ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        # Logits at positions prompt_len-1 ... prompt_len+answer_len-2 predict answer tokens.
        decision_positions = list(range(prompt_len - 1, prompt_len - 1 + len(answer_ids)))
        return EncodedSequence(input_ids, attention_mask, prompt_len, len(answer_ids), answer_ids, decision_positions)

    @staticmethod
    def _sequence_logprob_tensor(logits: torch.Tensor, encoded: EncodedSequence) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.tensor(encoded.decision_positions, dtype=torch.long, device=logits.device)
        targets = torch.tensor(encoded.answer_token_ids, dtype=torch.long, device=logits.device)
        selected = logits[0, positions, :].float()
        token_lp = F.log_softmax(selected, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        return token_lp.sum(), token_lp.mean()

    def score_answer(self, user_prompt: str, answer: str) -> SequenceScore:
        encoded = self.encode(user_prompt, answer)
        with torch.inference_mode():
            out = self.model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, use_cache=False)
            total, mean = self._sequence_logprob_tensor(out.logits, encoded)
            positions = torch.tensor(encoded.decision_positions, device=self.device)
            targets = torch.tensor(encoded.answer_token_ids, device=self.device)
            token_lp = F.log_softmax(out.logits[0, positions, :].float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        return SequenceScore(float(total.cpu()), float(mean.cpu()), encoded.answer_len, token_lp.cpu().tolist())

    def pair_score(self, user_prompt: str, right_answer: str, hallucinated_answer: str) -> PairScore:
        right = self.score_answer(user_prompt, right_answer)
        hall = self.score_answer(user_prompt, hallucinated_answer)
        margin = right.mean_logprob - hall.mean_logprob
        total_margin = right.total_logprob - hall.total_logprob
        return PairScore(right, hall, margin, total_margin, "right" if margin >= 0 else "hallucinated", margin >= 0)

    def surface_surprisal(self, text: str) -> float:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
            target = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="mean")
        return float(loss.cpu())

    def generate(
        self,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> str:
        rendered = self.chat_prompt(user_prompt, system_prompt=system_prompt)
        enc = self.tok(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        continuation = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(continuation, skip_special_tokens=True).strip()

    def classify_generation(self, generated: str, right: str, hall: str) -> dict[str, Any]:
        fr = token_f1(generated, right)
        fh = token_f1(generated, hall)
        if max(fr, fh) < 0.25 or abs(fr - fh) < 0.05:
            label = "other"
        else:
            label = "right" if fr > fh else "hallucinated"
        return {"label": label, "right_token_f1": fr, "hallucinated_token_f1": fh}

    # ----- pre-o_proj head-state capture / patching -------------------------

    @contextlib.contextmanager
    def capture_pre_o_proj(self, layer_indices: Iterable[int], retain_grad: bool = False) -> Iterator[dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in sorted(set(layer_indices)):
            module = self.o_proj(layer_idx)

            def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], li: int = layer_idx):
                x = args[0]
                if retain_grad:
                    x.retain_grad()
                    captured[li] = x
                else:
                    captured[li] = x.detach().clone()
                return None

            handles.append(module.register_forward_pre_hook(hook))
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _position_map(target_positions: Sequence[int], source_positions: Sequence[int]) -> list[tuple[int, int]]:
        if not target_positions or not source_positions:
            return []
        if len(target_positions) == 1:
            return [(target_positions[0], source_positions[-1])]
        pairs = []
        for i, t in enumerate(target_positions):
            frac = i / max(len(target_positions) - 1, 1)
            j = int(round(frac * (len(source_positions) - 1)))
            pairs.append((t, source_positions[j]))
        return pairs

    @contextlib.contextmanager
    def patch_pre_o_proj(
        self,
        source: HeadStatePackage,
        heads: Sequence[tuple[int, int]],
        target_positions: Sequence[int],
    ) -> Iterator[None]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for layer_idx, head_idx in heads:
            grouped[int(layer_idx)].append(int(head_idx))
        handles = []
        for layer_idx, head_ids in grouped.items():
            if layer_idx not in source.states:
                raise KeyError(f"source state missing layer {layer_idx}")
            src = source.states[layer_idx]
            module = self.o_proj(layer_idx)
            mapping = self._position_map(target_positions, source.decision_positions)

            def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], li: int = layer_idx, hs: list[int] = list(head_ids), mp: list[tuple[int, int]] = list(mapping)):
                x = args[0]
                if x.shape[-1] != self.hidden_size:
                    raise ValueError(f"unexpected pre-o_proj width {x.shape[-1]} at layer {li}")
                y = x.clone()
                yv = y.view(y.shape[0], y.shape[1], self.num_heads, self.head_dim)
                source_tensor = source.states[li].to(device=y.device, dtype=y.dtype)
                sv = source_tensor.view(source_tensor.shape[0], source_tensor.shape[1], self.num_heads, self.head_dim)
                for tp, sp in mp:
                    if tp >= yv.shape[1] or sp >= sv.shape[1]:
                        continue
                    yv[:, tp, hs, :] = sv[:, sp, hs, :]
                return (yv.reshape_as(y),) + args[1:]

            handles.append(module.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def ablate_pre_o_proj(
        self,
        heads: Sequence[tuple[int, int]],
        target_positions: Sequence[int],
    ) -> Iterator[None]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for layer_idx, head_idx in heads:
            grouped[int(layer_idx)].append(int(head_idx))
        handles = []
        for layer_idx, head_ids in grouped.items():
            module = self.o_proj(layer_idx)

            def hook(
                _module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                li: int = layer_idx,
                hs: list[int] = list(head_ids),
                positions: list[int] = list(target_positions),
            ):
                x = args[0]
                y = x.clone()
                yv = y.view(y.shape[0], y.shape[1], self.num_heads, self.head_dim)
                for pos in positions:
                    if pos < yv.shape[1]:
                        yv[:, pos, hs, :] = 0
                return (yv.reshape_as(y),) + args[1:]

            handles.append(module.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def score_answer_with_ablation(
        self,
        prompt: str,
        answer: str,
        heads: Sequence[tuple[int, int]],
    ) -> SequenceScore:
        encoded = self.encode(prompt, answer)
        with self.ablate_pre_o_proj(heads, encoded.decision_positions):
            with torch.inference_mode():
                out = self.model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    use_cache=False,
                )
                total, mean = self._sequence_logprob_tensor(out.logits, encoded)
                positions = torch.tensor(encoded.decision_positions, device=self.device)
                targets = torch.tensor(encoded.answer_token_ids, device=self.device)
                token_lp = F.log_softmax(
                    out.logits[0, positions, :].float(), dim=-1
                ).gather(-1, targets[:, None]).squeeze(-1)
        return SequenceScore(
            float(total.cpu()), float(mean.cpu()), encoded.answer_len, token_lp.cpu().tolist()
        )

    def pair_score_with_ablation(
        self,
        prompt: str,
        right_answer: str,
        hall_answer: str,
        heads: Sequence[tuple[int, int]],
    ) -> PairScore:
        right = self.score_answer_with_ablation(prompt, right_answer, heads)
        hall = self.score_answer_with_ablation(prompt, hall_answer, heads)
        margin = right.mean_logprob - hall.mean_logprob
        return PairScore(
            right, hall, margin, right.total_logprob - hall.total_logprob,
            "right" if margin >= 0 else "hallucinated", margin >= 0,
        )

    def capture_answer_states(self, user_prompt: str, answer: str, layers: Sequence[int]) -> HeadStatePackage:
        encoded = self.encode(user_prompt, answer)
        with self.capture_pre_o_proj(layers, retain_grad=False) as captured:
            with torch.inference_mode():
                self.model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, use_cache=False)
        return HeadStatePackage({k: v.cpu() for k, v in captured.items()}, encoded.decision_positions)

    def score_answer_with_patch(
        self,
        target_prompt: str,
        answer: str,
        source: HeadStatePackage,
        heads: Sequence[tuple[int, int]],
    ) -> SequenceScore:
        encoded = self.encode(target_prompt, answer)
        with self.patch_pre_o_proj(source, heads, encoded.decision_positions):
            with torch.inference_mode():
                out = self.model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, use_cache=False)
                total, mean = self._sequence_logprob_tensor(out.logits, encoded)
                positions = torch.tensor(encoded.decision_positions, device=self.device)
                targets = torch.tensor(encoded.answer_token_ids, device=self.device)
                token_lp = F.log_softmax(out.logits[0, positions, :].float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        return SequenceScore(float(total.cpu()), float(mean.cpu()), encoded.answer_len, token_lp.cpu().tolist())

    def pair_score_with_patch(
        self,
        target_prompt: str,
        right_answer: str,
        hall_answer: str,
        right_source: HeadStatePackage,
        hall_source: HeadStatePackage,
        heads: Sequence[tuple[int, int]],
    ) -> PairScore:
        r = self.score_answer_with_patch(target_prompt, right_answer, right_source, heads)
        h = self.score_answer_with_patch(target_prompt, hall_answer, hall_source, heads)
        margin = r.mean_logprob - h.mean_logprob
        return PairScore(r, h, margin, r.total_logprob - h.total_logprob, "right" if margin >= 0 else "hallucinated", margin >= 0)

    def answer_attribution_to_source(
        self,
        target_prompt: str,
        source_prompt: str,
        answer: str,
        layers: Sequence[int],
    ) -> np.ndarray:
        source = self.capture_answer_states(source_prompt, answer, layers)
        encoded = self.encode(target_prompt, answer)
        self.model.zero_grad(set_to_none=True)
        # Model parameters are frozen, so create a differentiable input-embedding
        # leaf; otherwise pre-o_proj tensors would not require gradients.
        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(encoded.input_ids).detach().requires_grad_(True)
        with self.capture_pre_o_proj(layers, retain_grad=True) as target_states:
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoded.attention_mask,
                use_cache=False,
            )
            _total, mean = self._sequence_logprob_tensor(out.logits, encoded)
            mean.backward()

        scores = np.zeros((len(self.layers), self.num_heads), dtype=np.float64)
        mapping = self._position_map(encoded.decision_positions, source.decision_positions)
        for layer_idx in layers:
            target = target_states[layer_idx]
            grad = target.grad
            if grad is None:
                continue
            src = source.states[layer_idx].to(target.device, target.dtype)
            tv = target.view(target.shape[0], target.shape[1], self.num_heads, self.head_dim)
            gv = grad.view(grad.shape[0], grad.shape[1], self.num_heads, self.head_dim)
            sv = src.view(src.shape[0], src.shape[1], self.num_heads, self.head_dim)
            per_pos = []
            for tp, sp in mapping:
                if tp < tv.shape[1] and sp < sv.shape[1]:
                    per_pos.append((gv[:, tp] * (sv[:, sp] - tv[:, tp])).sum(dim=-1))
            if per_pos:
                value = torch.stack(per_pos).mean(dim=(0, 1))
                scores[layer_idx] = value.detach().float().cpu().numpy()
        self.model.zero_grad(set_to_none=True)
        return scores

    def pair_bidirectional_attribution(
        self,
        original_prompt: str,
        pdp_prompt: str,
        right_answer: str,
        hall_answer: str,
        layers: Sequence[int],
    ) -> np.ndarray:
        # Forward predicts change in right-minus-hall margin when moving O -> PDP.
        f_r = self.answer_attribution_to_source(original_prompt, pdp_prompt, right_answer, layers)
        f_h = self.answer_attribution_to_source(original_prompt, pdp_prompt, hall_answer, layers)
        forward = f_r - f_h
        # Reverse is sign-normalized so positive also means the O/PDP difference carries improvement.
        r_r = self.answer_attribution_to_source(pdp_prompt, original_prompt, right_answer, layers)
        r_h = self.answer_attribution_to_source(pdp_prompt, original_prompt, hall_answer, layers)
        reverse_harm = -(r_r - r_h)
        return 0.5 * (forward + reverse_harm)


# ---------------------------------------------------------------------------
# Editor and semantic validator
# ---------------------------------------------------------------------------


class Editor:
    def __init__(self, args: argparse.Namespace, scorer: OpenAnswerScorer, cache: JsonCache):
        self.args = args
        self.scorer = scorer
        self.cache = cache
        self.client = None
        if args.editor_backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("pip install openai for --editor-backend openai") from exc
            self.client = OpenAI(api_key=args.editor_api_key or os.environ.get("OPENAI_API_KEY"), base_url=args.editor_base_url)

    def complete(self, namespace: str, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "backend": self.args.editor_backend,
            "model": self.args.editor_model or self.args.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cached = self.cache.get(namespace, payload)
        if cached is not None:
            return str(cached)
        if self.args.editor_backend == "openai":
            assert self.client is not None
            result = self.client.chat.completions.create(
                model=self.args.editor_model or self.args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = result.choices[0].message.content or ""
        else:
            text = self.scorer.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                system_prompt="Follow the user's transformation or validation instruction exactly. Return only the requested JSON.",
            )
        self.cache.set(namespace, payload, text)
        return text

    def paraphrases(self, span: str, n: int) -> list[str]:
        prompt = f"""Generate {n} natural English paraphrases of the SOURCE sentence.
Preserve every entity, number, negation, modality, temporal relation, spatial relation, and factual implication.
Do not add explanations, recommendations, answer hints, or new constraints.
Return JSON only in this format: {{\"paraphrases\": [\"...\"]}}.

SOURCE:
{span}
"""
        errors = []
        for attempt in range(1, self.args.editor_max_retries + 1):
            raw = self.complete(f"paraphrases_attempt_{attempt}", prompt, self.args.editor_temperature if attempt == 1 else 0.0, 1000)
            try:
                obj = extract_json_object(raw)
                values = obj.get("paraphrases")
                if not isinstance(values, list):
                    raise ValueError("paraphrases is not a list")
                out = []
                for value in values:
                    if isinstance(value, str) and value.strip() and value.strip() not in out:
                        out.append(value.strip())
                if out:
                    return out[:n]
                raise ValueError("no non-empty paraphrases")
            except Exception as exc:
                errors.append(str(exc))
        raise RuntimeError("paraphrase generation failed: " + " | ".join(errors))

    def review_paraphrase(self, item: HaluItem, source: str, candidate: str) -> dict[str, Any]:
        prompt = f"""Evaluate whether CANDIDATE is a safe semantic-preserving replacement for SOURCE inside the HaluEval open QA item.
Return JSON only. Use booleans for all fields and an integer naturalness score from 1 to 5.
Required fields: {', '.join(PARAPHRASE_REVIEW_FIELDS)}, naturalness, notes.
Be strict about entity identity, predicate arguments, temporal/spatial relations, modality, and not leaking either reference answer.

KNOWLEDGE:
{item.knowledge}

QUESTION:
{item.question}

RIGHT REFERENCE ANSWER (for preservation checking only):
{item.right_answer}

HALLUCINATED REFERENCE ANSWER (for leak checking only):
{item.hallucinated_answer}

SOURCE:
{source}

CANDIDATE:
{candidate}
"""
        errors = []
        for attempt in range(1, self.args.editor_max_retries + 1):
            raw = self.complete(f"review_attempt_{attempt}", prompt, 0.0, 700)
            try:
                obj = extract_json_object(raw)
                result: dict[str, Any] = {}
                for field in PARAPHRASE_REVIEW_FIELDS:
                    if not isinstance(obj.get(field), bool):
                        raise ValueError(f"{field} missing or not boolean")
                    result[field] = bool(obj[field])
                naturalness = int(obj.get("naturalness"))
                if not 1 <= naturalness <= 5:
                    raise ValueError("naturalness outside 1..5")
                result["naturalness"] = naturalness
                result["notes"] = str(obj.get("notes", ""))
                return result
            except Exception as exc:
                errors.append(str(exc))
        return {**{f: False for f in PARAPHRASE_REVIEW_FIELDS}, "naturalness": 0, "notes": "validator failed: " + " | ".join(errors)}

    def context_link(self, item: HaluItem, shortcut: Span, constraint: Span) -> str | None:
        prompt = f"""Write one natural English sentence, 16-45 words, that can be appended to the knowledge.
It must explicitly state that the SHORTCUT fact alone is insufficient to answer the question and that the CONSTRAINT fact is the relevant basis.
Preserve all facts. Do not mention experiments, shortcuts, constraints, candidate answers, correctness, or give the answer directly.
Return JSON only: {{\"sentence\": \"...\"}}.

QUESTION:
{item.question}

SHORTCUT FACT:
{shortcut.text}

CONSTRAINT FACT:
{constraint.text}
"""
        raw = self.complete("context_link", prompt, 0.2, 300)
        try:
            sentence = str(extract_json_object(raw).get("sentence", "")).strip()
        except Exception:
            return None
        if not sentence or len(WORD_RE.findall(sentence)) < 12:
            return None
        if max(token_f1(sentence, item.right_answer), token_f1(sentence, item.hallucinated_answer)) > self.args.ckl_max_answer_token_f1:
            return None
        if token_f1(sentence, shortcut.text) < self.args.ckl_min_anchor_f1:
            return None
        if token_f1(sentence, constraint.text) < self.args.ckl_min_anchor_f1:
            return None
        return sentence


def deterministic_paraphrase_checks(item: HaluItem, source: str, candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    source_numbers = NUMBER_RE.findall(source)
    candidate_numbers = NUMBER_RE.findall(candidate)
    source_neg = bool(NEGATION_RE.search(source))
    candidate_neg = bool(NEGATION_RE.search(candidate))
    source_modals = {x.lower() for x in MODAL_RE.findall(source)}
    candidate_modals = {x.lower() for x in MODAL_RE.findall(candidate)}

    # Capitalized and mixed-case tokens provide a conservative entity anchor.
    entity_tokens = {
        token for token in WORD_RE.findall(source)
        if any(c.isupper() for c in token) or any(c.isdigit() for c in token)
    }
    candidate_tokens_case = set(WORD_RE.findall(candidate))
    entity_anchor_preserved = entity_tokens.issubset(candidate_tokens_case)
    answer_overlap = max(token_f1(candidate, item.right_answer), token_f1(candidate, item.hallucinated_answer))

    return {
        "numbers_exactly_preserved": source_numbers == candidate_numbers,
        "negation_presence_preserved": source_neg == candidate_neg,
        "modal_anchors_preserved": not source_modals or source_modals.issubset(candidate_modals),
        "entity_anchor_preserved": entity_anchor_preserved,
        "token_f1": token_f1(source, candidate),
        "edit_ratio": edit_ratio(source, candidate),
        "max_answer_token_f1": answer_overlap,
        "no_near_answer_copy": answer_overlap <= args.max_answer_token_f1,
    }


def candidate_valid(review: Mapping[str, Any], checks: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return (
        all(bool(review.get(f)) for f in PARAPHRASE_REVIEW_FIELDS)
        and int(review.get("naturalness", 0)) >= args.min_naturalness
        and bool(checks.get("numbers_exactly_preserved"))
        and bool(checks.get("negation_presence_preserved"))
        and bool(checks.get("modal_anchors_preserved"))
        and bool(checks.get("entity_anchor_preserved"))
        and bool(checks.get("no_near_answer_copy"))
        and float(checks.get("token_f1", 0.0)) >= args.min_paraphrase_token_f1
        and float(checks.get("edit_ratio", 1.0)) <= args.max_edit_ratio
    )


# ---------------------------------------------------------------------------
# Localization and semantic counterfactual preparation
# ---------------------------------------------------------------------------


def occlusion_localize(item: HaluItem, scorer: OpenAnswerScorer, args: argparse.Namespace) -> dict[str, Any]:
    spans = propose_item_spans(item, args.span_mode, args.min_span_words, args.include_question_spans)
    base_prompt = open_qa_user_prompt(item.knowledge, item.question)
    base = scorer.pair_score(base_prompt, item.right_answer, item.hallucinated_answer)
    rows = []
    for span in spans[: args.max_localization_spans or None]:
        k, q = delete_container_span(item, span)
        prompt = open_qa_user_prompt(k, q)
        score = scorer.pair_score(prompt, item.right_answer, item.hallucinated_answer)
        rows.append({
            "span": asdict(span),
            "deleted_correct_margin": score.correct_margin,
            "margin_change": score.correct_margin - base.correct_margin,
        })
    if not rows:
        return {"shortcut": None, "constraint": None, "rows": [], "base": asdict(base)}
    shortcut_row = max(rows, key=lambda r: r["margin_change"])
    constraint_row = min(rows, key=lambda r: r["margin_change"])
    shortcut = Span(**shortcut_row["span"]) if shortcut_row["margin_change"] >= args.localization_min_effect else None
    constraint = Span(**constraint_row["span"]) if constraint_row["margin_change"] <= -args.localization_min_effect else None
    if shortcut and constraint and shortcut.container == constraint.container and shortcut.start == constraint.start and shortcut.end == constraint.end:
        constraint = None
    return {
        "shortcut": asdict(shortcut) if shortcut else None,
        "constraint": asdict(constraint) if constraint else None,
        "rows": rows,
        "base": asdict(base),
    }


def choose_spans(
    item: HaluItem,
    detector_record: dict[str, Any] | None,
    scorer: OpenAnswerScorer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    info = detector_info(detector_record)
    shortcut = locate_exact_or_fuzzy(item, info["shortcut_text"]) if info["shortcut_text"] else None
    constraint = locate_exact_or_fuzzy(item, info["constraint_text"]) if info["constraint_text"] else None
    source = "detector"
    fallback = None
    if shortcut is None and args.allow_occlusion_fallback:
        fallback = occlusion_localize(item, scorer, args)
        shortcut = Span(**fallback["shortcut"]) if fallback.get("shortcut") else None
        if constraint is None and fallback.get("constraint"):
            constraint = Span(**fallback["constraint"])
        source = "occlusion_fallback"
    if shortcut and constraint and shortcut.container == constraint.container and shortcut.start == constraint.start and shortcut.end == constraint.end:
        constraint = None
    return {
        "shortcut": asdict(shortcut) if shortcut else None,
        "constraint": asdict(constraint) if constraint else None,
        "localization_source": source if shortcut else None,
        "detector": info,
        "occlusion_fallback": fallback,
    }


def score_variant(item: HaluItem, scorer: OpenAnswerScorer, knowledge: str, question: str, generate: bool, args: argparse.Namespace) -> dict[str, Any]:
    prompt = open_qa_user_prompt(knowledge, question)
    pair = scorer.pair_score(prompt, item.right_answer, item.hallucinated_answer)
    out: dict[str, Any] = {
        "knowledge": knowledge,
        "question": question,
        "prompt": prompt,
        "pair": asdict(pair),
    }
    if generate:
        generated = scorer.generate(prompt, args.generation_max_tokens, temperature=0.0)
        out["generated_answer"] = generated
        out["generation_class"] = scorer.classify_generation(generated, item.right_answer, item.hallucinated_answer)
    return out


def prepare_semantic_item(
    item: HaluItem,
    detector_record: dict[str, Any] | None,
    scorer: OpenAnswerScorer,
    editor: Editor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    localization = choose_spans(item, detector_record, scorer, args)
    shortcut = Span(**localization["shortcut"]) if localization.get("shortcut") else None
    constraint = Span(**localization["constraint"]) if localization.get("constraint") else None
    original = score_variant(item, scorer, item.knowledge, item.question, args.generate_answers, args)
    result: dict[str, Any] = {
        "item_index": item.idx,
        "item_id": item.item_id,
        "split": item.split,
        "right_answer": item.right_answer,
        "hallucinated_answer": item.hallucinated_answer,
        "localization": localization,
        "original": original,
        "status": "prepared" if shortcut else "not_localized",
    }
    if shortcut is None:
        result["reason"] = "no_shortcut_span"
        return result

    original_surprisal = scorer.surface_surprisal(shortcut.text)
    generated = editor.paraphrases(shortcut.text, args.paraphrase_candidates)
    candidate_rows = []
    for i, text in enumerate(generated, 1):
        checks = deterministic_paraphrase_checks(item, shortcut.text, text, args)
        review = editor.review_paraphrase(item, shortcut.text, text)
        valid = candidate_valid(review, checks, args)
        row: dict[str, Any] = {
            "candidate_id": f"p{i:02d}",
            "text": text,
            "review": review,
            "automatic_checks": checks,
            "valid": valid,
        }
        if valid:
            surprisal = scorer.surface_surprisal(text)
            row["surface_surprisal"] = surprisal
            row["surprisal_increase"] = surprisal - original_surprisal
            if row["surprisal_increase"] > args.max_surprisal_increase:
                row["valid"] = False
                row["invalid_reason"] = "surprisal_increase"
            else:
                probe = scorer.pair_score(prior_probe_user_prompt(text, item.question), item.right_answer, item.hallucinated_answer)
                k, q = replace_container_span(item, shortcut, text)
                full = score_variant(item, scorer, k, q, False, args)
                row["prior_probe"] = asdict(probe)
                row["prior_shortcut_margin"] = -probe.correct_margin
                row["full"] = full
        candidate_rows.append(row)

    valid_rows = [r for r in candidate_rows if r.get("valid") and "prior_probe" in r]
    result["all_paraphrase_candidates"] = candidate_rows
    if len(valid_rows) < args.min_valid_paraphrases:
        result["status"] = "insufficient_valid_paraphrases"
        result["reason"] = f"only_{len(valid_rows)}_valid_paraphrases"
        return result

    by_prior = sorted(valid_rows, key=lambda r: float(r["prior_shortcut_margin"]))
    prior_low = by_prior[0]
    prior_high = by_prior[-1]
    prior_mid = by_prior[len(by_prior) // 2]
    original_prior = scorer.pair_score(prior_probe_user_prompt(shortcut.text, item.question), item.right_answer, item.hallucinated_answer)
    common = min(valid_rows, key=lambda r: abs(float(r["prior_shortcut_margin"]) - (-original_prior.correct_margin)))

    # PDP minimizes shortcut prior, then maximizes full correct margin within a small prior band.
    low_value = float(prior_low["prior_shortcut_margin"])
    band = [r for r in valid_rows if float(r["prior_shortcut_margin"]) <= low_value + args.pdp_prior_band]
    pdp = max(band, key=lambda r: float(r["full"]["pair"]["correct_margin"]))
    consensus = by_prior[: min(args.pdp_consensus_k, len(by_prior))]

    selected = {
        "prior_low": prior_low,
        "prior_mid": prior_mid,
        "prior_high": prior_high,
        "common_control": common,
        "pdp": pdp,
        "pdp_consensus": consensus,
    }
    result["selected"] = selected
    result["original_prior_probe"] = asdict(original_prior)

    if constraint:
        ckl = editor.context_link(item, shortcut, constraint)
        if ckl:
            context_knowledge = item.knowledge.rstrip() + " " + ckl if shortcut.container == "knowledge" else item.knowledge
            context_question = item.question if shortcut.container == "knowledge" else item.question.rstrip() + " " + ckl
            context = score_variant(item, scorer, context_knowledge, context_question, False, args)
            pdp_k, pdp_q = replace_container_span(item, shortcut, pdp["text"])
            joint_knowledge = pdp_k.rstrip() + " " + ckl if shortcut.container == "knowledge" else pdp_k
            joint_question = pdp_q if shortcut.container == "knowledge" else pdp_q.rstrip() + " " + ckl
            joint = score_variant(item, scorer, joint_knowledge, joint_question, False, args)
            result["context_link"] = {"available": True, "text": ckl, "variant": context}
            result["joint"] = {"available": True, "text": ckl, "variant": joint}
        else:
            result["context_link"] = {"available": False, "reason": "context_link_validation_failed"}
            result["joint"] = {"available": False}
    else:
        result["context_link"] = {"available": False, "reason": "no_distinct_constraint"}
        result["joint"] = {"available": False}

    result["status"] = "complete"
    return result


# ---------------------------------------------------------------------------
# Semantic metrics
# ---------------------------------------------------------------------------


def variant_pair(record: Mapping[str, Any], condition: str) -> Mapping[str, Any] | None:
    if condition == "original":
        return record.get("original", {}).get("pair")
    if condition in ("prior_low", "prior_mid", "prior_high", "common_control", "pdp"):
        return record.get("selected", {}).get(condition, {}).get("full", {}).get("pair")
    if condition == "context_link":
        return record.get("context_link", {}).get("variant", {}).get("pair")
    if condition == "joint":
        return record.get("joint", {}).get("variant", {}).get("pair")
    return None


def condition_metrics(records: Sequence[dict[str, Any]], condition: str, gated: bool = False) -> dict[str, Any]:
    rows = []
    for r in records:
        original = variant_pair(r, "original")
        candidate = variant_pair(r, condition)
        if not original or not candidate:
            continue
        trigger = bool(r.get("localization", {}).get("detector", {}).get("predicted_hallucination"))
        chosen = candidate if (not gated or trigger) else original
        rows.append((original, chosen, trigger))
    if not rows:
        return {"n": 0}
    orig_correct = [bool(x[0]["is_pair_correct"]) for x in rows]
    new_correct = [bool(x[1]["is_pair_correct"]) for x in rows]
    orig_m = [float(x[0]["correct_margin"]) for x in rows]
    new_m = [float(x[1]["correct_margin"]) for x in rows]
    w2c = sum((not o) and n for o, n in zip(orig_correct, new_correct))
    c2w = sum(o and (not n) for o, n in zip(orig_correct, new_correct))
    return {
        "n": len(rows),
        "pair_accuracy": float(np.mean(new_correct)),
        "base_pair_accuracy": float(np.mean(orig_correct)),
        "mean_correct_margin": float(np.mean(new_m)),
        "mean_margin_change": float(np.mean(np.asarray(new_m) - np.asarray(orig_m))),
        "wrong_to_correct": int(w2c),
        "correct_to_wrong": int(c2w),
        "net_corrections": int(w2c - c2w),
        "trigger_rate": float(np.mean([x[2] for x in rows])),
        "actual_intervention_rate": float(np.mean([x[2] for x in rows])) if gated else 1.0,
    }


def full_split_condition_metrics(
    records: Sequence[dict[str, Any]], condition: str, gated: bool
) -> dict[str, Any]:
    rows = []
    for r in records:
        original = variant_pair(r, "original")
        if not original:
            continue
        candidate = variant_pair(r, condition)
        trigger = bool(
            r.get("localization", {}).get("detector", {}).get("predicted_hallucination")
        )
        available = candidate is not None
        intervene = available and (not gated or trigger)
        chosen = candidate if intervene else original
        rows.append((original, chosen, trigger, available, intervene))
    if not rows:
        return {"n": 0}
    orig_correct = np.asarray([bool(x[0]["is_pair_correct"]) for x in rows])
    new_correct = np.asarray([bool(x[1]["is_pair_correct"]) for x in rows])
    orig_m = np.asarray([float(x[0]["correct_margin"]) for x in rows])
    new_m = np.asarray([float(x[1]["correct_margin"]) for x in rows])
    return {
        "n": len(rows),
        "pair_accuracy": float(new_correct.mean()),
        "base_pair_accuracy": float(orig_correct.mean()),
        "mean_correct_margin": float(new_m.mean()),
        "mean_margin_change": float((new_m - orig_m).mean()),
        "wrong_to_correct": int(((~orig_correct) & new_correct).sum()),
        "correct_to_wrong": int((orig_correct & (~new_correct)).sum()),
        "net_corrections": int(((~orig_correct) & new_correct).sum() - (orig_correct & (~new_correct)).sum()),
        "trigger_rate": float(np.mean([x[2] for x in rows])),
        "condition_coverage": float(np.mean([x[3] for x in rows])),
        "actual_intervention_rate": float(np.mean([x[4] for x in rows])),
    }


def fixed_effect_slope(xs_by_item: list[list[float]], ys_by_item: list[list[float]]) -> float | None:
    x_centered, y_centered = [], []
    for xs, ys in zip(xs_by_item, ys_by_item):
        if len(xs) < 2 or len(xs) != len(ys):
            continue
        xa, ya = np.asarray(xs), np.asarray(ys)
        x_centered.extend((xa - xa.mean()).tolist())
        y_centered.extend((ya - ya.mean()).tolist())
    if not x_centered:
        return None
    x = np.asarray(x_centered)
    y = np.asarray(y_centered)
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > EPS else None


def summarize_semantic(records: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    complete = [r for r in records if r.get("status") == "complete"]
    conditions = ("original", "common_control", "prior_low", "prior_mid", "prior_high", "pdp", "context_link", "joint")
    metrics = {c: condition_metrics(complete, c, gated=False) for c in conditions}
    gated = {c: condition_metrics(complete, c, gated=True) for c in ("pdp", "context_link", "joint")}
    full_split = {
        "always_on_where_available": {
            c: full_split_condition_metrics(records, c, gated=False)
            for c in ("pdp", "context_link", "joint")
        },
        "detector_gated": {
            c: full_split_condition_metrics(records, c, gated=True)
            for c in ("pdp", "context_link", "joint")
        },
    }

    xs_by_item: list[list[float]] = []
    ys_by_item: list[list[float]] = []
    surprisal_by_item: list[list[float]] = []
    all_x, all_y, all_s = [], [], []
    for r in complete:
        valid = [x for x in r.get("all_paraphrase_candidates", []) if x.get("valid") and "prior_shortcut_margin" in x]
        xs = [float(x["prior_shortcut_margin"]) for x in valid]
        ys = [float(x["full"]["pair"]["correct_margin"]) for x in valid]
        ss = [float(x["surface_surprisal"]) for x in valid]
        if len(xs) >= 2:
            xs_by_item.append(xs)
            ys_by_item.append(ys)
            surprisal_by_item.append(ss)
            all_x.extend(xs)
            all_y.extend(ys)
            all_s.extend(ss)

    slope = fixed_effect_slope(xs_by_item, ys_by_item)
    slope_s = fixed_effect_slope(surprisal_by_item, xs_by_item)
    rng = random.Random(args.seed)
    boot_slopes = []
    boot_s_slopes = []
    if xs_by_item:
        for _ in range(args.bootstrap_draws):
            inds = [rng.randrange(len(xs_by_item)) for _ in range(len(xs_by_item))]
            boot_slopes.append(fixed_effect_slope([xs_by_item[i] for i in inds], [ys_by_item[i] for i in inds]))
            boot_s_slopes.append(fixed_effect_slope([surprisal_by_item[i] for i in inds], [xs_by_item[i] for i in inds]))
    boot_slopes = [x for x in boot_slopes if x is not None]
    boot_s_slopes = [x for x in boot_s_slopes if x is not None]

    within = {
        "n_items": len(xs_by_item),
        "n_candidate_points": len(all_x),
        "spearman_prior_shortcut_margin_vs_full_correct_margin": float(spearmanr(all_x, all_y).statistic) if len(all_x) >= 3 else None,
        "naive_pointwise_p_value": float(spearmanr(all_x, all_y).pvalue) if len(all_x) >= 3 else None,
        "fixed_effect_slope_prior_to_correct_margin": {
            "slope": slope,
            "bootstrap_95_ci": [float(np.quantile(boot_slopes, .025)), float(np.quantile(boot_slopes, .975))] if boot_slopes else None,
        },
        "spearman_surface_surprisal_vs_prior_shortcut_margin": float(spearmanr(all_s, all_x).statistic) if len(all_x) >= 3 else None,
        "naive_surface_surprisal_p_value": float(spearmanr(all_s, all_x).pvalue) if len(all_x) >= 3 else None,
        "fixed_effect_slope_surprisal_to_prior": {
            "slope": slope_s,
            "bootstrap_95_ci": [float(np.quantile(boot_s_slopes, .025)), float(np.quantile(boot_s_slopes, .975))] if boot_s_slopes else None,
        },
    }

    return {
        "n_input": len(records),
        "n_localized": sum(bool(r.get("localization", {}).get("shortcut")) for r in records),
        "n_complete": len(complete),
        "localization_coverage": sum(bool(r.get("localization", {}).get("shortcut")) for r in records) / max(len(records), 1),
        "semantic_completion_coverage": len(complete) / max(len(records), 1),
        "condition_metrics": metrics,
        "detector_gated_metrics": gated,
        "full_requested_split": full_split,
        "within_item_relationship": within,
    }


# ---------------------------------------------------------------------------
# Internal causal validation
# ---------------------------------------------------------------------------


def consensus_pdp_prompts(record: Mapping[str, Any]) -> list[str]:
    prompts = []
    for candidate in record.get("selected", {}).get("pdp_consensus", []):
        prompt = candidate.get("full", {}).get("prompt") if isinstance(candidate, Mapping) else None
        if isinstance(prompt, str) and prompt not in prompts:
            prompts.append(prompt)
    main = record.get("selected", {}).get("pdp", {}).get("full", {}).get("prompt")
    if isinstance(main, str) and main not in prompts:
        prompts.insert(0, main)
    return prompts


def record_prompts(record: Mapping[str, Any]) -> tuple[str, str] | None:
    original = record.get("original", {}).get("prompt")
    pdp = record.get("selected", {}).get("pdp", {}).get("full", {}).get("prompt")
    if isinstance(original, str) and isinstance(pdp, str):
        return original, pdp
    return None


def internal_eligible(record: Mapping[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if record.get("status") != "complete":
        return False, "semantic_not_complete"
    original = variant_pair(record, "original")
    pdp = variant_pair(record, "pdp")
    if not original or not pdp:
        return False, "missing_pair_scores"
    trigger = bool(record.get("localization", {}).get("detector", {}).get("predicted_hallucination"))
    if args.causal_require_trigger and not trigger:
        return False, "detector_not_triggered"
    if args.internal_population in ("all_detected_errors", "responders") and bool(original["is_pair_correct"]):
        return False, "base_not_hallucinated_pair"
    effect = float(pdp["correct_margin"]) - float(original["correct_margin"])
    if args.internal_population == "responders" and effect < args.causal_min_pdp_effect:
        return False, "pdp_treatment_effect_too_small"
    return True, "eligible"


def single_head_patch_effect(
    scorer: OpenAnswerScorer,
    record: Mapping[str, Any],
    head: tuple[int, int],
    state_cache: dict[tuple[str, str, str, int], HeadStatePackage],
) -> dict[str, float]:
    prompts = record_prompts(record)
    if prompts is None:
        raise ValueError("record missing original/PDP prompts")
    original_prompt, pdp_prompt = prompts
    right = str(record["right_answer"])
    hall = str(record["hallucinated_answer"])
    item_id = str(record["item_id"])
    layer = head[0]

    def states(which: str, answer_name: str, prompt: str, answer: str) -> HeadStatePackage:
        key = (item_id, which, answer_name, layer)
        if key not in state_cache:
            state_cache[key] = scorer.capture_answer_states(prompt, answer, [layer])
        return state_cache[key]

    o = variant_pair(record, "original")
    p = variant_pair(record, "pdp")
    assert o and p
    pdp_r = states("pdp", "right", pdp_prompt, right)
    pdp_h = states("pdp", "hall", pdp_prompt, hall)
    orig_r = states("orig", "right", original_prompt, right)
    orig_h = states("orig", "hall", original_prompt, hall)
    forward = scorer.pair_score_with_patch(original_prompt, right, hall, pdp_r, pdp_h, [head])
    reverse = scorer.pair_score_with_patch(pdp_prompt, right, hall, orig_r, orig_h, [head])
    forward_effect = forward.correct_margin - float(o["correct_margin"])
    reverse_rescue = float(p["correct_margin"]) - reverse.correct_margin
    return {
        "forward_patch_effect": forward_effect,
        "reverse_rescue_effect": reverse_rescue,
        "bidirectional_patch_mean": 0.5 * (forward_effect + reverse_rescue),
    }


def multi_head_patch_effect(
    scorer: OpenAnswerScorer,
    record: Mapping[str, Any],
    heads: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    prompts = record_prompts(record)
    if prompts is None:
        raise ValueError("record missing prompts")
    original_prompt, pdp_prompt = prompts
    right, hall = str(record["right_answer"]), str(record["hallucinated_answer"])
    layers = sorted({h[0] for h in heads})
    pdp_r = scorer.capture_answer_states(pdp_prompt, right, layers)
    pdp_h = scorer.capture_answer_states(pdp_prompt, hall, layers)
    orig_r = scorer.capture_answer_states(original_prompt, right, layers)
    orig_h = scorer.capture_answer_states(original_prompt, hall, layers)
    o = variant_pair(record, "original")
    p = variant_pair(record, "pdp")
    assert o and p
    ablated = scorer.pair_score_with_ablation(original_prompt, right, hall, heads)
    forward = scorer.pair_score_with_patch(original_prompt, right, hall, pdp_r, pdp_h, heads)
    reverse = scorer.pair_score_with_patch(pdp_prompt, right, hall, orig_r, orig_h, heads)
    return {
        "ablation": asdict(ablated),
        "ablation_effect": ablated.correct_margin - float(o["correct_margin"]),
        "forward": asdict(forward),
        "reverse_run": asdict(reverse),
        "forward_patch_effect": forward.correct_margin - float(o["correct_margin"]),
        "reverse_rescue_effect": float(p["correct_margin"]) - reverse.correct_margin,
        "bidirectional_patch_mean": 0.5 * (
            forward.correct_margin - float(o["correct_margin"])
            + float(p["correct_margin"]) - reverse.correct_margin
        ),
    }


def discover_heads(
    scorer: OpenAnswerScorer,
    discovery: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_layers = list(range(len(scorer.layers)))
    attribution_rows = []
    for rec in discovery:
        original_prompt, pdp_prompt = record_prompts(rec) or (None, None)
        if not original_prompt:
            continue
        prompt_variants = consensus_pdp_prompts(rec) or [pdp_prompt]
        attrs = []
        for source_prompt in prompt_variants[: args.pdp_consensus_k]:
            attrs.append(
                scorer.pair_bidirectional_attribution(
                    original_prompt,
                    source_prompt,
                    str(rec["right_answer"]),
                    str(rec["hallucinated_answer"]),
                    all_layers,
                )
            )
        attribution_rows.append(np.mean(np.stack(attrs), axis=0))
    if not attribution_rows:
        return {"selected_heads": [], "reason": "no_attribution_rows"}
    mean_attr = np.mean(np.stack(attribution_rows), axis=0)
    layer_scores = np.maximum(mean_attr, 0).sum(axis=1)
    top_layer_count = min(args.causal_candidate_layers, len(all_layers))
    top_layers = np.argsort(layer_scores)[::-1][:top_layer_count].tolist()
    candidates = []
    for layer_idx in top_layers:
        for head_idx in range(scorer.num_heads):
            candidates.append((float(mean_attr[layer_idx, head_idx]), int(layer_idx), int(head_idx)))
    candidates.sort(reverse=True)
    candidate_heads = [(l, h) for _s, l, h in candidates[: args.causal_candidate_heads]]

    state_cache: dict[tuple[str, str, str, int], HeadStatePackage] = {}
    causal_rows = []
    for rank, head in enumerate(candidate_heads):
        effects = []
        for rec in discovery:
            try:
                effects.append(single_head_patch_effect(scorer, rec, head, state_cache))
            except Exception as exc:
                effects.append({"error": str(exc)})
        valid = [e for e in effects if "error" not in e]
        f = [float(e["forward_patch_effect"]) for e in valid]
        r = [float(e["reverse_rescue_effect"]) for e in valid]
        mean_f = float(np.mean(f)) if f else 0.0
        mean_r = float(np.mean(r)) if r else 0.0
        sf = float(np.mean(np.asarray(f) > 0)) if f else 0.0
        sr = float(np.mean(np.asarray(r) > 0)) if r else 0.0
        mean_bi = 0.5 * (mean_f + mean_r)
        score = max(mean_bi, 0.0) * math.sqrt(sf * sr)
        causal_rows.append({
            "rank_from_attribution": rank,
            "layer": head[0],
            "head": head[1],
            "attribution_score": float(mean_attr[head[0], head[1]]),
            "mean_forward_patch_effect": mean_f,
            "mean_reverse_rescue_effect": mean_r,
            "mean_bidirectional_patch_effect": mean_bi,
            "forward_directional_success": sf,
            "reverse_directional_success": sr,
            "causal_selection_score": score,
            "n_discovery_items": len(valid),
        })
    causal_rows.sort(key=lambda x: (x["causal_selection_score"], x["mean_bidirectional_patch_effect"]), reverse=True)
    selected = [
        (int(x["layer"]), int(x["head"]))
        for x in causal_rows
        if x["forward_directional_success"] >= args.causal_min_directional_success
        and x["reverse_directional_success"] >= args.causal_min_directional_success
        and x["mean_bidirectional_patch_effect"] > 0
    ][: args.top_heads]
    if not selected and causal_rows:
        best = causal_rows[0]
        if best["mean_bidirectional_patch_effect"] > 0:
            selected = [(int(best["layer"]), int(best["head"]))]
    return {
        "top_layers": top_layers,
        "layer_scores": layer_scores.tolist(),
        "mean_attribution": mean_attr.tolist(),
        "candidate_results": causal_rows,
        "selected_heads": [{"layer": l, "head": h} for l, h in selected],
    }


def matched_random_sets(
    selected: Sequence[tuple[int, int]],
    num_heads: int,
    runs: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    out = []
    selected_set = set(selected)
    by_layer = Counter(l for l, _ in selected)
    for _ in range(runs):
        heads = []
        used = set()
        for layer, count in by_layer.items():
            pool = [(layer, h) for h in range(num_heads) if (layer, h) not in selected_set and (layer, h) not in used]
            if len(pool) < count:
                pool = [(layer, h) for h in range(num_heads) if (layer, h) not in used]
            chosen = rng.sample(pool, k=min(count, len(pool)))
            heads.extend(chosen)
            used.update(chosen)
        out.append(heads)
    return out


def validate_head_set(
    scorer: OpenAnswerScorer,
    validation: Sequence[dict[str, Any]],
    selected: Sequence[tuple[int, int]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = []
    random_sets = matched_random_sets(selected, scorer.num_heads, args.random_head_runs, rng)
    dose_values = sorted(set(int(x) for x in args.causal_dose_k.split(",") if x.strip()))
    for rec in validation:
        row: dict[str, Any] = {
            "item_index": rec["item_index"],
            "item_id": rec["item_id"],
            "selected_heads": [{"layer": l, "head": h} for l, h in selected],
        }
        target = multi_head_patch_effect(scorer, rec, selected) if selected else None
        row["target_head_set"] = target
        controls = []
        for run_idx, random_heads in enumerate(random_sets):
            if not random_heads:
                continue
            effect = multi_head_patch_effect(scorer, rec, random_heads)
            effect["run_index"] = run_idx
            effect["heads"] = [{"layer": l, "head": h} for l, h in random_heads]
            controls.append(effect)
        row["layer_matched_random_controls"] = controls
        doses = []
        for k in dose_values:
            chosen = list(selected[: min(k, len(selected))])
            if chosen:
                effect = multi_head_patch_effect(scorer, rec, chosen)
                effect["k"] = len(chosen)
                effect["heads"] = [{"layer": l, "head": h} for l, h in chosen]
                doses.append(effect)
        row["dose_response"] = doses
        if target:
            original = variant_pair(rec, "original")
            pdp = variant_pair(rec, "pdp")
            total = float(pdp["correct_margin"]) - float(original["correct_margin"]) if original and pdp else 0.0
            mediated = max(0.0, min(1.0, float(target["bidirectional_patch_mean"]) / max(total, EPS))) if total > 0 else 0.0
            row["pdp_treatment_effect"] = total
            row["mediation_fraction_proxy"] = mediated
        rows.append(row)
    return rows


def repeated_crossfit_internal(
    records: Sequence[dict[str, Any]],
    scorer: OpenAnswerScorer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    eligible = []
    exclusions = []
    for r in records:
        ok, reason = internal_eligible(r, args)
        if ok:
            eligible.append(r)
        else:
            exclusions.append({"item_index": r.get("item_index"), "item_id": r.get("item_id"), "reason": reason})
    if args.internal_max_items > 0:
        eligible = eligible[: args.internal_max_items]
    if len(eligible) < max(args.causal_folds, 2):
        return {"n": len(eligible), "exclusions": exclusions, "error": "too_few_internal_items"}

    rng = random.Random(args.seed)
    runs = []
    validation_rows = []
    selection_counter: Counter[tuple[int, int]] = Counter()
    for repeat in range(args.causal_repeats):
        order = list(range(len(eligible)))
        rng.shuffle(order)
        folds = [order[i::args.causal_folds] for i in range(args.causal_folds)]
        for fold_idx in range(args.causal_folds):
            val_idx = set(folds[fold_idx])
            validation = [eligible[i] for i in sorted(val_idx)]
            discovery = [eligible[i] for i in order if i not in val_idx]
            discovered = discover_heads(scorer, discovery, args)
            selected = [(int(x["layer"]), int(x["head"])) for x in discovered.get("selected_heads", [])]
            for head in selected:
                selection_counter[head] += 1
            rows = validate_head_set(scorer, validation, selected, args, rng) if selected else []
            for row in rows:
                row["repeat"] = repeat
                row["fold"] = fold_idx
            validation_rows.extend(rows)
            runs.append({
                "repeat": repeat,
                "fold": fold_idx,
                "discovery_item_ids": [x["item_id"] for x in discovery],
                "validation_item_ids": [x["item_id"] for x in validation],
                "discovery": discovered,
            })

    target_a = []
    target_f = []
    target_r = []
    random_a = []
    random_f = []
    random_r = []
    mediation = []
    for row in validation_rows:
        target = row.get("target_head_set")
        if target:
            target_a.append(float(target["ablation_effect"]))
            target_f.append(float(target["forward_patch_effect"]))
            target_r.append(float(target["reverse_rescue_effect"]))
            mediation.append(float(row.get("mediation_fraction_proxy", 0.0)))
        controls = row.get("layer_matched_random_controls", [])
        if controls:
            random_a.append(float(np.mean([x["ablation_effect"] for x in controls])))
            random_f.append(float(np.mean([x["forward_patch_effect"] for x in controls])))
            random_r.append(float(np.mean([x["reverse_rescue_effect"] for x in controls])))

    n_runs = args.causal_repeats * args.causal_folds
    stability = [
        {"layer": l, "head": h, "selection_count": count, "selection_frequency": count / n_runs}
        for (l, h), count in selection_counter.most_common()
    ]
    target_minus_random_a = [a - b for a, b in zip(target_a, random_a)]
    target_minus_random_f = [a - b for a, b in zip(target_f, random_f)]
    target_minus_random_r = [a - b for a, b in zip(target_r, random_r)]
    return {
        "n": len(eligible),
        "eligible_item_ids": [x["item_id"] for x in eligible],
        "exclusions": exclusions,
        "n_crossfit_runs": n_runs,
        "runs": runs,
        "validation_rows": validation_rows,
        "head_stability": stability,
        "held_out_target_heads": {
            "ablation_mean_effect": mean_or_none(target_a),
            "ablation_directional_success": float(np.mean(np.asarray(target_a) > 0)) if target_a else None,
            "forward_patch_mean_effect": mean_or_none(target_f),
            "forward_patch_directional_success": float(np.mean(np.asarray(target_f) > 0)) if target_f else None,
            "reverse_rescue_mean_effect": mean_or_none(target_r),
            "reverse_rescue_directional_success": float(np.mean(np.asarray(target_r) > 0)) if target_r else None,
        },
        "layer_matched_random_controls": {
            "ablation_mean_effect": mean_or_none(random_a),
            "forward_patch_mean_effect": mean_or_none(random_f),
            "reverse_rescue_mean_effect": mean_or_none(random_r),
        },
        "target_minus_random": {
            "ablation": {
                "mean_difference": mean_or_none(target_minus_random_a),
                "bootstrap_95_ci": bootstrap_mean_ci(target_minus_random_a, args.bootstrap_draws, args.seed + 29),
            },
            "forward_patch": {
                "mean_difference": mean_or_none(target_minus_random_f),
                "bootstrap_95_ci": bootstrap_mean_ci(target_minus_random_f, args.bootstrap_draws, args.seed + 31),
            },
            "reverse_rescue": {
                "mean_difference": mean_or_none(target_minus_random_r),
                "bootstrap_95_ci": bootstrap_mean_ci(target_minus_random_r, args.bootstrap_draws, args.seed + 37),
            },
        },
        "mean_mediation_fraction_proxy": mean_or_none(mediation),
        "design": "repeated cross-fitted gradient-times-activation-difference pre-screen; discovery-fold bidirectional sequence patching; held-out sequence-level validation; layer-matched random controls",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def markdown_summary(summary: Mapping[str, Any]) -> str:
    sem = summary.get("semantic_counterfactuals", {})
    lines = [
        "# KeyShift v10 — HaluEval open-answer summary",
        "",
        f"- Input items: {sem.get('n_input', 0)}",
        f"- Shortcut localized: {sem.get('n_localized', 0)} ({100 * sem.get('localization_coverage', 0):.1f}%)",
        f"- Semantic counterfactual complete: {sem.get('n_complete', 0)} ({100 * sem.get('semantic_completion_coverage', 0):.1f}%)",
        "",
        "## Sequence-level semantic conditions",
        "",
        "| Condition | n | Pair accuracy | Mean correct margin | Margin change | W→C | C→W |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in sem.get("condition_metrics", {}).items():
        if not m or not m.get("n"):
            continue
        lines.append(
            f"| {name} | {m['n']} | {m.get('pair_accuracy', 0):.3f} | {m.get('mean_correct_margin', 0):.3f} | {m.get('mean_margin_change', 0):+.3f} | {m.get('wrong_to_correct', 0)} | {m.get('correct_to_wrong', 0)} |"
        )
    within = sem.get("within_item_relationship", {})
    lines += [
        "",
        "## Within-item relationship",
        "",
        f"- Candidate points: {within.get('n_candidate_points')}",
        f"- Spearman(prior shortcut margin, full correct margin): {within.get('spearman_prior_shortcut_margin_vs_full_correct_margin')}",
        f"- Fixed-effect slope: {within.get('fixed_effect_slope_prior_to_correct_margin', {}).get('slope')}",
        f"- Fixed-effect bootstrap 95% CI: {within.get('fixed_effect_slope_prior_to_correct_margin', {}).get('bootstrap_95_ci')}",
    ]
    internal = summary.get("internal_causal_validation")
    if internal:
        lines += [
            "",
            "## Internal causal validation",
            "",
            f"- Eligible items: {internal.get('n')}",
            f"- Cross-fit runs: {internal.get('n_crossfit_runs')}",
            f"- Forward target-minus-random: {internal.get('target_minus_random', {}).get('forward_patch')}",
            f"- Reverse target-minus-random: {internal.get('target_minus_random', {}).get('reverse_rescue')}",
            f"- Mean mediation proxy: {internal.get('mean_mediation_fraction_proxy')}",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Original HaluEval QA JSON/JSONL")
    p.add_argument("--detector-predictions", default=None, help="Frozen v7/v8 predictions JSONL; optional")
    p.add_argument("--detector-only", action="store_true", help="run only items covered by detector predictions")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=("semantic", "internal", "all"), default="all")
    p.add_argument("--only-split", default=None)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument(
        "--validate-data-only", action="store_true",
        help="validate/normalize HaluEval fields and exit without loading a model",
    )
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32", "fp16", "bf16", "fp32"))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--answer-prefix", default=" ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-cache", action="store_true")

    p.add_argument("--span-mode", choices=("sentence", "clause"), default="clause")
    p.add_argument("--min-span-words", type=int, default=3)
    p.add_argument("--include-question-spans", action="store_true")
    p.add_argument("--allow-occlusion-fallback", action="store_true")
    p.add_argument("--max-localization-spans", type=int, default=20)
    p.add_argument("--localization-min-effect", type=float, default=0.05)

    p.add_argument("--editor-backend", choices=("local", "openai"), default="local")
    p.add_argument("--editor-model", default=None)
    p.add_argument("--editor-base-url", default=None)
    p.add_argument("--editor-api-key", default=None)
    p.add_argument("--editor-temperature", type=float, default=0.7)
    p.add_argument("--editor-max-retries", type=int, default=3)
    p.add_argument("--paraphrase-candidates", type=int, default=10)
    p.add_argument("--min-valid-paraphrases", type=int, default=3)
    p.add_argument("--min-naturalness", type=int, default=4)
    p.add_argument("--min-paraphrase-token-f1", type=float, default=0.45)
    p.add_argument("--max-edit-ratio", type=float, default=0.55)
    p.add_argument("--max-answer-token-f1", type=float, default=0.80)
    p.add_argument("--max-surprisal-increase", type=float, default=2.0)
    p.add_argument("--pdp-prior-band", type=float, default=0.15)
    p.add_argument("--pdp-consensus-k", type=int, default=3)
    p.add_argument("--ckl-min-anchor-f1", type=float, default=0.30)
    p.add_argument("--ckl-max-answer-token-f1", type=float, default=0.45)
    p.add_argument("--generate-answers", action="store_true")
    p.add_argument("--generation-max-tokens", type=int, default=96)

    p.add_argument("--internal-population", choices=("all_detected_errors", "all_localized", "responders"), default="all_detected_errors")
    p.add_argument("--causal-require-trigger", action="store_true", default=True)
    p.add_argument("--no-causal-require-trigger", dest="causal_require_trigger", action="store_false")
    p.add_argument("--causal-min-pdp-effect", type=float, default=0.10)
    p.add_argument("--causal-folds", type=int, default=4)
    p.add_argument("--causal-repeats", type=int, default=3)
    p.add_argument("--causal-candidate-layers", type=int, default=6)
    p.add_argument("--causal-candidate-heads", type=int, default=48)
    p.add_argument("--top-heads", type=int, default=4)
    p.add_argument("--causal-min-directional-success", type=float, default=0.55)
    p.add_argument("--random-head-runs", type=int, default=10)
    p.add_argument("--causal-dose-k", default="1,2,4")
    p.add_argument("--internal-max-items", type=int, default=0)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "cache.json"
    if args.overwrite_cache and cache_path.exists():
        cache_path.unlink()
    cache = JsonCache(cache_path)

    raw = read_records(args.input)
    items = []
    failures = []
    for idx, obj in enumerate(raw):
        try:
            item = normalize_halueval_item(obj, idx)
            if args.only_split and item.split != args.only_split:
                continue
            items.append(item)
        except Exception as exc:
            failures.append({"index": idx, "error": str(exc)})
    if not items:
        raise RuntimeError("no valid HaluEval items after filtering")

    detector_by_id: dict[str, dict[str, Any]] = {}
    detector_by_index: dict[str, dict[str, Any]] = {}
    if args.detector_predictions:
        for rec in read_records(args.detector_predictions):
            for key in ("item_id", "id"):
                if key in rec:
                    value = str(rec[key])
                    detector_by_id[value] = rec
                    match = re.fullmatch(r"item_(\d+)", value)
                    if match:
                        detector_by_id[str(int(match.group(1)))] = rec
            for key in ("item_index", "idx", "index"):
                if key in rec:
                    detector_by_index[str(rec[key])] = rec
    if args.detector_only:
        if not args.detector_predictions:
            raise ValueError("--detector-only requires --detector-predictions")
        items = [item for item in items if item.item_id in detector_by_id or str(item.idx) in detector_by_index]
        if not items:
            raise RuntimeError("no HaluEval items matched detector predictions")
    if args.max_items > 0:
        items = items[: args.max_items]

    data_report = {
        "n_raw": len(raw),
        "n_valid": len(items),
        "n_invalid": len(failures),
        "canonical_fields": ["knowledge", "question", "right_answer", "hallucinated_answer"],
        "sample": {
            "item_id": items[0].item_id,
            "knowledge_preview": items[0].knowledge[:300],
            "question": items[0].question,
            "right_answer": items[0].right_answer,
            "hallucinated_answer": items[0].hallucinated_answer,
            "open_prompt_preview": open_qa_user_prompt(items[0].knowledge, items[0].question)[:700],
        },
        "failures": failures[:50],
    }
    write_json(outdir / "data_format_report.json", data_report)
    if args.validate_data_only:
        print(json.dumps(data_report, ensure_ascii=False, indent=2), file=sys.stdout)
        return 0

    scorer = OpenAnswerScorer(args.model, args.device, args.dtype, args.trust_remote_code, args.max_length, args.answer_prefix)
    editor = Editor(args, scorer, cache)
    semantic_path = outdir / "semantic_counterfactuals.jsonl"
    existing = load_jsonl_by_id(semantic_path) if args.resume else {}
    if not args.resume and semantic_path.exists() and args.stage in ("semantic", "all"):
        semantic_path.unlink()

    records: list[dict[str, Any]] = []
    if args.stage in ("semantic", "all"):
        for n, item in enumerate(items, 1):
            if args.resume and item.item_id in existing:
                rec = existing[item.item_id]
                records.append(rec)
                print(f"[{n}/{len(items)}] {item.item_id}: resumed", file=sys.stderr, flush=True)
                continue
            det = detector_by_id.get(item.item_id) or detector_by_index.get(str(item.idx))
            try:
                rec = prepare_semantic_item(item, det, scorer, editor, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec = {"item_index": item.idx, "item_id": item.item_id, "status": "error", "error": "CUDA OOM"}
            except Exception as exc:
                rec = {
                    "item_index": item.idx,
                    "item_id": item.item_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            records.append(rec)
            append_jsonl(semantic_path, rec)
            print(f"[{n}/{len(items)}] {item.item_id}: {rec.get('status')}", file=sys.stderr, flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        records = read_records(semantic_path)

    semantic_summary = summarize_semantic(records, args)
    internal_summary = None
    if args.stage in ("internal", "all"):
        internal_summary = repeated_crossfit_internal(records, scorer, args)
        write_json(outdir / "internal_causal_validation.json", internal_summary)
        with (outdir / "internal_causal_validation.jsonl").open("w", encoding="utf-8") as f:
            for row in internal_summary.get("validation_rows", []):
                f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        write_json(outdir / "internal_exclusions.json", internal_summary.get("exclusions", []))

    summary = {
        "method": "KeyShift v10 HaluEval open-answer sequence-level semantic counterfactual and causal validation",
        "data_format": {
            "canonical_fields": ["knowledge", "question", "right_answer", "hallucinated_answer"],
            "primary_observable": "length-normalized teacher-forced sequence log-probability margin: mean_logP(right)-mean_logP(hallucinated)",
            "prompt_exposes_answer_candidates": False,
        },
        "semantic_counterfactuals": semantic_summary,
        "internal_causal_validation": internal_summary,
        "configuration": vars(args),
        "input_failures": failures,
        "files": {
            "semantic_counterfactuals": semantic_path,
            "internal_causal_validation": outdir / "internal_causal_validation.json",
            "summary_json": outdir / "summary.json",
            "summary_markdown": outdir / "summary.md",
        },
    }
    write_json(outdir / "summary.json", summary)
    (outdir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(f"Wrote v10 outputs to {outdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
