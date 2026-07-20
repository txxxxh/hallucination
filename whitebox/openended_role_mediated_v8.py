#!/usr/bin/env python3
"""
Open-ended interventional multimodal span role-mediated hallucination detector v8.

Workflow:
  generate -> score every atomic span by answer-to-span attention
  -> select top-k attention spans only
  -> span interventions -> teacher-forced support/regeneration changes
  -> span role learning -> shortcut/constraint aggregation -> hallucination detection.

References are used only for training pseudo-role construction and evaluation.
No reference-derived quantity is used as a test-time detector feature.
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
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CACHE_SCHEMA_VERSION = "openended_v8_topk_attention_v1"
ROLE_NAMES = ["constraint", "shortcut", "irrelevant"]
ROLE_TO_ID = {x: i for i, x in enumerate(ROLE_NAMES)}
OPERATORS = ("delete", "neutralize", "mask")
FEATURE_SETS = {
    "structure_only": ("structural",),
    "behavior_only": ("behavior",),
    "attention_only": ("attention",),
    "gradient_only": ("gradient",),
    "spectral_only": ("spectral",),
    "behavior_attention": ("behavior", "attention"),
    "behavior_gradient": ("behavior", "gradient"),
    "behavior_spectral": ("behavior", "spectral"),
    "whitebox_combined": ("attention", "gradient", "spectral"),
    "all_combined": ("structural", "behavior", "attention", "gradient", "spectral"),
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


def intervene(source: str, span: Span, op: str, mask: str, neutral: str) -> tuple[str, Optional[Span]]:
    before, after = source[:span.start], source[span.end:]
    if op == "delete":
        x = re.sub(r"[ \t]+", " ", before.rstrip() + " " + after.lstrip())
        x = re.sub(r"\s+([,.;:!?])", r"\1", x)
        return x.strip(), None
    repl = neutral if op == "neutralize" else mask
    return before + repl + after, Span(span.index, repl, len(before), len(before) + len(repl))


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
        text = self.prompt_text(source, system)
        start = text.find(source)
        if start < 0: raise RuntimeError("Cannot locate source in prompt")
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

    def analyze(self, source: str, answer_ids_cpu: torch.Tensor, spans: Sequence[Span], grad: bool, attn: bool, spectral: bool) -> dict[str,Any]:
        p=self.encode_prompt(source); prompt_ids=p.prompt_ids.to(self.device); ans=answer_ids_cpu.to(self.device)
        full=torch.cat([prompt_ids,ans]).unsqueeze(0); mask=torch.ones_like(full); plen=len(prompt_ids); answer_idx=list(range(plen,plen+len(ans)))
        token_map={s.index:self.span_tokens(p,s) for s in spans}; self.model.zero_grad(set_to_none=True)
        ctx=torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out=self.model(input_ids=full,attention_mask=mask,output_attentions=attn or spectral,output_hidden_states=grad,use_cache=False,return_dict=True)
            logits=out.logits[:,plen-1:plen+len(ans)-1,:]; targets=ans.unsqueeze(0)
            lp=F.log_softmax(logits.float(),-1); selected=lp.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
            seq=selected.mean(); probs=lp.exp(); entropy=-(probs*lp).sum(-1).mean()
            if grad:
                for h in out.hidden_states: h.retain_grad()
                (-seq).backward()
        sf={}
        for s in spans:
            f={}; toks=token_map[s.index]
            if attn: f["attention"]=self.attention_features(out.attentions,toks,answer_idx)
            if grad: f["gradient"]=self.gradient_features(out.hidden_states,toks)
            if spectral: f["spectral"]=self.spectral_features(out.attentions,toks,answer_idx,plen)
            sf[s.index]=f
        result={"sequence_logprob":float(seq.detach().cpu()),"mean_token_entropy":float(entropy.detach().cpu()),"span_features":sf,"prompt_tokens":plen,"answer_tokens":len(ans)}
        del out,full,mask; self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

# -------------------------- extraction ------------------------------------

def change_metrics(original: str, regenerated: str, base: dict[str,Any], inter: dict[str,Any]) -> dict[str,float]:
    sim=token_f1(original,regenerated)
    return {
        "support_delta":float(base["sequence_logprob"])-float(inter["sequence_logprob"]),
        "semantic_similarity":sim,
        "answer_changed":float(sim<.80),
        "polarity_changed":float(polarity_sig(original)!=polarity_sig(regenerated)),
        "entity_changed":float(entity_set(original)!=entity_set(regenerated)) if entity_set(original) or entity_set(regenerated) else 0.,
        "number_changed":float(set(_NUMBER_RE.findall(original.replace(',','')))!=set(_NUMBER_RE.findall(regenerated.replace(',','')))) if _NUMBER_RE.search(original+regenerated) else 0.,
        "intervened_original_answer_logprob":float(inter["sequence_logprob"]),
        "entropy_delta":float(inter["mean_token_entropy"])-float(base["mean_token_entropy"]),
    }


def combine_whitebox(base: np.ndarray, rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows: return base.astype(np.float32,copy=False)
    d=np.stack(rows).astype(np.float32)-base[None,:]
    return np.concatenate([base,np.median(d,0),np.max(np.abs(d),0)]).astype(np.float32)


def process_item(ex: Example, engine: WhiteboxEngine, evaluator: CorrectnessEvaluator, args: argparse.Namespace) -> dict[str,Any]:
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

    base = engine.analyze(
        ex.source_text,
        answer_ids,
        spans,
        args.compute_gradient_features,
        True,
        True,
    )
    span_rows=[]
    for span in spans:
        bf=base["span_features"][span.index]; ops=[]; inter_family={}
        for op in OPERATORS:
            modified,repl=intervene(ex.source_text,span,op,args.mask_text,args.neutral_text)
            regenerated,_=engine.generate(modified)
            regenerated_correct=evaluator.evaluate(regenerated,ex.references,ex.question)
            target=[] if repl is None else [repl]
            analysis=engine.analyze(modified,answer_ids,target,False,args.intervention_whitebox,args.intervention_whitebox)
            m=change_metrics(original,regenerated,base,analysis)
            m.update(operator=op,regenerated_answer=regenerated,regenerated_correct=regenerated_correct)
            ops.append(m)
            if args.intervention_whitebox:
                if repl is None:
                    inter_family[op]={"attention":np.zeros(engine.attention_dim,np.float32),"spectral":np.zeros(engine.spectral_dim,np.float32)}
                else:
                    inter_family[op]={"attention":analysis["span_features"][span.index]["attention"],"spectral":analysis["span_features"][span.index]["spectral"]}
        family={
            "structural":structural_features(span,ex.source_text),
            "attention":bf["attention"],
            "gradient":bf.get("gradient",np.zeros(engine.gradient_dim,np.float32)),
            "spectral":bf["spectral"],
        }
        if args.intervention_whitebox:
            for name in ("attention","spectral"):
                family[name]=combine_whitebox(family[name],[inter_family[o][name] for o in OPERATORS])
        selection_row = attention_selection[span.index]
        span_rows.append({
            "span_uid":f"{ex.item_id}::span::{span.index}","span_index":span.index,"span_text":span.text,
            "span_start":span.start,"span_end":span.end,
            "attention_selection_rank":int(attention_rank[span.index]),
            "attention_selection_score":float(selection_row["score"]),
            "attention_selection_mass":float(selection_row["mass"]),
            "attention_selection_density":float(selection_row["density"]),
            "attention_selection_peak":float(selection_row["peak"]),
            "attention_selection_token_count":int(selection_row["token_count"]),
            "family_features":family,"operators":ops,
        })
    return {
        "item_id":ex.item_id,"raw_index":ex.raw_index,"source_text":ex.source_text,"question":ex.question,
        "references":ex.references,"generated_answer":original,"original_correct":correct,
        "hallucination_label":None if correct is None else int(not correct),
        "base_sequence_logprob":base["sequence_logprob"],"base_mean_token_entropy":base["mean_token_entropy"],
        "prompt_tokens":base["prompt_tokens"],"answer_tokens":base["answer_tokens"],
        "n_candidate_spans":len(all_spans),
        "n_selected_spans":len(spans),
        "spans":span_rows,"error":None,
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
        "intervention_whitebox": args.intervention_whitebox,
        "compute_gradient_features": args.compute_gradient_features,
        "lap_topk": args.lap_topk,
        "spectral_anchor_tokens": args.spectral_anchor_tokens,
        "spectral_max_nodes": args.spectral_max_nodes,
    }
    return stable_hash(json.dumps(fields, sort_keys=True, ensure_ascii=False))


def extract_all(examples: Sequence[Example], engine: WhiteboxEngine, evaluator: CorrectnessEvaluator, args: argparse.Namespace, outdir: Path) -> list[dict[str,Any]]:
    cache=outdir/"item_cache"; cache.mkdir(parents=True,exist_ok=True); records=[]
    cache_signature = extraction_cache_signature(args)
    for i,ex in enumerate(tqdm(examples,desc="Extracting")):
        path=cache/f"{i:06d}_{stable_hash(ex.item_id)}_{cache_signature}.pt"
        if path.exists() and not args.overwrite_cache:
            try: records.append(torch_load(path)); continue
            except Exception as e: warnings.warn(f"Bad cache {path}: {e}")
        try: rec=process_item(ex,engine,evaluator,args)
        except Exception as e:
            traceback.print_exc(); rec={"item_id":ex.item_id,"raw_index":ex.raw_index,"source_text":ex.source_text,"question":ex.question,"references":ex.references,"generated_answer":"","original_correct":None,"hallucination_label":None,"spans":[],"error":f"{type(e).__name__}: {e}"}
        atomic_torch_save(rec,path); records.append(rec); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
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


def attach_derived(records: Sequence[dict[str,Any]], scale: float, train_ids: set[str]) -> dict[str,int]:
    counts={x:0 for x in ROLE_NAMES}; counts["ambiguous"]=0
    for item in records:
        for span in item["spans"]:
            span["family_features"]["behavior"]=behavior_vector(span,scale); span["usage"]=usage(span,scale)
            role,rel,reason=pseudo_role(item,span,scale) if item["item_id"] in train_ids else (None,0.,"test_unlabeled")
            span["pseudo_role"],span["role_reliability"],span["pseudo_role_reason"]=role,rel,reason
            counts[role if role is not None else "ambiguous"]+=1
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


def role_metrics(y: Sequence[int],p: np.ndarray) -> dict[str,Any]:
    y=np.asarray(y); pred=p.argmax(1); out={"n":len(y),"accuracy":float(accuracy_score(y,pred)),"log_loss":float(log_loss(y,p,labels=[0,1,2])),"class_counts":{ROLE_NAMES[i]:int((y==i).sum()) for i in range(3)},"per_role":{}}
    aucs=[]
    for i,n in enumerate(ROLE_NAMES):
        b=(y==i).astype(int)
        if len(np.unique(b))<2: a=pr=None
        else: a=float(roc_auc_score(b,p[:,i])); pr=float(average_precision_score(b,p[:,i])); aucs.append(a)
        out["per_role"][n]={"auroc":a,"auprc":pr}
    out["macro_ovr_auroc"]=float(np.mean(aucs)) if aucs else None
    return out


def oof_roles(records: dict[str,dict[str,Any]],train_ids: list[str],labels: np.ndarray,feature_set: str,args: argparse.Namespace):
    minimum=int(np.bincount(labels).min()); folds=max(2,min(args.cv_folds,minimum)); sk=StratifiedKFold(folds,shuffle=True,random_state=args.seed)
    ids=np.asarray(train_ids); all_probs={}; true=[]; probs=[]
    for fold,(tr,va) in enumerate(sk.split(ids,labels)):
        X,y,w,_=labeled_arrays(records,ids[tr].tolist(),feature_set); model=fit_role(X,y,w,args.pca_dim,args.seed+fold)
        ss=spans_for(records,ids[va].tolist()); xv=np.stack([span_vector(s,feature_set) for s in ss]); pv=predict_role(model,xv)
        for s,p in zip(ss,pv): all_probs[s["span_uid"]]=p
        for s,p in zip(ss,pv):
            if s["pseudo_role"] is not None: true.append(ROLE_TO_ID[s["pseudo_role"]]); probs.append(p)
    return all_probs,role_metrics(true,np.stack(probs))


# ---------------- item mechanism ------------------------------------------

def aggregate(item: dict[str,Any], probs: dict[str,np.ndarray], topk: int):
    scont=[]; ccont=[]; us=[]; details=[]
    for span in item["spans"]:
        p=probs[span["span_uid"]]; u=float(span["usage"]); sc=u*float(p[1]); cc=u*float(p[0])
        us.append(u); scont.append(sc); ccont.append(cc)
        details.append({"span_uid":span["span_uid"],"span_index":span["span_index"],"span_text":span["span_text"],"usage":u,"role_probabilities":{ROLE_NAMES[i]:float(p[i]) for i in range(3)},"shortcut_contribution":sc,"constraint_contribution":cc,"operators":span["operators"]})
    if not us: x=np.zeros(4,np.float32)
    else:
        den=float(np.sum(us))+1e-8; order=sorted(scont,reverse=True); k=min(max(1,topk),len(order))
        x=np.asarray([np.sum(scont)/den,np.sum(ccont)/den,max(scont),np.mean(order[:k])],np.float32)
    return x,{"shortcut_evidence":float(x[0]),"constraint_evidence":float(x[1]),"max_shortcut_contribution":float(x[2]),"topk_shortcut_contribution":float(x[3]),"spans":details}


class Mechanism(nn.Module):
    def __init__(self,rate: float):
        super().__init__(); rate=float(np.clip(rate,1e-4,1-1e-4)); self.bias=nn.Parameter(torch.tensor(math.log(rate/(1-rate)),dtype=torch.float32)); self.raw=nn.Parameter(torch.zeros(4))
    def betas(self): return F.softplus(self.raw)
    def forward(self,x):
        b=self.betas(); return self.bias+b[0]*x[:,0]-b[1]*x[:,1]+b[2]*x[:,2]+b[3]*x[:,3]


def fit_mechanism(X: np.ndarray,y: np.ndarray,args: argparse.Namespace) -> Mechanism:
    torch.manual_seed(args.seed); xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32); m=Mechanism(float(y.mean()))
    pos=max(float(y.sum()),1); neg=max(float(len(y)-y.sum()),1); pw=torch.tensor(neg/pos); opt=torch.optim.Adam(m.parameters(),lr=args.mechanism_lr); best=(float("inf"),None)
    for _ in range(args.mechanism_epochs):
        opt.zero_grad(); logits=m(xt); loss=F.binary_cross_entropy_with_logits(logits,yt,pos_weight=pw)+args.mechanism_weight_decay*m.betas().pow(2).sum(); loss.backward(); opt.step()
        if float(loss)<best[0]: best=(float(loss),{k:v.detach().clone() for k,v in m.state_dict().items()})
    if best[1] is not None: m.load_state_dict(best[1])
    m.eval(); return m


def mech_predict(m: Mechanism,X: np.ndarray) -> np.ndarray:
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(X,dtype=torch.float32))).numpy()


def threshold_f1(y: np.ndarray,p: np.ndarray) -> float:
    pr,re,th=precision_recall_curve(y,p)
    if len(th)==0: return .5
    f=2*pr[:-1]*re[:-1]/np.clip(pr[:-1]+re[:-1],1e-12,None); return float(th[int(np.nanargmax(f))])


def binary_metrics(y: np.ndarray,p: np.ndarray,t: float) -> dict[str,Any]:
    pred=(p>=t).astype(int); out={"n":len(y),"positive_rate":float(y.mean()),"threshold":t,"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    out["auroc"]=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None; out["auprc"]=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None
    return out


def mech_params(m: Mechanism) -> dict[str,Any]:
    b=m.betas().detach().numpy(); return {"bias":float(m.bias.detach()),"beta_shortcut_mean":float(b[0]),"beta_constraint_mean":float(b[1]),"beta_shortcut_top1":float(b[2]),"beta_shortcut_topk":float(b[3]),"formula":"bias + b_s_mean*shortcut_mean - b_c_mean*constraint_mean + b_s_top1*shortcut_top1 + b_s_topk*shortcut_topk"}

# ---------------- evaluation / outputs ------------------------------------

def explain_stats(y: np.ndarray,s: np.ndarray,seed: int,draws: int) -> dict[str,Any]:
    rng=np.random.default_rng(seed); pos=s[y==1]; neg=s[y==0]; diff=float(pos.mean()-neg.mean())
    def boot(v):
        z=[rng.choice(v,len(v),replace=True).mean() for _ in range(draws)]; return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
    db=[]
    for _ in range(draws): db.append(rng.choice(pos,len(pos),replace=True).mean()-rng.choice(neg,len(neg),replace=True).mean())
    count=0
    for _ in range(draws):
        p=rng.permutation(y); d=s[p==1].mean()-s[p==0].mean(); count+=abs(d)>=abs(diff)
    pooled=(((len(pos)-1)*pos.var(ddof=1)+(len(neg)-1)*neg.var(ddof=1))/max(len(pos)+len(neg)-2,1)); dval=diff/math.sqrt(max(pooled,1e-12))
    order=np.argsort(s); groups=np.array_split(order,4); dose=[]
    for i,g in enumerate(groups,1): dose.append({"quartile":i,"n":len(g),"shortcut_evidence_min":float(s[g].min()),"shortcut_evidence_max":float(s[g].max()),"hallucination_rate":float(y[g].mean())})
    return {"shortcut_evidence_by_outcome":{"hallucination":{"n":len(pos),"mean":float(pos.mean()),"median":float(np.median(pos)),"std":float(pos.std()),"bootstrap_mean_95_ci":boot(pos)},"correct":{"n":len(neg),"mean":float(neg.mean()),"median":float(np.median(neg)),"std":float(neg.std()),"bootstrap_mean_95_ci":boot(neg)},"mean_difference_hallucination_minus_correct":diff,"difference_bootstrap_95_ci":[float(np.quantile(db,.025)),float(np.quantile(db,.975))],"label_permutation_p_value":float((count+1)/(draws+1)),"cohens_d":float(dval),"shortcut_evidence_auroc":float(roc_auc_score(y,s)),"shortcut_evidence_auprc":float(average_precision_score(y,s))},"shortcut_evidence_dose_response":dose}


def evaluate_feature_set(name: str,records: dict[str,dict[str,Any]],train_ids: list[str],test_ids: list[str],ytr: np.ndarray,yte: np.ndarray,args: argparse.Namespace) -> dict[str,Any]:
    oof,role_oof=oof_roles(records,train_ids,ytr,name,args)
    xtr=[]; tr_details={}
    for iid in train_ids:
        x,d=aggregate(records[iid],oof,args.item_top_k); xtr.append(x); tr_details[iid]=d
    xtr=np.stack(xtr); mechanism=fit_mechanism(xtr,ytr,args); ptr=mech_predict(mechanism,xtr); threshold=threshold_f1(ytr,ptr)
    Xr,yr,wr,_=labeled_arrays(records,train_ids,name); full_role=fit_role(Xr,yr,wr,args.pca_dim,args.seed)
    test_spans=spans_for(records,test_ids); xts=np.stack([span_vector(s,name) for s in test_spans]); pp=predict_role(full_role,xts); test_probs={s["span_uid"]:p for s,p in zip(test_spans,pp)}
    xte=[]; te_details={}
    for iid in test_ids:
        x,d=aggregate(records[iid],test_probs,args.item_top_k); xte.append(x); te_details[iid]=d
    xte=np.stack(xte); pte=mech_predict(mechanism,xte)
    train_spans=spans_for(records,train_ids); xfs=np.stack([span_vector(s,name) for s in train_spans]); pfs=predict_role(full_role,xfs); train_probs={s["span_uid"]:p for s,p in zip(train_spans,pfs)}
    xfull=np.stack([aggregate(records[iid],train_probs,args.item_top_k)[0] for iid in train_ids]); pfull=mech_predict(mechanism,xfull)
    pca=full_role["pca"]
    return {
        "feature_set":name,"feature_families":list(FEATURE_SETS[name]),
        "role_head":{"n_input_features":int(Xr.shape[1]),"n_training_spans":len(Xr),"classifier_classes":[int(x) for x in full_role["classifier"].classes_],"pca_dim":None if pca is None else int(pca.n_components_),"pca_explained_variance_ratio_sum":None if pca is None else float(pca.explained_variance_ratio_.sum())},
        "span_role_train_oof":role_oof,"selected_threshold_from_train_oof":threshold,"role_mechanism":mech_params(mechanism),
        "item_metrics":{"train_oof":binary_metrics(ytr,ptr,threshold),"train_full":binary_metrics(ytr,pfull,threshold),"test":binary_metrics(yte,pte,threshold)},
        "_artifacts":{"role_model":full_role,"mechanism_state":{k:v.detach().cpu() for k,v in mechanism.state_dict().items()},"threshold":threshold},
        "_predictions":{"test_prob":pte,"test_details":te_details},
    }


def base_row(item: dict[str,Any]) -> dict[str,Any]:
    return {k:item.get(k) for k in ("item_id","raw_index","question","generated_answer","references","original_correct","hallucination_label","base_sequence_logprob","base_mean_token_entropy","prompt_tokens","answer_tokens","n_candidate_spans","n_selected_spans","error")} | {"n_spans":len(item.get("spans",[]))}


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed); outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); write_json(outdir/"run_config.json",vars(args))
    rows=load_rows(args); examples=build_examples(rows,args); engine=WhiteboxEngine(args); evaluator=CorrectnessEvaluator(args.correctness_mode,args.token_f1_threshold,engine)
    records=extract_all(examples,engine,evaluator,args,outdir); del engine; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    failed=[x for x in records if x.get("error")]; valid=[x for x in records if not x.get("error") and x.get("hallucination_label") is not None and x.get("spans")]
    if len(valid)<20: raise RuntimeError(f"Only {len(valid)} valid items")
    labels=np.asarray([x["hallucination_label"] for x in valid],dtype=int)
    if len(np.unique(labels))<2: raise RuntimeError("Need correct and hallucinated outputs")
    ids=np.asarray([x["item_id"] for x in valid]); tr,te,ytr,yte=train_test_split(ids,labels,test_size=args.test_size,random_state=args.seed,stratify=labels)
    train_ids,test_ids=tr.tolist(),te.tolist(); train_set=set(train_ids); scale=estimate_scale(valid,train_set,args.minimum_support_scale); counts=attach_derived(valid,scale,train_set); byid={x["item_id"]:x for x in valid}
    base_path=outdir/"base_open_features.jsonl"; inter_path=outdir/"intervention_open_features.jsonl"
    for p in (base_path,inter_path):
        if p.exists(): p.unlink()
    for item in records:
        append_jsonl(base_path,base_row(item))
        for span in item.get("spans",[]): append_jsonl(inter_path,{"item_id":item["item_id"],"span_uid":span["span_uid"],"span_index":span["span_index"],"span_text":span["span_text"],"attention_selection_rank":span.get("attention_selection_rank"),"attention_selection_score":span.get("attention_selection_score"),"attention_selection_mass":span.get("attention_selection_mass"),"attention_selection_density":span.get("attention_selection_density"),"attention_selection_peak":span.get("attention_selection_peak"),"attention_selection_token_count":span.get("attention_selection_token_count"),"usage":span.get("usage"),"pseudo_role":span.get("pseudo_role"),"role_reliability":span.get("role_reliability"),"pseudo_role_reason":span.get("pseudo_role_reason"),"operators":span["operators"]})
    sets=[x.strip() for x in args.feature_sets.split(",") if x.strip()]
    if args.primary_feature_set not in sets: sets.insert(0,args.primary_feature_set)
    for x in sets:
        if x not in FEATURE_SETS: raise ValueError(f"Unknown feature set {x}")
    results={}; bundle_models={}
    for name in sets:
        print(f"\n=== {name} ===",flush=True); r=evaluate_feature_set(name,byid,train_ids,test_ids,ytr,yte,args); bundle_models[name]=r.pop("_artifacts"); results[name]=r
    primary_private=results[args.primary_feature_set].pop("_predictions")
    for name,r in results.items():
        if name!=args.primary_feature_set: r.pop("_predictions",None)
    pte=np.asarray(primary_private["test_prob"]); details=primary_private["test_details"]; threshold=float(results[args.primary_feature_set]["selected_threshold_from_train_oof"])
    pred_path=outdir/"predictions.jsonl"
    if pred_path.exists(): pred_path.unlink()
    for iid,y,p in zip(test_ids,yte,pte):
        item=byid[iid]; append_jsonl(pred_path,{"item_id":iid,"question":item["question"],"generated_answer":item["generated_answer"],"references":item["references"],"hallucination_label":int(y),"hallucination_probability":float(p),"predicted_hallucination":bool(p>=threshold),"threshold":threshold,"feature_set":args.primary_feature_set,**details[iid]})
    shortcut=np.asarray([details[i]["shortcut_evidence"] for i in test_ids])
    item_rank=sorted(sets,key=lambda n:results[n]["item_metrics"]["test"]["auroc"] or -1,reverse=True); role_rank=sorted(sets,key=lambda n:results[n]["span_role_train_oof"]["macro_ovr_auroc"] or -1,reverse=True)
    summary={
        "method":"open-ended test-time interventional multimodal span role-mediated detector v8 (top-k answer-attention selection)","model":args.model,"data":args.input or args.hf_dataset,
        "n_input":len(rows),"n_examples":len(examples),"n_extracted":len(records),"n_failed":len(failed),"n_refusal_or_unlabeled":len(records)-len(failed)-len(valid),"n_valid":len(valid),"n_train":len(train_ids),"n_test":len(test_ids),"train_positive_rate":float(ytr.mean()),"test_positive_rate":float(yte.mean()),
        "support_scale_from_train":scale,"interventions_used_for_prediction":list(OPERATORS),
        "span_selection":{
            "method":"top_k_answer_to_span_attention",
            "max_intervention_spans":args.max_intervention_spans,
            "control_spans_used":False,
            "layers":args.attention_selection_layers,
            "last_n":args.attention_selection_last_n if args.attention_selection_layers=="last_n" else None,
            "score":args.attention_selection_score,
            "hybrid_density_weight":args.attention_selection_density_weight if args.attention_selection_score=="hybrid" else None,
            "answer_token_aggregation":"mean",
            "head_aggregation":"mean",
        },
        "test_prediction_uses_reference_features":False,"references_used_for_train_pseudo_roles":True,"references_used_for_final_evaluation":True,
        "primary_feature_set":args.primary_feature_set,"feature_sets_evaluated":sets,"feature_set_comparison":results,"feature_set_item_test_auroc_ranking":item_rank,"feature_set_span_role_oof_auroc_ranking":role_rank,"role_pseudo_label_counts":counts,
        "primary_metrics":results[args.primary_feature_set]["item_metrics"],"primary_role_mechanism":results[args.primary_feature_set]["role_mechanism"],"primary_selected_threshold_from_train_oof":threshold,"shortcut_explanatory_statistics_primary":explain_stats(yte,shortcut,args.seed,args.bootstrap_draws),
        "files":{"base_open_features":str(base_path),"intervention_open_features":str(inter_path),"predictions":str(pred_path),"model_bundle":str(outdir/"openended_v8_bundle.joblib"),"item_cache":str(outdir/"item_cache")},
        "failed_items":[{"item_id":x["item_id"],"error":x["error"]} for x in failed],
        "method_notes":{"open_ended_generation":True,"teacher_forcing_target":"original generated answer","span_proposal":"top-k answer-to-span attention only; no control span","attention_selection_is_part_of_detector":True,"behavior_support_definition":"mean token logP(original answer|base)-mean token logP(original answer|intervention)","intervention_regeneration_used":True,"span_roles":ROLE_NAMES,"item_evidence_pooling":"usage-normalized means plus shortcut top1/topk","item_mechanism_sign_constraints":True,"detector_has_global_residual_channel":False,"feature_ablation_warning":"Because attention proposes the intervened spans, attention-based feature sets have a proposal-stage advantage. Use an all-span subset for a proposal-independent family ablation.","causal_audit_warning":"Prediction operators cannot also be independent causal validation operators."},
    }
    joblib.dump({"args":vars(args),"support_scale":scale,"feature_models":bundle_models,"role_names":ROLE_NAMES,"feature_sets":FEATURE_SETS},outdir/"openended_v8_bundle.joblib",compress=3); write_json(outdir/"summary.json",summary)
    print("\nPrimary test metrics:\n"+json.dumps(summary["primary_metrics"]["test"],indent=2)); print(f"Outputs: {outdir}")


# ---------------- CLI ------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input"); p.add_argument("--hf-dataset"); p.add_argument("--hf-subset"); p.add_argument("--hf-split",default="validation")
    p.add_argument("--question-field"); p.add_argument("--context-field"); p.add_argument("--answers-field"); p.add_argument("--prompt-field"); p.add_argument("--id-field"); p.add_argument("--max-samples",type=int,default=0); p.add_argument("--test-size",type=float,default=.25)
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); p.add_argument("--device",default="cuda"); p.add_argument("--dtype",default="bfloat16"); p.add_argument("--trust-remote-code",action="store_true"); p.add_argument("--attn-implementation",default="eager")
    p.add_argument("--max-input-tokens",type=int,default=2048); p.add_argument("--max-new-tokens",type=int,default=64); p.add_argument("--temperature",type=float,default=0.0); p.add_argument("--top-p",type=float,default=.95); p.add_argument("--system-prompt",default=DEFAULT_SYSTEM); p.add_argument("--answer-instruction",default="Provide a concise final answer.")
    p.add_argument("--correctness-mode",choices=("hybrid","exact","token_f1","numeric","llm_judge"),default="hybrid"); p.add_argument("--token-f1-threshold",type=float,default=.8)
    p.add_argument("--min-clause-words",type=int,default=12); p.add_argument("--min-span-words",type=int,default=2); p.add_argument("--max-intervention-spans",type=int,default=4,help="Top-k answer-attention spans to intervene; 0 means all spans")
    p.add_argument("--attention-selection-layers",choices=("all","last_half","last_quarter","last_n"),default="last_quarter",help="Layers used only for top-k attention proposal")
    p.add_argument("--attention-selection-last-n",type=int,default=4,help="Used when --attention-selection-layers last_n")
    p.add_argument("--attention-selection-score",choices=("hybrid","density","mass","peak"),default="hybrid",help="Answer-to-span attention score for top-k proposal")
    p.add_argument("--attention-selection-density-weight",type=float,default=.7,help="Density weight in hybrid z-scored density/peak proposal score")
    p.add_argument("--mask-text",default="[MASKED INFORMATION]"); p.add_argument("--neutral-text",default="This detail is unspecified."); p.add_argument("--intervention-whitebox",action="store_true"); p.add_argument("--minimum-support-scale",type=float,default=.05)
    p.add_argument("--compute-gradient-features",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--lap-topk",type=int,default=10); p.add_argument("--spectral-anchor-tokens",type=int,default=8); p.add_argument("--spectral-max-nodes",type=int,default=64)
    p.add_argument("--feature-sets",default="behavior_attention,behavior_spectral,behavior_gradient"); p.add_argument("--primary-feature-set",default="behavior_attention"); p.add_argument("--pca-dim",type=int,default=128); p.add_argument("--cv-folds",type=int,default=5); p.add_argument("--item-top-k",type=int,default=3); p.add_argument("--mechanism-epochs",type=int,default=2000); p.add_argument("--mechanism-lr",type=float,default=.03); p.add_argument("--mechanism-weight-decay",type=float,default=1e-3)
    p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--bootstrap-draws",type=int,default=2000); p.add_argument("--overwrite-cache",action="store_true")
    return p


def main() -> None:
    p=parser(); args=p.parse_args()
    if bool(args.input)==bool(args.hf_dataset): p.error("Provide exactly one of --input or --hf-dataset")
    if args.primary_feature_set not in FEATURE_SETS: p.error("Unknown primary feature set")
    if not 0.0 <= args.attention_selection_density_weight <= 1.0:
        p.error("--attention-selection-density-weight must be in [0, 1]")
    run(args)


if __name__=="__main__": main()
