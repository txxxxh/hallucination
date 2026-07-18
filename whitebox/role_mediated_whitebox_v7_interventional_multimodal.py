#!/usr/bin/env python3
"""
Test-time interventional multimodal span detector (v7).

This standalone script extends v6 from spectral-only span-role learning to a
controlled feature-family comparison.  Every candidate span may use:

- BEHAVIOR / LOGIT features: chosen-margin changes, normalized effects,
  alternative-probability gains, flips, and cross-operator consistency;
- ATTENTION features: decision-row attention to the span, per-layer/head
  distributions, and intervention-induced attention redistribution;
- GRADIENT features: raw input-gradient and gradient-times-input summaries for
  the model's original chosen-answer contrast, plus intervention deltas;
- SPECTRAL features: token-indexed causal-attention Laplacian eigenvalues for
  every layer/head and their intervention deltas;
- STRUCTURAL features: span type, position, and length.

At BOTH training and test time, all requested span interventions are executed.
Gold-relative effects are used only on the training split to create weak
CONSTRAINT / SHORTCUT / IRRELEVANT labels.  Test-time features are strictly
relative to the model's original choice and never use the gold answer.

The script trains and reports multiple feature-set ablations:

    structure_only
    behavior_only
    attention_only
    gradient_only
    spectral_only
    behavior_attention
    behavior_gradient
    behavior_spectral
    whitebox_combined       (attention + gradient + spectral)
    all_combined            (all families)

Each feature set is evaluated at two levels:
1. span-role pseudo-label prediction;
2. item-level role-mediated hallucination detection.

There is no global residual hallucination head and no chosen_is_a feature.
The final detector for every feature set is monotonic:

    bias + beta_shortcut * shortcut_evidence
         - beta_constraint * constraint_evidence.

Important causal-validation note
--------------------------------
The interventions consumed as detector inputs are not independent post-hoc
causal evidence.  Reserve separate semantic-preserving operators for a final
held-out audit.

Example
-------
CUDA_VISIBLE_DEVICES=0 python role_mediated_whitebox_v7_interventional_multimodal.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data /home/tong56/shuffled_prepend_profiles_question.json \
  --out-dir scientistqa_profiles_v7 \
  --span-mode atomic \
  --interventions delete,neutralize,mask \
  --max-intervention-spans 0 \
  --lap-topk 10 \
  --role-pca-dim 128 \
  --dtype bfloat16

Requirements
------------
pip install "transformers>=4.44" torch accelerate numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROLE_NAMES = ("constraint", "shortcut", "irrelevant")
ROLE_TO_ID = {name: i for i, name in enumerate(ROLE_NAMES)}
EPS = 1e-9

QUESTION_KEYS = ("question", "scenario", "prompt")
PROMPT_KEYS = ("benchmark_prompt", "prompt")
GOLD_KEYS = ("answer", "gold", "label", "gold_answer", "rgt_ans")

NUMBERED_OPTION_RE = re.compile(
    r"(?m)^\s*(?P<number>[12])\.\s*(?P<text>.+?)\s*$"
)
QUESTION_MARKER_RE = re.compile(r"(?im)^\s*Question:\s*")
FINAL_PERSON_QUESTION_RE = re.compile(
    r"(?is)\bWho\s+is\s+this\s+person\s*\?\s*$"
)

# Long alternatives must appear before their shorter components.
NEGATION_OPERATOR_RE = re.compile(
    r"""
    \b(?:
        did\s+not | does\s+not | do\s+not |
        was\s+not | were\s+not | is\s+not | are\s+not |
        has\s+not | have\s+not | had\s+not |
        could\s+not | would\s+not | should\s+not | can\s+not |
        cannot | can't | couldn't | wouldn't | shouldn't |
        wasn't | weren't | isn't | aren't |
        didn't | doesn't | don't | hasn't | haven't | hadn't |
        never | neither | nor | without |
        no(?!\s+longer\b)
    )\b
    """,
    re.I | re.X,
)

# Split broad prose into short semantic fragments without splitting ordinary
# proper-name phrases such as "Arts and Sciences" or "Physiology or Medicine".
ATOMIC_BOUNDARY_RE = re.compile(
    r"""
    (?:
        ;\s+ |
        :\s+ |
        \s+[—–]\s+ |
        ,\s+ |
        \s+(?=(?:but|yet|however|although|though|whereas|while)\b)
    )
    """,
    re.I | re.X,
)

WORD_TOKEN_RE = re.compile(r"\b[\w][\w'’.-]*\b", re.UNICODE)
ENTITY_CONNECTORS = {
    "at", "de", "del", "der", "des", "du", "for", "from", "in", "la",
    "le", "of", "on", "or", "the", "to", "van", "von", "with",
}
ATOMIC_SINGLE_WORD_STOPLIST = {
    "additionally", "although", "and", "but", "despite", "however", "nor",
    "or", "though", "throughout", "whereas", "while", "yet",
}
ENTITY_TRIGGER_WORDS = {
    "academy", "award", "chancellor", "college", "cross", "doctorate",
    "fellow", "fellowship", "foundation", "institute", "medal", "minister",
    "order", "prize", "society", "university",
}

SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
CLAUSE_BOUNDARY_RE = re.compile(
    r"""
    (?:
        ;\s+ |
        \s+[—–-]\s+ |
        ,\s+(?=(?:but|yet|however|although|though|while|whereas)\b) |
        \s+(?=(?:but|yet|however|although|though|whereas)\b)
    )
    """,
    re.I | re.X,
)
SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data structures and JSON helpers
# ---------------------------------------------------------------------------


@dataclass
class CandidateSpan:
    span_id: int
    start: int
    end: int
    text: str
    span_type: str = "structural"


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
        return records

    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Accept common wrappers.
        for key in ("items", "data", "records", "questions"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("input must be a JSON list, JSONL, or a dict containing a list")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def load_jsonl_index(path: Path, key: str = "idx") -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(rec[key])] = rec
    return out


def pick_key(item: dict, explicit: str | None, candidates: Sequence[str]) -> str:
    if explicit:
        if explicit not in item:
            raise KeyError(f"requested field {explicit!r} is absent")
        return explicit
    for key in candidates:
        if key in item:
            return key
    raise KeyError(f"none of the expected fields are present: {candidates}")


def _normalize_option_text(value: Any) -> str:
    """Normalize option/person text without discarding Unicode distinctions."""
    return SPACE_RE.sub(" ", str(value)).strip().casefold()


def parse_numbered_options(prompt_text: str) -> dict[str, str]:
    """Parse lines such as ``1. Name`` and ``2. Name`` from a full prompt."""
    parsed: dict[str, str] = {}
    for match in NUMBERED_OPTION_RE.finditer(prompt_text):
        number = match.group("number")
        if number not in parsed:
            parsed[number] = match.group("text").strip()
    return parsed


def normalize_gold(
    value: Any,
    choices: tuple[str, str],
    prompt_text: str | None = None,
) -> str:
    """Normalize a numeric gold label or map a gold option name to 1/2.

    ScientistQA records store ``rgt_ans`` as the correct person's name rather
    than as the option index.  When ``prompt_text`` contains numbered options,
    this function maps that name to ``choices[0]`` or ``choices[1]``.
    """
    text = str(value).strip()
    # Accept numeric JSON values and strings like "1".
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text in choices:
        return text

    if prompt_text is not None:
        options = parse_numbered_options(prompt_text)
        target = _normalize_option_text(text)
        option_1 = options.get("1")
        option_2 = options.get("2")
        if option_1 is not None and target == _normalize_option_text(option_1):
            return choices[0]
        if option_2 is not None and target == _normalize_option_text(option_2):
            return choices[1]

        raise ValueError(
            f"gold answer {value!r} is not one of {choices} and does not "
            f"match parsed options: {options}"
        )

    raise ValueError(f"gold answer {value!r} is not one of {choices}")


# ---------------------------------------------------------------------------
# Candidate spans: structural proposal only, no role keywords
# ---------------------------------------------------------------------------


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def clause_spans(text: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for sent_start, sent_end in sentence_spans(text):
        sent = text[sent_start:sent_end]
        pieces = []
        cursor = 0
        for match in CLAUSE_BOUNDARY_RE.finditer(sent):
            a, b = cursor, match.start()
            a, b = _trim_span(sent, a, b)
            if a < b:
                pieces.append((sent_start + a, sent_start + b))
            cursor = match.end()
        a, b = _trim_span(sent, cursor, len(sent))
        if a < b:
            pieces.append((sent_start + a, sent_start + b))

        # Avoid creating tiny fragments. If clause splitting is poor, keep
        # the full sentence as one candidate.
        useful = [
            span for span in pieces
            if len(re.findall(r"\w+", text[span[0]:span[1]])) >= 3
        ]
        output.extend(useful if len(useful) >= 2 else [(sent_start, sent_end)])
    return output


def scientistqa_body_bounds(text: str) -> tuple[int, int] | None:
    """Locate the evidence body in a numbered ScientistQA prompt.

    The model still receives the complete prompt. Only candidate-span
    proposal and subsequent interventions are restricted to this interval.
    """
    marker = QUESTION_MARKER_RE.search(text)
    option_matches = list(NUMBERED_OPTION_RE.finditer(text))
    numbers = {m.group("number") for m in option_matches}
    if marker is None or not {"1", "2"}.issubset(numbers):
        return None

    body_start = marker.end()
    final_match = FINAL_PERSON_QUESTION_RE.search(text, body_start)
    body_end = final_match.start() if final_match is not None else len(text)
    body_start, body_end = _trim_span(text, body_start, body_end)
    return (body_start, body_end) if body_start < body_end else None


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def atomic_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence spans that do not split after initials such as ``C.``."""
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    for i, char in enumerate(text):
        if char not in ".!?":
            continue

        if char == ".":
            prefix = text[start:i]
            word_match = re.search(r"([A-Za-z]+)$", prefix)
            previous_word = word_match.group(1) if word_match else ""
            if len(previous_word) == 1 and previous_word.isupper():
                continue
            if previous_word.casefold() in {
                "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs"
            }:
                continue
            if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue

        j = i + 1
        while j < n and text[j].isspace():
            j += 1
        if j < n and not text[j].isupper():
            continue

        a, b = _trim_span(text, start, i + 1)
        if a < b:
            spans.append((a, b))
        start = i + 1

    a, b = _trim_span(text, start, n)
    if a < b:
        spans.append((a, b))
    return spans


def _is_entity_content_token(token: str) -> bool:
    stripped = token.strip(".'’-")
    if not stripped:
        return False
    if any(ch.isdigit() for ch in stripped):
        return True
    first_alpha = next((ch for ch in stripped if ch.isalpha()), "")
    return bool(first_alpha and first_alpha.isupper()) or (
        len(stripped) >= 2 and stripped.isupper()
    )


def named_entity_spans(text: str, base_offset: int = 0) -> list[tuple[int, int]]:
    """Heuristically extract maximal proper-name / institution spans."""
    tokens = [
        (m.group(0), m.start(), m.end()) for m in WORD_TOKEN_RE.finditer(text)
    ]
    proposals: list[tuple[int, int]] = []

    def gap_allows(previous_index: int, current_index: int) -> bool:
        if previous_index < 0:
            return True
        previous = tokens[previous_index][0].strip(".'’-")
        gap = text[tokens[previous_index][2]:tokens[current_index][1]]
        if gap.strip() == "":
            return True
        # Permit the period after a one-letter initial: Arthur C. Cope.
        if (
            len(previous) == 1
            and previous.isupper()
            and re.fullmatch(r"\.\s*", gap)
        ):
            return True
        return False

    for i, (token, start, _) in enumerate(tokens):
        if not _is_entity_content_token(token):
            continue

        j = i
        content_count = 0
        has_trigger = False
        last_content_j = i - 1
        while j < len(tokens):
            if j > i and not gap_allows(j - 1, j):
                break

            word = tokens[j][0]
            lower = word.strip(".'’-").casefold()
            if _is_entity_content_token(word):
                content_count += 1
                has_trigger = has_trigger or lower in ENTITY_TRIGGER_WORDS
                last_content_j = j
                j += 1
                continue

            if lower in ENTITY_CONNECTORS:
                k = j
                while k < len(tokens):
                    if k > j and not gap_allows(k - 1, k):
                        break
                    connector = tokens[k][0].strip(".'’-").casefold()
                    if connector not in ENTITY_CONNECTORS:
                        break
                    k += 1
                if (
                    k < len(tokens)
                    and gap_allows(k - 1, k)
                    and _is_entity_content_token(tokens[k][0])
                ):
                    j += 1
                    continue
            break

        if last_content_j < i:
            continue
        phrase_start = start
        phrase_end = tokens[last_content_j][2]
        if content_count >= 2 or has_trigger:
            proposals.append(
                (base_offset + phrase_start, base_offset + phrase_end)
            )

    proposals = sorted(set(proposals), key=lambda x: (x[0], -(x[1] - x[0])))
    maximal: list[tuple[int, int]] = []
    for candidate in proposals:
        if any(a <= candidate[0] and b >= candidate[1] for a, b in maximal):
            continue
        maximal.append(candidate)
    return sorted(maximal)


def _split_by_boundaries(text: str, start: int, end: int) -> list[tuple[int, int]]:
    segment = text[start:end]
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for match in ATOMIC_BOUNDARY_RE.finditer(segment):
        a, b = _trim_span(segment, cursor, match.start())
        if a < b:
            pieces.append((start + a, start + b))
        cursor = match.end()
    a, b = _trim_span(segment, cursor, len(segment))
    if a < b:
        pieces.append((start + a, start + b))
    return pieces


def atomic_body_spans(
    full_text: str,
    body_start: int,
    body_end: int,
    min_words: int,
) -> list[tuple[int, int, str]]:
    """Create fine-grained, partly overlapping ScientistQA evidence spans."""
    body = full_text[body_start:body_end]
    raw: list[tuple[int, int, str]] = []

    def add_fragment(a: int, b: int) -> None:
        a, b = _trim_span(full_text, a, b)
        if a >= b:
            return
        fragment = full_text[a:b]
        words = re.findall(r"\w+", fragment, flags=re.UNICODE)
        if len(words) >= min_words:
            raw.append((a, b, "predicate_atom"))
        elif len(words) == 1:
            token = words[0].casefold()
            if len(token) >= 4 and token not in ATOMIC_SINGLE_WORD_STOPLIST:
                raw.append((a, b, "lexical_atom"))

    for local_sent_start, local_sent_end in atomic_sentence_spans(body):
        sent_start = body_start + local_sent_start
        sent_end = body_start + local_sent_end
        sent_start, sent_end = _trim_span(full_text, sent_start, sent_end)
        while sent_end > sent_start and full_text[sent_end - 1] in ".!?":
            sent_end -= 1
        sent_start, sent_end = _trim_span(full_text, sent_start, sent_end)
        if sent_start >= sent_end:
            continue

        for a, b in named_entity_spans(
            full_text[sent_start:sent_end], base_offset=sent_start
        ):
            raw.append((a, b, "named_entity"))

        for piece_start, piece_end in _split_by_boundaries(
            full_text, sent_start, sent_end
        ):
            piece = full_text[piece_start:piece_end]
            cursor = 0
            for match in NEGATION_OPERATOR_RE.finditer(piece):
                before_a, before_b = _trim_span(piece, cursor, match.start())
                if before_a < before_b:
                    a, b = piece_start + before_a, piece_start + before_b
                    add_fragment(a, b)

                op_a, op_b = _trim_span(piece, match.start(), match.end())
                if op_a < op_b:
                    raw.append(
                        (
                            piece_start + op_a,
                            piece_start + op_b,
                            "negation_operator",
                        )
                    )
                cursor = match.end()

            after_a, after_b = _trim_span(piece, cursor, len(piece))
            if after_a < after_b:
                a, b = piece_start + after_a, piece_start + after_b
                add_fragment(a, b)

    priority = {
        "negation_operator": 0,
        "named_entity": 1,
        "lexical_atom": 2,
        "predicate_atom": 3,
    }
    best: dict[tuple[int, int], tuple[int, int, str]] = {}
    for a, b, span_type in raw:
        a, b = _trim_span(full_text, a, b)
        if a >= b:
            continue
        key = (a, b)
        current = best.get(key)
        if current is None or priority[span_type] < priority[current[2]]:
            best[key] = (a, b, span_type)

    return sorted(
        best.values(),
        key=lambda x: (x[0], x[1], priority.get(x[2], 99)),
    )


def structured_numbered_prompt_spans(
    text: str,
    mode: str,
    min_words: int,
) -> list[tuple[int, int, str]] | None:
    """Propose evidence spans only inside the ScientistQA question body."""
    bounds = scientistqa_body_bounds(text)
    if bounds is None:
        return None
    body_start, body_end = bounds

    if mode == "atomic":
        return atomic_body_spans(text, body_start, body_end, min_words)

    body = text[body_start:body_end]
    body_spans = clause_spans(body) if mode == "clause" else sentence_spans(body)
    span_type = "body_clause" if mode == "clause" else "body_sentence"
    return [
        (body_start + a, body_start + b, span_type)
        for a, b in body_spans
    ]


def propose_spans(
    question: str,
    mode: str = "atomic",
    include_question_span: bool = False,
    min_words: int = 3,
) -> list[CandidateSpan]:
    structured_raw = structured_numbered_prompt_spans(
        question, mode, min_words
    )
    if structured_raw is not None:
        raw = structured_raw
    else:
        structural = (
            clause_spans(question)
            if mode in {"clause", "atomic"}
            else sentence_spans(question)
        )
        fallback_type = (
            "body_clause" if mode in {"clause", "atomic"} else "body_sentence"
        )
        raw = [(a, b, fallback_type) for a, b in structural]

    candidates: list[CandidateSpan] = []
    for start, end, span_type in raw:
        text = question[start:end].strip()
        required_words = {
            "negation_operator": 1,
            "named_entity": 2,
            "lexical_atom": 1,
        }.get(span_type, min_words)
        if _word_count(text) < required_words:
            continue
        if not include_question_span and text.rstrip().endswith("?"):
            continue

        candidates.append(
            CandidateSpan(
                span_id=len(candidates),
                start=start,
                end=end,
                text=text,
                span_type=span_type,
            )
        )

    if not candidates:
        for start, end, span_type in raw:
            text = question[start:end].strip()
            if _word_count(text) >= 1:
                candidates.append(
                    CandidateSpan(
                        span_id=len(candidates),
                        start=start,
                        end=end,
                        text=text,
                        span_type=span_type,
                    )
                )

    return candidates


# ---------------------------------------------------------------------------
# Prompt rendering and interventions
# ---------------------------------------------------------------------------


class PromptAdapter:
    def __init__(
        self,
        tokenizer,
        question_field: str | None,
        prompt_field: str | None,
        gold_field: str | None,
        answer_instruction: str,
        apply_chat_template: bool,
    ):
        self.tok = tokenizer
        self.question_field = question_field
        self.prompt_field = prompt_field
        self.gold_field = gold_field
        self.answer_instruction = answer_instruction
        self.apply_chat_template = apply_chat_template

    def unpack(self, item: dict) -> tuple[str, str | None, Any]:
        q_key = pick_key(item, self.question_field, QUESTION_KEYS)
        g_key = pick_key(item, self.gold_field, GOLD_KEYS)
        question = str(item[q_key])

        prompt_key = None
        if self.prompt_field:
            prompt_key = pick_key(item, self.prompt_field, PROMPT_KEYS)
        else:
            for key in PROMPT_KEYS:
                if key in item:
                    prompt_key = key
                    break

        base_prompt = str(item[prompt_key]) if prompt_key else None
        return question, base_prompt, item[g_key]

    def user_text(
        self,
        original_question: str,
        modified_question: str,
        base_prompt: str | None,
    ) -> str:
        if base_prompt is None:
            text = modified_question
        elif "{question}" in base_prompt:
            text = base_prompt.replace("{question}", modified_question)
        elif original_question in base_prompt:
            text = base_prompt.replace(original_question, modified_question, 1)
        else:
            raise ValueError(
                "prompt field does not contain the exact question and has no "
                "'{question}' placeholder; safe intervention is impossible"
            )

        instruction = self.answer_instruction.strip()
        if instruction and instruction.lower() not in text.lower():
            text = text.rstrip() + "\n" + instruction
        return text

    def render(
        self,
        original_question: str,
        modified_question: str,
        base_prompt: str | None,
    ) -> str:
        user_text = self.user_text(original_question, modified_question, base_prompt)
        if not self.apply_chat_template:
            return user_text
        if not hasattr(self.tok, "apply_chat_template"):
            raise ValueError("tokenizer does not provide apply_chat_template")
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )


def normalize_edited_text(text: str) -> str:
    # Preserve numbered-prompt line structure.  Collapsing every newline would
    # make intervention prompts differ from originals for an unrelated reason.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"[ \t]+([,.;:!?])", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def intervene(question: str, span: CandidateSpan, kind: str) -> str:
    before, target, after = (
        question[:span.start],
        question[span.start:span.end],
        question[span.end:],
    )
    span_type = getattr(span, "span_type", "structural")

    if kind == "delete":
        replacement = ""
    elif kind == "neutralize":
        if span_type == "negation_operator":
            replacement = "[POLARITY UNSPECIFIED]"
        elif span_type == "named_entity":
            replacement = "[NAMED ENTITY UNSPECIFIED]"
        else:
            replacement = "[DETAIL UNSPECIFIED]"
    elif kind == "mask":
        if span_type == "negation_operator":
            replacement = "[POLARITY OMITTED]"
        elif span_type == "named_entity":
            replacement = "[NAMED ENTITY OMITTED]"
        else:
            replacement = "[DETAIL OMITTED]"
    elif kind == "negate":
        stripped = target.strip()
        if span_type == "negation_operator":
            replacement = ""
        else:
            stripped = (
                stripped[:-1]
                if stripped.endswith((".", "?", "!"))
                else stripped
            )
            replacement = f"It is not true that {stripped}."
    else:
        raise ValueError(f"unknown intervention: {kind}")

    return normalize_edited_text(before + replacement + after)


# ---------------------------------------------------------------------------
# White-box extractor
# ---------------------------------------------------------------------------


def band_summary(values: np.ndarray, name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    thirds = np.array_split(values, 3)
    out = {
        f"{name}_early": float(np.mean(thirds[0])),
        f"{name}_mid": float(np.mean(thirds[1])),
        f"{name}_late": float(np.mean(thirds[2])),
        f"{name}_mean": float(np.mean(values)),
        f"{name}_max": float(np.max(values)),
        f"{name}_std": float(np.std(values)),
    }
    return out


def safe_entropy_binary(log_a: float, log_b: float) -> float:
    vals = torch.tensor([log_a, log_b], dtype=torch.float64)
    p = torch.softmax(vals, dim=0)
    return float(-(p * torch.log(p + 1e-12)).sum().item())


class WeakWhiteboxExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lap_topk: int = 10,
    ):
        self.device = device
        self.dtype = dtype
        self.lap_topk = lap_topk

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if not getattr(self.tok, "is_fast", False):
            raise ValueError("a fast tokenizer is required for offset mappings")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self._choice_id_cache: dict[str, list[int]] = {}

    def choice_token_ids(self, choice: str) -> list[int]:
        if choice in self._choice_id_cache:
            return self._choice_id_cache[choice]
        ids: set[int] = set()
        for variant in (choice, " " + choice):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            enc = self.tok.encode(choice, add_special_tokens=False)
            if not enc:
                raise ValueError(f"choice {choice!r} tokenized to nothing")
            warnings.warn(
                f"choice {choice!r} is not a single token; using first token only"
            )
            ids.add(enc[0])
        result = sorted(ids)
        self._choice_id_cache[choice] = result
        return result

    def _choice_logs(self, logits: torch.Tensor, choices: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids_a = self.choice_token_ids(choices[0])
        ids_b = self.choice_token_ids(choices[1])
        log_a = torch.logsumexp(logits[ids_a], dim=0)
        log_b = torch.logsumexp(logits[ids_b], dim=0)
        return log_a, log_b

    @torch.no_grad()
    def score_prompt(
        self,
        prompt: str,
        gold: str,
        choices: tuple[str, str],
    ) -> dict:
        enc = self.tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(self.device)
        out = self.model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -1].float()
        log_a, log_b = self._choice_logs(logits, choices)

        chosen = choices[0] if log_a >= log_b else choices[1]
        gold_log = log_a if gold == choices[0] else log_b
        other_log = log_b if gold == choices[0] else log_a
        chosen_log = torch.maximum(log_a, log_b)
        rejected_log = torch.minimum(log_a, log_b)

        return {
            "log_a": float(log_a.item()),
            "log_b": float(log_b.item()),
            "gold_margin": float((gold_log - other_log).item()),
            "chosen_margin": float((chosen_log - rejected_log).item()),
            "chosen": chosen,
            "choice_entropy": safe_entropy_binary(log_a.item(), log_b.item()),
        }

    def extract_original(
        self,
        prompt: str,
        question: str,
        spans: Sequence[CandidateSpan],
        choices: tuple[str, str],
    ) -> dict:
        """
        Extract features from the original prompt only.

        The decision-row attention is taken at the final prompt position,
        which is the causal position whose logits predict the first answer
        token. This avoids the one-token row shift in teacher-forced analyses.
        """
        enc = self.tok(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0]
        ids = enc["input_ids"].to(self.device)
        T = ids.shape[1]

        question_base = prompt.find(question)
        if question_base < 0:
            raise ValueError("rendered prompt does not contain the exact question")

        masks: list[torch.Tensor] = []
        for span in spans:
            absolute = (question_base + span.start, question_base + span.end)
            mask = self._char_mask(offsets, absolute, T)
            if not mask.any():
                raise ValueError(f"candidate span matched no tokens: {span.text!r}")
            masks.append(mask.to(self.device))

        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                output_attentions=True,
                use_cache=False,
            )
            logits = out.logits[0, -1].float()
            attns = out.attentions

        log_a, log_b = self._choice_logs(logits, choices)
        chosen = choices[0] if log_a >= log_b else choices[1]
        chosen_contrast = (log_a - log_b) if chosen == choices[0] else (log_b - log_a)

        global_features: dict[str, float] = {
            "choice_margin_abs": float(abs((log_a - log_b).item())),
            "choice_entropy": safe_entropy_binary(log_a.item(), log_b.item()),
            "chosen_is_a": float(chosen == choices[0]),
            "prompt_tokens": float(T),
            "num_candidate_spans": float(len(spans)),
        }

        # Gradient x input for the model's own chosen contrast.
        grad_norm, grad_signed = self._contrastive_input_gradient(
            ids=ids,
            choices=choices,
            chosen=chosen,
        )

        span_features: list[dict[str, float]] = []
        for span, mask in zip(spans, masks):
            idx = mask.nonzero().squeeze(-1)
            span_features.append(
                {
                    "span_words": float(len(re.findall(r"\w+", span.text))),
                    "span_tokens": float(idx.numel()),
                    "span_characters": float(span.end - span.start),
                    "span_type_negation_operator": float(
                        span.span_type == "negation_operator"
                    ),
                    "span_type_named_entity": float(
                        span.span_type == "named_entity"
                    ),
                    "span_type_predicate_atom": float(
                        span.span_type == "predicate_atom"
                    ),
                    "span_type_lexical_atom": float(
                        span.span_type == "lexical_atom"
                    ),
                    "span_type_body_clause": float(
                        span.span_type == "body_clause"
                    ),
                    "span_type_body_sentence": float(
                        span.span_type == "body_sentence"
                    ),
                    "span_relative_start": float(span.start / max(len(question), 1)),
                    "span_relative_end": float(span.end / max(len(question), 1)),
                    "span_relative_length": float((span.end - span.start) / max(len(question), 1)),
                    "grad_norm_sum": float(grad_norm[idx].sum().item()),
                    "grad_norm_density": float(grad_norm[idx].mean().item()),
                    "grad_signed_sum": float(grad_signed[idx].sum().item()),
                    "grad_signed_density": float(grad_signed[idx].mean().item()),
                }
            )

        layer_entropy = []
        layer_lapeigs = []
        # Temporary per-layer arrays for each span.
        per_span_attn_mass = [[] for _ in spans]
        per_span_attn_density = [[] for _ in spans]
        per_span_attn_headmax = [[] for _ in spans]
        per_span_lap = [[] for _ in spans]

        for A in attns:
            # [H, T, T]
            Ah = A[0].float()
            decision = Ah[:, T - 1, :T]  # [H, T], predicts the first answer token

            p = decision.mean(0)
            p = p / (p.sum() + EPS)
            layer_entropy.append(float((-(p * (p + 1e-12).log()).sum()).item()))

            # Head-averaged causal Laplacian diagonal. Token identity is kept
            # for span pooling, while sorted top-k is retained as a baseline.
            Amean = Ah.mean(0)
            col = torch.arange(T, device=Amean.device)
            denom = (T - 1 - col).clamp(min=1).float()
            received = Amean.tril(-1).sum(0) / denom
            lam = received - torch.diagonal(Amean)

            k = min(self.lap_topk, T)
            top = torch.sort(lam, descending=True).values[:k].cpu().numpy()
            layer_lapeigs.append(top)

            for j, mask in enumerate(masks):
                idx = mask.nonzero().squeeze(-1)
                head_mass = decision[:, idx].sum(-1)
                per_span_attn_mass[j].append(float(head_mass.mean().item()))
                per_span_attn_density[j].append(
                    float((decision[:, idx].mean(-1)).mean().item())
                )
                per_span_attn_headmax[j].append(float(head_mass.max().item()))
                per_span_lap[j].append(float(lam[idx].mean().item()))

        global_features.update(
            band_summary(np.asarray(layer_entropy), "decision_attn_entropy")
        )

        lap_arr = np.stack(layer_lapeigs) if layer_lapeigs else np.zeros((0, 0))
        if lap_arr.size:
            for rank in range(lap_arr.shape[1]):
                global_features.update(
                    band_summary(lap_arr[:, rank], f"lapeig{rank}")
                )

        for j in range(len(spans)):
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_mass[j]), "attn_mass")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_density[j]), "attn_density")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_headmax[j]), "attn_headmax")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_lap[j]), "lap_token")
            )

        return {
            "chosen": chosen,
            "log_a": float(log_a.item()),
            "log_b": float(log_b.item()),
            "chosen_margin": float(abs((log_a - log_b).item())),
            "global_features": global_features,
            "span_features": span_features,
        }

    @staticmethod
    def _char_mask(offsets: torch.Tensor, span: tuple[int, int], n_tokens: int) -> torch.Tensor:
        start, end = span
        mask = torch.zeros(n_tokens, dtype=torch.bool)
        for i, (a, b) in enumerate(offsets[:n_tokens].tolist()):
            if a < end and b > start and b > a:
                mask[i] = True
        return mask

    def _contrastive_input_gradient(
        self,
        ids: torch.Tensor,
        choices: tuple[str, str],
        chosen: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(ids).detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
        logits = out.logits[0, -1].float()
        log_a, log_b = self._choice_logs(logits, choices)
        contrast = (log_a - log_b) if chosen == choices[0] else (log_b - log_a)
        contrast.backward()

        gx = inputs_embeds.grad * inputs_embeds
        grad_norm = gx.norm(dim=-1)[0].detach()
        grad_signed = gx.sum(dim=-1)[0].detach()
        self.model.zero_grad(set_to_none=True)
        return grad_norm, grad_signed


# ---------------------------------------------------------------------------
# Pseudo-role construction
# ---------------------------------------------------------------------------


def softmax_np(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / np.sum(exp)


def build_soft_role(
    deltas: Sequence[float],
    deadzone: float,
    temperature: float,
) -> dict:
    """
    Convert intervention contributions into a soft role distribution.

    contribution > 0: constraint-like
    contribution < 0: shortcut-like
    near zero: irrelevant-like
    """
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("at least one intervention delta is required")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    logits = [
        median / max(temperature, EPS),
        -median / max(temperature, EPS),
        (deadzone - abs(median)) / max(temperature, EPS),
    ]
    probs = softmax_np(logits)

    if abs(median) >= deadzone:
        target_sign = 1.0 if median > 0 else -1.0
        informative = np.abs(values) >= deadzone * 0.25
        if informative.any():
            agreement = float(
                np.mean(np.sign(values[informative]) == target_sign)
            )
        else:
            agreement = 0.0
    else:
        agreement = float(np.mean(np.abs(values) <= deadzone))

    stability = math.exp(-mad / max(temperature, EPS))
    reliability = float(np.clip(agreement * stability, 0.0, 1.0))

    hard_id = int(np.argmax(probs))
    return {
        "median_contribution": median,
        "mad": mad,
        "agreement": agreement,
        "reliability": reliability,
        "role_probs": probs.tolist(),
        "hard_role": ROLE_NAMES[hard_id],
    }


# ---------------------------------------------------------------------------
# Feature matrix and classifier heads
# ---------------------------------------------------------------------------


class FeatureMatrix:
    def __init__(self, keys: Sequence[str] | None = None):
        self.keys = list(keys) if keys is not None else None

    def fit(self, dicts: Sequence[dict[str, float]]) -> "FeatureMatrix":
        self.keys = sorted(set().union(*(d.keys() for d in dicts)))
        return self

    def transform(self, dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if self.keys is None:
            raise RuntimeError("FeatureMatrix is not fitted")
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in dicts],
            dtype=float,
        )
        # Keep missing/invalid numerical values from crashing sklearn. Their
        # occurrence should still be audited in output logs.
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


class SpanRoleHead:
    def __init__(self, C: float = 1.0):
        self.matrix = FeatureMatrix()
        self.C = C
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
            ),
        )

    def fit(
        self,
        feat_dicts: Sequence[dict[str, float]],
        soft_labels: Sequence[Sequence[float]],
        reliabilities: Sequence[float],
    ) -> "SpanRoleHead":
        self.matrix.fit(feat_dicts)
        X_base = self.matrix.transform(feat_dicts)

        rows = []
        labels = []
        weights = []
        for x, probs, rel in zip(X_base, soft_labels, reliabilities):
            probs_arr = np.asarray(probs, dtype=float)
            for role_id in range(3):
                weight = float(max(rel, 1e-4) * max(probs_arr[role_id], 1e-6))
                rows.append(x)
                labels.append(role_id)
                weights.append(weight)

        X = np.asarray(rows)
        y = np.asarray(labels, dtype=int)
        w = np.asarray(weights, dtype=float)
        self.pipe.fit(X, y, logisticregression__sample_weight=w)
        return self

    def predict_proba(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        X = self.matrix.transform(feat_dicts)
        probs = self.pipe.predict_proba(X)

        # Guarantee fixed role order 0,1,2 even if sklearn changes class order.
        classes = self.pipe.named_steps["logisticregression"].classes_
        out = np.zeros((len(feat_dicts), 3), dtype=float)
        for col, cls in enumerate(classes):
            out[:, int(cls)] = probs[:, col]
        return out

    def coefficient_report(self, top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
        lr = self.pipe.named_steps["logisticregression"]
        report: dict[str, list[tuple[str, float]]] = {}
        for row, cls in zip(lr.coef_, lr.classes_):
            ranked = sorted(
                zip(self.matrix.keys or [], row),
                key=lambda pair: -abs(pair[1]),
            )[:top_n]
            report[ROLE_NAMES[int(cls)]] = [
                (name, float(coef)) for name, coef in ranked
            ]
        return report



class UsageNormalizer:
    """Convert original-prompt span features into a bounded usage score.

    The score is intentionally label-free. It combines standardized
    decision-row attention density, head-max attention, and contrastive
    gradient density. Higher usage means the model's chosen 1-vs-2 decision
    is more locally tied to the span.
    """

    DEFAULT_KEYS = (
        "attn_density_late",
        "attn_headmax_late",
        "grad_norm_density",
    )

    def __init__(self, keys: Sequence[str] | None = None, temperature: float = 1.0):
        self.keys = list(keys or self.DEFAULT_KEYS)
        self.temperature = float(temperature)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, span_feat_dicts: Sequence[dict[str, float]]) -> "UsageNormalizer":
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in span_feat_dicts],
            dtype=float,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, span_feat_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("UsageNormalizer is not fitted")
        if not span_feat_dicts:
            return np.zeros(0, dtype=float)
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in span_feat_dicts],
            dtype=float,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        Z = self.scaler.transform(X)
        raw = Z.mean(axis=1) / max(self.temperature, EPS)
        raw = np.clip(raw, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-raw))

    def report(self) -> dict:
        return {
            "keys": list(self.keys),
            "temperature": self.temperature,
            "means": self.scaler.mean_.tolist() if self.fitted else None,
            "scales": self.scaler.scale_.tolist() if self.fitted else None,
        }


def build_role_evidence(
    span_features: Sequence[dict[str, float]],
    role_probs: np.ndarray,
    usage_normalizer: UsageNormalizer,
) -> dict:
    """Return additive, span-indexed evidence before mechanism weights."""
    n = len(span_features)
    if n == 0:
        return {
            "usage": np.zeros(0),
            "shortcut_base": np.zeros(0),
            "constraint_base": np.zeros(0),
            "shortcut_evidence": 0.0,
            "constraint_evidence": 0.0,
            "role_ambiguity": 0.0,
        }

    usage = usage_normalizer.transform(span_features)
    shortcut_p = role_probs[:, ROLE_TO_ID["shortcut"]]
    constraint_p = role_probs[:, ROLE_TO_ID["constraint"]]

    # Divide by n so evidence is comparable across questions with different
    # numbers of candidate spans and remains exactly additive.
    shortcut_base = shortcut_p * usage / n
    constraint_base = constraint_p * usage / n
    entropy = -(role_probs * np.log(role_probs + 1e-12)).sum(axis=1)

    return {
        "usage": usage,
        "shortcut_base": shortcut_base,
        "constraint_base": constraint_base,
        "shortcut_evidence": float(shortcut_base.sum()),
        "constraint_evidence": float(constraint_base.sum()),
        "role_ambiguity": float(entropy.mean()),
    }


class RoleMechanismHead:
    """Monotonic role channel with exact signed span contributions."""

    def __init__(
        self,
        epochs: int = 2500,
        lr: float = 0.03,
        l2: float = 1e-3,
        min_shortcut_weight: float = 0.05,
        seed: int = 0,
    ):
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.min_shortcut_weight = float(min_shortcut_weight)
        self.seed = int(seed)
        self.bias = 0.0
        self.beta_shortcut = 1.0
        self.beta_constraint = 1.0
        self.fitted = False

    def fit(
        self,
        evidences: Sequence[dict],
        labels: Sequence[int],
    ) -> "RoleMechanismHead":
        X = np.asarray(
            [
                [
                    float(e["shortcut_evidence"]),
                    float(e["constraint_evidence"]),
                ]
                for e in evidences
            ],
            dtype=np.float32,
        )
        y = np.asarray(labels, dtype=np.float32)
        if len(np.unique(y)) < 2:
            raise ValueError("RoleMechanismHead requires both classes")

        torch.manual_seed(self.seed)
        x_t = torch.tensor(X)
        y_t = torch.tensor(y)

        bias = torch.nn.Parameter(torch.tensor(0.0))
        raw_s = torch.nn.Parameter(torch.tensor(0.0))
        raw_c = torch.nn.Parameter(torch.tensor(0.0))
        params = [bias, raw_s, raw_c]
        opt = torch.optim.Adam(params, lr=self.lr)

        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        sample_w = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
        w_t = torch.tensor(sample_w, dtype=torch.float32)

        best = float("inf")
        best_values = None
        patience = 250
        stale = 0

        for _ in range(self.epochs):
            opt.zero_grad()
            beta_s = F.softplus(raw_s)
            beta_c = F.softplus(raw_c)
            logits = bias + beta_s * x_t[:, 0] - beta_c * x_t[:, 1]
            per_item = F.binary_cross_entropy_with_logits(
                logits, y_t, reduction="none"
            )
            loss = (per_item * w_t).mean()
            loss = loss + self.l2 * (beta_s.square() + beta_c.square())
            loss = loss + 5.0 * F.relu(
                torch.tensor(self.min_shortcut_weight) - beta_s
            ).square()
            loss.backward()
            opt.step()

            value = float(loss.detach().item())
            if value + 1e-7 < best:
                best = value
                best_values = (
                    float(bias.detach().item()),
                    float(F.softplus(raw_s).detach().item()),
                    float(F.softplus(raw_c).detach().item()),
                )
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_values is None:
            raise RuntimeError("role mechanism optimization failed")
        self.bias, self.beta_shortcut, self.beta_constraint = best_values
        self.fitted = True
        return self

    def decision_function(self, evidences: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RoleMechanismHead is not fitted")
        s = np.asarray([float(e["shortcut_evidence"]) for e in evidences])
        c = np.asarray([float(e["constraint_evidence"]) for e in evidences])
        return self.bias + self.beta_shortcut * s - self.beta_constraint * c

    def predict_proba(self, evidences: Sequence[dict]) -> np.ndarray:
        z = np.clip(self.decision_function(evidences), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def span_contributions(self, evidence: dict) -> tuple[np.ndarray, np.ndarray]:
        positive = self.beta_shortcut * np.asarray(evidence["shortcut_base"])
        negative = -self.beta_constraint * np.asarray(evidence["constraint_base"])
        return positive, negative

    def report(self) -> dict:
        return {
            "bias": self.bias,
            "beta_shortcut": self.beta_shortcut,
            "beta_constraint": self.beta_constraint,
            "formula": (
                "bias + beta_shortcut * shortcut_evidence "
                "- beta_constraint * constraint_evidence"
            ),
        }


class ResidualHead:
    """Global white-box channel that never receives role features."""

    def __init__(self, cv: int = 5):
        self.matrix = FeatureMatrix()
        self.cv = int(cv)
        self.pipe = None

    def _make_pipe(self, y: np.ndarray):
        pos = int(y.sum())
        neg = int(len(y) - pos)
        folds = min(self.cv, pos, neg)
        if folds >= 2:
            return make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=10,
                    cv=folds,
                    scoring="roc_auc",
                    solver="liblinear",
                    penalty="l1",
                    class_weight="balanced",
                    max_iter=5000,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l1",
                class_weight="balanced",
                max_iter=5000,
            ),
        )

    def fit(
        self,
        feat_dicts: Sequence[dict[str, float]],
        labels: Sequence[int],
    ) -> "ResidualHead":
        self.matrix.fit(feat_dicts)
        X = self.matrix.transform(feat_dicts)
        y = np.asarray(labels, dtype=int)
        self.pipe = self._make_pipe(y)
        self.pipe.fit(X, y)
        return self

    def decision_function(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("ResidualHead is not fitted")
        X = self.matrix.transform(feat_dicts)
        return np.asarray(self.pipe.decision_function(X), dtype=float)

    def predict_proba(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        z = np.clip(self.decision_function(feat_dicts), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def coefficient_report(self, top_n: int = 30) -> list[tuple[str, float]]:
        if self.pipe is None:
            return []
        final_name = list(self.pipe.named_steps.keys())[-1]
        lr = self.pipe.named_steps[final_name]
        row = lr.coef_[0]
        ranked = sorted(
            zip(self.matrix.keys or [], row),
            key=lambda pair: -abs(pair[1]),
        )[:top_n]
        return [(name, float(coef)) for name, coef in ranked]


class FinalCalibrator:
    """Calibrate role + capped residual while preserving decomposition.

    The coefficient on role_logit is fixed to 1. The residual correction is
    bounded by residual_cap. Only a global bias and positive temperature are
    learned.
    """

    def __init__(
        self,
        residual_cap: float = 1.0,
        epochs: int = 2000,
        lr: float = 0.03,
        seed: int = 0,
    ):
        self.residual_cap = float(residual_cap)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.bias = 0.0
        self.temperature = 1.0
        self.fitted = False

    def residual_adjustment(self, residual_logits: Sequence[float]) -> np.ndarray:
        z = np.asarray(residual_logits, dtype=float)
        return self.residual_cap * np.tanh(z)

    def fit(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
        labels: Sequence[int],
    ) -> "FinalCalibrator":
        role = torch.tensor(np.asarray(role_logits), dtype=torch.float32)
        residual = torch.tensor(
            self.residual_adjustment(residual_logits), dtype=torch.float32
        )
        y_np = np.asarray(labels, dtype=np.float32)
        y = torch.tensor(y_np)

        torch.manual_seed(self.seed)
        bias = torch.nn.Parameter(torch.tensor(0.0))
        raw_temp = torch.nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam([bias, raw_temp], lr=self.lr)

        pos = max(float(y_np.sum()), 1.0)
        neg = max(float(len(y_np) - y_np.sum()), 1.0)
        sample_w = np.where(
            y_np > 0.5,
            len(y_np) / (2.0 * pos),
            len(y_np) / (2.0 * neg),
        )
        w = torch.tensor(sample_w, dtype=torch.float32)

        best = float("inf")
        best_values = None
        stale = 0
        for _ in range(self.epochs):
            opt.zero_grad()
            temp = F.softplus(raw_temp) + 0.05
            logits = (role + residual + bias) / temp
            per_item = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = (per_item * w).mean()
            loss.backward()
            opt.step()

            value = float(loss.detach().item())
            if value + 1e-7 < best:
                best = value
                best_values = (
                    float(bias.detach().item()),
                    float(temp.detach().item()),
                )
                stale = 0
            else:
                stale += 1
                if stale >= 200:
                    break

        if best_values is None:
            raise RuntimeError("final calibration failed")
        self.bias, self.temperature = best_values
        self.fitted = True
        return self

    def decision_function(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FinalCalibrator is not fitted")
        role = np.asarray(role_logits, dtype=float)
        adjustment = self.residual_adjustment(residual_logits)
        return (role + adjustment + self.bias) / self.temperature

    def predict_proba(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
    ) -> np.ndarray:
        z = np.clip(
            self.decision_function(role_logits, residual_logits), -40.0, 40.0
        )
        return 1.0 / (1.0 + np.exp(-z))

    def report(self) -> dict:
        return {
            "residual_cap": self.residual_cap,
            "bias": self.bias,
            "temperature": self.temperature,
            "formula": (
                "(role_logit + residual_cap*tanh(residual_logit) + bias)"
                " / temperature"
            ),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def choose_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + EPS
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_binary(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )
    result = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, p))
        result["auprc"] = float(average_precision_score(y, p))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def rank_spans_for_intervention(record: dict, max_spans: int) -> list[int]:
    spans = record["spans"]
    if max_spans <= 0 or len(spans) <= max_spans:
        return list(range(len(spans)))

    # Always retain logical operators and named concepts. Otherwise a pure
    # salience budget can omit exactly the pair we need to contrast, such as
    # "never" versus "Nobel Prize ...". The numerical budget limits only the
    # remaining predicate atoms.
    protected = [
        j
        for j, span in enumerate(spans)
        if span.get("span_type") in {"negation_operator", "named_entity"}
    ]

    scored = []
    protected_set = set(protected)
    for j, span in enumerate(spans):
        if j in protected_set:
            continue
        feat = span["features"]
        score = (
            float(feat.get("attn_density_late", 0.0))
            + float(feat.get("grad_norm_density", 0.0))
        )
        scored.append((score, j))
    scored.sort(reverse=True)
    remaining_budget = max(max_spans - len(protected), 0)
    selected = protected + [j for _, j in scored[:remaining_budget]]
    return sorted(set(selected))


def extract_base_records(
    items: list[dict],
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    args,
    cache_path: Path,
) -> list[dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    records: list[dict] = []

    for idx, item in enumerate(items):
        if idx in cached:
            records.append(cached[idx])
            continue

        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices, base_prompt or question)
        spans = propose_spans(
            question,
            mode=args.span_mode,
            include_question_span=args.include_question_span,
            min_words=args.min_span_words,
        )
        if not spans:
            warnings.warn(f"[{idx}] no candidate spans; skipped")
            continue

        prompt = adapter.render(question, question, base_prompt)
        try:
            extracted = extractor.extract_original(
                prompt=prompt,
                question=question,
                spans=spans,
                choices=choices,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn(f"[{idx}] CUDA OOM; skipped")
            continue
        except Exception as exc:
            warnings.warn(f"[{idx}] base extraction failed: {exc}")
            continue

        chosen = extracted["chosen"]
        record = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "chosen": chosen,
            "hallucinated": int(chosen != gold),
            "chosen_margin": extracted["chosen_margin"],
            "global_features": extracted["global_features"],
            "spans": [
                {
                    **asdict(span),
                    "features": features,
                }
                for span, features in zip(spans, extracted["span_features"])
            ],
        }
        append_jsonl(cache_path, record)
        records.append(record)

        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"base extraction: {idx + 1}/{len(items)}", flush=True)

    return records


def add_intervention_labels(
    records: list[dict],
    items: list[dict],
    train_indices: set[int],
    test_indices: set[int],
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    interventions: list[str],
    args,
    cache_path: Path,
) -> dict[int, dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    output = dict(cached)

    # Weak role labels are produced on training items only.
    # Test interventions are performed later by a separate frozen audit.
    target_indices = set(train_indices)

    for count, record in enumerate(records, 1):
        idx = int(record["idx"])
        if idx not in target_indices or idx in output:
            continue

        item = items[idx]
        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices, base_prompt or question)
        original_prompt = adapter.render(question, question, base_prompt)
        original_score = extractor.score_prompt(original_prompt, gold, choices)
        original_gold_margin = original_score["gold_margin"]

        chosen_span_ids = rank_spans_for_intervention(
            record,
            args.max_intervention_spans,
        )

        span_labels = []
        for j, span_record in enumerate(record["spans"]):
            if j not in chosen_span_ids:
                span_labels.append(
                    {
                        "evaluated": False,
                        "role_probs": [0.0, 0.0, 1.0],
                        "hard_role": "irrelevant",
                        "reliability": 0.0,
                        "interventions": [],
                    }
                )
                continue

            span = CandidateSpan(
                span_id=int(span_record["span_id"]),
                start=int(span_record["start"]),
                end=int(span_record["end"]),
                text=str(span_record["text"]),
                span_type=str(span_record.get("span_type", "structural")),
            )

            evidences = []
            deltas = []
            for kind in interventions:
                modified_question = intervene(question, span, kind)
                try:
                    variant_prompt = adapter.render(
                        question,
                        modified_question,
                        base_prompt,
                    )
                    score = extractor.score_prompt(variant_prompt, gold, choices)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    warnings.warn(f"[{idx}] OOM on {kind} span {j}")
                    continue
                except Exception as exc:
                    warnings.warn(f"[{idx}] intervention failed ({kind}, span {j}): {exc}")
                    continue

                delta = float(original_gold_margin - score["gold_margin"])
                deltas.append(delta)
                evidences.append(
                    {
                        "kind": kind,
                        "gold_margin": score["gold_margin"],
                        "chosen": score["chosen"],
                        "delta": delta,
                        "flip": bool(score["chosen"] != original_score["chosen"]),
                    }
                )

            if deltas:
                pseudo = build_soft_role(
                    deltas,
                    deadzone=args.role_deadzone,
                    temperature=args.role_temperature,
                )
                pseudo.update(
                    {
                        "evaluated": True,
                        "interventions": evidences,
                    }
                )
            else:
                pseudo = {
                    "evaluated": False,
                    "role_probs": [0.0, 0.0, 1.0],
                    "hard_role": "irrelevant",
                    "reliability": 0.0,
                    "interventions": evidences,
                }
            span_labels.append(pseudo)

        rec = {
            "idx": idx,
            "original_gold_margin": original_gold_margin,
            "original_chosen": original_score["chosen"],
            "span_pseudo_labels": span_labels,
        }
        append_jsonl(cache_path, rec)
        output[idx] = rec

        if count % 10 == 0 or count == len(records):
            print(
                f"interventions: processed base item {count}/{len(records)}",
                flush=True,
            )

    return output


def prepare_role_training(
    base_by_idx: dict[int, dict],
    intervention_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    min_reliability: float,
) -> tuple[list[dict], list[list[float]], list[float], list[int], list[int]]:
    features = []
    soft_labels = []
    reliabilities = []
    groups = []
    span_local_ids = []

    for idx in train_indices:
        base = base_by_idx[idx]
        pseudo = intervention_by_idx.get(idx)
        if pseudo is None:
            continue
        for local_id, (span, label) in enumerate(
            zip(base["spans"], pseudo["span_pseudo_labels"])
        ):
            if not label.get("evaluated", False):
                continue
            reliability = float(label.get("reliability", 0.0))
            if reliability < min_reliability:
                continue
            features.append(span["features"])
            soft_labels.append(label["role_probs"])
            reliabilities.append(reliability)
            groups.append(idx)
            span_local_ids.append(local_id)

    if not features:
        raise RuntimeError(
            "no reliable pseudo-labeled spans; lower --min-role-reliability "
            "or inspect intervention validity"
        )
    return features, soft_labels, reliabilities, groups, span_local_ids


def make_role_predictions_by_item(
    base_by_idx: dict[int, dict],
    indices: Sequence[int],
    role_head: SpanRoleHead,
) -> dict[int, np.ndarray]:
    result = {}
    for idx in indices:
        features = [span["features"] for span in base_by_idx[idx]["spans"]]
        result[idx] = role_head.predict_proba(features)
    return result


def make_oof_role_predictions_by_item(
    base_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    role_train_features: list[dict],
    role_train_soft: list[list[float]],
    role_train_rel: list[float],
    role_groups: list[int],
    role_local_ids: list[int],
    n_splits: int,
    full_role_head: SpanRoleHead,
) -> dict[int, np.ndarray]:
    """
    Predict every span of each training item with a role model that was fitted
    without any pseudo-labeled span from that item.

    role_local_ids is accepted for API compatibility and auditing; OOF
    prediction is deliberately performed for all spans, including spans whose
    pseudo-label was filtered for low reliability.
    """
    del role_local_ids
    indices = list(train_indices)
    splits = min(n_splits, len(indices))
    if splits < 2:
        return make_role_predictions_by_item(base_by_idx, indices, full_role_head)

    result: dict[int, np.ndarray] = {}
    splitter = KFold(n_splits=splits, shuffle=True, random_state=0)
    group_arr = np.asarray(role_groups, dtype=int)

    for train_pos, val_pos in splitter.split(indices):
        fit_items = {indices[i] for i in train_pos}
        val_items = [indices[i] for i in val_pos]
        fit_rows = [i for i, group in enumerate(group_arr) if int(group) in fit_items]

        if not fit_rows:
            fold_head = full_role_head
        else:
            fold_head = SpanRoleHead().fit(
                [role_train_features[i] for i in fit_rows],
                [role_train_soft[i] for i in fit_rows],
                [role_train_rel[i] for i in fit_rows],
            )

        for idx in val_items:
            all_span_features = [
                span["features"] for span in base_by_idx[idx]["spans"]
            ]
            result[idx] = fold_head.predict_proba(all_span_features)

    missing = [idx for idx in indices if idx not in result]
    if missing:
        result.update(
            make_role_predictions_by_item(base_by_idx, missing, full_role_head)
        )
    return result


def role_mechanism_oof_logits(
    evidences: list[dict],
    labels: list[int],
    n_splits: int,
    args,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    splits = min(n_splits, pos, neg)
    if splits < 2:
        head = RoleMechanismHead(
            epochs=args.mechanism_epochs,
            lr=args.mechanism_lr,
            l2=args.mechanism_l2,
            min_shortcut_weight=args.min_shortcut_weight,
            seed=args.seed,
        ).fit(evidences, labels)
        return head.decision_function(evidences)

    splitter = StratifiedKFold(
        n_splits=splits, shuffle=True, random_state=args.seed
    )
    oof = np.zeros(len(y), dtype=float)
    dummy = np.zeros((len(y), 1))

    for fold, (train_rows, val_rows) in enumerate(splitter.split(dummy, y)):
        head = RoleMechanismHead(
            epochs=args.mechanism_epochs,
            lr=args.mechanism_lr,
            l2=args.mechanism_l2,
            min_shortcut_weight=args.min_shortcut_weight,
            seed=args.seed + fold,
        ).fit(
            [evidences[i] for i in train_rows],
            [labels[i] for i in train_rows],
        )
        oof[val_rows] = head.decision_function(
            [evidences[i] for i in val_rows]
        )
    return oof


def compute_residual_oof_logits(
    feat_dicts: list[dict],
    labels: list[int],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    matrix = FeatureMatrix().fit(feat_dicts)
    X = matrix.transform(feat_dicts)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    splits = min(n_splits, pos, neg)
    if splits < 2:
        head = ResidualHead(cv=2).fit(feat_dicts, labels)
        return head.decision_function(feat_dicts)

    splitter = StratifiedKFold(
        n_splits=splits, shuffle=True, random_state=seed
    )
    oof = np.zeros(len(y), dtype=float)
    for train_rows, val_rows in splitter.split(X, y):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l1",
                class_weight="balanced",
                max_iter=5000,
            ),
        )
        pipe.fit(X[train_rows], y[train_rows])
        oof[val_rows] = pipe.decision_function(X[val_rows])
    return oof


def sigmoid_np(values: Sequence[float]) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def choose_distinct_explanation_spans(
    base: dict,
    role_probs: np.ndarray,
    evidence: dict,
    role_head: RoleMechanismHead,
    calibrator: FinalCalibrator,
    role_probability_threshold: float,
    usage_threshold: float,
    contribution_threshold: float,
) -> dict:
    """Choose distinct C/S spans and return exact signed contributions."""
    spans = base["spans"]
    usage = np.asarray(evidence["usage"], dtype=float)
    shortcut_pos, constraint_neg = role_head.span_contributions(evidence)

    # Convert exact role-logit contributions into final calibrated-logit units.
    shortcut_final = shortcut_pos / calibrator.temperature
    constraint_final = constraint_neg / calibrator.temperature
    total_final = shortcut_final + constraint_final

    s_prob = role_probs[:, ROLE_TO_ID["shortcut"]]
    c_prob = role_probs[:, ROLE_TO_ID["constraint"]]

    s_candidates = [
        i for i in range(len(spans))
        if s_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and shortcut_final[i] >= contribution_threshold
    ]
    c_candidates = [
        i for i in range(len(spans))
        if c_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and abs(constraint_final[i]) >= contribution_threshold
    ]

    best_pair = None
    best_score = -float("inf")
    for s_idx in s_candidates:
        for c_idx in c_candidates:
            if s_idx == c_idx:
                continue
            score = shortcut_final[s_idx] + abs(constraint_final[c_idx])
            if score > best_score:
                best_score = score
                best_pair = (s_idx, c_idx)

    shortcut_idx = None
    constraint_idx = None
    if best_pair is not None:
        shortcut_idx, constraint_idx = best_pair
    else:
        if s_candidates:
            shortcut_idx = max(s_candidates, key=lambda i: shortcut_final[i])
        remaining_c = [i for i in c_candidates if i != shortcut_idx]
        if remaining_c:
            constraint_idx = max(remaining_c, key=lambda i: abs(constraint_final[i]))

    per_span = []
    for i, span in enumerate(spans):
        per_span.append(
            {
                "span_id": int(span["span_id"]),
                "text": span["text"],
                "span_type": span.get("span_type", "structural"),
                "constraint_probability": float(c_prob[i]),
                "shortcut_probability": float(s_prob[i]),
                "irrelevant_probability": float(
                    role_probs[i, ROLE_TO_ID["irrelevant"]]
                ),
                "decision_usage": float(usage[i]),
                "shortcut_logit_contribution": float(shortcut_final[i]),
                "constraint_logit_contribution": float(constraint_final[i]),
                "net_role_logit_contribution": float(total_final[i]),
            }
        )

    def span_payload(idx: int | None, role: str):
        if idx is None:
            return {
                "resolved": False,
                "reason": (
                    f"no distinct {role} span passed probability, usage, "
                    "and contribution thresholds"
                ),
            }
        row = per_span[idx]
        return {"resolved": True, **row}

    return {
        "predicted_shortcut": span_payload(shortcut_idx, "shortcut"),
        "predicted_constraint": span_payload(constraint_idx, "constraint"),
        "distinct_pair_resolved": bool(
            shortcut_idx is not None and constraint_idx is not None
        ),
        "per_span_contributions": per_span,
    }


def explanation_status(
    combined_probability: float,
    role_probability: float,
    residual_probability: float,
    explanation: dict,
    decision_threshold: float,
) -> str:
    if combined_probability < decision_threshold:
        return "not_flagged"
    if (
        role_probability >= decision_threshold
        and explanation["predicted_shortcut"].get("resolved", False)
    ):
        if explanation["predicted_constraint"].get("resolved", False):
            return "role_supported_with_constraint_competition"
        return "role_supported_shortcut_only"
    if residual_probability >= decision_threshold:
        return "residual_only_no_faithful_role_explanation"
    return "insufficient_role_evidence"



def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _safe_median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> list[float] | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        stats[b] = rng.choice(arr, size=arr.size, replace=True).mean()
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(stats, alpha)),
        float(np.quantile(stats, 1.0 - alpha)),
    ]


def bootstrap_group_difference_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    n_boot: int,
    seed: int,
) -> list[float] | None:
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        stats[k] = aa.mean() - bb.mean()
    return [
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    ]


def paired_bootstrap_difference_ci(
    target: Sequence[float],
    baseline: Sequence[float],
    n_boot: int,
    seed: int,
) -> list[float] | None:
    a = np.asarray(target, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.size == 0 or a.size != b.size:
        return None
    diff = a - b
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        stats[k] = rng.choice(diff, size=diff.size, replace=True).mean()
    return [
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    ]


def paired_signflip_pvalue(
    target: Sequence[float],
    baseline: Sequence[float],
    n_perm: int,
    seed: int,
) -> float | None:
    a = np.asarray(target, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.size == 0 or a.size != b.size:
        return None
    diff = a - b
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=diff.size)
        if abs(float((diff * signs).mean())) >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def label_permutation_pvalue(
    values: Sequence[float],
    labels: Sequence[int],
    n_perm: int,
    seed: int,
) -> float | None:
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.size == 0 or len(np.unique(y)) < 2:
        return None
    observed = abs(float(x[y == 1].mean() - x[y == 0].mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        stat = abs(float(x[yp == 1].mean() - x[yp == 0].mean()))
        if stat >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 3 or a.size != b.size:
        return None
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    if np.std(ra) <= EPS or np.std(rb) <= EPS:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def paired_auc_bootstrap(
    labels: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    n_boot: int,
    seed: int,
) -> dict | None:
    y = np.asarray(labels, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if len(np.unique(y)) < 2:
        return None

    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    diffs = []
    attempts = 0
    while len(diffs) < n_boot and attempts < n_boot * 10:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            float(
                roc_auc_score(y[idx], a[idx])
                - roc_auc_score(y[idx], b[idx])
            )
        )
    if not diffs:
        return None
    arr = np.asarray(diffs)
    return {
        "delta_auroc": observed,
        "bootstrap_95_ci": [
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        ],
        "bootstrap_probability_delta_le_zero": float(np.mean(arr <= 0)),
        "n_bootstrap": int(len(arr)),
    }


def _cohens_d(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    pooled = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / max(a.size + b.size - 2, 1)
    )
    if pooled <= EPS:
        return None
    return float((a.mean() - b.mean()) / pooled)


def shortcut_explanatory_statistics(
    test_records: Sequence[dict],
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    records = list(test_records)
    labels = np.asarray(
        [int(bool(r["hallucinated"])) for r in records], dtype=int
    )
    shortcut_evidence = np.asarray(
        [
            float(r["mechanism_decomposition"]["shortcut_evidence"])
            for r in records
        ],
        dtype=float,
    )
    detected = np.asarray(
        [
            bool(
                r["explanation"]["predicted_shortcut"].get(
                    "resolved", False
                )
            )
            for r in records
        ],
        dtype=bool,
    )
    max_probability = np.asarray(
        [
            max(
                (
                    float(x["shortcut_probability"])
                    for x in r["explanation"]["per_span_contributions"]
                ),
                default=0.0,
            )
            for r in records
        ],
        dtype=float,
    )
    max_contribution = np.asarray(
        [
            max(
                (
                    float(x["shortcut_logit_contribution"])
                    for x in r["explanation"]["per_span_contributions"]
                ),
                default=0.0,
            )
            for r in records
        ],
        dtype=float,
    )

    def group_summary(values: np.ndarray, group: int) -> dict:
        arr = values[labels == group]
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                arr, n_boot=n_boot, seed=seed + group
            ),
        }

    hall = shortcut_evidence[labels == 1]
    correct = shortcut_evidence[labels == 0]
    a = int(np.sum((labels == 1) & detected))
    b = int(np.sum((labels == 1) & ~detected))
    c = int(np.sum((labels == 0) & detected))
    d = int(np.sum((labels == 0) & ~detected))

    # Haldane-Anscombe correction only when needed.
    aa, bb, cc, dd = map(float, (a, b, c, d))
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    log_se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    odds_ci = [
        float(math.exp(math.log(odds_ratio) - 1.96 * log_se)),
        float(math.exp(math.log(odds_ratio) + 1.96 * log_se)),
    ]

    # Dose-response quartiles. qcut-like ranking avoids duplicate-boundary loss.
    order = np.argsort(shortcut_evidence, kind="mergesort")
    bins = np.empty(len(records), dtype=int)
    for rank, idx in enumerate(order):
        bins[idx] = min(3, (4 * rank) // max(len(records), 1))
    quartiles = []
    for q in range(4):
        mask = bins == q
        quartiles.append(
            {
                "quartile": q + 1,
                "n": int(mask.sum()),
                "shortcut_evidence_min": (
                    float(shortcut_evidence[mask].min()) if mask.any() else None
                ),
                "shortcut_evidence_max": (
                    float(shortcut_evidence[mask].max()) if mask.any() else None
                ),
                "hallucination_rate": (
                    float(labels[mask].mean()) if mask.any() else None
                ),
            }
        )

    detection_rate_difference = (
        a / max(a + b, 1) - c / max(c + d, 1)
    )
    detection_perm_p = label_permutation_pvalue(
        detected.astype(float), labels, n_perm=n_perm, seed=seed + 19
    )

    return {
        "shortcut_evidence_by_outcome": {
            "hallucination": group_summary(shortcut_evidence, 1),
            "correct": group_summary(shortcut_evidence, 0),
            "mean_difference_hallucination_minus_correct": float(
                hall.mean() - correct.mean()
            ),
            "difference_bootstrap_95_ci": bootstrap_group_difference_ci(
                hall, correct, n_boot=n_boot, seed=seed + 3
            ),
            "label_permutation_p_value": label_permutation_pvalue(
                shortcut_evidence, labels, n_perm=n_perm, seed=seed + 5
            ),
            "cohens_d": _cohens_d(hall, correct),
            "shortcut_evidence_auroc": float(
                roc_auc_score(labels, shortcut_evidence)
            ),
            "shortcut_evidence_auprc": float(
                average_precision_score(labels, shortcut_evidence)
            ),
        },
        "additional_shortcut_quantities": {
            "max_shortcut_probability": {
                "hallucination": group_summary(max_probability, 1),
                "correct": group_summary(max_probability, 0),
                "auroc": float(roc_auc_score(labels, max_probability)),
            },
            "max_shortcut_logit_contribution": {
                "hallucination": group_summary(max_contribution, 1),
                "correct": group_summary(max_contribution, 0),
                "auroc": float(roc_auc_score(labels, max_contribution)),
            },
        },
        "shortcut_detection_prevalence": {
            "contingency": {
                "hallucination_detected": a,
                "hallucination_not_detected": b,
                "correct_detected": c,
                "correct_not_detected": d,
            },
            "detected_rate_given_hallucination": a / max(a + b, 1),
            "detected_rate_given_correct": c / max(c + d, 1),
            "rate_difference": detection_rate_difference,
            "label_permutation_p_value": detection_perm_p,
            "hallucination_rate_given_detected": a / max(a + c, 1),
            "hallucination_rate_given_not_detected": b / max(b + d, 1),
            "odds_ratio": float(odds_ratio),
            "odds_ratio_95_ci": odds_ci,
        },
        "shortcut_evidence_dose_response": quartiles,
    }


def audit_one_span(
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    item: dict,
    span_record: dict,
    choices: tuple[str, str],
    interventions: Sequence[str],
    normalized_deadzone: float,
    consistency_threshold: float,
) -> dict:
    question, base_prompt, gold_raw = adapter.unpack(item)
    gold = normalize_gold(gold_raw, choices, base_prompt or question)
    original_prompt = adapter.render(question, question, base_prompt)
    original = extractor.score_prompt(original_prompt, gold, choices)

    span = CandidateSpan(
        span_id=int(span_record["span_id"]),
        start=int(span_record["start"]),
        end=int(span_record["end"]),
        text=str(span_record["text"]),
        span_type=str(span_record.get("span_type", "structural")),
    )
    effects = []
    for kind in interventions:
        modified = intervene(question, span, kind)
        variant_prompt = adapter.render(question, modified, base_prompt)
        score = extractor.score_prompt(variant_prompt, gold, choices)
        improvement = float(score["gold_margin"] - original["gold_margin"])
        normalized = float(
            improvement
            / (
                abs(score["gold_margin"])
                + abs(original["gold_margin"])
                + EPS
            )
        )
        effects.append(
            {
                "kind": kind,
                "gold_margin_after": float(score["gold_margin"]),
                "gold_margin_improvement": improvement,
                "normalized_gold_margin_improvement": normalized,
                "chosen_after": score["chosen"],
                "answer_flip": bool(score["chosen"] != original["chosen"]),
                "wrong_to_right": bool(
                    original["chosen"] != gold and score["chosen"] == gold
                ),
                "right_to_wrong": bool(
                    original["chosen"] == gold and score["chosen"] != gold
                ),
            }
        )

    normalized_values = np.asarray(
        [e["normalized_gold_margin_improvement"] for e in effects],
        dtype=float,
    )
    raw_values = np.asarray(
        [e["gold_margin_improvement"] for e in effects], dtype=float
    )
    positive_fraction = float(
        np.mean(normalized_values > normalized_deadzone)
    )
    median_normalized = float(np.median(normalized_values))
    causal_success = bool(
        median_normalized > normalized_deadzone
        and positive_fraction >= consistency_threshold
    )

    return {
        "span_id": int(span.span_id),
        "text": span.text,
        "span_type": span.span_type,
        "original_gold_margin": float(original["gold_margin"]),
        "original_chosen": original["chosen"],
        "gold": gold,
        "original_correct": bool(original["chosen"] == gold),
        "interventions": effects,
        "summary": {
            "mean_gold_margin_improvement": float(raw_values.mean()),
            "median_gold_margin_improvement": float(np.median(raw_values)),
            "mean_normalized_improvement": float(normalized_values.mean()),
            "median_normalized_improvement": median_normalized,
            "positive_consistency_fraction": positive_fraction,
            "answer_flip_any": bool(any(e["answer_flip"] for e in effects)),
            "wrong_to_right_any": bool(
                any(e["wrong_to_right"] for e in effects)
            ),
            "wrong_to_right_majority": bool(
                np.mean([e["wrong_to_right"] for e in effects]) >= 0.5
            ),
            "right_to_wrong_any": bool(
                any(e["right_to_wrong"] for e in effects)
            ),
            "causal_success": causal_success,
        },
    }


def _best_other_span(
    spans: Sequence[dict],
    predicted_shortcut_id: int,
    feature_key: str,
) -> int | None:
    candidates = [
        (
            float(span["features"].get(feature_key, 0.0)),
            int(span["span_id"]),
        )
        for span in spans
        if int(span["span_id"]) != predicted_shortcut_id
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def run_postprediction_causal_audit(
    test_indices: Sequence[int],
    items: list[dict],
    base_by_idx: dict[int, dict],
    explanations_by_idx: dict[int, dict],
    evidence_by_idx: dict[int, dict],
    role_logits_by_idx: dict[int, float],
    residual_logits_by_idx: dict[int, float],
    final_probabilities_by_idx: dict[int, float],
    role_mechanism: RoleMechanismHead,
    calibrator: FinalCalibrator,
    decision_threshold: float,
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    interventions: Sequence[str],
    random_repeats: int,
    normalized_deadzone: float,
    consistency_threshold: float,
    audit_max_items: int,
    seed: int,
    resume: bool,
    cache_path: Path,
) -> dict[int, dict]:
    cached = load_jsonl_index(cache_path) if resume else {}
    output: dict[int, dict] = {}
    signature = json.dumps(
        {
            "interventions": list(interventions),
            "random_repeats": random_repeats,
            "normalized_deadzone": normalized_deadzone,
            "consistency_threshold": consistency_threshold,
        },
        sort_keys=True,
    )

    eligible = [
        idx
        for idx in test_indices
        if explanations_by_idx[idx]["predicted_shortcut"].get(
            "resolved", False
        )
    ]
    if audit_max_items > 0:
        eligible = eligible[:audit_max_items]

    for count, idx in enumerate(eligible, 1):
        old = cached.get(idx)
        if old is not None and old.get("audit_signature") == signature:
            output[idx] = old
            continue

        base = base_by_idx[idx]
        explanation = explanations_by_idx[idx]
        shortcut = explanation["predicted_shortcut"]
        shortcut_id = int(shortcut["span_id"])
        spans = base["spans"]
        by_id = {int(s["span_id"]): s for s in spans}

        predicted_audit = audit_one_span(
            extractor,
            adapter,
            items[idx],
            by_id[shortcut_id],
            choices,
            interventions,
            normalized_deadzone,
            consistency_threshold,
        )

        rng = random.Random(seed + idx * 1009)
        other_ids = [
            int(s["span_id"])
            for s in spans
            if int(s["span_id"]) != shortcut_id
        ]
        rng.shuffle(other_ids)
        random_ids = other_ids[: min(random_repeats, len(other_ids))]
        random_audits = [
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[sid],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            for sid in random_ids
        ]

        attn_id = _best_other_span(
            spans, shortcut_id, "attn_density_late"
        )
        grad_id = _best_other_span(
            spans, shortcut_id, "grad_norm_density"
        )
        attn_audit = (
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[attn_id],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            if attn_id is not None
            else None
        )
        grad_audit = (
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[grad_id],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            if grad_id is not None
            else None
        )

        constraint_audit = None
        constraint = explanation["predicted_constraint"]
        if constraint.get("resolved", False):
            cid = int(constraint["span_id"])
            if cid in by_id and cid != shortcut_id:
                constraint_audit = audit_one_span(
                    extractor,
                    adapter,
                    items[idx],
                    by_id[cid],
                    choices,
                    interventions,
                    normalized_deadzone,
                    consistency_threshold,
                )

        evidence = evidence_by_idx[idx]
        shortcut_pos, _ = role_mechanism.span_contributions(evidence)
        role_logit = float(role_logits_by_idx[idx])
        residual_logit = float(residual_logits_by_idx[idx])
        original_probability = float(final_probabilities_by_idx[idx])

        predicted_contribution = float(shortcut_pos[shortcut_id])
        role_without_predicted = role_logit - predicted_contribution
        probability_without_predicted = float(
            calibrator.predict_proba(
                [role_without_predicted], [residual_logit]
            )[0]
        )
        role_without_all = role_logit - float(np.sum(shortcut_pos))
        probability_without_all = float(
            calibrator.predict_proba([role_without_all], [residual_logit])[0]
        )

        record = {
            "idx": idx,
            "audit_signature": signature,
            "hallucinated": bool(base["hallucinated"]),
            "predicted_shortcut": predicted_audit,
            "baselines": {
                "random_nonshortcut_spans": random_audits,
                "max_attention_nonshortcut": attn_audit,
                "max_gradient_nonshortcut": grad_audit,
            },
            "predicted_constraint": constraint_audit,
            "detector_mediation": {
                "original_probability": original_probability,
                "probability_without_predicted_shortcut": (
                    probability_without_predicted
                ),
                "probability_without_all_shortcut_evidence": (
                    probability_without_all
                ),
                "probability_drop_predicted_shortcut": float(
                    original_probability - probability_without_predicted
                ),
                "probability_drop_all_shortcuts": float(
                    original_probability - probability_without_all
                ),
                "original_flagged": bool(
                    original_probability >= decision_threshold
                ),
                "flagged_without_predicted_shortcut": bool(
                    probability_without_predicted >= decision_threshold
                ),
                "flagged_without_all_shortcuts": bool(
                    probability_without_all >= decision_threshold
                ),
                "predicted_shortcut_detector_critical": bool(
                    original_probability >= decision_threshold
                    and probability_without_predicted < decision_threshold
                ),
                "all_shortcuts_detector_critical": bool(
                    original_probability >= decision_threshold
                    and probability_without_all < decision_threshold
                ),
                "predicted_shortcut_role_logit_contribution": (
                    predicted_contribution
                ),
            },
        }
        record["causal_alignment"] = bool(
            record["detector_mediation"][
                "predicted_shortcut_detector_critical"
            ]
            and predicted_audit["summary"]["causal_success"]
        )
        append_jsonl(cache_path, record)
        output[idx] = record

        if count % 10 == 0 or count == len(eligible):
            print(
                f"post-prediction causal audit: {count}/{len(eligible)}",
                flush=True,
            )

    return output


def _mean_random_baseline(record: dict) -> float | None:
    audits = record["baselines"]["random_nonshortcut_spans"]
    if not audits:
        return None
    return float(
        np.mean(
            [
                x["summary"]["median_normalized_improvement"]
                for x in audits
            ]
        )
    )


def aggregate_postprediction_causal_audit(
    audit_by_idx: dict[int, dict],
    all_test_records: Sequence[dict],
    evidence_by_idx: dict[int, dict],
    role_logits_by_idx: dict[int, float],
    residual_logits_by_idx: dict[int, float],
    final_probabilities_by_idx: dict[int, float],
    decision_threshold: float,
    role_mechanism: RoleMechanismHead,
    calibrator: FinalCalibrator,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict | None:
    audits = list(audit_by_idx.values())
    if not audits:
        return None

    def method_value(rec: dict, method: str) -> float | None:
        if method == "predicted_shortcut":
            return float(
                rec["predicted_shortcut"]["summary"][
                    "median_normalized_improvement"
                ]
            )
        if method == "random_nonshortcut":
            return _mean_random_baseline(rec)
        audit = rec["baselines"].get(method)
        if audit is None:
            return None
        return float(audit["summary"]["median_normalized_improvement"])

    def summarize_method(method: str, label_filter: int | None) -> dict:
        selected = [
            rec
            for rec in audits
            if label_filter is None
            or int(bool(rec["hallucinated"])) == label_filter
        ]
        values = [
            method_value(rec, method)
            for rec in selected
            if method_value(rec, method) is not None
        ]
        causal_success = []
        wrong_to_right = []
        answer_flip = []
        for rec in selected:
            if method == "predicted_shortcut":
                audit = rec["predicted_shortcut"]
                causal_success.append(
                    bool(audit["summary"]["causal_success"])
                )
                wrong_to_right.append(
                    bool(audit["summary"]["wrong_to_right_any"])
                )
                answer_flip.append(bool(audit["summary"]["answer_flip_any"]))
            elif method == "random_nonshortcut":
                for audit in rec["baselines"]["random_nonshortcut_spans"]:
                    causal_success.append(
                        bool(audit["summary"]["causal_success"])
                    )
                    wrong_to_right.append(
                        bool(audit["summary"]["wrong_to_right_any"])
                    )
                    answer_flip.append(
                        bool(audit["summary"]["answer_flip_any"])
                    )
            elif method in (
                "max_attention_nonshortcut",
                "max_gradient_nonshortcut",
            ):
                audit = rec["baselines"].get(method)
                if audit is not None:
                    causal_success.append(
                        bool(audit["summary"]["causal_success"])
                    )
                    wrong_to_right.append(
                        bool(audit["summary"]["wrong_to_right_any"])
                    )
                    answer_flip.append(
                        bool(audit["summary"]["answer_flip_any"])
                    )
        return {
            "n": len(values),
            "mean_median_normalized_improvement": _safe_mean(values),
            "median_median_normalized_improvement": _safe_median(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values, n_boot=n_boot, seed=seed + {"predicted_shortcut": 11, "random_nonshortcut": 23, "max_attention_nonshortcut": 37, "max_gradient_nonshortcut": 53}[method]
            ),
            "causal_success_rate": (
                float(np.mean(causal_success)) if causal_success else None
            ),
            "answer_flip_any_rate": (
                float(np.mean(answer_flip)) if answer_flip else None
            ),
            "wrong_to_right_any_rate": (
                float(np.mean(wrong_to_right)) if wrong_to_right else None
            ),
        }

    methods = (
        "predicted_shortcut",
        "random_nonshortcut",
        "max_attention_nonshortcut",
        "max_gradient_nonshortcut",
    )
    method_summaries = {}
    for method in methods:
        method_summaries[method] = {
            "all": summarize_method(method, None),
            "hallucination": summarize_method(method, 1),
            "correct": summarize_method(method, 0),
        }

    paired_comparisons = {}
    for baseline in (
        "random_nonshortcut",
        "max_attention_nonshortcut",
        "max_gradient_nonshortcut",
    ):
        target_values = []
        baseline_values = []
        labels = []
        for rec in audits:
            target = method_value(rec, "predicted_shortcut")
            other = method_value(rec, baseline)
            if target is None or other is None:
                continue
            target_values.append(target)
            baseline_values.append(other)
            labels.append(int(bool(rec["hallucinated"])))
        paired_comparisons[baseline] = {
            "n": len(target_values),
            "mean_improvement_difference": (
                float(
                    np.mean(
                        np.asarray(target_values)
                        - np.asarray(baseline_values)
                    )
                )
                if target_values
                else None
            ),
            "paired_bootstrap_95_ci": paired_bootstrap_difference_ci(
                target_values,
                baseline_values,
                n_boot=n_boot,
                seed=seed + 101,
            ),
            "paired_signflip_p_value": paired_signflip_pvalue(
                target_values,
                baseline_values,
                n_perm=n_perm,
                seed=seed + 103,
            ),
            "hallucination_only": {},
        }
        hall_target = [
            t for t, y in zip(target_values, labels) if y == 1
        ]
        hall_base = [
            b for b, y in zip(baseline_values, labels) if y == 1
        ]
        paired_comparisons[baseline]["hallucination_only"] = {
            "n": len(hall_target),
            "mean_improvement_difference": (
                float(np.mean(np.asarray(hall_target) - np.asarray(hall_base)))
                if hall_target
                else None
            ),
            "paired_bootstrap_95_ci": paired_bootstrap_difference_ci(
                hall_target,
                hall_base,
                n_boot=n_boot,
                seed=seed + 107,
            ),
            "paired_signflip_p_value": paired_signflip_pvalue(
                hall_target,
                hall_base,
                n_perm=n_perm,
                seed=seed + 109,
            ),
        }

    # Detector mediation is computed for every test item directly from the
    # frozen role evidence. Behavioral interventions are not required here.
    labels = []
    original_prob = []
    without_predicted = []
    without_all = []
    for record in all_test_records:
        idx = int(record["idx"])
        labels.append(int(bool(record["hallucinated"])))
        p = float(final_probabilities_by_idx[idx])
        original_prob.append(p)

        evidence = evidence_by_idx[idx]
        shortcut_pos, _ = role_mechanism.span_contributions(evidence)
        role_logit = float(role_logits_by_idx[idx])
        residual_logit = float(residual_logits_by_idx[idx])

        role_without_all = role_logit - float(np.sum(shortcut_pos))
        without_all.append(
            float(
                calibrator.predict_proba(
                    [role_without_all], [residual_logit]
                )[0]
            )
        )

        predicted = record["explanation"]["predicted_shortcut"]
        if predicted.get("resolved", False):
            sid = int(predicted["span_id"])
            role_without_predicted = role_logit - float(shortcut_pos[sid])
            without_predicted.append(
                float(
                    calibrator.predict_proba(
                        [role_without_predicted], [residual_logit]
                    )[0]
                )
            )
        else:
            without_predicted.append(p)

    original_metrics = evaluate_binary(
        labels, original_prob, decision_threshold
    )
    targeted_metrics = evaluate_binary(
        labels, without_predicted, decision_threshold
    )
    all_zero_metrics = evaluate_binary(
        labels, without_all, decision_threshold
    )

    def lost_flags(counterfactual: Sequence[float]) -> dict:
        y = np.asarray(labels, dtype=int)
        p0 = np.asarray(original_prob)
        p1 = np.asarray(counterfactual)
        lost = (p0 >= decision_threshold) & (p1 < decision_threshold)
        return {
            "all_flags_lost": int(lost.sum()),
            "true_positive_flags_lost": int(np.sum(lost & (y == 1))),
            "false_positive_flags_lost": int(np.sum(lost & (y == 0))),
            "fraction_of_original_flags_lost": float(
                lost.sum() / max(np.sum(p0 >= decision_threshold), 1)
            ),
            "fraction_of_original_true_positives_lost": float(
                np.sum(lost & (y == 1))
                / max(np.sum((p0 >= decision_threshold) & (y == 1)), 1)
            ),
        }

    contribution = []
    improvement = []
    shortcut_probability = []
    usage = []
    hall_contribution = []
    hall_improvement = []
    for rec in audits:
        span = rec["predicted_shortcut"]
        contribution.append(
            float(
                rec["detector_mediation"][
                    "predicted_shortcut_role_logit_contribution"
                ]
            )
        )
        improvement.append(
            float(span["summary"]["median_normalized_improvement"])
        )
        # Probability and usage are supplied by the prediction file below.
        record = next(
            r for r in all_test_records if int(r["idx"]) == int(rec["idx"])
        )
        pred_span = record["explanation"]["predicted_shortcut"]
        shortcut_probability.append(
            float(pred_span["shortcut_probability"])
        )
        usage.append(float(pred_span["decision_usage"]))
        if rec["hallucinated"]:
            hall_contribution.append(contribution[-1])
            hall_improvement.append(improvement[-1])

    causal_alignment = [
        rec for rec in audits if bool(rec.get("causal_alignment", False))
    ]
    causal_alignment_hall = [
        rec
        for rec in causal_alignment
        if bool(rec["hallucinated"])
    ]

    return {
        "audit_population": {
            "n_test": len(all_test_records),
            "n_predicted_shortcut_resolved_and_audited": len(audits),
            "n_hallucination_audited": int(
                sum(bool(x["hallucinated"]) for x in audits)
            ),
            "n_correct_audited": int(
                sum(not bool(x["hallucinated"]) for x in audits)
            ),
        },
        "behavioral_effects": method_summaries,
        "predicted_shortcut_vs_baselines": paired_comparisons,
        "detector_mediation": {
            "structural_derivative_final_logit_per_unit_shortcut_evidence": (
                float(
                    role_mechanism.beta_shortcut
                    / calibrator.temperature
                )
            ),
            "original_metrics": original_metrics,
            "without_predicted_shortcut_metrics": targeted_metrics,
            "without_all_shortcut_evidence_metrics": all_zero_metrics,
            "predicted_shortcut_ablation": lost_flags(without_predicted),
            "all_shortcut_evidence_ablation": lost_flags(without_all),
            "mean_probability_drop_predicted_shortcut": float(
                np.mean(
                    np.asarray(original_prob)
                    - np.asarray(without_predicted)
                )
            ),
            "mean_probability_drop_all_shortcuts": float(
                np.mean(
                    np.asarray(original_prob) - np.asarray(without_all)
                )
            ),
        },
        "causal_alignment": {
            "definition": (
                "predicted shortcut is detector-critical AND its held-out "
                "interventions consistently improve the target model's "
                "gold-answer margin"
            ),
            "n": len(causal_alignment),
            "rate_among_audited": float(
                len(causal_alignment) / max(len(audits), 1)
            ),
            "n_hallucination": len(causal_alignment_hall),
            "rate_among_audited_hallucinations": float(
                len(causal_alignment_hall)
                / max(
                    sum(bool(x["hallucinated"]) for x in audits),
                    1,
                )
            ),
        },
        "predicted_strength_vs_observed_behavior": {
            "spearman_detector_contribution_vs_margin_improvement": (
                spearman_correlation(contribution, improvement)
            ),
            "spearman_shortcut_probability_vs_margin_improvement": (
                spearman_correlation(shortcut_probability, improvement)
            ),
            "spearman_decision_usage_vs_margin_improvement": (
                spearman_correlation(usage, improvement)
            ),
            "hallucination_only_spearman_contribution_vs_improvement": (
                spearman_correlation(hall_contribution, hall_improvement)
            ),
        },
    }


# ---------------------------------------------------------------------------
# v7: test-time interventional multimodal span detector
# ---------------------------------------------------------------------------

from sklearn.decomposition import PCA

PROFILE_NAME_RE = re.compile(r"(?im)^\s*name:\s*(?P<name>.+?)\s*$")
PROFILE_QUESTION_RE = re.compile(
    r"(?is)Choose\s+exactly\s+one\s+profile\s+from\s+the\s+two,.*?"
    r"following\s+question:\s*"
)
PROFILE_MAPPING_BLOCK_RE = re.compile(
    r"(?ims)^\s*Profile number mapping:\s*\n\s*1\.\s*.*?\n\s*2\.\s*.*?\n"
)

_BASE_READ_RECORDS = read_records
_BASE_SCIENTISTQA_BODY_BOUNDS = scientistqa_body_bounds


def profile_names(prompt: str) -> tuple[str, str]:
    names: list[str] = []
    for match in PROFILE_NAME_RE.finditer(prompt):
        name = match.group("name").strip()
        if name not in names:
            names.append(name)
        if len(names) == 2:
            break
    if len(names) != 2:
        raise ValueError(f"expected two profile names, found {names!r}")
    return names[0], names[1]


def add_profile_number_mapping(prompt: str) -> str:
    if PROFILE_MAPPING_BLOCK_RE.search(prompt):
        return prompt
    marker = PROFILE_QUESTION_RE.search(prompt)
    if marker is None:
        return prompt
    first, second = profile_names(prompt)
    mapping = (
        "Profile number mapping:\n"
        f"1. {first}\n"
        f"2. {second}\n"
    )
    return prompt[: marker.start()] + mapping + prompt[marker.start() :]


def read_records(path: str | Path) -> list[dict]:
    records = _BASE_READ_RECORDS(path)
    output: list[dict] = []
    for item in records:
        adapted = dict(item)
        if "prompt" in adapted and PROFILE_QUESTION_RE.search(str(adapted["prompt"])):
            adapted["prompt"] = add_profile_number_mapping(str(adapted["prompt"]))
        output.append(adapted)
    return output


def scientistqa_body_bounds(text: str) -> tuple[int, int] | None:
    marker = PROFILE_QUESTION_RE.search(text)
    if marker is None:
        return _BASE_SCIENTISTQA_BODY_BOUNDS(text)
    body_start = marker.end()
    final_match = FINAL_PERSON_QUESTION_RE.search(text, body_start)
    body_end = final_match.start() if final_match is not None else len(text)
    body_start, body_end = _trim_span(text, body_start, body_end)
    return (body_start, body_end) if body_start < body_end else None


def swap_profile_order(prompt: str) -> str | None:
    """Swap the two structured profile blocks and update the 1/2 mapping."""
    mapping = PROFILE_MAPPING_BLOCK_RE.search(prompt)
    question_marker = PROFILE_QUESTION_RE.search(prompt)
    if question_marker is None:
        return None
    profile_end = mapping.start() if mapping is not None else question_marker.start()
    name_matches = [m for m in PROFILE_NAME_RE.finditer(prompt, 0, profile_end)]
    if len(name_matches) < 2:
        return None
    first_start = name_matches[0].start()
    second_start = name_matches[1].start()
    first_name = name_matches[0].group("name").strip()
    second_name = name_matches[1].group("name").strip()
    prefix = prompt[:first_start]
    block1 = prompt[first_start:second_start]
    block2 = prompt[second_start:profile_end]
    suffix = prompt[profile_end:]
    swapped = prefix + block2 + block1 + suffix
    new_mapping = (
        "Profile number mapping:\n"
        f"1. {second_name}\n"
        f"2. {first_name}\n"
    )
    if PROFILE_MAPPING_BLOCK_RE.search(swapped):
        swapped = PROFILE_MAPPING_BLOCK_RE.sub(new_mapping, swapped, count=1)
    else:
        marker2 = PROFILE_QUESTION_RE.search(swapped)
        if marker2 is None:
            return None
        swapped = swapped[:marker2.start()] + new_mapping + swapped[marker2.start():]
    return swapped


def intervene_with_range(
    question: str,
    span: CandidateSpan,
    kind: str,
) -> tuple[str, tuple[int, int] | None, str]:
    """Apply a local intervention and return the replacement character range.

    Unlike the legacy helper, this deliberately avoids whole-string whitespace
    normalization so that the replacement range remains exact for offset-based
    span spectral extraction.
    """
    before = question[:span.start]
    target = question[span.start:span.end]
    after = question[span.end:]
    span_type = span.span_type

    if kind == "delete":
        replacement = ""
    elif kind == "neutralize":
        if span_type == "negation_operator":
            replacement = "[POLARITY UNSPECIFIED]"
        elif span_type == "named_entity":
            replacement = "[NAMED ENTITY UNSPECIFIED]"
        else:
            replacement = "[DETAIL UNSPECIFIED]"
    elif kind == "mask":
        if span_type == "negation_operator":
            replacement = "[POLARITY OMITTED]"
        elif span_type == "named_entity":
            replacement = "[NAMED ENTITY OMITTED]"
        else:
            replacement = "[DETAIL OMITTED]"
    elif kind == "negate":
        stripped = target.strip()
        if span_type == "negation_operator":
            replacement = ""
        else:
            stripped = stripped[:-1] if stripped.endswith((".", "?", "!")) else stripped
            replacement = f"It is not true that {stripped}."
    else:
        raise ValueError(f"unknown intervention: {kind}")

    modified = before + replacement + after
    if replacement:
        replacement_range = (len(before), len(before) + len(replacement))
    else:
        replacement_range = None
    return modified, replacement_range, replacement


def _fixed_choice_margin(log_a: float, log_b: float, choice: str, choices: tuple[str, str]) -> float:
    return (log_a - log_b) if choice == choices[0] else (log_b - log_a)


def _gold_margin(log_a: float, log_b: float, gold: str, choices: tuple[str, str]) -> float:
    return (log_a - log_b) if gold == choices[0] else (log_b - log_a)


def _binary_choice(log_a: float, log_b: float, choices: tuple[str, str]) -> str:
    return choices[0] if log_a >= log_b else choices[1]


class InterventionalMultimodalExtractor(WeakWhiteboxExtractor):
    """Extract span-indexed attention, gradient, logit and spectral features.

    For layer ``l`` and head ``h`` with causal attention matrix ``A``:

        d_i = sum_{j=i}^{T-1} A[j, i] / (T-i)
        lambda_i = d_i - A[i, i]

    The causal Laplacian is triangular, so ``lambda_i`` is the token-indexed
    eigenvalue.  Token identity is retained until features are pooled over a
    candidate span.  Attention and gradient features use the same span masks.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        dtype: torch.dtype,
        lap_topk: int = 10,
        keep_head_identities: bool = False,
        compute_gradient_features: bool = True,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            lap_topk=lap_topk,
        )
        self.keep_head_identities = bool(keep_head_identities)
        self.compute_gradient_features = bool(compute_gradient_features)

    def _multimodal_input_gradient(
        self,
        ids: torch.Tensor,
        choices: tuple[str, str],
        chosen: str,
    ) -> dict[str, torch.Tensor]:
        """Gradient features for the model's original chosen-answer contrast."""
        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(ids).detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
        logits = out.logits[0, -1].float()
        log_a, log_b = self._choice_logs(logits, choices)
        contrast = (log_a - log_b) if chosen == choices[0] else (log_b - log_a)
        contrast.backward()

        grad = inputs_embeds.grad[0].detach().float()
        emb = inputs_embeds.detach()[0].float()
        gx = grad * emb
        result = {
            "raw_norm": grad.norm(dim=-1),
            "gx_norm": gx.norm(dim=-1),
            "gx_signed": gx.sum(dim=-1),
            "gx_abs_signed": gx.abs().sum(dim=-1),
        }
        self.model.zero_grad(set_to_none=True)
        del out, inputs_embeds, grad, emb, gx
        return result

    @staticmethod
    def _add_vector_stats(
        target: dict[str, float],
        prefix: str,
        values: torch.Tensor,
        total: float | None = None,
    ) -> None:
        values = values.float()
        target[f"{prefix}_sum"] = float(values.sum().item())
        target[f"{prefix}_mean"] = float(values.mean().item())
        target[f"{prefix}_max"] = float(values.max().item())
        target[f"{prefix}_std"] = float(values.std(unbiased=False).item())
        if total is not None:
            target[f"{prefix}_share"] = float(values.sum().item() / (total + EPS))

    def extract_multimodal(
        self,
        prompt: str,
        question: str,
        spans: Sequence[CandidateSpan],
        choices: tuple[str, str],
    ) -> dict:
        enc = self.tok(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0]
        ids = enc["input_ids"].to(self.device)
        T = int(ids.shape[1])
        question_base = prompt.find(question)
        if question_base < 0:
            raise ValueError("rendered prompt does not contain the exact question")

        masks: list[torch.Tensor] = []
        for span in spans:
            absolute = (question_base + span.start, question_base + span.end)
            mask = self._char_mask(offsets, absolute, T)
            if not mask.any():
                raise ValueError(f"candidate span matched no tokens: {span.text!r}")
            masks.append(mask.to(self.device))

        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                output_attentions=True,
                use_cache=False,
            )
            logits = out.logits[0, -1].float()
            log_a_t, log_b_t = self._choice_logs(logits, choices)
            log_a = float(log_a_t.item())
            log_b = float(log_b_t.item())
            chosen = _binary_choice(log_a, log_b, choices)

            span_features: list[dict[str, float]] = []
            for span, mask in zip(spans, masks):
                idx = mask.nonzero().squeeze(-1)
                span_features.append(
                    {
                        "span_words": float(len(re.findall(r"\w+", span.text))),
                        "span_tokens": float(idx.numel()),
                        "span_characters": float(span.end - span.start),
                        "span_relative_start": float(span.start / max(len(question), 1)),
                        "span_relative_end": float(span.end / max(len(question), 1)),
                        "span_relative_length": float((span.end - span.start) / max(len(question), 1)),
                        "span_type_negation_operator": float(span.span_type == "negation_operator"),
                        "span_type_named_entity": float(span.span_type == "named_entity"),
                        "span_type_predicate_atom": float(span.span_type == "predicate_atom"),
                        "span_type_lexical_atom": float(span.span_type == "lexical_atom"),
                        "span_type_body_clause": float(span.span_type == "body_clause"),
                        "span_type_body_sentence": float(span.span_type == "body_sentence"),
                        "span_type_profile_swap": float(span.span_type == "profile_swap"),
                    }
                )

            global_features: dict[str, float] = {
                "global_logit_choice_margin_abs": float(abs(log_a - log_b)),
                "global_logit_choice_entropy": safe_entropy_binary(log_a, log_b),
                "global_structure_prompt_tokens": float(T),
            }

            denom = (T - torch.arange(T, device=self.device)).clamp(min=1).float()

            for layer_idx, A in enumerate(out.attentions):
                # A: [batch=1, heads, query, key]
                Ah = A[0].float()
                H = int(Ah.shape[0])
                lower = torch.tril(Ah)
                received = lower.sum(dim=1) / denom.unsqueeze(0)
                diag = torch.diagonal(Ah, dim1=-2, dim2=-1)
                lam = received - diag
                decision = Ah[:, T - 1, :T]

                p = decision / (decision.sum(dim=-1, keepdim=True) + EPS)
                ent = -(p * (p + 1e-12).log()).sum(dim=-1)
                global_features[f"global_attn_l{layer_idx}_decision_entropy_headmean"] = float(ent.mean().item())
                global_features[f"global_attn_l{layer_idx}_decision_entropy_headstd"] = float(ent.std(unbiased=False).item())
                global_features[f"global_attn_l{layer_idx}_decision_peak_headmean"] = float(decision.max(dim=-1).values.mean().item())

                k = min(self.lap_topk, T)
                top_vals, top_idx = torch.topk(lam, k=k, dim=-1)
                for rank in range(k):
                    vals = top_vals[:, rank]
                    global_features[f"global_spec_l{layer_idx}_top{rank}_headmean"] = float(vals.mean().item())
                    global_features[f"global_spec_l{layer_idx}_top{rank}_headmax"] = float(vals.max().item())
                    if self.keep_head_identities:
                        for head_idx in range(H):
                            global_features[f"global_spec_l{layer_idx}_h{head_idx}_top{rank}"] = float(vals[head_idx].item())

                for span_idx, mask in enumerate(masks):
                    idx = mask.nonzero().squeeze(-1)
                    vals = lam[:, idx]
                    per_head_mean = vals.mean(dim=-1)
                    per_head_max = vals.max(dim=-1).values
                    topk_hits = mask[top_idx].float().mean(dim=-1)
                    decision_slice = decision[:, idx]
                    decision_mass = decision_slice.sum(dim=-1)
                    decision_density = decision_slice.mean(dim=-1)
                    decision_tokenmax = decision_slice.max(dim=-1).values
                    decision_weighted = (decision_slice * vals).sum(dim=-1)
                    row = span_features[span_idx]

                    sp = f"spec_l{layer_idx}"
                    row[f"{sp}_span_headmean"] = float(per_head_mean.mean().item())
                    row[f"{sp}_span_headmax"] = float(per_head_mean.max().item())
                    row[f"{sp}_span_headstd"] = float(per_head_mean.std(unbiased=False).item())
                    row[f"{sp}_tokenmax_headmean"] = float(per_head_max.mean().item())
                    row[f"{sp}_topk_hit_headmean"] = float(topk_hits.mean().item())
                    row[f"{sp}_topk_any_headrate"] = float((topk_hits > 0).float().mean().item())
                    row[f"{sp}_decision_weighted_headmean"] = float(decision_weighted.mean().item())

                    ap = f"attn_l{layer_idx}"
                    row[f"{ap}_mass_headmean"] = float(decision_mass.mean().item())
                    row[f"{ap}_mass_headmax"] = float(decision_mass.max().item())
                    row[f"{ap}_mass_headstd"] = float(decision_mass.std(unbiased=False).item())
                    row[f"{ap}_density_headmean"] = float(decision_density.mean().item())
                    row[f"{ap}_tokenmax_headmean"] = float(decision_tokenmax.mean().item())

                    if self.keep_head_identities:
                        for head_idx in range(H):
                            shp = f"{sp}_h{head_idx}"
                            row[f"{shp}_span_mean"] = float(per_head_mean[head_idx].item())
                            row[f"{shp}_span_max"] = float(per_head_max[head_idx].item())
                            row[f"{shp}_topk_hit"] = float(topk_hits[head_idx].item())
                            row[f"{shp}_decision_weighted"] = float(decision_weighted[head_idx].item())
                            ahp = f"{ap}_h{head_idx}"
                            row[f"{ahp}_mass"] = float(decision_mass[head_idx].item())
                            row[f"{ahp}_density"] = float(decision_density[head_idx].item())
                            row[f"{ahp}_tokenmax"] = float(decision_tokenmax[head_idx].item())

                # Release the current layer's large attention tensors before
                # the separate gradient forward/backward pass.
                del A, Ah, lower, received, diag, lam, decision, p, ent, top_vals, top_idx

            del out

        if self.compute_gradient_features:
            grad = self._multimodal_input_gradient(ids, choices, chosen)
            for name, values in grad.items():
                self._add_vector_stats(global_features, f"global_grad_{name}", values)
            totals = {name: float(values.sum().item()) for name, values in grad.items()}
            for row, mask in zip(span_features, masks):
                idx = mask.nonzero().squeeze(-1)
                for name, values in grad.items():
                    self._add_vector_stats(
                        row,
                        f"grad_{name}",
                        values[idx],
                        total=totals[name],
                    )
            del grad

        return {
            "chosen": chosen,
            "log_a": log_a,
            "log_b": log_b,
            "chosen_margin": float(abs(log_a - log_b)),
            "global_multimodal_features": global_features,
            "span_multimodal_features": span_features,
        }

    # Backward-compatible alias used by the shared extraction workflow while the
    # implementation and returned namespaces are now multimodal.
    def extract_spectral(
        self,
        prompt: str,
        question: str,
        spans: Sequence[CandidateSpan],
        choices: tuple[str, str],
    ) -> dict:
        result = self.extract_multimodal(prompt, question, spans, choices)
        return {
            "chosen": result["chosen"],
            "log_a": result["log_a"],
            "log_b": result["log_b"],
            "chosen_margin": result["chosen_margin"],
            "global_spectral_features": result["global_multimodal_features"],
            "span_spectral_features": result["span_multimodal_features"],
        }


# Compatibility name for cached code paths and old bundles.
InterventionalSpectralExtractor = InterventionalMultimodalExtractor

def rank_spans_for_intervention(record: dict, max_spans: int) -> list[int]:
    """Rank spans with the new original spectral/decision features.

    A zero budget means all spans.  Negation operators and named entities are
    always protected so the logical operator and its lexical object can be
    contrasted directly.
    """
    spans = record["spans"]
    if max_spans <= 0 or len(spans) <= max_spans:
        return list(range(len(spans)))
    protected = [
        i for i, span in enumerate(spans)
        if span.get("span_type") in {"negation_operator", "named_entity"}
    ]
    protected_set = set(protected)
    scored: list[tuple[float, int]] = []
    for i, span in enumerate(spans):
        if i in protected_set:
            continue
        feat = span.get("features", {})
        attention_salience = max(
            (abs(float(v)) for k, v in feat.items() if k.startswith("attn_")),
            default=0.0,
        )
        spectral_salience = max(
            (abs(float(v)) for k, v in feat.items() if k.startswith("spec_")),
            default=0.0,
        )
        gradient_salience = max(
            (abs(float(v)) for k, v in feat.items() if k.startswith("grad_")),
            default=0.0,
        )
        scored.append((attention_salience + spectral_salience + gradient_salience, i))
    scored.sort(reverse=True)
    remaining = max(max_spans - len(protected), 0)
    return sorted(set(protected + [i for _, i in scored[:remaining]]))


FEATURE_SET_FAMILIES: dict[str, tuple[str, ...]] = {
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
DEFAULT_FEATURE_SET_ORDER = tuple(FEATURE_SET_FAMILIES)


def feature_family(key: str) -> str:
    """Assign every raw span feature to exactly one interpretable family."""
    if (
        key.startswith("spec_")
        or "global_spec" in key
        or "span_spec" in key
        or "spectral" in key
    ):
        return "spectral"
    if (
        key.startswith("attn_")
        or "global_attn" in key
        or "span_attn" in key
        or "attention" in key
    ):
        return "attention"
    if (
        key.startswith("grad_")
        or "global_grad" in key
        or "span_grad" in key
        or "gradient" in key
    ):
        return "gradient"
    if key.startswith("span_"):
        return "structural"
    # All remaining values are intervention behavior / logits / flips,
    # including profile-swap responses and operator-consistency statistics.
    return "behavior"


def select_feature_families(
    features: dict[str, float],
    families: Sequence[str],
) -> dict[str, float]:
    allowed = set(families)
    return {
        key: float(value)
        for key, value in features.items()
        if feature_family(key) in allowed
    }


class MultimodalSpanRoleHead:
    """Soft-label role probe restricted to a declared feature-family set."""

    def __init__(
        self,
        feature_set_name: str,
        C: float = 1.0,
        pca_dim: int = 128,
        seed: int = 0,
    ):
        if feature_set_name not in FEATURE_SET_FAMILIES:
            raise ValueError(f"unknown feature set: {feature_set_name}")
        self.feature_set_name = str(feature_set_name)
        self.feature_families = tuple(FEATURE_SET_FAMILIES[feature_set_name])
        self.matrix = FeatureMatrix()
        self.C = float(C)
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.lr = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
        )
        self.fitted = False

    def new_unfitted(self, seed: int | None = None) -> "MultimodalSpanRoleHead":
        return MultimodalSpanRoleHead(
            feature_set_name=self.feature_set_name,
            C=self.C,
            pca_dim=self.pca_dim,
            seed=self.seed if seed is None else int(seed),
        )

    def _selected(self, feat_dicts: Sequence[dict[str, float]]) -> list[dict[str, float]]:
        selected = [select_feature_families(x, self.feature_families) for x in feat_dicts]
        if not selected or not any(selected):
            raise RuntimeError(
                f"feature set {self.feature_set_name!r} selected no raw features"
            )
        return selected

    def _fit_transform(self, feat_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        selected = self._selected(feat_dicts)
        self.matrix.fit(selected)
        X = self.matrix.transform(selected)
        Xs = self.scaler.fit_transform(X)
        max_dim = min(max(Xs.shape[0] - 1, 1), Xs.shape[1])
        dim = min(self.pca_dim, max_dim)
        if dim > 0 and dim < Xs.shape[1]:
            self.pca = PCA(
                n_components=dim,
                svd_solver="randomized",
                random_state=self.seed,
            )
            return self.pca.fit_transform(Xs)
        self.pca = None
        return Xs

    def _transform(self, feat_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        selected = self._selected(feat_dicts)
        X = self.matrix.transform(selected)
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs) if self.pca is not None else Xs

    def fit(
        self,
        feat_dicts: Sequence[dict[str, float]],
        soft_labels: Sequence[Sequence[float]],
        reliabilities: Sequence[float],
    ) -> "MultimodalSpanRoleHead":
        Z_base = self._fit_transform(feat_dicts)
        rows: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for z, probs, rel in zip(Z_base, soft_labels, reliabilities):
            probs_arr = np.asarray(probs, dtype=float)
            for role_id in range(3):
                rows.append(z)
                labels.append(role_id)
                weights.append(float(max(rel, 1e-4) * max(probs_arr[role_id], 1e-6)))
        self.lr.fit(
            np.asarray(rows),
            np.asarray(labels),
            sample_weight=np.asarray(weights),
        )
        self.fitted = True
        return self

    def predict_proba(self, feat_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MultimodalSpanRoleHead is not fitted")
        Z = self._transform(feat_dicts)
        probs = self.lr.predict_proba(Z)
        out = np.zeros((len(feat_dicts), 3), dtype=float)
        for col, cls in enumerate(self.lr.classes_):
            out[:, int(cls)] = probs[:, col]
        return out

    def report(self) -> dict:
        raw_keys = list(self.matrix.keys or [])
        family_counts = {name: 0 for name in self.feature_families}
        for key in raw_keys:
            family_counts[feature_family(key)] = family_counts.get(feature_family(key), 0) + 1
        return {
            "feature_set": self.feature_set_name,
            "feature_families": list(self.feature_families),
            "n_input_features": len(raw_keys),
            "input_feature_counts_by_family": family_counts,
            "pca_dim": int(self.pca.n_components_) if self.pca is not None else None,
            "pca_explained_variance_ratio_sum": (
                float(self.pca.explained_variance_ratio_.sum())
                if self.pca is not None
                else None
            ),
            "classifier_classes": [int(x) for x in self.lr.classes_],
            "note": (
                "coefficients live in PCA space; use the feature-set ablation "
                "table and permutation importance for family-level interpretation"
            ),
        }


# Backward-compatible symbol for old references inside the file.
SpectralSpanRoleHead = MultimodalSpanRoleHead

def _prefixed_delta(
    original: dict[str, float],
    modified: dict[str, float],
    prefix: str,
    include_absolute: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted(set(original) | set(modified))
    for key in keys:
        a = float(original.get(key, 0.0))
        b = float(modified.get(key, 0.0))
        out[f"{prefix}_delta_{key}"] = b - a
        if include_absolute:
            out[f"{prefix}_abs_{key}"] = b
    return out


def extract_interventional_base_records(
    items: list[dict],
    extractor: InterventionalSpectralExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    args,
    cache_path: Path,
) -> list[dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    records: list[dict] = []
    for idx, item in enumerate(items):
        if idx in cached:
            records.append(cached[idx])
            continue
        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices, base_prompt or question)
        spans = propose_spans(
            question,
            mode=args.span_mode,
            include_question_span=args.include_question_span,
            min_words=args.min_span_words,
        )
        if not spans:
            warnings.warn(f"[{idx}] no candidate spans; skipped")
            continue
        prompt = adapter.render(question, question, base_prompt)
        try:
            extracted = extractor.extract_spectral(prompt, question, spans, choices)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn(f"[{idx}] CUDA OOM during original spectral extraction; skipped")
            continue
        except Exception as exc:
            warnings.warn(f"[{idx}] original spectral extraction failed: {exc}")
            continue
        chosen = extracted["chosen"]
        record = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "chosen": chosen,
            "hallucinated": int(chosen != gold),
            "log_a": extracted["log_a"],
            "log_b": extracted["log_b"],
            "chosen_margin": extracted["chosen_margin"],
            "gold_margin": _gold_margin(extracted["log_a"], extracted["log_b"], gold, choices),
            "global_spectral_features": extracted["global_spectral_features"],
            "spans": [
                {**asdict(span), "features": feat}
                for span, feat in zip(spans, extracted["span_spectral_features"])
            ],
        }
        append_jsonl(cache_path, record)
        records.append(record)
        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"original spectral extraction: {idx + 1}/{len(items)}", flush=True)
    return records


def _profile_swap_row(
    record: dict,
    item: dict,
    extractor: InterventionalSpectralExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    is_train: bool,
    args,
) -> dict | None:
    if not args.include_profile_swap_span:
        return None
    question, base_prompt, gold_raw = adapter.unpack(item)
    swapped_question = swap_profile_order(question)
    if swapped_question is None:
        return None
    swapped_base = swap_profile_order(base_prompt) if base_prompt is not None else None
    if base_prompt is not None and swapped_base is None:
        swapped_base = swapped_question
    try:
        prompt = adapter.render(swapped_question, swapped_question, swapped_base)
        ext = extractor.extract_spectral(prompt, swapped_question, [], choices)
    except Exception as exc:
        warnings.warn(f"[{record['idx']}] profile swap failed: {exc}")
        return None

    # Map swapped option logits back to original person identities.
    mapped_log_a = float(ext["log_b"])
    mapped_log_b = float(ext["log_a"])
    original_choice = str(record["chosen"])
    fixed_margin = _fixed_choice_margin(mapped_log_a, mapped_log_b, original_choice, choices)
    chosen_effect = float(record["chosen_margin"] - fixed_margin)
    norm = chosen_effect / (abs(float(record["chosen_margin"])) + EPS)
    swapped_person_choice = _binary_choice(mapped_log_a, mapped_log_b, choices)
    features: dict[str, float] = {
        "span_type_profile_swap": 1.0,
        "intervention_evaluated": 1.0,
        "profile_swap_chosen_margin_effect": chosen_effect,
        "profile_swap_normalized_effect": norm,
        "profile_swap_same_original_person": float(swapped_person_choice == original_choice),
        "profile_swap_same_position": float(ext["chosen"] == original_choice),
        "intervention_usage": float(np.clip(abs(norm), 0.0, 1.0)),
        "operator_sign_agreement": 1.0,
        "operator_effect_mad": 0.0,
        "operator_flip_rate": float(swapped_person_choice != original_choice),
    }
    for family in ("spectral", "attention", "gradient"):
        original_global = {
            k: v
            for k, v in record["global_spectral_features"].items()
            if feature_family(k) == family
        }
        swapped_global = {
            k: v
            for k, v in ext["global_spectral_features"].items()
            if feature_family(k) == family
        }
        features.update(
            _prefixed_delta(
                original_global,
                swapped_global,
                f"profile_swap_global_{family}",
                include_absolute=False,
            )
        )

    pseudo = None
    if is_train:
        gold = normalize_gold(gold_raw, choices, base_prompt or question)
        swapped_gold_margin = _gold_margin(mapped_log_a, mapped_log_b, gold, choices)
        gold_delta = float(record["gold_margin"] - swapped_gold_margin)
        pseudo = build_soft_role(
            [gold_delta],
            deadzone=args.role_deadzone,
            temperature=args.role_temperature,
        )
        pseudo["evaluated"] = True
        pseudo["interventions"] = [
            {
                "kind": "profile_swap",
                "gold_delta": gold_delta,
                "chosen_effect": chosen_effect,
                "normalized_effect": norm,
                "person_flip": bool(swapped_person_choice != original_choice),
            }
        ]

    return {
        "span_id": -1,
        "start": -1,
        "end": -1,
        "text": "[PROFILE ORDER]",
        "span_type": "profile_swap",
        "features": features,
        "pseudo_label": pseudo,
        "interventions": [
            {
                "kind": "profile_swap",
                "mapped_log_a": mapped_log_a,
                "mapped_log_b": mapped_log_b,
                "mapped_chosen": swapped_person_choice,
                "raw_swapped_chosen": ext["chosen"],
                "chosen_effect": chosen_effect,
                "normalized_effect": norm,
            }
        ],
    }


def extract_all_intervention_responses(
    base_records: list[dict],
    items: list[dict],
    train_set: set[int],
    extractor: InterventionalSpectralExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    interventions: list[str],
    args,
    cache_path: Path,
) -> dict[int, dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    output = dict(cached)

    for count, record in enumerate(base_records, 1):
        idx = int(record["idx"])
        if idx in output:
            continue
        item = items[idx]
        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices, base_prompt or question)
        original_choice = str(record["chosen"])
        selected_ids = set(rank_spans_for_intervention(record, args.max_intervention_spans))
        span_rows: list[dict] = []

        for local_id, span_record in enumerate(record["spans"]):
            original_features = dict(span_record["features"])
            span = CandidateSpan(
                span_id=int(span_record["span_id"]),
                start=int(span_record["start"]),
                end=int(span_record["end"]),
                text=str(span_record["text"]),
                span_type=str(span_record.get("span_type", "structural")),
            )
            if local_id not in selected_ids:
                features = dict(original_features)
                features.update(
                    {
                        "intervention_evaluated": 0.0,
                        "intervention_usage": 0.0,
                        "operator_sign_agreement": 0.0,
                        "operator_effect_mad": 0.0,
                        "operator_flip_rate": 0.0,
                    }
                )
                span_rows.append(
                    {
                        **{k: span_record[k] for k in ("span_id", "start", "end", "text", "span_type")},
                        "features": features,
                        "pseudo_label": None,
                        "interventions": [],
                    }
                )
                continue

            combined_features = dict(original_features)
            combined_features.update(
                {
                    "original_logit_chosen_margin": float(record["chosen_margin"]),
                    "original_logit_choice_entropy": safe_entropy_binary(
                        float(record["log_a"]), float(record["log_b"])
                    ),
                    "original_logit_probability_max": float(
                        torch.softmax(
                            torch.tensor([record["log_a"], record["log_b"]]), dim=0
                        ).max().item()
                    ),
                }
            )
            chosen_effects: list[float] = []
            normalized_effects: list[float] = []
            gold_deltas: list[float] = []
            evidences: list[dict] = []

            for kind in interventions:
                modified_question, replacement_range, replacement = intervene_with_range(question, span, kind)
                modified_spans: list[CandidateSpan] = []
                if replacement_range is not None:
                    modified_spans = [
                        CandidateSpan(
                            span_id=span.span_id,
                            start=replacement_range[0],
                            end=replacement_range[1],
                            text=replacement,
                            span_type=span.span_type,
                        )
                    ]
                try:
                    variant_prompt = adapter.render(question, modified_question, base_prompt)
                    ext = extractor.extract_spectral(
                        variant_prompt,
                        modified_question,
                        modified_spans,
                        choices,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    warnings.warn(f"[{idx}] OOM on {kind} span {local_id}")
                    continue
                except Exception as exc:
                    warnings.warn(f"[{idx}] intervention failed ({kind}, span {local_id}): {exc}")
                    continue

                fixed_margin = _fixed_choice_margin(ext["log_a"], ext["log_b"], original_choice, choices)
                chosen_effect = float(record["chosen_margin"] - fixed_margin)
                normalized_effect = chosen_effect / (abs(float(record["chosen_margin"])) + EPS)
                variant_choice = str(ext["chosen"])
                gold_margin_variant = None
                gold_delta = None
                if idx in train_set:
                    gold_margin_variant = _gold_margin(
                        ext["log_a"], ext["log_b"], gold, choices
                    )
                    gold_delta = float(record["gold_margin"] - gold_margin_variant)
                    gold_deltas.append(gold_delta)

                chosen_effects.append(chosen_effect)
                normalized_effects.append(normalized_effect)
                evidence = {
                    "kind": kind,
                    "log_a": ext["log_a"],
                    "log_b": ext["log_b"],
                    "chosen": variant_choice,
                    "flip": bool(variant_choice != original_choice),
                    "chosen_margin_effect": chosen_effect,
                    "normalized_effect": normalized_effect,
                }
                if idx in train_set:
                    evidence["gold_margin"] = float(gold_margin_variant)
                    evidence["gold_delta"] = float(gold_delta)
                evidences.append(evidence)

                combined_features[f"{kind}_chosen_margin_effect"] = chosen_effect
                combined_features[f"{kind}_normalized_effect"] = normalized_effect
                combined_features[f"{kind}_abs_normalized_effect"] = abs(normalized_effect)
                combined_features[f"{kind}_flip"] = float(variant_choice != original_choice)
                combined_features[f"{kind}_alternative_probability_gain"] = float(
                    torch.softmax(torch.tensor([ext["log_a"], ext["log_b"]]), dim=0)[
                        1 if original_choice == choices[0] else 0
                    ].item()
                    - torch.softmax(torch.tensor([record["log_a"], record["log_b"]]), dim=0)[
                        1 if original_choice == choices[0] else 0
                    ].item()
                )
                # Global redistribution is available even for deletion,
                # where the original span no longer has a token-aligned match.
                for family in ("spectral", "attention", "gradient"):
                    original_global = {
                        k: v
                        for k, v in record["global_spectral_features"].items()
                        if feature_family(k) == family
                    }
                    variant_global = {
                        k: v
                        for k, v in ext["global_spectral_features"].items()
                        if feature_family(k) == family
                    }
                    combined_features.update(
                        _prefixed_delta(
                            original_global,
                            variant_global,
                            f"{kind}_global_{family}",
                            include_absolute=False,
                        )
                    )

                # Neutralize/mask/negate retain a replacement span, allowing
                # direct span-to-span attention, gradient, and spectral deltas.
                if modified_spans and ext["span_spectral_features"]:
                    variant_span_features = ext["span_spectral_features"][0]
                    for family in ("spectral", "attention", "gradient"):
                        original_family = {
                            k: v for k, v in original_features.items()
                            if feature_family(k) == family
                        }
                        variant_family = {
                            k: v for k, v in variant_span_features.items()
                            if feature_family(k) == family
                        }
                        combined_features.update(
                            _prefixed_delta(
                                original_family,
                                variant_family,
                                f"{kind}_span_{family}",
                                include_absolute=(
                                    args.include_variant_absolute_spectral
                                    if family == "spectral"
                                    else args.include_variant_absolute_whitebox
                                ),
                            )
                        )

            if normalized_effects:
                arr = np.asarray(normalized_effects, dtype=float)
                median = float(np.median(arr))
                mad = float(np.median(np.abs(arr - median)))
                informative = np.abs(arr) >= args.response_deadzone
                if informative.any() and abs(median) >= args.response_deadzone:
                    agreement = float(np.mean(np.sign(arr[informative]) == np.sign(median)))
                else:
                    agreement = float(np.mean(np.abs(arr) < args.response_deadzone))
                flip_rate = float(np.mean([bool(x["flip"]) for x in evidences]))
                usage = float(
                    np.clip(np.median(np.abs(arr)), 0.0, 1.0)
                    * agreement
                    * math.exp(-mad / max(args.response_temperature, EPS))
                )
                combined_features.update(
                    {
                        "intervention_evaluated": 1.0,
                        "response_median": median,
                        "response_abs_median": abs(median),
                        "response_mean": float(arr.mean()),
                        "response_std": float(arr.std()),
                        "operator_sign_agreement": agreement,
                        "operator_effect_mad": mad,
                        "operator_flip_rate": flip_rate,
                        "intervention_usage": usage,
                    }
                )
            else:
                combined_features.update(
                    {
                        "intervention_evaluated": 0.0,
                        "intervention_usage": 0.0,
                        "operator_sign_agreement": 0.0,
                        "operator_effect_mad": 0.0,
                        "operator_flip_rate": 0.0,
                    }
                )

            pseudo = None
            if idx in train_set and gold_deltas:
                pseudo = build_soft_role(
                    gold_deltas,
                    deadzone=args.role_deadzone,
                    temperature=args.role_temperature,
                )
                pseudo["evaluated"] = True
                pseudo["interventions"] = evidences

            span_rows.append(
                {
                    **{k: span_record[k] for k in ("span_id", "start", "end", "text", "span_type")},
                    "features": combined_features,
                    "pseudo_label": pseudo,
                    "interventions": evidences,
                }
            )

        swap_row = _profile_swap_row(
            record=record,
            item=item,
            extractor=extractor,
            adapter=adapter,
            choices=choices,
            is_train=idx in train_set,
            args=args,
        )
        if swap_row is not None:
            span_rows.append(swap_row)

        rec = {
            "idx": idx,
            "original_chosen": original_choice,
            "original_chosen_margin": record["chosen_margin"],
            "span_rows": span_rows,
        }
        append_jsonl(cache_path, rec)
        output[idx] = rec
        if count % 10 == 0 or count == len(base_records):
            print(f"test-time intervention features: {count}/{len(base_records)}", flush=True)
    return output


def prepare_interventional_role_training(
    intervention_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    min_reliability: float,
) -> tuple[list[dict], list[list[float]], list[float], list[int]]:
    features: list[dict] = []
    soft: list[list[float]] = []
    rels: list[float] = []
    groups: list[int] = []
    for idx in train_indices:
        rec = intervention_by_idx[idx]
        for row in rec["span_rows"]:
            pseudo = row.get("pseudo_label")
            if not pseudo or not pseudo.get("evaluated", False):
                continue
            reliability = float(pseudo.get("reliability", 0.0))
            if reliability < min_reliability:
                continue
            features.append(row["features"])
            soft.append(pseudo["role_probs"])
            rels.append(reliability)
            groups.append(int(idx))
    if not features:
        raise RuntimeError("no reliable interventional span labels were produced")
    return features, soft, rels, groups


def make_interventional_role_predictions(
    intervention_by_idx: dict[int, dict],
    indices: Sequence[int],
    role_head: MultimodalSpanRoleHead,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for idx in indices:
        feats = [row["features"] for row in intervention_by_idx[idx]["span_rows"]]
        out[int(idx)] = role_head.predict_proba(feats)
    return out


def make_interventional_oof_role_predictions(
    intervention_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    role_features: list[dict],
    role_soft: list[list[float]],
    role_rel: list[float],
    role_groups: list[int],
    n_splits: int,
    pca_dim: int,
    seed: int,
    full_head: MultimodalSpanRoleHead,
) -> dict[int, np.ndarray]:
    indices = list(train_indices)
    splits = min(n_splits, len(indices))
    if splits < 2:
        return make_interventional_role_predictions(intervention_by_idx, indices, full_head)
    result: dict[int, np.ndarray] = {}
    splitter = KFold(n_splits=splits, shuffle=True, random_state=seed)
    group_arr = np.asarray(role_groups, dtype=int)
    for fold, (fit_pos, val_pos) in enumerate(splitter.split(indices)):
        fit_items = {indices[i] for i in fit_pos}
        val_items = [indices[i] for i in val_pos]
        fit_rows = [i for i, group in enumerate(group_arr) if int(group) in fit_items]
        if not fit_rows:
            head = full_head
        else:
            head = full_head.new_unfitted(seed=seed + fold).fit(
                [role_features[i] for i in fit_rows],
                [role_soft[i] for i in fit_rows],
                [role_rel[i] for i in fit_rows],
            )
        for idx in val_items:
            feats = [row["features"] for row in intervention_by_idx[idx]["span_rows"]]
            result[int(idx)] = head.predict_proba(feats)
    for idx in indices:
        if int(idx) not in result:
            result[int(idx)] = full_head.predict_proba(
                [row["features"] for row in intervention_by_idx[int(idx)]["span_rows"]]
            )
    return result


def evaluate_span_role_predictions(
    intervention_by_idx: dict[int, dict],
    indices: Sequence[int],
    predictions_by_idx: dict[int, np.ndarray],
    min_reliability: float,
) -> dict:
    """Evaluate OOF span-role predictions against reliable hard pseudo-labels."""
    labels: list[int] = []
    probs: list[np.ndarray] = []
    reliabilities: list[float] = []
    for idx in indices:
        rows = intervention_by_idx[int(idx)]["span_rows"]
        pred = predictions_by_idx[int(idx)]
        if len(rows) != len(pred):
            raise ValueError("span row / role probability length mismatch")
        for row, p in zip(rows, pred):
            pseudo = row.get("pseudo_label")
            if not pseudo or not pseudo.get("evaluated", False):
                continue
            reliability = float(pseudo.get("reliability", 0.0))
            if reliability < min_reliability:
                continue
            labels.append(ROLE_TO_ID[str(pseudo["hard_role"])])
            probs.append(np.asarray(p, dtype=float))
            reliabilities.append(reliability)
    if not labels:
        return {"n": 0}
    y = np.asarray(labels, dtype=int)
    P = np.vstack(probs)
    pred_y = P.argmax(axis=1)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred_y)),
        "weighted_accuracy": float(
            np.average((pred_y == y).astype(float), weights=np.asarray(reliabilities))
        ),
        "log_loss": float(log_loss(y, P, labels=[0, 1, 2])),
        "class_counts": {
            ROLE_NAMES[c]: int(np.sum(y == c)) for c in range(3)
        },
    }
    per_role: dict[str, Any] = {}
    aucs: list[float] = []
    for role_id, role_name in enumerate(ROLE_NAMES):
        binary = (y == role_id).astype(int)
        if len(np.unique(binary)) < 2:
            auc = None
            ap = None
        else:
            auc = float(roc_auc_score(binary, P[:, role_id]))
            ap = float(average_precision_score(binary, P[:, role_id]))
            aucs.append(auc)
        per_role[role_name] = {"auroc": auc, "auprc": ap}
    result["per_role"] = per_role
    result["macro_ovr_auroc"] = float(np.mean(aucs)) if aucs else None
    return result


def build_interventional_role_evidence(
    span_rows: Sequence[dict],
    role_probs: np.ndarray,
) -> dict:
    n = len(span_rows)
    if n == 0:
        return {
            "usage": np.zeros(0),
            "shortcut_base": np.zeros(0),
            "constraint_base": np.zeros(0),
            "shortcut_evidence": 0.0,
            "constraint_evidence": 0.0,
            "role_ambiguity": 0.0,
        }
    usage = np.asarray(
        [float(row["features"].get("intervention_usage", 0.0)) for row in span_rows],
        dtype=float,
    )
    shortcut_p = role_probs[:, ROLE_TO_ID["shortcut"]]
    constraint_p = role_probs[:, ROLE_TO_ID["constraint"]]
    active = max(int(np.sum(usage > 0)), 1)
    shortcut_base = shortcut_p * usage / active
    constraint_base = constraint_p * usage / active
    entropy = -(role_probs * np.log(role_probs + 1e-12)).sum(axis=1)
    return {
        "usage": usage,
        "shortcut_base": shortcut_base,
        "constraint_base": constraint_base,
        "shortcut_evidence": float(shortcut_base.sum()),
        "constraint_evidence": float(constraint_base.sum()),
        "role_ambiguity": float(entropy.mean()),
    }


def choose_interventional_explanation(
    span_rows: Sequence[dict],
    role_probs: np.ndarray,
    evidence: dict,
    mechanism: RoleMechanismHead,
    role_probability_threshold: float,
    usage_threshold: float,
    contribution_threshold: float,
) -> dict:
    usage = np.asarray(evidence["usage"], dtype=float)
    shortcut_pos, constraint_neg = mechanism.span_contributions(evidence)
    s_prob = role_probs[:, ROLE_TO_ID["shortcut"]]
    c_prob = role_probs[:, ROLE_TO_ID["constraint"]]
    s_candidates = [
        i for i in range(len(span_rows))
        if s_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and shortcut_pos[i] >= contribution_threshold
    ]
    c_candidates = [
        i for i in range(len(span_rows))
        if c_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and abs(constraint_neg[i]) >= contribution_threshold
    ]
    s_idx = max(s_candidates, key=lambda i: shortcut_pos[i]) if s_candidates else None
    remaining = [i for i in c_candidates if i != s_idx]
    c_idx = max(remaining, key=lambda i: abs(constraint_neg[i])) if remaining else None

    per_span = []
    for i, row in enumerate(span_rows):
        per_span.append(
            {
                "span_id": int(row["span_id"]),
                "text": row["text"],
                "span_type": row["span_type"],
                "constraint_probability": float(c_prob[i]),
                "shortcut_probability": float(s_prob[i]),
                "irrelevant_probability": float(role_probs[i, ROLE_TO_ID["irrelevant"]]),
                "intervention_usage": float(usage[i]),
                "response_median": float(row["features"].get("response_median", row["features"].get("profile_swap_normalized_effect", 0.0))),
                "operator_sign_agreement": float(row["features"].get("operator_sign_agreement", 0.0)),
                "operator_flip_rate": float(row["features"].get("operator_flip_rate", 0.0)),
                "shortcut_logit_contribution": float(shortcut_pos[i]),
                "constraint_logit_contribution": float(constraint_neg[i]),
                "net_role_logit_contribution": float(shortcut_pos[i] + constraint_neg[i]),
                "interventions": row.get("interventions", []),
            }
        )

    def payload(idx: int | None, role: str) -> dict:
        if idx is None:
            return {"resolved": False, "reason": f"no {role} span passed thresholds"}
        return {"resolved": True, **per_span[idx]}

    return {
        "predicted_shortcut": payload(s_idx, "shortcut"),
        "predicted_constraint": payload(c_idx, "constraint"),
        "distinct_pair_resolved": bool(s_idx is not None and c_idx is not None and s_idx != c_idx),
        "per_span_contributions": per_span,
    }


def interventional_role_oof_logits(
    evidences: list[dict],
    labels: list[int],
    n_splits: int,
    args,
) -> np.ndarray:
    return role_mechanism_oof_logits(evidences, labels, n_splits, args)


def _role_pseudo_counts(
    intervention_by_idx: dict[int, dict],
    train_indices: Sequence[int],
) -> dict[str, int]:
    counts = {name: 0 for name in ROLE_NAMES}
    for idx in train_indices:
        for row in intervention_by_idx[idx]["span_rows"]:
            pseudo = row.get("pseudo_label")
            if pseudo and pseudo.get("evaluated", False):
                counts[str(pseudo["hard_role"])] += 1
    return counts


def train_feature_set_pipeline(
    feature_set_name: str,
    intervention_by_idx: dict[int, dict],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    role_features: list[dict],
    role_soft: list[list[float]],
    role_rel: list[float],
    role_groups: list[int],
    train_labels: list[int],
    test_labels: list[int],
    args,
) -> dict:
    """Fit one span-role feature ablation and its monotonic item detector."""
    role_head = MultimodalSpanRoleHead(
        feature_set_name=feature_set_name,
        pca_dim=args.role_pca_dim,
        seed=args.seed,
    ).fit(role_features, role_soft, role_rel)

    train_role_probs = make_interventional_oof_role_predictions(
        intervention_by_idx,
        train_idx,
        role_features,
        role_soft,
        role_rel,
        role_groups,
        n_splits=args.role_oof_folds,
        pca_dim=args.role_pca_dim,
        seed=args.seed,
        full_head=role_head,
    )
    test_role_probs = make_interventional_role_predictions(
        intervention_by_idx,
        test_idx,
        role_head,
    )

    span_role_metrics = evaluate_span_role_predictions(
        intervention_by_idx,
        train_idx,
        train_role_probs,
        min_reliability=args.min_role_reliability,
    )

    train_evidence = [
        build_interventional_role_evidence(
            intervention_by_idx[idx]["span_rows"], train_role_probs[idx]
        )
        for idx in train_idx
    ]
    test_evidence = [
        build_interventional_role_evidence(
            intervention_by_idx[idx]["span_rows"], test_role_probs[idx]
        )
        for idx in test_idx
    ]

    role_oof_logits = interventional_role_oof_logits(
        train_evidence,
        train_labels,
        args.hall_oof_folds,
        args,
    )
    role_oof_prob = sigmoid_np(role_oof_logits)
    threshold = choose_f1_threshold(np.asarray(train_labels), role_oof_prob)

    mechanism = RoleMechanismHead(
        epochs=args.mechanism_epochs,
        lr=args.mechanism_lr,
        l2=args.mechanism_l2,
        min_shortcut_weight=args.min_shortcut_weight,
        seed=args.seed,
    ).fit(train_evidence, train_labels)
    train_full_prob = mechanism.predict_proba(train_evidence)
    test_prob = mechanism.predict_proba(test_evidence)

    metrics = {
        "train_oof": evaluate_binary(train_labels, role_oof_prob, threshold),
        "train_full": evaluate_binary(train_labels, train_full_prob, threshold),
        "test": evaluate_binary(test_labels, test_prob, threshold),
    }
    return {
        "feature_set": feature_set_name,
        "role_head": role_head,
        "train_role_probs": train_role_probs,
        "test_role_probs": test_role_probs,
        "span_role_metrics_train_oof": span_role_metrics,
        "train_evidence": train_evidence,
        "test_evidence": test_evidence,
        "role_oof_prob": role_oof_prob,
        "threshold": float(threshold),
        "mechanism": mechanism,
        "train_full_prob": train_full_prob,
        "test_prob": test_prob,
        "metrics": metrics,
    }


def compact_feature_set_report(result: dict, primary_test_auroc: float | None) -> dict:
    test_metrics = result["metrics"]["test"]
    test_auc = test_metrics.get("auroc")
    delta = None
    if test_auc is not None and primary_test_auroc is not None:
        delta = float(test_auc - primary_test_auroc)
    return {
        "feature_families": list(FEATURE_SET_FAMILIES[result["feature_set"]]),
        "role_head": result["role_head"].report(),
        "span_role_train_oof": result["span_role_metrics_train_oof"],
        "selected_threshold_from_train_oof": result["threshold"],
        "role_mechanism": result["mechanism"].report(),
        "item_metrics": result["metrics"],
        "test_auroc_delta_vs_primary": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test-time interventional multimodal span detector v7 with "
            "attention/gradient/logit/spectral feature ablations"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="interventional_multimodal_v7_output")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--gold-field")
    parser.add_argument("--answer-instruction", default="Reply with a single character: 1 or 2.")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--choice-a", default="1")
    parser.add_argument("--choice-b", default="2")
    parser.add_argument("--span-mode", choices=["sentence", "clause", "atomic"], default="atomic")
    parser.add_argument("--include-question-span", action="store_true")
    parser.add_argument("--min-span-words", type=int, default=2)
    parser.add_argument("--interventions", default="delete,neutralize,mask")
    parser.add_argument(
        "--max-intervention-spans",
        type=int,
        default=0,
        help="0 evaluates every candidate span; positive values apply a multimodal salience budget",
    )
    parser.add_argument("--include-profile-swap-span", action="store_true")
    parser.add_argument(
        "--no-profile-swap-span",
        dest="include_profile_swap_span",
        action="store_false",
    )
    parser.set_defaults(include_profile_swap_span=True)
    parser.add_argument("--include-variant-absolute-spectral", action="store_true")
    parser.add_argument(
        "--include-variant-absolute-whitebox",
        action="store_true",
        help="also retain absolute variant attention/gradient values, not only deltas",
    )
    parser.add_argument("--keep-head-identities", action="store_true")
    parser.add_argument("--no-gradient-features", dest="compute_gradient_features", action="store_false")
    parser.set_defaults(compute_gradient_features=True)
    parser.add_argument("--lap-topk", type=int, default=10)
    parser.add_argument("--role-pca-dim", type=int, default=128)
    parser.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SET_ORDER),
        help=(
            "comma-separated feature ablations; available: "
            + ",".join(DEFAULT_FEATURE_SET_ORDER)
        ),
    )
    parser.add_argument("--primary-feature-set", default="all_combined")
    parser.add_argument("--role-deadzone", type=float, default=0.25)
    parser.add_argument("--role-temperature", type=float, default=0.75)
    parser.add_argument("--min-role-reliability", type=float, default=0.20)
    parser.add_argument("--response-deadzone", type=float, default=0.05)
    parser.add_argument("--response-temperature", type=float, default=0.50)
    parser.add_argument("--mechanism-epochs", type=int, default=2500)
    parser.add_argument("--mechanism-lr", type=float, default=0.03)
    parser.add_argument("--mechanism-l2", type=float, default=1e-3)
    parser.add_argument("--min-shortcut-weight", type=float, default=0.05)
    parser.add_argument("--role-prob-threshold", type=float, default=0.45)
    parser.add_argument("--usage-threshold", type=float, default=0.20)
    parser.add_argument("--span-contribution-threshold", type=float, default=0.002)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-oof-folds", type=int, default=5)
    parser.add_argument("--hall-oof-folds", type=int, default=5)
    parser.add_argument("--statistics-bootstrap", type=int, default=2000)
    parser.add_argument("--statistics-permutations", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    interventions = [x.strip() for x in args.interventions.split(",") if x.strip()]
    allowed_interventions = {"delete", "neutralize", "mask", "negate"}
    invalid = set(interventions) - allowed_interventions
    if invalid:
        parser.error(f"invalid interventions: {sorted(invalid)}")
    if len(interventions) < 2:
        warnings.warn("at least two intervention operators are recommended")

    feature_sets = []
    for name in (x.strip() for x in args.feature_sets.split(",")):
        if name and name not in feature_sets:
            feature_sets.append(name)
    unknown_sets = set(feature_sets) - set(FEATURE_SET_FAMILIES)
    if unknown_sets:
        parser.error(f"unknown feature sets: {sorted(unknown_sets)}")
    if not feature_sets:
        parser.error("at least one feature set is required")
    if args.primary_feature_set not in feature_sets:
        parser.error("primary-feature-set must also appear in --feature-sets")
    if not args.compute_gradient_features:
        gradient_sets = [
            name for name in feature_sets
            if "gradient" in FEATURE_SET_FAMILIES[name]
        ]
        if gradient_sets:
            parser.error(
                "--no-gradient-features is incompatible with selected feature sets: "
                + ",".join(gradient_sets)
            )

    choices = (str(args.choice_a), str(args.choice_b))
    if choices[0] == choices[1]:
        parser.error("choice-a and choice-b must differ")

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device == "cpu" and dtype != torch.float32:
        warnings.warn("CPU selected; forcing float32")
        dtype = torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / "base_multimodal_features.jsonl"
    response_path = out_dir / "intervention_multimodal_features.jsonl"
    prediction_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    bundle_path = out_dir / "interventional_multimodal_bundle.joblib"
    if not args.resume:
        for path in (base_path, response_path):
            if path.exists():
                path.unlink()
    if prediction_path.exists():
        prediction_path.unlink()

    items = read_records(args.data)
    if args.limit > 0:
        items = items[:args.limit]
    if len(items) < 10:
        warnings.warn("very small dataset; estimates will be unstable")

    print(f"loading model {args.model} on {device} ...", flush=True)
    extractor = InterventionalMultimodalExtractor(
        args.model,
        device=device,
        dtype=dtype,
        lap_topk=args.lap_topk,
        keep_head_identities=args.keep_head_identities,
        compute_gradient_features=args.compute_gradient_features,
    )
    adapter = PromptAdapter(
        extractor.tok,
        question_field=args.question_field,
        prompt_field=args.prompt_field,
        gold_field=args.gold_field,
        answer_instruction=args.answer_instruction,
        apply_chat_template=not args.no_chat_template,
    )

    base_records = extract_interventional_base_records(
        items, extractor, adapter, choices, args, base_path
    )
    if len(base_records) < 4:
        raise RuntimeError("too few successfully extracted records")
    base_by_idx = {int(r["idx"]): r for r in base_records}
    valid_indices = np.asarray(sorted(base_by_idx), dtype=int)
    labels_all = np.asarray(
        [int(base_by_idx[i]["hallucinated"]) for i in valid_indices],
        dtype=int,
    )
    if len(np.unique(labels_all)) < 2:
        raise RuntimeError("model produced only one hallucination class")
    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels_all,
    )
    train_idx = sorted(int(x) for x in train_idx)
    test_idx = sorted(int(x) for x in test_idx)
    train_set = set(train_idx)

    intervention_by_idx = extract_all_intervention_responses(
        base_records=base_records,
        items=items,
        train_set=train_set,
        extractor=extractor,
        adapter=adapter,
        choices=choices,
        interventions=interventions,
        args=args,
        cache_path=response_path,
    )

    role_features, role_soft, role_rel, role_groups = prepare_interventional_role_training(
        intervention_by_idx,
        train_idx,
        min_reliability=args.min_role_reliability,
    )
    train_labels = [int(base_by_idx[idx]["hallucinated"]) for idx in train_idx]
    test_labels = [int(base_by_idx[idx]["hallucinated"]) for idx in test_idx]

    feature_results: dict[str, dict] = {}
    for feature_set_name in feature_sets:
        print(f"\n=== fitting feature set: {feature_set_name} ===", flush=True)
        feature_results[feature_set_name] = train_feature_set_pipeline(
            feature_set_name=feature_set_name,
            intervention_by_idx=intervention_by_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            role_features=role_features,
            role_soft=role_soft,
            role_rel=role_rel,
            role_groups=role_groups,
            train_labels=train_labels,
            test_labels=test_labels,
            args=args,
        )
        print(
            json.dumps(
                feature_results[feature_set_name]["metrics"]["test"],
                indent=2,
            ),
            flush=True,
        )

    primary = feature_results[args.primary_feature_set]
    primary_test_auroc = primary["metrics"]["test"].get("auroc")

    prediction_records: list[dict] = []
    for split, indices in (("train", train_idx), ("test", test_idx)):
        for pos, idx in enumerate(indices):
            base = base_by_idx[idx]
            span_rows = intervention_by_idx[idx]["span_rows"]
            primary_role_probs = (
                primary["train_role_probs"][idx]
                if split == "train"
                else primary["test_role_probs"][idx]
            )
            primary_evidence = (
                primary["train_evidence"][pos]
                if split == "train"
                else primary["test_evidence"][pos]
            )
            primary_probability = float(
                primary["train_full_prob"][pos]
                if split == "train"
                else primary["test_prob"][pos]
            )
            explanation = choose_interventional_explanation(
                span_rows,
                primary_role_probs,
                primary_evidence,
                primary["mechanism"],
                role_probability_threshold=args.role_prob_threshold,
                usage_threshold=args.usage_threshold,
                contribution_threshold=args.span_contribution_threshold,
            )

            family_outputs: dict[str, dict] = {}
            for feature_set_name, result in feature_results.items():
                evidence = (
                    result["train_evidence"][pos]
                    if split == "train"
                    else result["test_evidence"][pos]
                )
                probability = float(
                    result["train_full_prob"][pos]
                    if split == "train"
                    else result["test_prob"][pos]
                )
                family_outputs[feature_set_name] = {
                    "hallucination_probability": probability,
                    "predicted_hallucination": int(probability >= result["threshold"]),
                    "shortcut_evidence": float(evidence["shortcut_evidence"]),
                    "constraint_evidence": float(evidence["constraint_evidence"]),
                    "role_ambiguity": float(evidence["role_ambiguity"]),
                    "role_logit": float(
                        result["mechanism"].decision_function([evidence])[0]
                    ),
                }

            row = {
                "idx": idx,
                "split": split,
                "gold": base["gold"],
                "chosen": base["chosen"],
                "hallucinated": int(base["hallucinated"]),
                "primary_feature_set": args.primary_feature_set,
                "hallucination_probability": primary_probability,
                "predicted_hallucination": int(
                    primary_probability >= primary["threshold"]
                ),
                "mechanism_decomposition": family_outputs[args.primary_feature_set],
                "feature_set_outputs": family_outputs,
                "explanation": explanation,
            }
            append_jsonl(prediction_path, row)
            prediction_records.append(row)

    test_prediction_records = [r for r in prediction_records if r["split"] == "test"]
    shortcut_stats = shortcut_explanatory_statistics(
        test_prediction_records,
        n_boot=args.statistics_bootstrap,
        n_perm=args.statistics_permutations,
        seed=args.seed,
    )

    feature_set_reports = {
        name: compact_feature_set_report(result, primary_test_auroc)
        for name, result in feature_results.items()
    }
    item_ranking = sorted(
        feature_sets,
        key=lambda name: (
            feature_results[name]["metrics"]["test"].get("auroc")
            if feature_results[name]["metrics"]["test"].get("auroc") is not None
            else -1.0
        ),
        reverse=True,
    )
    span_ranking = sorted(
        feature_sets,
        key=lambda name: (
            feature_results[name]["span_role_metrics_train_oof"].get("macro_ovr_auroc")
            if feature_results[name]["span_role_metrics_train_oof"].get("macro_ovr_auroc") is not None
            else -1.0
        ),
        reverse=True,
    )

    role_reliabilities = np.asarray(role_rel, dtype=float)
    summary = {
        "method": "test-time interventional multimodal span role-mediated detector v7",
        "model": args.model,
        "data": args.data,
        "device": device,
        "dtype": str(dtype),
        "choices": list(choices),
        "n_input": len(items),
        "n_extracted": len(base_records),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "interventions_used_for_prediction": interventions,
        "span_mode": args.span_mode,
        "max_intervention_spans": args.max_intervention_spans,
        "keep_head_identities": args.keep_head_identities,
        "compute_gradient_features": args.compute_gradient_features,
        "lap_topk": args.lap_topk,
        "feature_sets_evaluated": feature_sets,
        "primary_feature_set": args.primary_feature_set,
        "feature_set_comparison": feature_set_reports,
        "feature_set_item_test_auroc_ranking": item_ranking,
        "feature_set_span_role_oof_auroc_ranking": span_ranking,
        "role_pseudo_label_counts": _role_pseudo_counts(intervention_by_idx, train_idx),
        "mean_role_reliability": float(role_reliabilities.mean()),
        "n_role_training_spans": len(role_features),
        "primary_metrics": primary["metrics"],
        "primary_role_head": primary["role_head"].report(),
        "primary_role_mechanism": primary["mechanism"].report(),
        "primary_selected_threshold_from_train_oof": primary["threshold"],
        "shortcut_explanatory_statistics_primary": shortcut_stats,
        "files": {
            "base_multimodal_features": str(base_path),
            "intervention_multimodal_features": str(response_path),
            "predictions": str(prediction_path),
            "model_bundle": str(bundle_path),
        },
        "method_notes": {
            "test_prediction_uses_span_interventions": True,
            "test_prediction_uses_gold_answer": False,
            "detector_has_global_residual_channel": False,
            "span_role_feature_families": [
                "structural", "behavior/logit", "attention", "gradient", "spectral"
            ],
            "logit_features_are_span_level_intervention_responses": True,
            "gradient_features_are_span_level_chosen_contrast_gradients": True,
            "attention_features_are_span_level_decision_row_features": True,
            "spectral_features_use_per_layer_per_head_causal_attention_laplacian": True,
            "all_feature_set_item_detectors_share_the_same_behavior_derived_usage_weight": True,
            "cleanest_feature_family_comparison_is_span_role_train_oof_metrics": True,
            "profile_swap_is_a_synthetic_intervention_span": bool(args.include_profile_swap_span),
            "causal_audit_warning": (
                "delete/neutralize/mask are part of prediction and cannot also serve as independent post-hoc causal validation; reserve held-out semantic-preserving operators"
            ),
        },
    }

    joblib.dump(
        {
            "feature_sets": {
                name: {
                    "role_head": result["role_head"],
                    "mechanism": result["mechanism"],
                    "threshold": result["threshold"],
                }
                for name, result in feature_results.items()
            },
            "primary_feature_set": args.primary_feature_set,
            "choices": choices,
            "args": vars(args),
        },
        bundle_path,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print("\n=== v7 multimodal feature comparison complete ===")
    for rank, name in enumerate(item_ranking, 1):
        test_metrics = feature_results[name]["metrics"]["test"]
        span_metrics = feature_results[name]["span_role_metrics_train_oof"]
        print(
            f"{rank:2d}. {name:20s} "
            f"item_AUROC={test_metrics.get('auroc')} "
            f"span_macro_AUROC={span_metrics.get('macro_ovr_auroc')}"
        )
    print(f"\nprimary={args.primary_feature_set}")
    print(json.dumps(primary["metrics"]["test"], indent=2))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
