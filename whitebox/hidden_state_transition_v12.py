#!/usr/bin/env python3
"""
Open-ended static-hidden-state and behavior hallucination detector v12.

Core workflow
-------------
1. Generate the model's original answer.
2. Teacher-force the same original answer and extract every layer's hidden state
   at the last answer token (paper-style static hidden-state representation).
3. Segment the prompt into atomic spans.
4. Rank spans without attention, using the similarity between each span's
   mean hidden state and the original answer hidden state.
5. Intervene on the selected spans with delete / neutralize / mask.
6. Keep the v8 behavior definition:

       support_delta = mean_logP(original answer | original prompt)
                     - mean_logP(original answer | intervened prompt)

7. Use behavior plus static span-hidden relation features to learn
   constraint / shortcut / irrelevant pseudo roles.
8. Scan all hidden-state layers with train-only OOF linear probes and fold-local
   Top-k hidden-coordinate selection, then select the best layer without using
   the test set.
9. Compare hidden-only, hidden+behavior, and hidden+behavior+shortcut models.

No hidden-state transition is computed in v12. Intervened prompts are used only
for behavior measurements and pseudo-role construction.

References are used only for:
  * correctness labels,
  * training pseudo-role construction,
  * final evaluation.
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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CACHE_SCHEMA_VERSION = "openended_v12_static_hidden_v2"
ROLE_NAMES = ["constraint", "shortcut", "irrelevant"]
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
OPERATORS = ("delete", "neutralize", "mask")

# "hidden_relation" is span-specific. It does not contain the full raw answer
# hidden vector; instead it summarizes how a span representation relates to the
# answer representation at each layer. Item-level probes select a small number
# of answer-hidden coordinates using training data inside each OOF fold.
ROLE_FEATURE_SETS = {
    "structure_only": ("structural",),
    "behavior_only": ("behavior",),
    "hidden_relation_only": ("hidden_relation",),
    "behavior_hidden": ("behavior", "hidden_relation"),
    "structure_behavior_hidden": ("structural", "behavior", "hidden_relation"),
}

ITEM_MODES = {
    "behavior_only",
    "hidden_only",
    "hidden_behavior",
    "behavior_shortcut",
    "hidden_shortcut",
    "hidden_behavior_shortcut",
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


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        handle.flush()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def safe_auroc(y_true: np.ndarray, probability: np.ndarray) -> Optional[float]:
    return (
        float(roc_auc_score(y_true, probability))
        if len(np.unique(y_true)) > 1
        else None
    )


def safe_auprc(y_true: np.ndarray, probability: np.ndarray) -> Optional[float]:
    return (
        float(average_precision_score(y_true, probability))
        if len(np.unique(y_true)) > 1
        else None
    )


# -------------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------------

def get_nested(row: dict[str, Any], field: Optional[str]) -> Any:
    if not field:
        return None
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_present(row: dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = get_nested(row, field)
        if value is not None:
            return value
    return None


def flatten_refs(value: Any) -> list[str]:
    output: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            if item.strip():
                output.append(item.strip())
        elif isinstance(item, (int, float, np.integer, np.floating)):
            output.append(str(item))
        elif isinstance(item, (list, tuple, set, np.ndarray)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            preferred = (
                "aliases",
                "correct_answers",
                "answers",
                "answer",
                "text",
                "value",
                "normalized_value",
            )
            found = False
            for key in preferred:
                if key in item:
                    visit(item[key])
                    found = True
            if not found:
                for child in item.values():
                    visit(child)

    visit(value)
    result: list[str] = []
    seen: set[str] = set()
    for text in output:
        key = text.lower().strip()
        if key and key not in seen:
            result.append(text.strip())
            seen.add(key)
    return result


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.hf_dataset:
        from datasets import load_dataset

        if args.hf_subset:
            dataset = load_dataset(
                args.hf_dataset,
                args.hf_subset,
                split=args.hf_split,
            )
        else:
            dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        rows = [dict(item) for item in dataset]
    else:
        path = Path(args.input)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif suffix == ".json":
            raw_text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                rows = [
                    json.loads(line)
                    for line in raw_text.splitlines()
                    if line.strip()
                ]
            else:
                if isinstance(data, list):
                    rows = data
                elif isinstance(data, dict):
                    rows = next(
                        (
                            data[key]
                            for key in ("data", "items", "examples", "questions")
                            if isinstance(data.get(key), list)
                        ),
                        None,
                    )
                    if rows is None:
                        raise ValueError(
                            "No list-valued data/items/examples/questions key."
                        )
                else:
                    raise ValueError("Unsupported JSON root")
        elif suffix in {".parquet", ".pq"}:
            rows = pd.read_parquet(path).to_dict("records")
        elif suffix in {".csv", ".tsv"}:
            rows = pd.read_csv(
                path,
                sep="\t" if suffix == ".tsv" else ",",
            ).to_dict("records")
        else:
            raise ValueError(f"Unsupported suffix: {suffix}")

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    return rows


def build_examples(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Example]:
    examples: list[Example] = []
    for index, row in enumerate(rows):
        question = (
            get_nested(row, args.question_field)
            if args.question_field
            else None
        )
        if question is None:
            question = first_present(
                row,
                ("question", "query", "prompt", "input", "instruction", "problem"),
            )
        context = (
            get_nested(row, args.context_field)
            if args.context_field
            else None
        )
        if context is None:
            context = first_present(
                row,
                ("knowledge", "context", "passage", "story", "article", "document", "evidence"),
            )
        answers = (
            get_nested(row, args.answers_field)
            if args.answers_field
            else None
        )
        if answers is None:
            answers = first_present(
                row,
                (
                    "right_answer",
                    "correct_answers",
                    "answers",
                    "answer.aliases",
                    "answer",
                    "reference_answer",
                    "gold_answer",
                    "target",
                    "output",
                ),
            )
        references = flatten_refs(answers)
        if question is None or not references:
            warnings.warn(f"Skipping row {index}: missing question or references")
            continue

        question_text = str(question).strip()
        context_text = "" if context is None else str(context).strip()
        if args.prompt_field:
            source_text = str(get_nested(row, args.prompt_field) or "").strip()
        else:
            source_text = (
                f"Context:\n{context_text}\n\nQuestion:\n{question_text}"
                if context_text
                else f"Question:\n{question_text}"
            )

        item_id = (
            get_nested(row, args.id_field)
            if args.id_field
            else None
        )
        if item_id is None:
            item_id = first_present(row, ("id", "key", "question_id", "qid"))
        examples.append(
            Example(
                item_id=str(item_id) if item_id is not None else f"item_{index:06d}",
                source_text=source_text,
                question=question_text,
                context=context_text,
                references=references,
                raw_index=index,
            )
        )

    if not examples:
        raise ValueError("No usable examples")
    return examples


# -------------------------------------------------------------------------
# Correctness evaluation
# -------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def token_f1(first: str, second: str) -> float:
    first_tokens = normalize_answer(first).split()
    second_tokens = normalize_answer(second).split()
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0
    first_counts: dict[str, int] = {}
    second_counts: dict[str, int] = {}
    for token in first_tokens:
        first_counts[token] = first_counts.get(token, 0) + 1
    for token in second_tokens:
        second_counts[token] = second_counts.get(token, 0) + 1
    common = sum(
        min(count, second_counts.get(token, 0))
        for token, count in first_counts.items()
    )
    if common == 0:
        return 0.0
    precision = common / len(first_tokens)
    recall = common / len(second_tokens)
    return 2 * precision * recall / (precision + recall)


def max_ref_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, reference) for reference in references), default=0.0)


def exact_or_contained(prediction: str, reference: str) -> bool:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    return (
        pred == ref
        or (len(ref.split()) >= 2 and ref in pred)
        or (len(pred.split()) >= 2 and pred in ref)
    )


def last_number(text: str) -> Optional[str]:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        value = float(matches[-1])
        return str(int(value)) if value.is_integer() else f"{value:.12g}"
    except ValueError:
        return matches[-1]


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def polarity_signature(text: str) -> tuple[int, int]:
    normalized = normalize_answer(text)
    yes = int(bool(re.search(r"\b(?:yes|true|correct|does|will|is)\b", normalized)))
    no = int(
        bool(_NEG_RE.search(normalized))
        or bool(re.search(r"\b(?:no|false|incorrect)\b", normalized))
    )
    return yes, no


def entity_set(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)*\b",
            text,
        )
    )


class CorrectnessEvaluator:
    def __init__(
        self,
        mode: str,
        threshold: float,
        engine: Optional["HiddenStateEngine"] = None,
    ):
        self.mode = mode
        self.threshold = threshold
        self.engine = engine

    def evaluate(
        self,
        prediction: str,
        references: Sequence[str],
        question: str,
    ) -> Optional[bool]:
        if is_refusal(prediction):
            return None
        if self.mode == "llm_judge":
            if self.engine is None:
                raise RuntimeError("llm_judge requires an engine")
            return self.engine.judge_answer(question, references, prediction)
        if self.mode == "numeric":
            pred_number = last_number(prediction)
            ref_numbers = {last_number(reference) for reference in references}
            ref_numbers.discard(None)
            return pred_number is not None and pred_number in ref_numbers
        exact = any(
            exact_or_contained(prediction, reference)
            for reference in references
        )
        if self.mode == "exact":
            return exact
        f1 = max_ref_f1(prediction, references)
        if self.mode == "token_f1":
            return f1 >= self.threshold
        return exact or f1 >= self.threshold


# -------------------------------------------------------------------------
# Span segmentation and interventions
# -------------------------------------------------------------------------

def segment_atomic(
    source: str,
    min_clause_words: int,
    min_span_words: int,
) -> list[Span]:
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT.split(source):
        if not sentence:
            continue
        start = source.find(sentence, cursor)
        if start < 0:
            continue
        end = start + len(sentence)
        cursor = end
        clean = sentence.strip()
        if not clean:
            continue
        clean_start = source.find(clean, start, end + 1)
        clean_end = clean_start + len(clean)
        if len(_WORD_RE.findall(clean)) > min_clause_words:
            sub_cursor = 0
            for clause in _CLAUSE_SPLIT.split(clean):
                if not clause.strip():
                    continue
                local_start = clean.find(clause, sub_cursor)
                if local_start < 0:
                    continue
                local_end = local_start + len(clause)
                sub_cursor = local_end
                stripped = clause.strip()
                offset = clause.find(stripped)
                pieces.append(
                    (
                        clean_start + local_start + offset,
                        clean_start + local_start + offset + len(stripped),
                    )
                )
        else:
            pieces.append((clean_start, clean_end))

    if not pieces and source.strip():
        start = source.find(source.strip())
        pieces = [(start, start + len(source.strip()))]

    merged: list[tuple[int, int]] = []
    for start, end in pieces:
        if merged and len(_WORD_RE.findall(source[start:end])) < min_span_words:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    spans: list[Span] = []
    for index, (start, end) in enumerate(merged):
        text = source[start:end].strip()
        if text:
            real_start = source.find(text, start, end + 1)
            spans.append(
                Span(
                    index=index,
                    text=text,
                    start=real_start,
                    end=real_start + len(text),
                )
            )
    return spans


def select_topk_hidden_spans(
    spans: list[Span],
    score_rows: dict[int, dict[str, float]],
    maximum: int,
) -> tuple[list[Span], dict[int, int]]:
    if not spans:
        return [], {}
    ranked = sorted(
        spans,
        key=lambda span: (
            float(score_rows.get(span.index, {}).get("score", float("-inf"))),
            float(score_rows.get(span.index, {}).get("cosine", float("-inf"))),
            -span.start,
        ),
        reverse=True,
    )
    rank_by_index = {
        span.index: rank + 1 for rank, span in enumerate(ranked)
    }
    if maximum > 0:
        ranked = ranked[:maximum]
    return sorted(ranked, key=lambda span: span.start), rank_by_index


def intervene(
    source: str,
    span: Span,
    operator: str,
    mask_text: str,
    neutral_text: str,
) -> str:
    before = source[: span.start]
    after = source[span.end :]
    if operator == "delete":
        modified = re.sub(
            r"[ \t]+",
            " ",
            before.rstrip() + " " + after.lstrip(),
        )
        modified = re.sub(r"\s+([,.;:!?])", r"\1", modified)
        return modified.strip()
    replacement = neutral_text if operator == "neutralize" else mask_text
    return before + replacement + after


def structural_features(span: Span, source: str) -> np.ndarray:
    source_chars = max(len(source), 1)
    source_words = max(len(_WORD_RE.findall(source)), 1)
    words = _WORD_RE.findall(span.text)
    values = [
        span.start / source_chars,
        span.end / source_chars,
        (span.end - span.start) / source_chars,
        len(words) / source_words,
        len(words),
        len(span.text),
        span.index,
        bool(_NEG_RE.search(span.text)),
        bool(_NUMBER_RE.search(span.text)),
        bool(re.search(r"\b[A-Z][\w'-]*\b", span.text)),
        "?" in span.text,
        ":" in span.text,
        span.text.lower().startswith(("question:", "context:", "evidence:")),
    ]
    return np.asarray(values, dtype=np.float32)


# -------------------------------------------------------------------------
# Model engine and hidden-state extraction
# -------------------------------------------------------------------------

class HiddenStateEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_from_name(args.dtype),
            "trust_remote_code": args.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False

        self.transformer_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0)
        )
        self.hidden_size = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        if self.transformer_layers <= 0 or self.hidden_size <= 0:
            raise RuntimeError(
                "Cannot infer num_hidden_layers or hidden_size from model config"
            )
        # Transformers returns embedding output plus every block output.
        self.hidden_state_count = self.transformer_layers + 1
        self.hidden_relation_dim = 6 * self.hidden_state_count

    def prompt_text(
        self,
        source: str,
        system: Optional[str] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system or self.args.system_prompt,
            },
            {
                "role": "user",
                "content": source + "\n\n" + self.args.answer_instruction,
            },
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"System: {messages[0]['content']}\n"
            f"User: {messages[1]['content']}\nAssistant:"
        )

    def encode_prompt(
        self,
        source: str,
        system: Optional[str] = None,
    ) -> PromptEncoding:
        # Render unique boundary markers through the chat template so source
        # offsets do not depend on a fragile search for an unmarked string.
        # Some templates trim message-edge whitespace; in that case locate the
        # trimmed source in the real render and compensate in source coordinates.
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        marker_start = f"<|source_start_{digest}|>"
        marker_end = f"<|source_end_{digest}|>"
        counter = 0
        while marker_start in source or marker_end in source:
            counter += 1
            marker_start = f"<|source_start_{digest}_{counter}|>"
            marker_end = f"<|source_end_{digest}_{counter}|>"

        marked = self.prompt_text(marker_start + source + marker_end, system)
        marker_start_position = marked.find(marker_start)
        marker_end_position = marked.find(
            marker_end,
            marker_start_position + len(marker_start),
        )
        if marker_start_position < 0 or marker_end_position < 0:
            raise RuntimeError(
                "Chat template did not preserve source boundary markers"
            )
        rendered_source = marked[
            marker_start_position + len(marker_start):marker_end_position
        ]
        if rendered_source != source:
            raise RuntimeError(
                "Chat template modified source content between boundary markers"
            )

        text = (
            marked[:marker_start_position]
            + source
            + marked[marker_end_position + len(marker_end):]
        )
        expected = self.prompt_text(source, system)
        if text == expected:
            source_start = marker_start_position
        else:
            trimmed_source = source.strip()
            if not trimmed_source:
                raise RuntimeError(
                    "Source became empty after chat-template trimming"
                )
            rendered_position = expected.find(trimmed_source)
            if (
                rendered_position < 0
                or expected.find(trimmed_source, rendered_position + 1) >= 0
            ):
                raise RuntimeError(
                    "Could not uniquely locate trimmed source in chat template output"
                )
            leading_trim = len(source) - len(source.lstrip())
            text = expected
            source_start = rendered_position - leading_trim
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = torch.tensor(
            encoding["input_ids"],
            dtype=torch.long,
        )
        if len(prompt_ids) > self.args.max_input_tokens:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeding "
                f"--max-input-tokens={self.args.max_input_tokens}"
            )
        return PromptEncoding(
            prompt_text=text,
            prompt_ids=prompt_ids,
            offsets=[
                (int(start), int(end))
                for start, end in encoding["offset_mapping"]
            ],
            source_start=source_start,
        )

    def span_tokens(
        self,
        prompt: PromptEncoding,
        span: Span,
    ) -> list[int]:
        absolute_start = prompt.source_start + span.start
        absolute_end = prompt.source_start + span.end
        return [
            index
            for index, (start, end) in enumerate(prompt.offsets)
            if end > start and end > absolute_start and start < absolute_end
        ]

    def generate(
        self,
        source: str,
        max_new_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> tuple[str, torch.Tensor]:
        prompt = self.encode_prompt(source, system)
        input_ids = prompt.prompt_ids.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.args.temperature > 0:
            generation_kwargs.update(
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        answer_ids = output[0, input_ids.shape[1] :].detach().cpu()
        answer_list = answer_ids.tolist()
        if self.tokenizer.eos_token_id in answer_list:
            answer_ids = answer_ids[
                : answer_list.index(self.tokenizer.eos_token_id)
            ]
        answer_text = self.tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip()
        if answer_ids.numel() == 0:
            fallback_ids = self.tokenizer(
                "I do not know.",
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = torch.tensor(fallback_ids, dtype=torch.long)
            answer_text = "I do not know."
        return answer_text, answer_ids

    def judge_answer(
        self,
        question: str,
        references: Sequence[str],
        prediction: str,
    ) -> Optional[bool]:
        source = (
            "Return exactly CORRECT, INCORRECT, or REFUSAL.\n"
            f"Question: {question}\n"
            f"References: {json.dumps(list(references), ensure_ascii=False)}\n"
            f"Candidate: {prediction}"
        )
        old_temperature = self.args.temperature
        try:
            self.args.temperature = 0.0
            verdict, _ = self.generate(
                source,
                max_new_tokens=5,
                system="You are a strict semantic answer evaluator.",
            )
        finally:
            self.args.temperature = old_temperature
        uppercase = verdict.upper()
        if "REFUSAL" in uppercase:
            return None
        if "INCORRECT" in uppercase:
            return False
        return "CORRECT" in uppercase

    @staticmethod
    def _zscore_within_item(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        standard_deviation = float(values.std())
        if not np.isfinite(standard_deviation) or standard_deviation < 1e-12:
            return np.zeros_like(values)
        return (values - float(values.mean())) / standard_deviation

    def hidden_selection_layer_indices(self) -> list[int]:
        """Return indices in output.hidden_states, including embedding index 0."""
        mode = self.args.hidden_selection_layers
        count = self.hidden_state_count
        if mode == "all":
            return list(range(count))
        if mode == "last_half":
            return list(range(max(0, count // 2), count))
        if mode == "last_n":
            number = max(
                1,
                min(int(self.args.hidden_selection_last_n), count),
            )
            return list(range(count - number, count))
        # Default: last quarter of returned hidden states.
        number = max(1, count // 4)
        return list(range(count - number, count))

    def _answer_representation(
        self,
        hidden_state: torch.Tensor,
        answer_indices: Sequence[int],
    ) -> torch.Tensor:
        if not answer_indices:
            raise ValueError("No answer token indices")
        mode = self.args.answer_hidden_pooling
        if mode == "mean":
            return hidden_state[0, list(answer_indices), :].mean(dim=0)
        if mode == "last_k_mean":
            number = max(
                1,
                min(self.args.answer_hidden_last_k, len(answer_indices)),
            )
            return hidden_state[0, list(answer_indices)[-number:], :].mean(dim=0)
        return hidden_state[0, int(answer_indices[-1]), :]

    def analyze_base_hidden(
        self,
        source: str,
        answer_ids_cpu: torch.Tensor,
        spans: Sequence[Span],
    ) -> dict[str, Any]:
        """Extract static hidden states from the original prompt and answer.

        Returns:
          answer_hidden_states: [embedding + every block, hidden_size]
          span_hidden_features:  per-span fixed-size relation vector
          span_selection:        hidden-similarity ranking statistics
        """
        prompt = self.encode_prompt(source)
        prompt_ids = prompt.prompt_ids.to(self.device)
        answer_ids = answer_ids_cpu.to(self.device)
        full_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        prompt_length = int(prompt_ids.numel())
        answer_indices = list(
            range(prompt_length, prompt_length + int(answer_ids.numel()))
        )
        token_map = {
            span.index: self.span_tokens(prompt, span)
            for span in spans
        }

        with torch.inference_mode():
            output = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )

        if output.hidden_states is None:
            raise RuntimeError("The model did not return hidden states")
        if len(output.hidden_states) != self.hidden_state_count:
            raise RuntimeError(
                f"Expected {self.hidden_state_count} hidden-state tensors, "
                f"received {len(output.hidden_states)}"
            )

        logits = output.logits[
            :,
            prompt_length - 1 : prompt_length + len(answer_ids) - 1,
            :,
        ]
        targets = answer_ids.unsqueeze(0)
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        selected = log_probabilities.gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        sequence_logprob = selected.mean()
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(-1).mean()

        cache_dtype = (
            np.float16
            if self.args.hidden_cache_dtype == "float16"
            else np.float32
        )
        answer_hidden_rows: list[np.ndarray] = []
        span_feature_parts: dict[int, list[float]] = {
            span.index: [] for span in spans
        }
        cosine_by_span: dict[int, list[float]] = {
            span.index: [] for span in spans
        }
        distance_by_span: dict[int, list[float]] = {
            span.index: [] for span in spans
        }
        hidden_dimension_scale = math.sqrt(max(self.hidden_size, 1))

        for hidden_state in output.hidden_states:
            state = hidden_state.detach().float()
            answer_vector = self._answer_representation(
                state,
                answer_indices,
            )
            answer_hidden_rows.append(
                answer_vector.cpu().numpy().astype(cache_dtype, copy=False)
            )
            answer_norm = float(torch.linalg.vector_norm(answer_vector).cpu())

            for span in spans:
                token_indices = token_map[span.index]
                if not token_indices:
                    span_feature_parts[span.index].extend([0.0] * 6)
                    cosine_by_span[span.index].append(0.0)
                    distance_by_span[span.index].append(0.0)
                    continue
                span_token_states = state[0, token_indices, :]
                span_vector = span_token_states.mean(dim=0)
                span_norm_tensor = torch.linalg.vector_norm(span_vector)
                span_norm = float(span_norm_tensor.cpu())
                cosine = float(
                    F.cosine_similarity(
                        span_vector.unsqueeze(0),
                        answer_vector.unsqueeze(0),
                        dim=-1,
                        eps=1e-8,
                    )[0].cpu()
                )
                distance = float(
                    (
                        torch.linalg.vector_norm(span_vector - answer_vector)
                        / hidden_dimension_scale
                    ).cpu()
                )
                scaled_dot = float(
                    (torch.dot(span_vector, answer_vector) / self.hidden_size).cpu()
                )
                # Mean within-span dispersion around the span mean. This is a
                # static hidden-state property and remains span-specific.
                dispersion = float(
                    torch.linalg.vector_norm(
                        span_token_states - span_vector.unsqueeze(0),
                        dim=-1,
                    ).mean().cpu()
                    / hidden_dimension_scale
                )
                span_feature_parts[span.index].extend(
                    [
                        cosine,
                        distance,
                        span_norm / hidden_dimension_scale,
                        answer_norm / hidden_dimension_scale,
                        scaled_dot,
                        dispersion,
                    ]
                )
                cosine_by_span[span.index].append(cosine)
                distance_by_span[span.index].append(distance)

        selected_layers = self.hidden_selection_layer_indices()
        cosine_values = np.asarray(
            [
                np.mean(
                    np.asarray(cosine_by_span[span.index])[selected_layers]
                )
                for span in spans
            ],
            dtype=np.float64,
        )
        negative_distance_values = np.asarray(
            [
                -np.mean(
                    np.asarray(distance_by_span[span.index])[selected_layers]
                )
                for span in spans
            ],
            dtype=np.float64,
        )
        if self.args.hidden_selection_score == "cosine":
            selection_scores = cosine_values
        elif self.args.hidden_selection_score == "negative_distance":
            selection_scores = negative_distance_values
        else:
            weight = float(
                np.clip(self.args.hidden_selection_cosine_weight, 0.0, 1.0)
            )
            selection_scores = (
                weight * self._zscore_within_item(cosine_values)
                + (1.0 - weight)
                * self._zscore_within_item(negative_distance_values)
            )

        selection: dict[int, dict[str, float]] = {}
        span_hidden_features: dict[int, np.ndarray] = {}
        for span, cosine, negative_distance, score in zip(
            spans,
            cosine_values,
            negative_distance_values,
            selection_scores,
        ):
            selection[span.index] = {
                "score": float(score),
                "cosine": float(cosine),
                "negative_distance": float(negative_distance),
                "token_count": float(len(token_map[span.index])),
            }
            span_hidden_features[span.index] = np.nan_to_num(
                np.asarray(
                    span_feature_parts[span.index],
                    dtype=np.float32,
                )
            )

        result = {
            "sequence_logprob": float(sequence_logprob.cpu()),
            "mean_token_entropy": float(entropy.cpu()),
            "answer_hidden_states": np.stack(answer_hidden_rows, axis=0),
            "span_hidden_features": span_hidden_features,
            "span_selection": selection,
            "prompt_tokens": prompt_length,
            "answer_tokens": int(answer_ids.numel()),
            "hidden_state_count": self.hidden_state_count,
            "hidden_size": self.hidden_size,
        }

        del output, full_ids, attention_mask, logits, log_probabilities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def score_original_answer(
        self,
        source: str,
        answer_ids_cpu: torch.Tensor,
    ) -> dict[str, float]:
        """Teacher-force the original answer without extracting hidden states."""
        prompt = self.encode_prompt(source)
        prompt_ids = prompt.prompt_ids.to(self.device)
        answer_ids = answer_ids_cpu.to(self.device)
        full_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        prompt_length = int(prompt_ids.numel())
        with torch.inference_mode():
            output = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[
            :,
            prompt_length - 1 : prompt_length + len(answer_ids) - 1,
            :,
        ]
        targets = answer_ids.unsqueeze(0)
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        selected = log_probabilities.gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        sequence_logprob = selected.mean()
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(-1).mean()
        result = {
            "sequence_logprob": float(sequence_logprob.cpu()),
            "mean_token_entropy": float(entropy.cpu()),
            "prompt_tokens": prompt_length,
            "answer_tokens": int(answer_ids.numel()),
        }
        del output, full_ids, attention_mask, logits, log_probabilities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


# -------------------------------------------------------------------------
# Extraction
# -------------------------------------------------------------------------

def change_metrics(
    original_answer: str,
    regenerated_answer: str,
    base: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, float]:
    similarity = token_f1(original_answer, regenerated_answer)
    original_entities = entity_set(original_answer)
    regenerated_entities = entity_set(regenerated_answer)
    return {
        "support_delta": (
            float(base["sequence_logprob"])
            - float(intervened["sequence_logprob"])
        ),
        "semantic_similarity": similarity,
        "answer_changed": float(similarity < 0.80),
        "polarity_changed": float(
            polarity_signature(original_answer)
            != polarity_signature(regenerated_answer)
        ),
        "entity_changed": float(
            original_entities != regenerated_entities
        ) if original_entities or regenerated_entities else 0.0,
        "number_changed": float(
            set(_NUMBER_RE.findall(original_answer.replace(",", "")))
            != set(_NUMBER_RE.findall(regenerated_answer.replace(",", "")))
        ) if _NUMBER_RE.search(original_answer + regenerated_answer) else 0.0,
        "intervened_original_answer_logprob": float(
            intervened["sequence_logprob"]
        ),
        "entropy_delta": (
            float(intervened["mean_token_entropy"])
            - float(base["mean_token_entropy"])
        ),
    }


def process_item(
    example: Example,
    engine: HiddenStateEngine,
    evaluator: CorrectnessEvaluator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_spans = segment_atomic(
        example.source_text,
        args.min_clause_words,
        args.min_span_words,
    )
    if not all_spans:
        raise ValueError("No spans")

    original_answer, answer_ids = engine.generate(example.source_text)
    original_correct = evaluator.evaluate(
        original_answer,
        example.references,
        example.question,
    )

    base = engine.analyze_base_hidden(
        example.source_text,
        answer_ids,
        all_spans,
    )
    selected_spans, hidden_rank = select_topk_hidden_spans(
        all_spans,
        base["span_selection"],
        args.max_intervention_spans,
    )
    if not selected_spans:
        raise ValueError("No spans survived hidden-state selection")

    span_rows: list[dict[str, Any]] = []
    for span in selected_spans:
        operator_rows: list[dict[str, Any]] = []
        for operator in OPERATORS:
            modified = intervene(
                example.source_text,
                span,
                operator,
                args.mask_text,
                args.neutral_text,
            )
            regenerated_answer, _ = engine.generate(modified)
            regenerated_correct = evaluator.evaluate(
                regenerated_answer,
                example.references,
                example.question,
            )
            intervened_score = engine.score_original_answer(
                modified,
                answer_ids,
            )
            metrics = change_metrics(
                original_answer,
                regenerated_answer,
                base,
                intervened_score,
            )
            metrics.update(
                operator=operator,
                regenerated_answer=regenerated_answer,
                regenerated_correct=regenerated_correct,
            )
            operator_rows.append(metrics)

        selection_row = base["span_selection"][span.index]
        span_rows.append(
            {
                "span_uid": f"{example.item_id}::span::{span.index}",
                "span_index": span.index,
                "span_text": span.text,
                "span_start": span.start,
                "span_end": span.end,
                "hidden_selection_rank": int(hidden_rank[span.index]),
                "hidden_selection_score": float(selection_row["score"]),
                "hidden_selection_cosine": float(selection_row["cosine"]),
                "hidden_selection_negative_distance": float(
                    selection_row["negative_distance"]
                ),
                "hidden_selection_token_count": int(
                    selection_row["token_count"]
                ),
                "family_features": {
                    "structural": structural_features(
                        span,
                        example.source_text,
                    ),
                    "hidden_relation": base["span_hidden_features"][span.index],
                },
                "operators": operator_rows,
            }
        )

    return {
        "item_id": example.item_id,
        "raw_index": example.raw_index,
        "source_text": example.source_text,
        "question": example.question,
        "references": example.references,
        "generated_answer": original_answer,
        "original_correct": original_correct,
        "hallucination_label": (
            None if original_correct is None else int(not original_correct)
        ),
        "base_sequence_logprob": base["sequence_logprob"],
        "base_mean_token_entropy": base["mean_token_entropy"],
        "prompt_tokens": base["prompt_tokens"],
        "answer_tokens": base["answer_tokens"],
        "hidden_state_count": base["hidden_state_count"],
        "hidden_size": base["hidden_size"],
        "answer_hidden_states": base["answer_hidden_states"],
        "n_candidate_spans": len(all_spans),
        "n_selected_spans": len(selected_spans),
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
        "hidden_selection_layers": args.hidden_selection_layers,
        "hidden_selection_last_n": args.hidden_selection_last_n,
        "hidden_selection_score": args.hidden_selection_score,
        "hidden_selection_cosine_weight": args.hidden_selection_cosine_weight,
        "answer_hidden_pooling": args.answer_hidden_pooling,
        "answer_hidden_last_k": args.answer_hidden_last_k,
        "hidden_cache_dtype": args.hidden_cache_dtype,
        "mask_text": args.mask_text,
        "neutral_text": args.neutral_text,
    }
    return stable_hash(json.dumps(fields, sort_keys=True, ensure_ascii=False))


def extract_all(
    examples: Sequence[Example],
    engine: HiddenStateEngine,
    evaluator: CorrectnessEvaluator,
    args: argparse.Namespace,
    output_directory: Path,
) -> list[dict[str, Any]]:
    cache_directory = output_directory / "item_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_signature = extraction_cache_signature(args)
    records: list[dict[str, Any]] = []

    for position, example in enumerate(tqdm(examples, desc="Extracting")):
        cache_path = cache_directory / (
            f"{position:06d}_{stable_hash(example.item_id)}_"
            f"{cache_signature}.pt"
        )
        if cache_path.exists() and not args.overwrite_cache:
            try:
                cached = torch_load(cache_path)
                if cached.get("error"):
                    warnings.warn(
                        f"Retrying cached error for {example.item_id}: "
                        f"{cached['error']}"
                    )
                else:
                    records.append(cached)
                    continue
            except Exception as error:
                warnings.warn(f"Bad cache {cache_path}: {error}")

        try:
            record = process_item(example, engine, evaluator, args)
        except Exception as error:
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
                "error": f"{type(error).__name__}: {error}",
            }
        atomic_torch_save(record, cache_path)
        records.append(record)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


# -------------------------------------------------------------------------
# Behavior features and pseudo-role construction
# -------------------------------------------------------------------------

def estimate_support_scale(
    records: Sequence[dict[str, Any]],
    training_ids: set[str],
    minimum: float,
) -> float:
    values: list[float] = []
    for item in records:
        if item["item_id"] not in training_ids:
            continue
        for span in item["spans"]:
            values.extend(
                abs(float(row["support_delta"]))
                for row in span["operators"]
            )
    if not values:
        return minimum
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(minimum, median + 1.4826 * mad, 1e-4)


def behavior_vector(span: dict[str, Any], scale: float) -> np.ndarray:
    by_operator = {
        row["operator"]: row for row in span["operators"]
    }
    values: list[float] = []
    support_deltas: list[float] = []
    similarities: list[float] = []
    changes: list[float] = []
    for operator in OPERATORS:
        row = by_operator[operator]
        delta = float(row["support_delta"])
        similarity = float(row["semantic_similarity"])
        changed = float(row["answer_changed"])
        values.extend(
            [
                delta,
                math.tanh(delta / scale),
                similarity,
                changed,
                float(row["polarity_changed"]),
                float(row["entity_changed"]),
                float(row["number_changed"]),
                float(row["entropy_delta"]),
            ]
        )
        support_deltas.append(delta)
        similarities.append(similarity)
        changes.append(changed)

    delta_array = np.asarray(support_deltas, dtype=np.float64)
    nonzero_signs = np.sign(delta_array[np.abs(delta_array) > 1e-8])
    sign_agreement = (
        abs(float(nonzero_signs.mean())) if len(nonzero_signs) else 0.0
    )
    values.extend(
        [
            float(np.median(delta_array)),
            float(np.std(delta_array)),
            float(np.max(np.abs(delta_array))),
            sign_agreement,
            float(np.mean(changes)),
            float(np.mean(similarities)),
        ]
    )
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def usage_score(span: dict[str, Any], scale: float) -> float:
    deltas = np.asarray(
        [float(row["support_delta"]) for row in span["operators"]],
        dtype=np.float64,
    )
    change = np.mean(
        [float(row["answer_changed"]) for row in span["operators"]]
    )
    contradiction = np.mean(
        [
            max(
                float(row["polarity_changed"]),
                float(row["entity_changed"]),
                float(row["number_changed"]),
            )
            for row in span["operators"]
        ]
    )
    value = (
        np.median(np.abs(np.tanh(deltas / scale)))
        + 0.35 * change
        + 0.20 * contradiction
    )
    return float(np.clip(value, 0.0, 1.0))


def pseudo_role(
    item: dict[str, Any],
    span: dict[str, Any],
    scale: float,
) -> tuple[Optional[str], float, str]:
    original_correct = item["original_correct"]
    if original_correct is None:
        return None, 0.0, "original_refusal"

    rows = span["operators"]
    intervention_correctness = [
        row["regenerated_correct"]
        for row in rows
        if row["regenerated_correct"] is not None
    ]
    deltas = np.asarray(
        [float(row["support_delta"]) for row in rows],
        dtype=np.float64,
    )
    similarities = np.asarray(
        [float(row["semantic_similarity"]) for row in rows],
        dtype=np.float64,
    )
    change_rate = float(
        np.mean([float(row["answer_changed"]) for row in rows])
    )
    median_delta = float(np.median(deltas))
    median_absolute_delta = float(np.median(np.abs(deltas)))
    median_similarity = float(np.median(similarities))

    if not original_correct and any(
        value is True for value in intervention_correctness
    ):
        reliability = min(
            1.0,
            0.85
            + 0.15
            * sum(value is True for value in intervention_correctness)
            / max(len(intervention_correctness), 1),
        )
        return "shortcut", reliability, "wrong_to_correct"

    if original_correct and any(
        value is False for value in intervention_correctness
    ):
        reliability = min(
            1.0,
            0.85
            + 0.15
            * sum(value is False for value in intervention_correctness)
            / max(len(intervention_correctness), 1),
        )
        return "constraint", reliability, "correct_to_wrong"

    if (
        median_absolute_delta <= 0.25 * scale
        and median_similarity >= 0.90
        and change_rate <= 1 / 3
    ):
        return "irrelevant", 0.75, "stable_low_effect"

    if median_delta >= 0.50 * scale and change_rate >= 1 / 3:
        if original_correct:
            return "constraint", 0.70, "supports_correct"
        return "shortcut", 0.70, "supports_wrong"

    if median_delta <= -0.50 * scale and change_rate >= 1 / 3:
        if original_correct:
            return "shortcut", 0.55, "suppresses_correct"
        return "constraint", 0.55, "suppresses_wrong"

    return None, 0.0, "ambiguous"


def attach_derived_features(
    records: Sequence[dict[str, Any]],
    support_scale: float,
    training_ids: set[str],
) -> dict[str, int]:
    counts = {name: 0 for name in ROLE_NAMES}
    counts["ambiguous"] = 0
    for item in records:
        for span in item["spans"]:
            span["family_features"]["behavior"] = behavior_vector(
                span,
                support_scale,
            )
            span["usage"] = usage_score(span, support_scale)
            if item["item_id"] in training_ids:
                role, reliability, reason = pseudo_role(
                    item,
                    span,
                    support_scale,
                )
            else:
                role, reliability, reason = None, 0.0, "test_unlabeled"
            span["pseudo_role"] = role
            span["role_reliability"] = reliability
            span["pseudo_role_reason"] = reason
            counts[role if role is not None else "ambiguous"] += 1
    return counts


# -------------------------------------------------------------------------
# Span-role models
# -------------------------------------------------------------------------

def span_vector(
    span: dict[str, Any],
    feature_set: str,
) -> np.ndarray:
    parts = [
        np.asarray(
            span["family_features"][family],
            dtype=np.float32,
        ).ravel()
        for family in ROLE_FEATURE_SETS[feature_set]
    ]
    return np.nan_to_num(
        np.concatenate(parts),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )


def spans_for(
    records: dict[str, dict[str, Any]],
    item_ids: Iterable[str],
) -> list[dict[str, Any]]:
    return [
        span
        for item_id in item_ids
        for span in records[item_id]["spans"]
    ]


def labeled_span_arrays(
    records: dict[str, dict[str, Any]],
    item_ids: Iterable[str],
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    spans = [
        span
        for span in spans_for(records, item_ids)
        if span["pseudo_role"] is not None
    ]
    if not spans:
        raise RuntimeError("No pseudo-labeled spans")
    features = np.stack(
        [span_vector(span, feature_set) for span in spans]
    )
    labels = np.asarray(
        [ROLE_TO_ID[span["pseudo_role"]] for span in spans],
        dtype=np.int64,
    )
    weights = np.asarray(
        [float(span["role_reliability"]) for span in spans],
        dtype=np.float64,
    )
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Need at least two pseudo-role classes")
    return features, labels, weights, spans


def fit_role_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    pca_dimension: int,
    seed: int,
    logistic_c: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    maximum_components = min(
        pca_dimension,
        scaled.shape[1],
        max(1, scaled.shape[0] - 1),
    )
    pca = (
        PCA(
            n_components=maximum_components,
            svd_solver="randomized",
            random_state=seed,
        )
        if pca_dimension > 0
        and scaled.shape[1] > maximum_components
        and maximum_components >= 2
        else None
    )
    transformed = pca.fit_transform(scaled) if pca is not None else scaled
    classifier = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        C=logistic_c,
        random_state=seed,
    )
    classifier.fit(transformed, labels, sample_weight=weights)
    return {
        "scaler": scaler,
        "pca": pca,
        "classifier": classifier,
    }


def predict_role_model(
    model: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    if model["pca"] is not None:
        transformed = model["pca"].transform(transformed)
    partial = model["classifier"].predict_proba(transformed)
    probabilities = np.zeros((len(features), len(ROLE_NAMES)), dtype=np.float64)
    for column, class_id in enumerate(model["classifier"].classes_):
        probabilities[:, int(class_id)] = partial[:, column]
    row_sum = probabilities.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 0
    probabilities[empty] = 1.0 / len(ROLE_NAMES)
    probabilities[~empty] /= row_sum[~empty]
    return probabilities


def role_metrics(
    labels: Sequence[int],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    output: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1, 2])
        ),
        "class_counts": {
            ROLE_NAMES[index]: int((y_true == index).sum())
            for index in range(len(ROLE_NAMES))
        },
        "per_role": {},
    }
    aurocs: list[float] = []
    auprcs: list[float] = []
    for index, name in enumerate(ROLE_NAMES):
        binary = (y_true == index).astype(int)
        if len(np.unique(binary)) < 2:
            auroc = None
            auprc = None
        else:
            auroc = float(roc_auc_score(binary, probabilities[:, index]))
            auprc = float(
                average_precision_score(binary, probabilities[:, index])
            )
            aurocs.append(auroc)
            auprcs.append(auprc)
        output["per_role"][name] = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": float(
                precision_score(
                    binary,
                    (predictions == index).astype(int),
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    binary,
                    (predictions == index).astype(int),
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    binary,
                    (predictions == index).astype(int),
                    zero_division=0,
                )
            ),
        }
    output["macro_ovr_auroc"] = (
        float(np.mean(aurocs)) if aurocs else None
    )
    output["macro_ovr_auprc"] = (
        float(np.mean(auprcs)) if auprcs else None
    )
    return output


def resolve_folds(labels: np.ndarray, requested: int) -> int:
    counts = np.bincount(labels.astype(int))
    nonzero = counts[counts > 0]
    if len(nonzero) < 2 or int(nonzero.min()) < 2:
        raise RuntimeError(
            "Need at least two examples in every item-level class for OOF CV"
        )
    return max(2, min(requested, int(nonzero.min())))


def oof_role_predictions(
    records: dict[str, dict[str, Any]],
    training_ids: list[str],
    item_labels: np.ndarray,
    feature_set: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folds = resolve_folds(item_labels, args.cv_folds)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=args.seed,
    )
    item_id_array = np.asarray(training_ids)
    probabilities_by_span: dict[str, np.ndarray] = {}
    true_roles: list[int] = []
    predicted_roles: list[np.ndarray] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(item_id_array, item_labels)
    ):
        fold_train_ids = item_id_array[train_indices].tolist()
        fold_validation_ids = item_id_array[validation_indices].tolist()
        features, labels, weights, _ = labeled_span_arrays(
            records,
            fold_train_ids,
            feature_set,
        )
        model = fit_role_model(
            features,
            labels,
            weights,
            args.role_pca_dim,
            args.seed + fold,
            args.role_logistic_c,
        )
        validation_spans = spans_for(records, fold_validation_ids)
        validation_features = np.stack(
            [span_vector(span, feature_set) for span in validation_spans]
        )
        validation_probabilities = predict_role_model(
            model,
            validation_features,
        )
        for span, probability in zip(
            validation_spans,
            validation_probabilities,
        ):
            probabilities_by_span[span["span_uid"]] = probability
            if span["pseudo_role"] is not None:
                true_roles.append(ROLE_TO_ID[span["pseudo_role"]])
                predicted_roles.append(probability)

    if not predicted_roles:
        raise RuntimeError("No pseudo-labeled validation spans in role OOF")
    return (
        probabilities_by_span,
        role_metrics(true_roles, np.stack(predicted_roles)),
    )


# -------------------------------------------------------------------------
# Item-level hidden probe and detector
# -------------------------------------------------------------------------

def item_behavior_summary(
    item: dict[str, Any],
    support_scale: float,
) -> tuple[np.ndarray, list[str]]:
    deltas = np.asarray(
        [
            float(operator["support_delta"])
            for span in item["spans"]
            for operator in span["operators"]
        ],
        dtype=np.float64,
    )
    changed = np.asarray(
        [
            float(operator["answer_changed"])
            for span in item["spans"]
            for operator in span["operators"]
        ],
        dtype=np.float64,
    )
    normalized = (
        np.tanh(deltas / max(support_scale, 1e-8))
        if deltas.size
        else np.zeros(0, dtype=np.float64)
    )
    values = [
        float(item["base_sequence_logprob"]),
        float(item["base_mean_token_entropy"]),
        float(np.mean(deltas)) if deltas.size else 0.0,
        float(np.median(deltas)) if deltas.size else 0.0,
        float(np.std(deltas)) if deltas.size else 0.0,
        float(np.max(np.abs(deltas))) if deltas.size else 0.0,
        float(np.mean(normalized)) if normalized.size else 0.0,
        float(np.mean(changed)) if changed.size else 0.0,
    ]
    names = [
        "base_sequence_logprob",
        "base_mean_token_entropy",
        "support_delta_mean",
        "support_delta_median",
        "support_delta_std",
        "support_delta_max_abs",
        "normalized_support_mean",
        "answer_change_rate",
    ]
    return np.asarray(values, dtype=np.float32), names


def aggregate_role_evidence(
    item: dict[str, Any],
    probabilities_by_span: dict[str, np.ndarray],
    top_k: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    shortcut_contributions: list[float] = []
    constraint_contributions: list[float] = []
    usage_values: list[float] = []
    shortcut_probabilities: list[float] = []
    span_details: list[dict[str, Any]] = []

    for span in item["spans"]:
        probability = probabilities_by_span[span["span_uid"]]
        usage = float(span["usage"])
        shortcut_probability = float(probability[ROLE_TO_ID["shortcut"]])
        constraint_probability = float(probability[ROLE_TO_ID["constraint"]])
        shortcut_contribution = usage * shortcut_probability
        constraint_contribution = usage * constraint_probability
        usage_values.append(usage)
        shortcut_probabilities.append(shortcut_probability)
        shortcut_contributions.append(shortcut_contribution)
        constraint_contributions.append(constraint_contribution)
        span_details.append(
            {
                "span_uid": span["span_uid"],
                "span_index": span["span_index"],
                "span_text": span["span_text"],
                "usage": usage,
                "hidden_selection_rank": span["hidden_selection_rank"],
                "hidden_selection_score": span["hidden_selection_score"],
                "role_probabilities": {
                    ROLE_NAMES[index]: float(probability[index])
                    for index in range(len(ROLE_NAMES))
                },
                "shortcut_contribution": shortcut_contribution,
                "constraint_contribution": constraint_contribution,
                "operators": span["operators"],
            }
        )

    if not usage_values:
        values = np.zeros(8, dtype=np.float32)
    else:
        denominator = float(np.sum(usage_values)) + 1e-8
        sorted_shortcut = sorted(shortcut_contributions, reverse=True)
        selected_k = min(max(1, top_k), len(sorted_shortcut))
        values = np.asarray(
            [
                np.sum(shortcut_contributions) / denominator,
                np.sum(constraint_contributions) / denominator,
                max(shortcut_contributions),
                np.mean(sorted_shortcut[:selected_k]),
                max(shortcut_probabilities),
                np.mean(shortcut_probabilities),
                float(np.sum(np.asarray(shortcut_probabilities) >= 0.5)),
                (
                    np.sum(shortcut_contributions)
                    - np.sum(constraint_contributions)
                ) / denominator,
            ],
            dtype=np.float32,
        )

    names = [
        "shortcut_evidence_mean",
        "constraint_evidence_mean",
        "max_shortcut_contribution",
        "topk_shortcut_contribution",
        "max_shortcut_probability",
        "mean_shortcut_probability",
        "n_shortcut_probability_ge_0.5",
        "shortcut_minus_constraint_evidence",
    ]
    details = {
        name: float(value) for name, value in zip(names, values)
    }
    details["spans"] = span_details
    return values, names, details


def answer_hidden_matrix(
    records: dict[str, dict[str, Any]],
    item_ids: Sequence[str],
    layer_index: int,
) -> np.ndarray:
    rows = []
    for item_id in item_ids:
        hidden_states = np.asarray(
            records[item_id]["answer_hidden_states"]
        )
        if layer_index < 0 or layer_index >= hidden_states.shape[0]:
            raise IndexError(
                f"Layer {layer_index} unavailable for item {item_id}; "
                f"shape={hidden_states.shape}"
            )
        rows.append(hidden_states[layer_index].astype(np.float32))
    return np.stack(rows)


def fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    logistic_c: float,
    pca_dimension: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    maximum_components = min(
        pca_dimension,
        scaled.shape[1],
        max(1, scaled.shape[0] - 1),
    )
    pca = (
        PCA(
            n_components=maximum_components,
            svd_solver="randomized",
            random_state=seed,
        )
        if pca_dimension > 0
        and scaled.shape[1] > maximum_components
        and maximum_components >= 2
        else None
    )
    transformed = pca.fit_transform(scaled) if pca is not None else scaled
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        C=logistic_c,
        random_state=seed,
    )
    classifier.fit(transformed, labels)
    return {
        "scaler": scaler,
        "pca": pca,
        "classifier": classifier,
    }


def predict_binary_logistic(
    model: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    transformed = model["scaler"].transform(features)
    if model["pca"] is not None:
        transformed = model["pca"].transform(transformed)
    return model["classifier"].predict_proba(transformed)[:, 1]


def binary_oof_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int = 0,
    hidden_feature_count: int = 0,
) -> np.ndarray:
    folds = resolve_folds(labels, args.cv_folds)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=args.seed + seed_offset,
    )
    probabilities = np.zeros(len(labels), dtype=np.float64)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels)
    ):
        selected_columns = select_item_feature_columns(
            features[train_indices],
            labels[train_indices],
            hidden_feature_count,
            args.hidden_top_k,
        )
        model = fit_binary_logistic(
            features[train_indices][:, selected_columns],
            labels[train_indices],
            args.seed + seed_offset + fold,
            args.item_logistic_c,
            args.item_pca_dim,
        )
        probabilities[validation_indices] = predict_binary_logistic(
            model,
            features[validation_indices][:, selected_columns],
        )
    return probabilities


def select_item_feature_columns(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_feature_count: int,
    hidden_top_k: int,
) -> np.ndarray:
    """Select hidden coordinates on training data and retain every tail feature.

    Hidden features must occupy the leading columns. Selection uses the
    absolute standardized class-mean difference and is called independently
    inside every OOF fold, preventing validation/test-label leakage.
    """
    total_features = int(features.shape[1])
    hidden_count = min(max(0, int(hidden_feature_count)), total_features)
    if hidden_count == 0:
        return np.arange(total_features, dtype=np.int64)
    number = min(max(1, int(hidden_top_k)), hidden_count)
    hidden = np.asarray(features[:, :hidden_count], dtype=np.float64)
    negative = hidden[labels == 0]
    positive = hidden[labels == 1]
    if len(negative) == 0 or len(positive) == 0:
        raise RuntimeError("Hidden selection requires both item classes")
    pooled_variance = 0.5 * (negative.var(axis=0) + positive.var(axis=0))
    score = np.abs(positive.mean(axis=0) - negative.mean(axis=0)) / np.sqrt(
        pooled_variance + 1e-12
    )
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    # Stable tie-breaking prefers the lower coordinate index.
    selected_hidden = np.lexsort((np.arange(hidden_count), -score))[:number]
    tail = np.arange(hidden_count, total_features, dtype=np.int64)
    return np.concatenate([selected_hidden.astype(np.int64), tail])


def threshold_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "auroc": safe_auroc(labels, probabilities),
        "auprc": safe_auprc(labels, probabilities),
    }


def resolve_hidden_layer_candidates(
    hidden_state_count: int,
    args: argparse.Namespace,
) -> list[int]:
    start = max(0, int(args.hidden_layer_start))
    stop = (
        hidden_state_count - 1
        if args.hidden_layer_end < 0
        else min(int(args.hidden_layer_end), hidden_state_count - 1)
    )
    stride = max(1, int(args.hidden_layer_stride))
    if stop < start:
        raise ValueError(
            f"Invalid hidden layer range: start={start}, end={stop}"
        )
    candidates = list(range(start, stop + 1, stride))
    if not candidates:
        raise ValueError("No hidden-state layer candidates")
    return candidates


def scan_hidden_layers(
    records: dict[str, dict[str, Any]],
    training_ids: list[str],
    test_ids: list[str],
    training_labels: np.ndarray,
    test_labels: np.ndarray,
    hidden_state_count: int,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], dict[int, dict[str, Any]]]:
    """Paper-style layer-wise linear probes.

    Layer selection uses only train OOF AUROC/AUPRC. Test metrics are reported
    for analysis but are never used to choose the selected layer.
    """
    candidates = resolve_hidden_layer_candidates(hidden_state_count, args)
    results: dict[int, dict[str, Any]] = {}
    best_key: Optional[tuple[float, float, int]] = None
    selected_layer: Optional[int] = None

    for layer_index in tqdm(candidates, desc="Scanning hidden layers"):
        training_features = answer_hidden_matrix(
            records,
            training_ids,
            layer_index,
        )
        test_features = answer_hidden_matrix(
            records,
            test_ids,
            layer_index,
        )
        train_oof_probability = binary_oof_predictions(
            training_features,
            training_labels,
            args,
            seed_offset=10000 + layer_index * 100,
            hidden_feature_count=int(training_features.shape[1]),
        )
        threshold = threshold_f1(training_labels, train_oof_probability)
        selected_columns = select_item_feature_columns(
            training_features,
            training_labels,
            int(training_features.shape[1]),
            args.hidden_top_k,
        )
        model = fit_binary_logistic(
            training_features[:, selected_columns],
            training_labels,
            args.seed + 20000 + layer_index,
            args.item_logistic_c,
            args.item_pca_dim,
        )
        test_probability = predict_binary_logistic(
            model,
            test_features[:, selected_columns],
        )
        train_auroc = safe_auroc(training_labels, train_oof_probability)
        train_auprc = safe_auprc(training_labels, train_oof_probability)
        ranking_key = (
            -1.0 if train_auroc is None else train_auroc,
            -1.0 if train_auprc is None else train_auprc,
            layer_index,
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            selected_layer = layer_index
        results[layer_index] = {
            "layer_index": layer_index,
            "layer_kind": "embedding" if layer_index == 0 else "transformer_block",
            "transformer_block_index": None if layer_index == 0 else layer_index - 1,
            "selected_threshold_from_train_oof": threshold,
            "selected_hidden_coordinate_count": int(len(selected_columns)),
            "selected_hidden_coordinates": selected_columns.tolist(),
            "train_oof": binary_metrics(
                training_labels,
                train_oof_probability,
                threshold,
            ),
            "test": binary_metrics(
                test_labels,
                test_probability,
                threshold,
            ),
        }

    assert selected_layer is not None
    selection_summary = {
        "selection_rule": "maximize train-OOF AUROC, then AUPRC, then later layer",
        "test_metrics_used_for_layer_selection": False,
        "candidate_layers": candidates,
        "selected_layer_index": selected_layer,
        "selected_layer_kind": (
            "embedding" if selected_layer == 0 else "transformer_block"
        ),
        "selected_transformer_block_index": (
            None if selected_layer == 0 else selected_layer - 1
        ),
        "selected_train_oof_metrics": results[selected_layer]["train_oof"],
        "selected_test_metrics": results[selected_layer]["test"],
    }
    return selected_layer, selection_summary, results


def item_vector(
    item: dict[str, Any],
    selected_hidden_layer: int,
    support_scale: float,
    role_probabilities: Optional[dict[str, np.ndarray]],
    mode: str,
    item_top_k: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if mode not in ITEM_MODES:
        raise ValueError(f"Unknown item mode: {mode}")

    parts: list[np.ndarray] = []
    names: list[str] = []
    details: dict[str, Any] = {
        "selected_hidden_layer": selected_hidden_layer,
    }

    if "hidden" in mode:
        hidden = np.asarray(
            item["answer_hidden_states"][selected_hidden_layer],
            dtype=np.float32,
        ).ravel()
        parts.append(hidden)
        names.extend(
            [f"hidden_{index:05d}" for index in range(len(hidden))]
        )

    if "behavior" in mode:
        behavior, behavior_names = item_behavior_summary(
            item,
            support_scale,
        )
        parts.append(behavior)
        names.extend(behavior_names)
        details.update(
            {
                name: float(value)
                for name, value in zip(behavior_names, behavior)
            }
        )

    if "shortcut" in mode:
        if role_probabilities is None:
            raise ValueError(
                f"Mode {mode} requires role probabilities"
            )
        evidence, evidence_names, evidence_details = aggregate_role_evidence(
            item,
            role_probabilities,
            item_top_k,
        )
        parts.append(evidence)
        names.extend(evidence_names)
        details.update(evidence_details)

    if not parts:
        raise RuntimeError(f"Mode {mode} produced no features")
    return (
        np.nan_to_num(
            np.concatenate(parts).astype(np.float32),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        ),
        names,
        details,
    )


def top_logistic_coefficients(
    model: dict[str, Any],
    feature_names: Sequence[str],
    maximum: int = 20,
) -> dict[str, Any]:
    classifier = model["classifier"]
    if model["pca"] is not None:
        return {
            "intercept": float(classifier.intercept_[0]),
            "coefficients_in_pca_space": True,
            "pca_components": int(model["pca"].n_components_),
        }
    coefficients = classifier.coef_[0]
    order_positive = np.argsort(coefficients)[::-1][:maximum]
    order_negative = np.argsort(coefficients)[:maximum]
    return {
        "intercept": float(classifier.intercept_[0]),
        "coefficients_in_pca_space": False,
        "top_positive": [
            {
                "feature": str(feature_names[index]),
                "coefficient": float(coefficients[index]),
            }
            for index in order_positive
        ],
        "top_negative": [
            {
                "feature": str(feature_names[index]),
                "coefficient": float(coefficients[index]),
            }
            for index in order_negative
        ],
    }


def evaluate_item_configuration(
    configuration_name: str,
    mode: str,
    records: dict[str, dict[str, Any]],
    training_ids: list[str],
    test_ids: list[str],
    training_labels: np.ndarray,
    test_labels: np.ndarray,
    selected_hidden_layer: int,
    support_scale: float,
    oof_role_probabilities: Optional[dict[str, np.ndarray]],
    test_role_probabilities: Optional[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    training_vectors: list[np.ndarray] = []
    training_details: dict[str, Any] = {}
    feature_names: Optional[list[str]] = None
    for item_id in training_ids:
        vector, names, details = item_vector(
            records[item_id],
            selected_hidden_layer,
            support_scale,
            oof_role_probabilities,
            mode,
            args.item_top_k,
        )
        training_vectors.append(vector)
        training_details[item_id] = details
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Item feature names changed across training items")
    assert feature_names is not None
    training_features = np.stack(training_vectors)
    hidden_feature_count = sum(
        name.startswith("hidden_") for name in feature_names
    )

    train_oof_probability = binary_oof_predictions(
        training_features,
        training_labels,
        args,
        seed_offset=30000 + stable_int_hash(configuration_name),
        hidden_feature_count=hidden_feature_count,
    )
    threshold = threshold_f1(training_labels, train_oof_probability)
    selected_columns = select_item_feature_columns(
        training_features,
        training_labels,
        hidden_feature_count,
        args.hidden_top_k,
    )
    selected_feature_names = [feature_names[index] for index in selected_columns]
    item_model = fit_binary_logistic(
        training_features[:, selected_columns],
        training_labels,
        args.seed + 40000 + stable_int_hash(configuration_name),
        args.item_logistic_c,
        args.item_pca_dim,
    )
    train_full_probability = predict_binary_logistic(
        item_model,
        training_features[:, selected_columns],
    )

    test_vectors: list[np.ndarray] = []
    test_details: dict[str, Any] = {}
    for item_id in test_ids:
        vector, names, details = item_vector(
            records[item_id],
            selected_hidden_layer,
            support_scale,
            test_role_probabilities,
            mode,
            args.item_top_k,
        )
        if names != feature_names:
            raise RuntimeError("Item feature names changed between train and test")
        test_vectors.append(vector)
        test_details[item_id] = details
    test_features = np.stack(test_vectors)
    test_probability = predict_binary_logistic(
        item_model,
        test_features[:, selected_columns],
    )

    return {
        "configuration": configuration_name,
        "item_mode": mode,
        "selected_hidden_layer": selected_hidden_layer,
        "item_feature_count": int(len(selected_columns)),
        "raw_item_feature_count": int(training_features.shape[1]),
        "selected_hidden_coordinate_count": int(
            min(hidden_feature_count, args.hidden_top_k)
        ),
        "selected_hidden_coordinates": selected_columns[
            :min(hidden_feature_count, args.hidden_top_k)
        ].tolist(),
        "selected_threshold_from_train_oof": threshold,
        "item_logistic": top_logistic_coefficients(
            item_model,
            selected_feature_names,
        ),
        "item_metrics": {
            "train_oof": binary_metrics(
                training_labels,
                train_oof_probability,
                threshold,
            ),
            "train_full": binary_metrics(
                training_labels,
                train_full_probability,
                threshold,
            ),
            "test": binary_metrics(
                test_labels,
                test_probability,
                threshold,
            ),
        },
        "_artifacts": {
            "item_model": item_model,
            "threshold": threshold,
            "feature_names": selected_feature_names,
            "selected_columns": selected_columns,
            "item_mode": mode,
            "selected_hidden_layer": selected_hidden_layer,
        },
        "_predictions": {
            "test_probability": test_probability,
            "test_details": test_details,
        },
    }


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:6], 16)


# -------------------------------------------------------------------------
# Explanatory statistics
# -------------------------------------------------------------------------

def explain_shortcut_statistics(
    labels: np.ndarray,
    shortcut_evidence: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    hallucinated = shortcut_evidence[labels == 1]
    correct = shortcut_evidence[labels == 0]
    if len(hallucinated) == 0 or len(correct) == 0:
        return {"available": False}
    rng = np.random.default_rng(seed)
    difference = float(hallucinated.mean() - correct.mean())

    difference_bootstrap = [
        rng.choice(hallucinated, len(hallucinated), replace=True).mean()
        - rng.choice(correct, len(correct), replace=True).mean()
        for _ in range(draws)
    ]
    permutation_count = 0
    for _ in range(draws):
        permuted = rng.permutation(labels)
        permuted_difference = (
            shortcut_evidence[permuted == 1].mean()
            - shortcut_evidence[permuted == 0].mean()
        )
        permutation_count += abs(permuted_difference) >= abs(difference)

    return {
        "available": True,
        "hallucination": {
            "n": int(len(hallucinated)),
            "mean": float(hallucinated.mean()),
            "median": float(np.median(hallucinated)),
            "std": float(hallucinated.std()),
        },
        "correct": {
            "n": int(len(correct)),
            "mean": float(correct.mean()),
            "median": float(np.median(correct)),
            "std": float(correct.std()),
        },
        "mean_difference_hallucination_minus_correct": difference,
        "difference_bootstrap_95_ci": [
            float(np.quantile(difference_bootstrap, 0.025)),
            float(np.quantile(difference_bootstrap, 0.975)),
        ],
        "label_permutation_p_value": float(
            (permutation_count + 1) / (draws + 1)
        ),
        "shortcut_evidence_auroc": safe_auroc(labels, shortcut_evidence),
        "shortcut_evidence_auprc": safe_auprc(labels, shortcut_evidence),
    }


# -------------------------------------------------------------------------
# Output and main run
# -------------------------------------------------------------------------

def base_row(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "item_id",
        "raw_index",
        "question",
        "generated_answer",
        "references",
        "original_correct",
        "hallucination_label",
        "base_sequence_logprob",
        "base_mean_token_entropy",
        "prompt_tokens",
        "answer_tokens",
        "hidden_state_count",
        "hidden_size",
        "n_candidate_spans",
        "n_selected_spans",
        "error",
    )
    return {key: item.get(key) for key in keys} | {
        "n_spans": len(item.get("spans", [])),
    }


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "run_config.json", vars(args))

    rows = load_rows(args)
    examples = build_examples(rows, args)
    engine = HiddenStateEngine(args)
    evaluator = CorrectnessEvaluator(
        args.correctness_mode,
        args.token_f1_threshold,
        engine,
    )
    records = extract_all(
        examples,
        engine,
        evaluator,
        args,
        output_directory,
    )
    hidden_state_count = engine.hidden_state_count
    hidden_size = engine.hidden_size
    hidden_relation_dim = engine.hidden_relation_dim
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    failed = [item for item in records if item.get("error")]
    valid = [
        item
        for item in records
        if not item.get("error")
        and item.get("hallucination_label") is not None
        and item.get("spans")
    ]
    if len(valid) < 20:
        raise RuntimeError(f"Only {len(valid)} valid items")

    labels = np.asarray(
        [int(item["hallucination_label"]) for item in valid],
        dtype=np.int64,
    )
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Need both correct and hallucinated outputs")
    item_ids = np.asarray([item["item_id"] for item in valid])
    (
        training_id_array,
        test_id_array,
        training_labels,
        test_labels,
    ) = train_test_split(
        item_ids,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    training_ids = training_id_array.tolist()
    test_ids = test_id_array.tolist()
    training_id_set = set(training_ids)

    support_scale = estimate_support_scale(
        valid,
        training_id_set,
        args.minimum_support_scale,
    )
    pseudo_role_counts = attach_derived_features(
        valid,
        support_scale,
        training_id_set,
    )
    records_by_id = {item["item_id"]: item for item in valid}

    # ------------------ role feature-set comparison ------------------
    role_feature_sets = [
        value.strip()
        for value in args.role_feature_sets.split(",")
        if value.strip()
    ]
    if args.primary_role_feature_set not in role_feature_sets:
        role_feature_sets.insert(0, args.primary_role_feature_set)
    for feature_set in role_feature_sets:
        if feature_set not in ROLE_FEATURE_SETS:
            raise ValueError(f"Unknown role feature set: {feature_set}")

    role_results: dict[str, Any] = {}
    role_models: dict[str, Any] = {}
    primary_oof_role_probabilities: Optional[dict[str, np.ndarray]] = None
    primary_test_role_probabilities: Optional[dict[str, np.ndarray]] = None

    for feature_set in role_feature_sets:
        print(f"\n=== span role: {feature_set} ===", flush=True)
        try:
            oof_probabilities, oof_metrics = oof_role_predictions(
                records_by_id,
                training_ids,
                training_labels,
                feature_set,
                args,
            )
            features, role_labels, role_weights, _ = labeled_span_arrays(
                records_by_id,
                training_ids,
                feature_set,
            )
            model = fit_role_model(
                features,
                role_labels,
                role_weights,
                args.role_pca_dim,
                args.seed,
                args.role_logistic_c,
            )
            test_spans = spans_for(records_by_id, test_ids)
            test_features = np.stack(
                [span_vector(span, feature_set) for span in test_spans]
            )
            test_probabilities_array = predict_role_model(
                model,
                test_features,
            )
            test_probabilities = {
                span["span_uid"]: probability
                for span, probability in zip(
                    test_spans,
                    test_probabilities_array,
                )
            }
            role_results[feature_set] = {
                "feature_families": list(ROLE_FEATURE_SETS[feature_set]),
                "n_input_features": int(features.shape[1]),
                "n_training_spans": int(len(features)),
                "train_oof": oof_metrics,
                "pca_dim": (
                    None
                    if model["pca"] is None
                    else int(model["pca"].n_components_)
                ),
            }
            role_models[feature_set] = model
            if feature_set == args.primary_role_feature_set:
                primary_oof_role_probabilities = oof_probabilities
                primary_test_role_probabilities = test_probabilities
        except RuntimeError as error:
            warnings.warn(
                f"Skipping role feature set {feature_set}: {error}"
            )
            role_results[feature_set] = {
                "error": str(error),
                "feature_families": list(ROLE_FEATURE_SETS[feature_set]),
            }

    if primary_oof_role_probabilities is None or primary_test_role_probabilities is None:
        raise RuntimeError(
            "Primary role feature set failed; cannot build shortcut evidence"
        )

    # ------------------ paper-style hidden layer scan ------------------
    (
        selected_hidden_layer,
        hidden_layer_selection,
        hidden_layer_results,
    ) = scan_hidden_layers(
        records_by_id,
        training_ids,
        test_ids,
        training_labels,
        test_labels,
        hidden_state_count,
        args,
    )

    # ------------------ item configuration comparison ------------------
    configurations = [
        value.strip()
        for value in args.item_configurations.split(",")
        if value.strip()
    ]
    if args.primary_configuration not in configurations:
        configurations.insert(0, args.primary_configuration)
    for configuration in configurations:
        if configuration not in ITEM_MODES:
            raise ValueError(f"Unknown item configuration: {configuration}")

    configuration_results: dict[str, Any] = {}
    configuration_models: dict[str, Any] = {}
    primary_predictions: Optional[dict[str, Any]] = None

    for configuration in configurations:
        print(f"\n=== item detector: {configuration} ===", flush=True)
        needs_shortcut = "shortcut" in configuration
        result = evaluate_item_configuration(
            configuration_name=configuration,
            mode=configuration,
            records=records_by_id,
            training_ids=training_ids,
            test_ids=test_ids,
            training_labels=training_labels,
            test_labels=test_labels,
            selected_hidden_layer=selected_hidden_layer,
            support_scale=support_scale,
            oof_role_probabilities=(
                primary_oof_role_probabilities if needs_shortcut else None
            ),
            test_role_probabilities=(
                primary_test_role_probabilities if needs_shortcut else None
            ),
            args=args,
        )
        configuration_models[configuration] = result.pop("_artifacts")
        predictions = result.pop("_predictions")
        configuration_results[configuration] = result
        if configuration == args.primary_configuration:
            primary_predictions = predictions

    assert primary_predictions is not None
    primary_result = configuration_results[args.primary_configuration]
    primary_threshold = float(
        primary_result["selected_threshold_from_train_oof"]
    )
    primary_test_probability = np.asarray(
        primary_predictions["test_probability"],
        dtype=np.float64,
    )
    primary_test_details = primary_predictions["test_details"]

    # ------------------ output files ------------------
    base_path = output_directory / "base_open_features.jsonl"
    intervention_path = output_directory / "intervention_open_features.jsonl"
    prediction_path = output_directory / "predictions.jsonl"
    hidden_index_path = output_directory / "hidden_state_index.jsonl"
    for path in (
        base_path,
        intervention_path,
        prediction_path,
        hidden_index_path,
    ):
        if path.exists():
            path.unlink()

    for item in records:
        append_jsonl(base_path, base_row(item))
        if not item.get("error") and item.get("answer_hidden_states") is not None:
            append_jsonl(
                hidden_index_path,
                {
                    "item_id": item["item_id"],
                    "hidden_shape": list(
                        np.asarray(item["answer_hidden_states"]).shape
                    ),
                    "cache_storage": "item_cache/*.pt",
                    "answer_hidden_pooling": args.answer_hidden_pooling,
                },
            )
        for span in item.get("spans", []):
            append_jsonl(
                intervention_path,
                {
                    "item_id": item["item_id"],
                    "span_uid": span["span_uid"],
                    "span_index": span["span_index"],
                    "span_text": span["span_text"],
                    "hidden_selection_rank": span.get("hidden_selection_rank"),
                    "hidden_selection_score": span.get("hidden_selection_score"),
                    "hidden_selection_cosine": span.get("hidden_selection_cosine"),
                    "hidden_selection_negative_distance": span.get(
                        "hidden_selection_negative_distance"
                    ),
                    "hidden_selection_token_count": span.get(
                        "hidden_selection_token_count"
                    ),
                    "usage": span.get("usage"),
                    "pseudo_role": span.get("pseudo_role"),
                    "role_reliability": span.get("role_reliability"),
                    "pseudo_role_reason": span.get("pseudo_role_reason"),
                    "operators": span["operators"],
                },
            )

    shortcut_evidence_values: list[float] = []
    for item_id, label, probability in zip(
        test_ids,
        test_labels,
        primary_test_probability,
    ):
        item = records_by_id[item_id]
        details = primary_test_details[item_id]
        shortcut_evidence = float(
            details.get("shortcut_evidence_mean", 0.0)
        )
        shortcut_evidence_values.append(shortcut_evidence)
        append_jsonl(
            prediction_path,
            {
                "item_id": item_id,
                "question": item["question"],
                "generated_answer": item["generated_answer"],
                "references": item["references"],
                "hallucination_label": int(label),
                "hallucination_probability": float(probability),
                "predicted_hallucination": bool(
                    probability >= primary_threshold
                ),
                "threshold": primary_threshold,
                "configuration": args.primary_configuration,
                "selected_hidden_layer": selected_hidden_layer,
                **details,
            },
        )

    hidden_ranking = sorted(
        configuration_results,
        key=lambda name: (
            configuration_results[name]["item_metrics"]["test"]["auroc"]
            if configuration_results[name]["item_metrics"]["test"]["auroc"]
            is not None
            else -1.0
        ),
        reverse=True,
    )
    role_ranking = sorted(
        role_results,
        key=lambda name: (
            role_results[name].get("train_oof", {}).get(
                "macro_ovr_auroc"
            )
            if role_results[name].get("train_oof", {}).get(
                "macro_ovr_auroc"
            )
            is not None
            else -1.0
        ),
        reverse=True,
    )

    shortcut_explanation = explain_shortcut_statistics(
        test_labels,
        np.asarray(shortcut_evidence_values, dtype=np.float64),
        args.seed,
        args.bootstrap_draws,
    )

    bundle_path = output_directory / "openended_v12_bundle.joblib"
    summary = {
        "method": (
            "open-ended static-hidden-state + behavior + role-mediated "
            "hallucination detector v12"
        ),
        "model": args.model,
        "data": args.input or args.hf_dataset,
        "n_input": len(rows),
        "n_examples": len(examples),
        "n_extracted": len(records),
        "n_failed": len(failed),
        "n_refusal_or_unlabeled": (
            len(records) - len(failed) - len(valid)
        ),
        "n_valid": len(valid),
        "n_train": len(training_ids),
        "n_test": len(test_ids),
        "train_positive_rate": float(training_labels.mean()),
        "test_positive_rate": float(test_labels.mean()),
        "support_scale_from_train": support_scale,
        "hidden_state_extraction": {
            "hidden_state_count": hidden_state_count,
            "transformer_layer_count": hidden_state_count - 1,
            "hidden_size": hidden_size,
            "answer_pooling": args.answer_hidden_pooling,
            "answer_last_k": (
                args.answer_hidden_last_k
                if args.answer_hidden_pooling == "last_k_mean"
                else None
            ),
            "cache_dtype": args.hidden_cache_dtype,
            "intervened_hidden_states_extracted": False,
            "hidden_transition_used": False,
            "raw_answer_hidden_used_for_item_probe": False,
            "item_hidden_selection": (
                "fold-local absolute standardized class-mean difference"
            ),
            "item_hidden_top_k": args.hidden_top_k,
            "item_behavior_feature_count": 8,
            "item_shortcut_feature_count": 8,
            "primary_item_feature_count_expected": (
                args.hidden_top_k + 8 + 8
            ),
            "span_hidden_relation_dim": hidden_relation_dim,
        },
        "span_selection": {
            "method": "top_k_span_to_answer_hidden_similarity",
            "attention_used": False,
            "max_intervention_spans": args.max_intervention_spans,
            "layers": args.hidden_selection_layers,
            "last_n": (
                args.hidden_selection_last_n
                if args.hidden_selection_layers == "last_n"
                else None
            ),
            "score": args.hidden_selection_score,
            "hybrid_cosine_weight": (
                args.hidden_selection_cosine_weight
                if args.hidden_selection_score == "hybrid"
                else None
            ),
        },
        "hidden_layer_selection": hidden_layer_selection,
        "hidden_layer_probe_results": {
            str(layer): result
            for layer, result in hidden_layer_results.items()
        },
        "test_prediction_uses_reference_features": False,
        "references_used_for_train_pseudo_roles": True,
        "references_used_for_final_evaluation": True,
        "interventions_used_for_prediction": list(OPERATORS),
        "behavior_support_definition": (
            "mean token logP(original answer|base) - mean token "
            "logP(original answer|intervention)"
        ),
        "role_names": ROLE_NAMES,
        "role_feature_sets": role_results,
        "role_feature_set_ranking_train_oof_auroc": role_ranking,
        "primary_role_feature_set": args.primary_role_feature_set,
        "role_pseudo_label_counts": pseudo_role_counts,
        "item_configurations": configuration_results,
        "item_configuration_test_auroc_ranking": hidden_ranking,
        "primary_configuration": args.primary_configuration,
        "primary_metrics": primary_result["item_metrics"],
        "primary_selected_threshold_from_train_oof": primary_threshold,
        "shortcut_explanatory_statistics_primary": shortcut_explanation,
        "files": {
            "base_open_features": str(base_path),
            "intervention_open_features": str(intervention_path),
            "hidden_state_index": str(hidden_index_path),
            "predictions": str(prediction_path),
            "model_bundle": str(bundle_path),
            "item_cache": str(output_directory / "item_cache"),
        },
        "failed_items": [
            {"item_id": item["item_id"], "error": item["error"]}
            for item in failed
        ],
        "method_notes": {
            "open_ended_generation": True,
            "teacher_forcing_target": "original generated answer",
            "static_hidden_state_position": (
                "last answer token by default; configurable pooling"
            ),
            "layer_selection": (
                "train-only OOF; test layer metrics reported but not used "
                "for selection"
            ),
            "span_proposal": (
                "hidden similarity only; no answer-to-span attention"
            ),
            "shortcut_role_is_auxiliary_to_item_detector": True,
            "hidden_transition": False,
            "intervention_hidden_state": False,
            "causal_interpretation_warning": (
                "Behavior interventions are causal perturbations, but the "
                "static hidden probe is correlational. v12 does not claim "
                "that hidden states themselves are causal mechanisms."
            ),
        },
    }

    joblib.dump(
        {
            "args": vars(args),
            "support_scale": support_scale,
            "selected_hidden_layer": selected_hidden_layer,
            "role_names": ROLE_NAMES,
            "role_feature_sets": ROLE_FEATURE_SETS,
            "role_models": role_models,
            "item_models": configuration_models,
            "hidden_layer_selection": hidden_layer_selection,
        },
        bundle_path,
        compress=3,
    )
    write_json(output_directory / "summary.json", summary)

    print(
        "\nPrimary test metrics:\n"
        + json.dumps(
            summary["primary_metrics"]["test"],
            indent=2,
        )
    )
    print(f"Selected hidden layer: {selected_hidden_layer}")
    print(f"Outputs: {output_directory}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    argument_parser.add_argument("--input")
    argument_parser.add_argument("--hf-dataset")
    argument_parser.add_argument("--hf-subset")
    argument_parser.add_argument("--hf-split", default="validation")
    argument_parser.add_argument("--question-field")
    argument_parser.add_argument("--context-field")
    argument_parser.add_argument("--answers-field")
    argument_parser.add_argument("--prompt-field")
    argument_parser.add_argument("--id-field")
    argument_parser.add_argument("--max-samples", type=int, default=0)
    argument_parser.add_argument("--test-size", type=float, default=0.25)

    # Model and generation
    argument_parser.add_argument(
        "--model",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    argument_parser.add_argument("--device", default="cuda")
    argument_parser.add_argument("--dtype", default="bfloat16")
    argument_parser.add_argument("--trust-remote-code", action="store_true")
    argument_parser.add_argument("--max-input-tokens", type=int, default=2048)
    argument_parser.add_argument("--max-new-tokens", type=int, default=64)
    argument_parser.add_argument("--temperature", type=float, default=0.0)
    argument_parser.add_argument("--top-p", type=float, default=0.95)
    argument_parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    argument_parser.add_argument(
        "--answer-instruction",
        default="Provide a concise final answer.",
    )

    # Evaluation
    argument_parser.add_argument(
        "--correctness-mode",
        choices=("hybrid", "exact", "token_f1", "numeric", "llm_judge"),
        default="hybrid",
    )
    argument_parser.add_argument(
        "--token-f1-threshold",
        type=float,
        default=0.8,
    )

    # Span construction and hidden-based proposal
    argument_parser.add_argument("--min-clause-words", type=int, default=12)
    argument_parser.add_argument("--min-span-words", type=int, default=2)
    argument_parser.add_argument(
        "--max-intervention-spans",
        type=int,
        default=4,
        help="Top-k hidden-similarity spans; 0 means all spans",
    )
    argument_parser.add_argument(
        "--hidden-selection-layers",
        choices=("all", "last_half", "last_quarter", "last_n"),
        default="last_quarter",
    )
    argument_parser.add_argument(
        "--hidden-selection-last-n",
        type=int,
        default=4,
    )
    argument_parser.add_argument(
        "--hidden-selection-score",
        choices=("hybrid", "cosine", "negative_distance"),
        default="hybrid",
    )
    argument_parser.add_argument(
        "--hidden-selection-cosine-weight",
        type=float,
        default=0.7,
    )
    argument_parser.add_argument(
        "--answer-hidden-pooling",
        choices=("last", "mean", "last_k_mean"),
        default="last",
        help="Paper-style default is last answer token",
    )
    argument_parser.add_argument(
        "--answer-hidden-last-k",
        type=int,
        default=4,
    )
    argument_parser.add_argument(
        "--hidden-cache-dtype",
        choices=("float16", "float32"),
        default="float16",
    )

    # Interventions
    argument_parser.add_argument(
        "--mask-text",
        default="[MASKED INFORMATION]",
    )
    argument_parser.add_argument(
        "--neutral-text",
        default="This detail is unspecified.",
    )
    argument_parser.add_argument(
        "--minimum-support-scale",
        type=float,
        default=0.05,
    )

    # Hidden layer scan
    argument_parser.add_argument(
        "--hidden-layer-start",
        type=int,
        default=0,
        help="Index in output.hidden_states; 0 is embedding output",
    )
    argument_parser.add_argument(
        "--hidden-layer-end",
        type=int,
        default=-1,
        help="Inclusive; -1 means final transformer block",
    )
    argument_parser.add_argument(
        "--hidden-layer-stride",
        type=int,
        default=1,
    )
    argument_parser.add_argument(
        "--hidden-top-k",
        type=int,
        default=16,
        help=(
            "Hidden coordinates retained by fold-local supervised selection; "
            "selection never uses validation or test labels"
        ),
    )

    # Role and item models
    argument_parser.add_argument(
        "--role-feature-sets",
        default="behavior_only,hidden_relation_only,behavior_hidden",
    )
    argument_parser.add_argument(
        "--primary-role-feature-set",
        default="behavior_hidden",
    )
    argument_parser.add_argument(
        "--role-pca-dim",
        type=int,
        default=128,
        help="0 disables PCA for the span-role model",
    )
    argument_parser.add_argument(
        "--role-logistic-c",
        type=float,
        default=1.0,
    )
    argument_parser.add_argument(
        "--item-configurations",
        default=(
            "behavior_only,hidden_only,hidden_behavior,behavior_shortcut,"
            "hidden_shortcut,hidden_behavior_shortcut"
        ),
    )
    argument_parser.add_argument(
        "--primary-configuration",
        default="hidden_behavior_shortcut",
    )
    argument_parser.add_argument(
        "--item-pca-dim",
        type=int,
        default=0,
        help="Optional PCA after Top-k hidden selection; 0 disables PCA",
    )
    argument_parser.add_argument(
        "--item-logistic-c",
        type=float,
        default=1.0,
    )
    argument_parser.add_argument("--cv-folds", type=int, default=5)
    argument_parser.add_argument("--item-top-k", type=int, default=3)

    # Output
    argument_parser.add_argument("--output-dir", required=True)
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--bootstrap-draws", type=int, default=2000)
    argument_parser.add_argument("--overwrite-cache", action="store_true")
    return argument_parser


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if bool(args.input) == bool(args.hf_dataset):
        argument_parser.error(
            "Provide exactly one of --input or --hf-dataset"
        )
    if args.primary_role_feature_set not in ROLE_FEATURE_SETS:
        argument_parser.error("Unknown --primary-role-feature-set")
    if args.primary_configuration not in ITEM_MODES:
        argument_parser.error("Unknown --primary-configuration")
    if not 0.0 <= args.hidden_selection_cosine_weight <= 1.0:
        argument_parser.error(
            "--hidden-selection-cosine-weight must be in [0, 1]"
        )
    if not 0.0 < args.test_size < 1.0:
        argument_parser.error("--test-size must be in (0, 1)")
    if args.hidden_top_k < 1:
        argument_parser.error("--hidden-top-k must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
