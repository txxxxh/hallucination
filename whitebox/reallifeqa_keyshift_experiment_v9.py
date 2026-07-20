#!/usr/bin/env python3
"""
KeyShift: theory-guided causal validation and mitigation for RealLifeQA.

This script starts from an already trained v7/v8 detector and implements four
experiments requested for frequency-induced key-selection hallucination:

1. Frequency-controlled semantic counterfactuals
   * A local editor LLM proposes semantically equivalent rewrites of the detector's
     predicted shortcut span.
   * A separate local target model measures each rewrite's shortcut-prior
     activation in a constraint-free carrier prompt.
   * Valid candidates are split into low / medium / high prior conditions, plus
     a common-paraphrase control.

2. Paraphrase effect linked to internal ablation and activation rescue
   * Instance-specific heads are ranked by the reduction in answer-to-shortcut
     attention between the original and prior-disrupting paraphrase runs.
   * The selected heads are zero-ablated and compared with layer-matched random
     head ablations.
   * Final-token head activations are patched in both directions:
       original -> low-prior prompt (rescue of shortcut tendency)
       low-prior -> original prompt (suppression of shortcut tendency)

3. Prior-Disrupting Paraphrase (PDP) mitigation
   * Select the lowest shortcut-prior candidate that passes semantic,
     naturalness, answer-preservation, and surprisal constraints.
   * Evaluate both always-on and detector-gated PDP.

4. Contextual Key Linking (CKL) mitigation
   * The editor LLM creates one bridge sentence connecting the shortcut cue to
     the predicted constraint without stating an option number or direct answer.
   * Evaluate CKL alone and PDP+CKL, always-on and detector-gated.

Inputs
------
RealLifeQA JSON list with the existing schema:
    id, question, options, answer, correct_option, benchmark_prompt,
    short_justification, ...

Detector predictions JSONL with the v7/v8 schema:
    idx, split, hallucination_probability, predicted_hallucination,
    explanation.predicted_shortcut,
    explanation.predicted_constraint,
    explanation.per_span_contributions

Important experimental separation
---------------------------------
* The editor LLM only proposes and validates language variants.
* Prior tiers are determined by the target model's independent prior probe,
  never by the editor's labels and never by the full-prompt answer outcome.
* Mitigation triggering uses detector output, not the gold answer.
* Gold answers are used only for candidate answer-preservation validation and
  final evaluation.
* Internal head selection uses the original-vs-PDP attention difference; zero
  ablation and activation patching are held-out causal operations.

Example
-------
python reallifeqa_keyshift_experiment.py all \
  --input /home/tong56/whitebox/question_and_result.json \
  --detector-predictions /home/tong56/.../predictions.jsonl \
  --target-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --editor-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --output-dir outputs/keyshift_reallifeqa \
  --only-split test

Both target and editor models are loaded locally with Transformers. By default,
the editor reuses the target weights to avoid a second model copy in memory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import hashlib
import json
import math
import random
import re
import statistics
import sys
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm.auto import tqdm


SYSTEM_CHOICE = "Answer with exactly one character: 1 or 2. Do not explain."
EPS = 1e-12
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|can't|won't|isn't|"
    r"aren't|didn't|doesn't|don't)\b",
    flags=re.IGNORECASE,
)
DIRECT_ANSWER_RE = re.compile(
    r"\b(?:choose|select|answer|option)\s*(?:1|2|one|two)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealLifeItem:
    index: int
    item_id: str
    question: str
    options: tuple[str, str]
    answer: int
    correct_option: str
    shortcut_option: str
    benchmark_prompt: str
    short_justification: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Detection:
    index: int
    split: str
    hallucination_probability: float
    predicted_hallucination: bool
    shortcut_text: Optional[str]
    constraint_text: Optional[str]
    shortcut_source: str
    constraint_source: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LocatedSpan:
    text: str
    start: int
    end: int
    match_method: str
    match_score: float


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_error(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def token_f1(a: str, b: str) -> float:
    aa = normalize_text(a).split()
    bb = normalize_text(b).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    ca: dict[str, int] = defaultdict(int)
    cb: dict[str, int] = defaultdict(int)
    for x in aa:
        ca[x] += 1
    for x in bb:
        cb[x] += 1
    overlap = sum(min(ca[k], cb[k]) for k in ca)
    if overlap == 0:
        return 0.0
    precision = overlap / len(aa)
    recall = overlap / len(bb)
    return 2 * precision * recall / (precision + recall)


def edit_distance_ratio(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def bootstrap_mean_difference(
    x: Sequence[float],
    y: Sequence[float],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) != len(ya) or len(xa) == 0:
        return {"n": 0, "mean_difference": None, "bootstrap_95_ci": None}
    diff = xa - ya
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for i in range(draws):
        samples[i] = rng.choice(diff, size=len(diff), replace=True).mean()
    return {
        "n": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def clustered_within_item_slope(
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Fixed-effect slope with item-cluster bootstrap confidence intervals."""
    usable = [(np.asarray(x, float), np.asarray(y, float)) for x, y in groups if len(x) >= 2]
    if not usable:
        return {"n_items": 0, "n_points": 0, "slope": None, "bootstrap_95_ci": None}

    def slope(sampled: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
        centered_x = []
        centered_y = []
        for x, y in sampled:
            centered_x.append(x - x.mean())
            centered_y.append(y - y.mean())
        xx = np.concatenate(centered_x)
        yy = np.concatenate(centered_y)
        denominator = float(np.dot(xx, xx))
        return float(np.dot(xx, yy) / denominator) if denominator > 1e-12 else float("nan")

    observed = slope(usable)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        indices = rng.integers(0, len(usable), size=len(usable))
        value = slope([usable[int(i)] for i in indices])
        if np.isfinite(value):
            boot.append(value)
    return {
        "n_items": len(usable),
        "n_points": int(sum(len(x) for x, _ in usable)),
        "slope": observed,
        "bootstrap_95_ci": (
            [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if boot
            else None
        ),
    }


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            try:
                self.data: dict[str, Any] = read_json(path)
            except Exception:
                backup = path.with_suffix(path.suffix + ".broken")
                path.replace(backup)
                self.data = {}
        else:
            self.data = {}

    def key(self, namespace: str, payload: Any) -> str:
        return f"{namespace}:{stable_hash(payload)}"

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = json_safe(value)
        write_json(self.path, self.data)


class ItemStageCache:
    def __init__(self, root: Path, stage: str, signature: str) -> None:
        self.root = root / "item_cache" / stage / signature[:16]
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, item: RealLifeItem) -> Path:
        return self.root / f"{item.index:06d}_{stable_hash(item.item_id)[:12]}.json"

    def load(self, item: RealLifeItem) -> Optional[dict[str, Any]]:
        path = self.path(item)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    def save(self, item: RealLifeItem, result: dict[str, Any]) -> None:
        write_json(self.path(item), result)

    def all_rows(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*.json")):
            try:
                rows.append(read_json(path))
            except Exception:
                pass
        return rows


# ---------------------------------------------------------------------------
# RealLifeQA and detector loading
# ---------------------------------------------------------------------------

def load_items(path: Path) -> list[RealLifeItem]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError("RealLifeQA input must be a JSON list.")
    items: list[RealLifeItem] = []
    required = ("question", "options", "answer", "benchmark_prompt")
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Item {index} missing keys: {missing}")
        options = row["options"]
        answer = int(row["answer"])
        if not isinstance(options, list) or len(options) != 2 or answer not in (1, 2):
            raise ValueError(f"Invalid options/answer in item {index}")
        item_id = str(row.get("id", row.get("key", index)))
        correct_option = str(row.get("correct_option", options[answer - 1]))
        shortcut_number = 1 if answer == 2 else 2
        items.append(
            RealLifeItem(
                index=index,
                item_id=item_id,
                question=str(row["question"]),
                options=(str(options[0]), str(options[1])),
                answer=answer,
                correct_option=correct_option,
                shortcut_option=str(options[shortcut_number - 1]),
                benchmark_prompt=str(row["benchmark_prompt"]),
                short_justification=str(row.get("short_justification", "")),
                raw=row,
            )
        )
    return items


def _resolved_prediction(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict) or not obj.get("resolved"):
        return None
    text = obj.get("text")
    return str(text).strip() if text else None


def _fallback_role_span(
    explanation: dict[str, Any],
    role: str,
) -> Optional[str]:
    rows = explanation.get("per_span_contributions", [])
    if not isinstance(rows, list) or not rows:
        return None
    if role == "shortcut":
        def score(row: dict[str, Any]) -> float:
            return float(
                row.get(
                    "shortcut_logit_contribution",
                    float(row.get("shortcut_probability", 0.0))
                    * float(row.get("intervention_usage", 0.0)),
                )
            )
    else:
        def score(row: dict[str, Any]) -> float:
            value = row.get("constraint_logit_contribution")
            if value is not None:
                return -float(value)
            return float(row.get("constraint_probability", 0.0)) * float(
                row.get("intervention_usage", 0.0)
            )
    best = max((r for r in rows if isinstance(r, dict)), key=score, default=None)
    if not best or not best.get("text"):
        return None
    return str(best["text"]).strip()


def load_detections(
    path: Path,
    allow_role_fallback: bool,
) -> dict[int, Detection]:
    rows = read_jsonl(path)
    detections: dict[int, Detection] = {}
    for row in rows:
        if "idx" not in row:
            continue
        idx = int(row["idx"])
        explanation = row.get("explanation") or {}
        shortcut = _resolved_prediction(explanation.get("predicted_shortcut"))
        constraint = _resolved_prediction(explanation.get("predicted_constraint"))
        shortcut_source = "predicted_shortcut"
        constraint_source = "predicted_constraint"
        if shortcut is None and allow_role_fallback:
            shortcut = _fallback_role_span(explanation, "shortcut")
            shortcut_source = "top_span_fallback" if shortcut else "missing"
        if constraint is None and allow_role_fallback:
            constraint = _fallback_role_span(explanation, "constraint")
            constraint_source = "top_span_fallback" if constraint else "missing"
        detections[idx] = Detection(
            index=idx,
            split=str(row.get("split", "unknown")),
            hallucination_probability=float(row.get("hallucination_probability", 0.0)),
            predicted_hallucination=bool(row.get("predicted_hallucination", False)),
            shortcut_text=shortcut,
            constraint_text=constraint,
            shortcut_source=shortcut_source,
            constraint_source=constraint_source,
            raw=row,
        )
    return detections


# ---------------------------------------------------------------------------
# Span matching and prompt editing
# ---------------------------------------------------------------------------

def locate_scenario(prompt: str) -> tuple[int, int]:
    start_match = re.search(r"^\s*Scenario:\s*", prompt)
    start = start_match.end() if start_match else 0
    option_match = re.search(r"\n\s*Option1\s*:", prompt)
    end = option_match.start() if option_match else len(prompt)
    if end <= start:
        return 0, len(prompt)
    return start, end


def candidate_chunks(prompt: str) -> list[LocatedSpan]:
    start, end = locate_scenario(prompt)
    scenario = prompt[start:end]
    chunks: list[LocatedSpan] = []
    sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", flags=re.DOTALL)
    for match in sentence_pattern.finditer(scenario):
        text = match.group(0).strip()
        if not text:
            continue
        local_start = scenario.find(text, match.start(), match.end() + 1)
        abs_start = start + local_start
        chunks.append(LocatedSpan(text, abs_start, abs_start + len(text), "segment", 1.0))
    if not chunks and scenario.strip():
        text = scenario.strip()
        local = scenario.find(text)
        chunks = [LocatedSpan(text, start + local, start + local + len(text), "scenario", 1.0)]
    return chunks


def locate_span(prompt: str, span_text: str, minimum_ratio: float = 0.45) -> LocatedSpan:
    exact = prompt.find(span_text)
    if exact >= 0:
        return LocatedSpan(span_text, exact, exact + len(span_text), "exact", 1.0)

    target = normalize_text(span_text)
    best: Optional[LocatedSpan] = None
    for chunk in candidate_chunks(prompt):
        ratio = difflib.SequenceMatcher(None, target, normalize_text(chunk.text)).ratio()
        if best is None or ratio > best.match_score:
            best = LocatedSpan(chunk.text, chunk.start, chunk.end, "fuzzy_segment", ratio)
    if best is None or best.match_score < minimum_ratio:
        best_text = "none" if best is None else f"{best.match_score:.3f}"
        raise ValueError(
            f"Could not locate detector span in prompt (best={best_text}): {span_text!r}"
        )
    return best


def replace_located(prompt: str, span: LocatedSpan, replacement: str) -> tuple[str, LocatedSpan]:
    replacement = replacement.strip()
    new_prompt = prompt[: span.start] + replacement + prompt[span.end :]
    new_span = LocatedSpan(
        text=replacement,
        start=span.start,
        end=span.start + len(replacement),
        match_method="replacement",
        match_score=1.0,
    )
    return new_prompt, new_span


def insert_after_located(prompt: str, span: LocatedSpan, insertion: str) -> str:
    insertion = insertion.strip()
    punctuation = "" if prompt[span.end - 1 : span.end] in ".!?" else "."
    prefix = prompt[: span.end]
    suffix = prompt[span.end :]
    return prefix + punctuation + " " + insertion + " " + suffix.lstrip()


def prompt_preserves_options(prompt: str, item: RealLifeItem) -> bool:
    return (
        f"Option1: {item.options[0]}" in prompt
        and f"Option2: {item.options[1]}" in prompt
        and "Answer 1" in prompt
        and "2" in prompt
    )


# ---------------------------------------------------------------------------
# Local editor LLM
# ---------------------------------------------------------------------------

class EditorLLM:
    def __init__(
        self,
        backend: "TargetModel",
        model_name: str,
        cache: JsonCache,
        temperature: float,
        max_retries: int,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.cache = cache
        self.temperature = temperature
        self.max_retries = max_retries

    @staticmethod
    def _extract_json(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start_obj = cleaned.find("{")
            end_obj = cleaned.rfind("}")
            start_arr = cleaned.find("[")
            end_arr = cleaned.rfind("]")
            candidates = []
            if start_obj >= 0 and end_obj > start_obj:
                candidates.append(cleaned[start_obj : end_obj + 1])
            if start_arr >= 0 and end_arr > start_arr:
                candidates.append(cleaned[start_arr : end_arr + 1])
            for candidate in candidates:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            raise

    def call_json(
        self,
        namespace: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> Any:
        payload = {
            "model": self.model_name,
            "system": system,
            "user": user,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        key = self.cache.key(namespace, payload)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        errors = []
        repair_suffix = ""
        for attempt in range(self.max_retries):
            temperature = self.temperature if attempt == 0 else 0.0
            try:
                content = self.backend.generate_chat(
                    system=system, user=user + repair_suffix,
                    max_new_tokens=max_tokens, temperature=temperature,
                )
                parsed = self._extract_json(content)
                self.cache.set(key, parsed)
                return parsed
            except Exception as exc:
                errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                repair_suffix = (
                    "\n\nYour previous response was not valid JSON. Try again and "
                    "output only one JSON value matching the requested schema."
                )
        raise RuntimeError("Editor call failed: " + " | ".join(errors))

    def generate_pack(
        self,
        item: RealLifeItem,
        shortcut: str,
        constraint: str,
        n_candidates: int,
    ) -> dict[str, Any]:
        user = f"""
You are constructing controlled linguistic interventions for a binary-choice
reasoning experiment. Return only valid JSON.

Original RealLifeQA prompt:
{item.benchmark_prompt}

Detected shortcut span:
{shortcut}

Detected constraint span:
{constraint}

Correct option number (for preservation checking only): {item.answer}
Correct option text: {item.correct_option}
Other/shortcut option text: {item.shortcut_option}
Factual justification: {item.short_justification}

Generate exactly {n_candidates} candidate replacements for ONLY the detected
shortcut span. All candidates must:
- preserve the shortcut span's truth conditions, entities, quantities,
  negation, modality, and temporal meaning;
- preserve the correct answer and the decision problem;
- add no new constraint, explanation, recommendation, or answer hint;
- remain grammatical in the original scenario;
- vary substantially in lexical conventionality and association strength,
  including ordinary direct wording and less stereotyped but still natural
  wording; do not label which candidate is low or high frequency;
- not use intentionally archaic, nonsensical, or obscure vocabulary.

Also generate one contextual_key_link sentence. It will be inserted immediately
after the shortcut span. It must:
- explain how the shortcut cue should be interpreted jointly with the detected
  constraint;
- make the relevant relationship explicit without saying "choose", "answer",
  "option 1", or "option 2";
- not quote or repeat an option verbatim;
- preserve all facts and remain one concise sentence.

Return exactly this JSON schema:
{{
  "paraphrases": [
    {{"candidate_id": "p01", "text": "..."}}
  ],
  "contextual_key_link": "..."
}}
""".strip()
        parsed = self.call_json(
            namespace="generate_keyshift_pack_v3",
            system="You produce controlled experimental text and return only valid JSON.",
            user=user,
            max_tokens=2200,
        )
        if not isinstance(parsed, dict):
            raise ValueError("Editor pack is not a JSON object.")
        paraphrases = parsed.get("paraphrases")
        link = parsed.get("contextual_key_link")
        if not isinstance(paraphrases, list) or not isinstance(link, str):
            raise ValueError("Editor pack lacks paraphrases/contextual_key_link.")
        cleaned = []
        seen = set()
        for idx, row in enumerate(paraphrases):
            if isinstance(row, str):
                text = row.strip()
                candidate_id = f"p{idx + 1:02d}"
            elif isinstance(row, dict):
                text = str(row.get("text", "")).strip()
                candidate_id = str(row.get("candidate_id", f"p{idx + 1:02d}"))
            else:
                continue
            key = normalize_text(text)
            if text and key and key not in seen:
                cleaned.append({"candidate_id": candidate_id, "text": text})
                seen.add(key)
        if len(cleaned) < 3:
            raise ValueError("Fewer than three unique paraphrases were produced.")
        return {"paraphrases": cleaned, "contextual_key_link": link.strip()}

    def validate_pack(
        self,
        item: RealLifeItem,
        shortcut: str,
        constraint: str,
        pack: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = pack["paraphrases"]
        user = f"""
Audit controlled text variants. Return only valid JSON.

Original prompt:
{item.benchmark_prompt}

Original shortcut span:
{shortcut}

Detected constraint span:
{constraint}

Correct option number: {item.answer}
Correct option text: {item.correct_option}
Other option text: {item.shortcut_option}
Factual justification: {item.short_justification}

Candidate paraphrases:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Contextual key-link sentence:
{pack['contextual_key_link']}

For each paraphrase, evaluate independently:
- semantic_equivalent: same truth conditions as the original shortcut span;
- entities_preserved;
- quantities_preserved;
- negation_modality_preserved;
- correct_answer_preserved;
- no_new_constraint;
- no_answer_leak;
- naturalness: integer 1 to 5.

For the contextual key link evaluate:
- constraint_relation_correct;
- facts_preserved;
- no_direct_answer_or_option_number;
- no_contradiction;
- naturalness: integer 1 to 5.

Return exactly:
{{
  "paraphrase_reviews": [
    {{
      "candidate_id": "p01",
      "semantic_equivalent": true,
      "entities_preserved": true,
      "quantities_preserved": true,
      "negation_modality_preserved": true,
      "correct_answer_preserved": true,
      "no_new_constraint": true,
      "no_answer_leak": true,
      "naturalness": 5,
      "notes": ""
    }}
  ],
  "contextual_key_link_review": {{
    "constraint_relation_correct": true,
    "facts_preserved": true,
    "no_direct_answer_or_option_number": true,
    "no_contradiction": true,
    "naturalness": 5,
    "notes": ""
  }}
}}
""".strip()
        parsed = self.call_json(
            namespace="validate_keyshift_pack_v3",
            system="You are a strict semantic equivalence and answer-leak auditor. Return only JSON.",
            user=user,
            max_tokens=2400,
        )
        if not isinstance(parsed, dict):
            raise ValueError("Validation output is not a JSON object.")
        return parsed

    def infer_constraint(
        self,
        item: RealLifeItem,
        shortcut: str,
    ) -> str:
        user = f"""
Return only JSON.

RealLifeQA prompt:
{item.benchmark_prompt}

Detected shortcut span:
{shortcut}

Correct option number: {item.answer}
Correct option text: {item.correct_option}
Factual justification: {item.short_justification}

Identify the shortest exact or near-exact span from the scenario that states the
physical, procedural, spatial, or factual constraint that makes the correct
option appropriate. Do not invent a new fact.

Return: {{"constraint_span": "..."}}
""".strip()
        parsed = self.call_json(
            namespace="infer_constraint_v2",
            system="You locate an existing constraint span and return only JSON.",
            user=user,
            max_tokens=400,
        )
        if not isinstance(parsed, dict) or not parsed.get("constraint_span"):
            raise ValueError("Could not infer constraint span.")
        return str(parsed["constraint_span"]).strip()


# ---------------------------------------------------------------------------
# Local target model: choice scoring, prior probes, attention, ablation, patching
# ---------------------------------------------------------------------------

def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class TargetModel:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        trust_remote_code: bool,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers>=4.45 before loading the local target model."
            ) from exc
        self.device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for span alignment.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.layers = self._resolve_layers()
        self.num_layers = len(self.layers)
        self.num_heads = int(getattr(self.model.config, "num_attention_heads"))
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError("hidden_size is not divisible by num_attention_heads.")
        self.head_dim = self.hidden_size // self.num_heads
        self.choice_ids = {
            "1": self._choice_token_ids("1"),
            "2": self._choice_token_ids("2"),
        }

    def _resolve_layers(self) -> list[Any]:
        candidates = [
            ("model.layers", lambda m: m.model.layers),
            ("transformer.h", lambda m: m.transformer.h),
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ]
        for _, getter in candidates:
            try:
                layers = list(getter(self.model))
                if layers:
                    return layers
            except Exception:
                pass
        raise RuntimeError("Unsupported model architecture: cannot resolve transformer layers.")

    def _o_proj(self, layer: Any) -> Any:
        paths = [
            lambda x: x.self_attn.o_proj,
            lambda x: x.attn.o_proj,
            lambda x: x.attention.dense,
        ]
        for getter in paths:
            try:
                return getter(layer)
            except Exception:
                pass
        raise RuntimeError("Unsupported attention module: cannot find attention output projection.")

    def _choice_token_ids(self, choice: str) -> list[int]:
        ids: list[int] = []
        for surface in (choice, " " + choice, "\n" + choice):
            encoded = self.tokenizer(surface, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise RuntimeError(f"Could not find a single-token representation for choice {choice}.")
        return ids

    def render_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_CHOICE},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {SYSTEM_CHOICE}\nUser: {prompt}\nAssistant:"

    @torch.inference_mode()
    def generate_chat(self, system: str, user: str, max_new_tokens: int,
                      temperature: float) -> str:
        """Generate an editor response entirely in the local process."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = f"System: {system}\nUser: {user}\nAssistant:"
        encoded = self.tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        do_sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            **encoded, "max_new_tokens": max_new_tokens, "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        output = self.model.generate(**kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True).strip()

    def encode_with_offsets(self, prompt: str) -> tuple[str, torch.Tensor, list[tuple[int, int]], int]:
        rendered = self.render_prompt(prompt)
        prompt_start = rendered.find(prompt)
        if prompt_start < 0:
            raise RuntimeError("Could not locate raw prompt inside rendered chat prompt.")
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        return rendered, ids, offsets, prompt_start

    def _choice_scores(self, final_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = {}
        for choice in ("1", "2"):
            token_ids = torch.tensor(self.choice_ids[choice], device=final_logits.device)
            scores[choice] = torch.logsumexp(final_logits[token_ids].float(), dim=0)
        return scores

    def _result_from_logits(self, final_logits: torch.Tensor, item: RealLifeItem) -> dict[str, Any]:
        scores = self._choice_scores(final_logits)
        score1 = float(scores["1"].detach().cpu())
        score2 = float(scores["2"].detach().cpu())
        pair = torch.stack([scores["1"], scores["2"]])
        probs = torch.softmax(pair, dim=0).detach().cpu().numpy()
        correct = str(item.answer)
        shortcut = "1" if correct == "2" else "2"
        correct_margin = float((scores[correct] - scores[shortcut]).detach().cpu())
        prediction = "1" if score1 >= score2 else "2"
        return {
            "prediction": prediction,
            "is_correct": prediction == correct,
            "logit_1": score1,
            "logit_2": score2,
            "prob_1": float(probs[0]),
            "prob_2": float(probs[1]),
            "correct_margin": correct_margin,
            "shortcut_margin": -correct_margin,
        }

    @torch.inference_mode()
    def score(self, prompt: str, item: RealLifeItem) -> dict[str, Any]:
        _, ids, _, _ = self.encode_with_offsets(prompt)
        outputs = self.model(input_ids=ids, use_cache=False, return_dict=True)
        return self._result_from_logits(outputs.logits[0, -1], item)

    @torch.inference_mode()
    def surface_surprisal(self, text: str) -> float:
        prefix = "The situation is described as follows: "
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prefix + text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prefix_ids):
            return float("nan")
        ids = torch.tensor(full_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        outputs = self.model(input_ids=ids, use_cache=False, return_dict=True)
        logits = outputs.logits[:, :-1, :].float()
        targets = ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Target token t is predicted at t-1. Keep only continuation targets.
        start = max(len(prefix_ids) - 1, 0)
        continuation = logp[:, start:]
        return float(-continuation.mean().cpu())

    @staticmethod
    def make_prior_probe(item: RealLifeItem, expression: str) -> str:
        return (
            f"Scenario cue: {expression}\n"
            f"Option1: {item.options[0]}\n"
            f"Option2: {item.options[1]}\n"
            "Question: Based only on this cue, which option is more immediately "
            "associated with the situation? Answer 1 for Option1 and 2 for Option2."
        )

    def prior_score(self, item: RealLifeItem, expression: str) -> dict[str, Any]:
        result = self.score(self.make_prior_probe(item, expression), item)
        return {
            **result,
            "prior_shortcut_margin": result["shortcut_margin"],
            "prior_correct_margin": result["correct_margin"],
        }

    def token_indices_for_span(
        self,
        prompt: str,
        located: LocatedSpan,
    ) -> tuple[torch.Tensor, list[int]]:
        _, ids, offsets, prompt_start = self.encode_with_offsets(prompt)
        absolute_start = prompt_start + located.start
        absolute_end = prompt_start + located.end
        indices = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > a and b > absolute_start and a < absolute_end
        ]
        if not indices:
            raise RuntimeError("No tokenizer tokens align with the selected span.")
        return ids, indices

    @torch.inference_mode()
    def head_attention_to_span(
        self,
        prompt: str,
        located: LocatedSpan,
    ) -> dict[str, np.ndarray]:
        ids, span_indices = self.token_indices_for_span(prompt, located)
        outputs = self.model(
            input_ids=ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions; eager attention is required.")
        mass = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
        peak = np.zeros_like(mass)
        for layer_idx, layer_attention in enumerate(outputs.attentions):
            # [batch, heads, query, key]
            row = layer_attention[0, :, -1, span_indices].detach().float()
            mass[layer_idx] = row.sum(dim=-1).cpu().numpy()
            peak[layer_idx] = row.max(dim=-1).values.cpu().numpy()
        density = mass / max(len(span_indices), 1)
        return {"mass": mass, "density": density, "peak": peak}

    @contextlib.contextmanager
    def _head_hooks(
        self,
        ablate: Optional[set[HeadRef]] = None,
        patch: Optional[dict[HeadRef, torch.Tensor]] = None,
        capture: Optional[dict[int, torch.Tensor]] = None,
        ablation_scope: str = "all_tokens",
    ) -> Iterator[None]:
        ablate = ablate or set()
        patch = patch or {}
        by_layer_ablate: dict[int, list[int]] = defaultdict(list)
        by_layer_patch: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        for ref in ablate:
            by_layer_ablate[ref.layer].append(ref.head)
        for ref, value in patch.items():
            by_layer_patch[ref.layer][ref.head] = value

        handles = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in by_layer_ablate and layer_idx not in by_layer_patch and capture is None:
                continue
            module = self._o_proj(layer)

            def hook(
                _module: Any,
                inputs: tuple[Any, ...],
                layer_index: int = layer_idx,
            ) -> tuple[Any, ...]:
                x = inputs[0]
                if x.dim() == 2:
                    x_view = x.unsqueeze(0)
                    squeeze = True
                else:
                    x_view = x
                    squeeze = False
                if capture is not None:
                    capture[layer_index] = (
                        x_view[0, -1]
                        .detach()
                        .float()
                        .reshape(self.num_heads, self.head_dim)
                        .cpu()
                    )
                needs_change = layer_index in by_layer_ablate or layer_index in by_layer_patch
                if not needs_change:
                    return inputs
                modified = x_view.clone()
                for head in by_layer_ablate.get(layer_index, []):
                    left = head * self.head_dim
                    right = (head + 1) * self.head_dim
                    if ablation_scope == "all_tokens":
                        modified[:, :, left:right] = 0
                    elif ablation_scope == "final_token":
                        modified[:, -1, left:right] = 0
                    else:
                        raise ValueError(f"Unknown ablation scope: {ablation_scope}")
                for head, source in by_layer_patch.get(layer_index, {}).items():
                    left = head * self.head_dim
                    right = (head + 1) * self.head_dim
                    modified[:, -1, left:right] = source.to(
                        device=modified.device,
                        dtype=modified.dtype,
                    )
                new_x = modified.squeeze(0) if squeeze else modified
                return (new_x,) + tuple(inputs[1:])

            handles.append(module.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def score_with_heads(
        self,
        prompt: str,
        item: RealLifeItem,
        ablate: Optional[set[HeadRef]] = None,
        patch: Optional[dict[HeadRef, torch.Tensor]] = None,
        capture: bool = False,
        ablation_scope: str = "all_tokens",
    ) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
        _, ids, _, _ = self.encode_with_offsets(prompt)
        captured: dict[int, torch.Tensor] = {}
        with self._head_hooks(
            ablate=ablate,
            patch=patch,
            capture=captured if capture else None,
            ablation_scope=ablation_scope,
        ):
            outputs = self.model(input_ids=ids, use_cache=False, return_dict=True)
        return self._result_from_logits(outputs.logits[0, -1], item), captured


# ---------------------------------------------------------------------------
# Candidate validation and selection
# ---------------------------------------------------------------------------

def review_map(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = validation.get("paraphrase_reviews", [])
    result = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("candidate_id"):
                result[str(row["candidate_id"])] = row
    return result


def automatic_surface_checks(original: str, candidate: str) -> dict[str, Any]:
    original_numbers = set(NUMBER_RE.findall(original.replace(",", "")))
    candidate_numbers = set(NUMBER_RE.findall(candidate.replace(",", "")))
    original_neg = bool(NEGATION_RE.search(original))
    candidate_neg = bool(NEGATION_RE.search(candidate))
    return {
        "numbers_exactly_preserved": original_numbers == candidate_numbers,
        "negation_presence_preserved": original_neg == candidate_neg,
        "token_f1": token_f1(original, candidate),
        "edit_ratio": edit_distance_ratio(original, candidate),
    }


def paraphrase_is_valid(
    review: dict[str, Any],
    auto: dict[str, Any],
    min_naturalness: int,
) -> bool:
    required = (
        "semantic_equivalent",
        "entities_preserved",
        "quantities_preserved",
        "negation_modality_preserved",
        "correct_answer_preserved",
        "no_new_constraint",
        "no_answer_leak",
    )
    return (
        all(bool(review.get(key)) for key in required)
        and int(review.get("naturalness", 0)) >= min_naturalness
        and bool(auto["numbers_exactly_preserved"])
        and bool(auto["negation_presence_preserved"])
    )


def link_is_valid(
    review: dict[str, Any],
    text: str,
    min_naturalness: int,
    item: RealLifeItem,
) -> bool:
    required = (
        "constraint_relation_correct",
        "facts_preserved",
        "no_direct_answer_or_option_number",
        "no_contradiction",
    )
    normalized_link = normalize_text(text)
    option_leak = any(
        len(normalize_text(option).split()) >= 1
        and normalize_text(option) in normalized_link
        for option in item.options
    )
    return (
        all(bool(review.get(key)) for key in required)
        and int(review.get("naturalness", 0)) >= min_naturalness
        and not DIRECT_ANSWER_RE.search(text)
        and not option_leak
    )


def choose_closest(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(float(row["prior_shortcut_margin"]) - target))


def select_candidate_conditions(
    valid_rows: list[dict[str, Any]],
    original_prior: float,
    original_surprisal: float,
    max_surprisal_increase: float,
) -> dict[str, dict[str, Any]]:
    if len(valid_rows) < 3:
        raise ValueError("At least three valid paraphrases are required for prior tiers.")
    ordered = sorted(valid_rows, key=lambda row: float(row["prior_shortcut_margin"]))
    low = ordered[0]
    high = ordered[-1]

    # Prefer distinct experimental cells whenever the candidate pool permits it.
    remaining_for_mid = [row for row in ordered if row is not low and row is not high]
    median_value = float(np.median([row["prior_shortcut_margin"] for row in ordered]))
    mid = choose_closest(remaining_for_mid or ordered, median_value)

    remaining_for_control = [
        row for row in ordered if row is not low and row is not high and row is not mid
    ]
    common = choose_closest(remaining_for_control or ordered, original_prior)

    pdp_pool = [
        row
        for row in ordered
        if float(row["surface_surprisal"]) <= original_surprisal + max_surprisal_increase
    ]
    if not pdp_pool:
        pdp_pool = ordered
    # Primary objective: reduce shortcut prior. Tie-break by smaller semantic edit
    # and less excessive surprisal.
    pdp = min(
        pdp_pool,
        key=lambda row: (
            float(row["prior_shortcut_margin"]),
            float(row["automatic_checks"]["edit_ratio"]),
            float(row["surface_surprisal"]),
        ),
    )
    return {
        "prior_low": low,
        "prior_mid": mid,
        "prior_high": high,
        "common_control": common,
        "pdp": pdp,
    }


# ---------------------------------------------------------------------------
# Stage 1: generate, validate, score, and select semantic counterfactuals
# ---------------------------------------------------------------------------

def prepare_item_selection(
    item: RealLifeItem,
    detection: Detection,
    editor: EditorLLM,
    target: TargetModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not detection.shortcut_text:
        raise ValueError("No predicted shortcut span.")
    shortcut_loc = locate_span(item.benchmark_prompt, detection.shortcut_text)
    shortcut_text = shortcut_loc.text

    constraint_text = detection.constraint_text
    constraint_source = detection.constraint_source
    if not constraint_text:
        if not args.editor_constraint_fallback:
            raise ValueError("No predicted constraint span and fallback disabled.")
        constraint_text = editor.infer_constraint(item, shortcut_text)
        constraint_source = "editor_fallback"
    constraint_loc = locate_span(item.benchmark_prompt, constraint_text)
    constraint_text = constraint_loc.text

    pack = editor.generate_pack(
        item,
        shortcut=shortcut_text,
        constraint=constraint_text,
        n_candidates=args.paraphrase_candidates,
    )
    validation = editor.validate_pack(item, shortcut_text, constraint_text, pack)
    reviews = review_map(validation)

    original_prior = target.prior_score(item, shortcut_text)
    original_surprisal = target.surface_surprisal(shortcut_text)
    original_full = target.score(item.benchmark_prompt, item)

    candidate_rows = []
    for row in pack["paraphrases"]:
        candidate_id = str(row["candidate_id"])
        candidate = str(row["text"]).strip()
        review = reviews.get(candidate_id, {})
        auto = automatic_surface_checks(shortcut_text, candidate)
        valid = paraphrase_is_valid(review, auto, args.min_naturalness)
        candidate_prompt, candidate_loc = replace_located(
            item.benchmark_prompt,
            shortcut_loc,
            candidate,
        )
        if not prompt_preserves_options(candidate_prompt, item):
            valid = False
        prior = target.prior_score(item, candidate)
        full = target.score(candidate_prompt, item)
        surprisal = target.surface_surprisal(candidate)
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "text": candidate,
                "valid": valid,
                "review": review,
                "automatic_checks": auto,
                "prior_shortcut_margin": prior["prior_shortcut_margin"],
                "prior_probe": prior,
                "surface_surprisal": surprisal,
                "full_prompt": full,
                "prompt": candidate_prompt,
                "located_span": asdict(candidate_loc),
            }
        )

    valid_rows = [row for row in candidate_rows if row["valid"]]
    selections = select_candidate_conditions(
        valid_rows,
        original_prior=float(original_prior["prior_shortcut_margin"]),
        original_surprisal=float(original_surprisal),
        max_surprisal_increase=args.max_surprisal_increase,
    )

    link = str(pack["contextual_key_link"]).strip()
    link_review = validation.get("contextual_key_link_review", {})
    link_valid = link_is_valid(link_review, link, args.min_naturalness, item)
    context_prompt = insert_after_located(item.benchmark_prompt, shortcut_loc, link)
    if not prompt_preserves_options(context_prompt, item):
        link_valid = False
    if not link_valid:
        raise ValueError("Contextual key-link sentence failed semantic/answer-leak validation.")
    context_full = target.score(context_prompt, item)

    pdp_row = selections["pdp"]
    pdp_prompt = pdp_row["prompt"]
    pdp_loc = LocatedSpan(**pdp_row["located_span"])
    joint_prompt = insert_after_located(pdp_prompt, pdp_loc, link)
    joint_full = target.score(joint_prompt, item)

    return {
        "item_index": item.index,
        "item_id": item.item_id,
        "split": detection.split,
        "gold": str(item.answer),
        "detector": {
            "hallucination_probability": detection.hallucination_probability,
            "predicted_hallucination": detection.predicted_hallucination,
            "shortcut_source": detection.shortcut_source,
            "constraint_source": constraint_source,
        },
        "shortcut": {
            "text": shortcut_text,
            "located": asdict(shortcut_loc),
        },
        "constraint": {
            "text": constraint_text,
            "located": asdict(constraint_loc),
        },
        "original": {
            "prompt": item.benchmark_prompt,
            "prior_probe": original_prior,
            "surface_surprisal": original_surprisal,
            "full_prompt": original_full,
        },
        "all_paraphrase_candidates": candidate_rows,
        "selected": {
            name: {
                key: value
                for key, value in row.items()
                if key not in {"review"}
            }
            for name, row in selections.items()
        },
        "contextual_key_link": {
            "text": link,
            "valid": link_valid,
            "review": link_review,
            "prompt": context_prompt,
            "full_prompt": context_full,
        },
        "joint_pdp_context": {
            "prompt": joint_prompt,
            "full_prompt": joint_full,
        },
    }


def run_prepare_stage(
    items: list[RealLifeItem],
    detections: dict[int, Detection],
    editor: EditorLLM,
    target: TargetModel,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    signature = stable_hash(
        {
            "version": 4,
            "target_model": args.target_model,
            "editor_model": args.editor_model,
            "only_split": args.only_split,
            "paraphrase_candidates": args.paraphrase_candidates,
            "min_naturalness": args.min_naturalness,
            "max_surprisal_increase": args.max_surprisal_increase,
            "allow_role_fallback": args.allow_role_fallback,
            "editor_constraint_fallback": args.editor_constraint_fallback,
        }
    )
    cache = ItemStageCache(output_dir, "prepare", signature)
    errors_path = output_dir / "errors_prepare.jsonl"

    universe_items = []
    for item in items:
        detection = detections.get(item.index)
        if not detection:
            continue
        if args.only_split != "all" and detection.split != args.only_split:
            continue
        universe_items.append((item, detection))
    if args.max_items > 0:
        universe_items = universe_items[: args.max_items]
    selected_items = [
        (item, detection)
        for item, detection in universe_items
        if detection.shortcut_text
    ]

    for item, detection in tqdm(selected_items, desc="Preparing semantic counterfactuals"):
        if not args.overwrite_cache:
            existing = cache.load(item)
            if existing is not None:
                continue
        try:
            result = prepare_item_selection(item, detection, editor, target, args)
            cache.save(item, result)
        except Exception as exc:
            traceback.print_exc()
            append_error(
                errors_path,
                {
                    "item_index": item.index,
                    "item_id": item.item_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    rows = cache.all_rows()
    write_jsonl(output_dir / "semantic_counterfactuals.jsonl", rows)
    return rows


def run_baseline_coverage_stage(
    items: list[RealLifeItem],
    detections: dict[int, Detection],
    prepared_rows: list[dict[str, Any]],
    target: TargetModel,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Score the full requested detector split, including non-localizable items.

    This prevents mitigation results from being reported only on the subset for
    which the detector successfully resolved a shortcut/constraint pair.
    """
    signature = stable_hash(
        {
            "version": 2,
            "target_model": args.target_model,
            "only_split": args.only_split,
            "max_items": args.max_items,
        }
    )
    cache = ItemStageCache(output_dir, "baseline_coverage", signature)
    prepared_by_index = {int(row["item_index"]): row for row in prepared_rows}

    universe: list[tuple[RealLifeItem, Detection]] = []
    for item in items:
        detection = detections.get(item.index)
        if detection is None:
            continue
        if args.only_split != "all" and detection.split != args.only_split:
            continue
        universe.append((item, detection))
    if args.max_items > 0:
        universe = universe[: args.max_items]

    for item, detection in tqdm(universe, desc="Scoring full-split baseline coverage"):
        if not args.overwrite_cache and cache.load(item) is not None:
            continue
        prepared = prepared_by_index.get(item.index)
        if prepared is not None:
            base_score = prepared["original"]["full_prompt"]
        else:
            base_score = target.score(item.benchmark_prompt, item)
        cache.save(
            item,
            {
                "item_index": item.index,
                "item_id": item.item_id,
                "split": detection.split,
                "gold": str(item.answer),
                "detector": {
                    "hallucination_probability": detection.hallucination_probability,
                    "predicted_hallucination": detection.predicted_hallucination,
                    "shortcut_resolved": detection.shortcut_text is not None,
                    "constraint_resolved": detection.constraint_text is not None,
                },
                "prepared": prepared is not None,
                "original": base_score,
            },
        )
    rows = cache.all_rows()
    write_jsonl(output_dir / "baseline_coverage.jsonl", rows)
    return rows


# ---------------------------------------------------------------------------
# Stage 2: internal causal validation
# ---------------------------------------------------------------------------

def select_mediated_heads(
    original_attention: dict[str, np.ndarray],
    pdp_attention: dict[str, np.ndarray],
    top_k: int,
    mode: str,
) -> tuple[list[HeadRef], list[dict[str, Any]]]:
    orig = original_attention["density"]
    low = pdp_attention["density"]
    if mode == "positive_difference":
        score = orig - low
    elif mode == "original_attention":
        score = orig
    elif mode == "difference_times_original":
        score = np.maximum(orig - low, 0.0) * orig
    else:
        raise ValueError(f"Unknown head score mode: {mode}")

    flat = []
    for layer in range(score.shape[0]):
        for head in range(score.shape[1]):
            flat.append(
                {
                    "layer": layer,
                    "head": head,
                    "score": float(score[layer, head]),
                    "original_density": float(orig[layer, head]),
                    "pdp_density": float(low[layer, head]),
                    "density_difference": float(orig[layer, head] - low[layer, head]),
                    "original_peak": float(original_attention["peak"][layer, head]),
                    "pdp_peak": float(pdp_attention["peak"][layer, head]),
                }
            )
    flat.sort(key=lambda row: row["score"], reverse=True)
    eligible = flat if mode == "original_attention" else [row for row in flat if row["score"] > 0]
    if not eligible:
        raise ValueError("No heads showed a positive original-to-PDP shortcut-attention reduction.")
    selected_rows = eligible[:top_k]
    refs = [HeadRef(int(row["layer"]), int(row["head"])) for row in selected_rows]
    return refs, selected_rows


def layer_matched_random_heads(
    selected: Sequence[HeadRef],
    num_heads: int,
    seed: int,
) -> set[HeadRef]:
    rng = random.Random(seed)
    selected_set = set(selected)
    counts: dict[int, int] = defaultdict(int)
    for ref in selected:
        counts[ref.layer] += 1
    result: set[HeadRef] = set()
    for layer, count in counts.items():
        candidates = [
            HeadRef(layer, head)
            for head in range(num_heads)
            if HeadRef(layer, head) not in selected_set
        ]
        if len(candidates) < count:
            candidates = [HeadRef(layer, head) for head in range(num_heads)]
        result.update(rng.sample(candidates, k=min(count, len(candidates))))
    return result


def build_patch(
    refs: Sequence[HeadRef],
    captured: dict[int, torch.Tensor],
) -> dict[HeadRef, torch.Tensor]:
    patch = {}
    for ref in refs:
        if ref.layer not in captured:
            raise RuntimeError(f"Missing captured layer {ref.layer}")
        patch[ref] = captured[ref.layer][ref.head].clone()
    return patch


def run_internal_item(
    item: RealLifeItem,
    row: dict[str, Any],
    target: TargetModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    original_prompt = row["original"]["prompt"]
    shortcut_original = LocatedSpan(**row["shortcut"]["located"])
    pdp = row["selected"]["pdp"]
    pdp_prompt = pdp["prompt"]
    pdp_span = LocatedSpan(**pdp["located_span"])

    orig_attention = target.head_attention_to_span(original_prompt, shortcut_original)
    pdp_attention = target.head_attention_to_span(pdp_prompt, pdp_span)
    selected_heads, selected_head_rows = select_mediated_heads(
        orig_attention,
        pdp_attention,
        top_k=args.top_heads,
        mode=args.head_score_mode,
    )
    selected_set = set(selected_heads)

    original_base, original_states = target.score_with_heads(
        original_prompt,
        item,
        capture=True,
    )
    pdp_base, pdp_states = target.score_with_heads(
        pdp_prompt,
        item,
        capture=True,
    )
    original_ablation, _ = target.score_with_heads(
        original_prompt,
        item,
        ablate=selected_set,
        ablation_scope=args.ablation_scope,
    )

    random_runs = []
    for random_index in range(args.random_head_runs):
        random_heads = layer_matched_random_heads(
            selected_heads,
            target.num_heads,
            seed=args.seed + 1000 * item.index + random_index,
        )
        random_score, _ = target.score_with_heads(
            original_prompt,
            item,
            ablate=random_heads,
            ablation_scope=args.ablation_scope,
        )
        random_runs.append(
            {
                "seed_index": random_index,
                "heads": [asdict(ref) for ref in sorted(random_heads, key=lambda x: (x.layer, x.head))],
                "score": random_score,
                "ablation_effect": random_score["correct_margin"] - original_base["correct_margin"],
            }
        )

    original_patch = build_patch(selected_heads, original_states)
    pdp_patch = build_patch(selected_heads, pdp_states)
    pdp_with_original, _ = target.score_with_heads(
        pdp_prompt,
        item,
        patch=original_patch,
    )
    original_with_pdp, _ = target.score_with_heads(
        original_prompt,
        item,
        patch=pdp_patch,
    )

    total_paraphrase_effect = pdp_base["correct_margin"] - original_base["correct_margin"]
    rescue_effect = pdp_with_original["correct_margin"] - pdp_base["correct_margin"]
    suppression_effect = original_with_pdp["correct_margin"] - original_base["correct_margin"]
    denominator = total_paraphrase_effect
    mediation_fraction = (
        float((pdp_base["correct_margin"] - pdp_with_original["correct_margin"]) / denominator)
        if abs(denominator) > 1e-6
        else None
    )

    return {
        "item_index": item.index,
        "item_id": item.item_id,
        "gold": str(item.answer),
        "detector": row["detector"],
        "selected_heads": selected_head_rows,
        "head_score_mode": args.head_score_mode,
        "ablation_scope": args.ablation_scope,
        "original": original_base,
        "pdp": pdp_base,
        "target_head_ablation": {
            "score": original_ablation,
            "effect_on_correct_margin": original_ablation["correct_margin"] - original_base["correct_margin"],
        },
        "random_matched_ablation": random_runs,
        "activation_patching": {
            "pdp_with_original_shortcut_head_states": pdp_with_original,
            "original_with_pdp_head_states": original_with_pdp,
            "rescue_effect_on_correct_margin": rescue_effect,
            "suppression_effect_on_correct_margin": suppression_effect,
            "total_paraphrase_effect_on_correct_margin": total_paraphrase_effect,
            "mediation_fraction_proxy": mediation_fraction,
        },
    }


def run_internal_stage(
    items_by_index: dict[int, RealLifeItem],
    prepared_rows: list[dict[str, Any]],
    target: TargetModel,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    signature = stable_hash(
        {
            "version": 4,
            "target_model": args.target_model,
            "top_heads": args.top_heads,
            "head_score_mode": args.head_score_mode,
            "random_head_runs": args.random_head_runs,
            "ablation_scope": args.ablation_scope,
            "internal_subset": args.internal_subset,
        }
    )
    cache = ItemStageCache(output_dir, "internal", signature)
    errors_path = output_dir / "errors_internal.jsonl"

    candidates = []
    for row in prepared_rows:
        item = items_by_index.get(int(row["item_index"]))
        if item is None:
            continue
        original_correct = bool(row["original"]["full_prompt"]["is_correct"])
        pdp_correct = bool(row["selected"]["pdp"]["full_prompt"]["is_correct"])
        triggered = bool(row["detector"]["predicted_hallucination"])
        if args.internal_subset == "triggered" and not triggered:
            continue
        if args.internal_subset == "base_wrong" and original_correct:
            continue
        if args.internal_subset == "repaired" and not (not original_correct and pdp_correct):
            continue
        candidates.append((item, row))
    if args.internal_max_items > 0:
        candidates = candidates[: args.internal_max_items]

    for item, row in tqdm(candidates, desc="Running head ablation and activation rescue"):
        if not args.overwrite_cache:
            existing = cache.load(item)
            if existing is not None:
                continue
        try:
            result = run_internal_item(item, row, target, args)
            cache.save(item, result)
        except Exception as exc:
            traceback.print_exc()
            append_error(
                errors_path,
                {
                    "item_index": item.index,
                    "item_id": item.item_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    rows = cache.all_rows()
    write_jsonl(output_dir / "internal_causal_validation.jsonl", rows)
    return rows


# ---------------------------------------------------------------------------
# Summaries: semantic counterfactuals, mitigation, and internal validation
# ---------------------------------------------------------------------------

def condition_from_prepared(row: dict[str, Any], condition: str) -> dict[str, Any]:
    if condition == "original":
        return row["original"]["full_prompt"]
    if condition in {"prior_low", "prior_mid", "prior_high", "common_control", "pdp"}:
        return row["selected"][condition]["full_prompt"]
    if condition == "context_link":
        return row["contextual_key_link"]["full_prompt"]
    if condition == "joint":
        return row["joint_pdp_context"]["full_prompt"]
    raise KeyError(condition)


def outcome_metrics(
    prepared_rows: list[dict[str, Any]],
    condition: str,
    gated: bool = False,
) -> dict[str, Any]:
    scores = []
    for row in prepared_rows:
        base = condition_from_prepared(row, "original")
        use_intervention = (
            not gated
            or bool(row["detector"]["predicted_hallucination"])
        )
        result = condition_from_prepared(row, condition) if use_intervention else base
        scores.append(
            {
                "base_correct": bool(base["is_correct"]),
                "is_correct": bool(result["is_correct"]),
                "margin": float(result["correct_margin"]),
                "base_margin": float(base["correct_margin"]),
                "triggered": bool(row["detector"]["predicted_hallucination"]),
            }
        )
    if not scores:
        return {"n": 0}
    wrong_to_correct = sum(not x["base_correct"] and x["is_correct"] for x in scores)
    correct_to_wrong = sum(x["base_correct"] and not x["is_correct"] for x in scores)
    return {
        "n": len(scores),
        "accuracy": float(np.mean([x["is_correct"] for x in scores])),
        "mean_correct_margin": float(np.mean([x["margin"] for x in scores])),
        "mean_margin_change": float(np.mean([x["margin"] - x["base_margin"] for x in scores])),
        "wrong_to_correct": int(wrong_to_correct),
        "correct_to_wrong": int(correct_to_wrong),
        "net_corrections": int(wrong_to_correct - correct_to_wrong),
        "trigger_rate": float(np.mean([x["triggered"] for x in scores])),
    }


def outcome_metrics_full_split(
    baseline_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    condition: str,
    gated: bool,
) -> dict[str, Any]:
    """Evaluate mitigation over the complete requested detector split.

    Items without a valid localized intervention, and detector-negative items in
    gated mode, retain their original output.
    """
    prepared_by_index = {int(row["item_index"]): row for row in prepared_rows}
    scores = []
    for base_row in baseline_rows:
        item_index = int(base_row["item_index"])
        base = base_row["original"]
        prepared = prepared_by_index.get(item_index)
        trigger = bool(base_row["detector"]["predicted_hallucination"])
        use_intervention = prepared is not None and (not gated or trigger)
        result = condition_from_prepared(prepared, condition) if use_intervention else base
        scores.append(
            {
                "base_correct": bool(base["is_correct"]),
                "is_correct": bool(result["is_correct"]),
                "margin": float(result["correct_margin"]),
                "base_margin": float(base["correct_margin"]),
                "triggered": trigger,
                "localizable": prepared is not None,
                "intervened": use_intervention,
            }
        )
    if not scores:
        return {"n": 0}
    wrong_to_correct = sum(not x["base_correct"] and x["is_correct"] for x in scores)
    correct_to_wrong = sum(x["base_correct"] and not x["is_correct"] for x in scores)
    return {
        "n": len(scores),
        "accuracy": float(np.mean([x["is_correct"] for x in scores])),
        "base_accuracy": float(np.mean([x["base_correct"] for x in scores])),
        "mean_correct_margin": float(np.mean([x["margin"] for x in scores])),
        "mean_margin_change": float(np.mean([x["margin"] - x["base_margin"] for x in scores])),
        "wrong_to_correct": int(wrong_to_correct),
        "correct_to_wrong": int(correct_to_wrong),
        "net_corrections": int(wrong_to_correct - correct_to_wrong),
        "trigger_rate": float(np.mean([x["triggered"] for x in scores])),
        "localization_coverage": float(np.mean([x["localizable"] for x in scores])),
        "actual_intervention_rate": float(np.mean([x["intervened"] for x in scores])),
    }


def semantic_counterfactual_summary(
    prepared_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    condition_names = (
        "original",
        "common_control",
        "prior_low",
        "prior_mid",
        "prior_high",
        "pdp",
        "context_link",
        "joint",
    )
    condition_metrics = {
        condition: outcome_metrics(prepared_rows, condition)
        for condition in condition_names
    }

    paired = {}
    for left, right in (
        ("prior_low", "prior_high"),
        ("prior_low", "common_control"),
        ("pdp", "original"),
        ("context_link", "original"),
        ("joint", "original"),
    ):
        x = [condition_from_prepared(row, left)["correct_margin"] for row in prepared_rows]
        y = [condition_from_prepared(row, right)["correct_margin"] for row in prepared_rows]
        paired[f"{left}_minus_{right}"] = bootstrap_mean_difference(
            x, y, draws=args.bootstrap_draws, seed=args.seed
        )

    # Within-item centered relationship across all valid paraphrases.
    prior_centered = []
    margin_centered = []
    surprisal_centered = []
    prior_margin_groups: list[tuple[np.ndarray, np.ndarray]] = []
    surprisal_prior_groups: list[tuple[np.ndarray, np.ndarray]] = []
    for row in prepared_rows:
        candidates = [x for x in row["all_paraphrase_candidates"] if x["valid"]]
        if len(candidates) < 3:
            continue
        prior = np.asarray([x["prior_shortcut_margin"] for x in candidates], dtype=float)
        margin = np.asarray([x["full_prompt"]["correct_margin"] for x in candidates], dtype=float)
        surprisal = np.asarray([x["surface_surprisal"] for x in candidates], dtype=float)
        prior_centered.extend((prior - prior.mean()).tolist())
        margin_centered.extend((margin - margin.mean()).tolist())
        surprisal_centered.extend((surprisal - surprisal.mean()).tolist())
        prior_margin_groups.append((prior, margin))
        surprisal_prior_groups.append((surprisal, prior))
    if len(prior_centered) >= 3:
        rho_prior, p_prior = spearmanr(prior_centered, margin_centered)
        rho_surprisal, p_surprisal = spearmanr(surprisal_centered, prior_centered)
    else:
        rho_prior = p_prior = rho_surprisal = p_surprisal = float("nan")

    mitigation = {
        "localized_subset": {
            "always_on": {
                "pdp": outcome_metrics(prepared_rows, "pdp", gated=False),
                "context_link": outcome_metrics(prepared_rows, "context_link", gated=False),
                "joint": outcome_metrics(prepared_rows, "joint", gated=False),
            },
            "detector_gated": {
                "pdp": outcome_metrics(prepared_rows, "pdp", gated=True),
                "context_link": outcome_metrics(prepared_rows, "context_link", gated=True),
                "joint": outcome_metrics(prepared_rows, "joint", gated=True),
            },
        },
        "full_requested_split": {
            "always_on_where_localizable": {
                "pdp": outcome_metrics_full_split(baseline_rows, prepared_rows, "pdp", gated=False),
                "context_link": outcome_metrics_full_split(baseline_rows, prepared_rows, "context_link", gated=False),
                "joint": outcome_metrics_full_split(baseline_rows, prepared_rows, "joint", gated=False),
            },
            "detector_gated": {
                "pdp": outcome_metrics_full_split(baseline_rows, prepared_rows, "pdp", gated=True),
                "context_link": outcome_metrics_full_split(baseline_rows, prepared_rows, "context_link", gated=True),
                "joint": outcome_metrics_full_split(baseline_rows, prepared_rows, "joint", gated=True),
            },
        },
    }
    return {
        "condition_metrics": condition_metrics,
        "paired_margin_effects": paired,
        "within_item_frequency_relationship": {
            "n_candidate_points": len(prior_centered),
            "spearman_prior_shortcut_margin_vs_full_correct_margin": float(rho_prior),
            "naive_pointwise_p_value": float(p_prior),
            "expected_direction": "negative",
            "fixed_effect_slope_prior_to_correct_margin": clustered_within_item_slope(
                prior_margin_groups,
                draws=args.bootstrap_draws,
                seed=args.seed,
            ),
            "spearman_surface_surprisal_vs_prior_shortcut_margin": float(rho_surprisal),
            "naive_surface_surprisal_p_value": float(p_surprisal),
            "fixed_effect_slope_surprisal_to_prior": clustered_within_item_slope(
                surprisal_prior_groups,
                draws=args.bootstrap_draws,
                seed=args.seed + 17,
            ),
        },
        "mitigation": mitigation,
    }


def internal_summary(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    target_effect = np.asarray(
        [row["target_head_ablation"]["effect_on_correct_margin"] for row in rows],
        dtype=float,
    )
    random_item_means = np.asarray(
        [
            np.mean([run["ablation_effect"] for run in row["random_matched_ablation"]])
            for row in rows
        ],
        dtype=float,
    )
    rescue = np.asarray(
        [row["activation_patching"]["rescue_effect_on_correct_margin"] for row in rows],
        dtype=float,
    )
    suppression = np.asarray(
        [row["activation_patching"]["suppression_effect_on_correct_margin"] for row in rows],
        dtype=float,
    )
    paraphrase = np.asarray(
        [row["activation_patching"]["total_paraphrase_effect_on_correct_margin"] for row in rows],
        dtype=float,
    )
    fractions = [
        row["activation_patching"]["mediation_fraction_proxy"]
        for row in rows
        if row["activation_patching"]["mediation_fraction_proxy"] is not None
        and np.isfinite(row["activation_patching"]["mediation_fraction_proxy"])
    ]
    return {
        "n": len(rows),
        "target_head_ablation": {
            "mean_effect_on_correct_margin": float(target_effect.mean()),
            "directional_success_rate_effect_positive": float(np.mean(target_effect > 0)),
        },
        "layer_matched_random_ablation": {
            "mean_effect_on_correct_margin": float(random_item_means.mean()),
        },
        "target_minus_random": bootstrap_mean_difference(
            target_effect,
            random_item_means,
            draws=args.bootstrap_draws,
            seed=args.seed,
        ),
        "activation_rescue": {
            "mean_original_into_pdp_effect": float(rescue.mean()),
            "expected_direction": "negative",
            "directional_success_rate": float(np.mean(rescue < 0)),
        },
        "activation_suppression": {
            "mean_pdp_into_original_effect": float(suppression.mean()),
            "expected_direction": "positive",
            "directional_success_rate": float(np.mean(suppression > 0)),
        },
        "mean_total_paraphrase_effect": float(paraphrase.mean()),
        "mean_mediation_fraction_proxy": float(np.mean(fractions)) if fractions else None,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    semantic = summary["semantic_counterfactual_and_mitigation"]
    internal = summary["internal_causal_validation"]
    lines = [
        "# KeyShift RealLifeQA Experiment Summary",
        "",
        f"- Prepared items: {summary['n_prepared_items']}",
        f"- Internal causal items: {internal.get('n', 0)}",
        "",
        "## Frequency-controlled semantic counterfactuals",
        "",
        "| Condition | Accuracy | Mean correct margin | Margin change | W→C | C→W |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in semantic["condition_metrics"].items():
        if metrics.get("n", 0) == 0:
            continue
        lines.append(
            f"| {condition} | {metrics['accuracy']:.4f} | "
            f"{metrics['mean_correct_margin']:.4f} | {metrics['mean_margin_change']:.4f} | "
            f"{metrics['wrong_to_correct']} | {metrics['correct_to_wrong']} |"
        )
    rel = semantic["within_item_frequency_relationship"]
    lines.extend(
        [
            "",
            "## Frequency relationship",
            "",
            f"- Within-item Spearman(prior shortcut margin, full correct margin): "
            f"{rel['spearman_prior_shortcut_margin_vs_full_correct_margin']:.4f} "
            f"(naive p={rel['naive_pointwise_p_value']:.4g}; expected negative).",
            f"- Item-fixed-effect slope: {rel['fixed_effect_slope_prior_to_correct_margin']['slope']} "
            f"with cluster-bootstrap CI {rel['fixed_effect_slope_prior_to_correct_margin']['bootstrap_95_ci']}.",
            "",
            "## Detector-gated mitigation",
            "",
            "| Method | Accuracy | Net corrections | Trigger rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, metrics in semantic["mitigation"]["full_requested_split"]["detector_gated"].items():
        lines.append(
            f"| {method} | {metrics['accuracy']:.4f} | {metrics['net_corrections']} | "
            f"{metrics['trigger_rate']:.4f} |"
        )
    lines.extend(["", "## Internal causal validation", ""])
    if internal.get("n", 0):
        lines.extend(
            [
                f"- Target-head ablation mean effect: "
                f"{internal['target_head_ablation']['mean_effect_on_correct_margin']:.4f}.",
                f"- Layer-matched random ablation mean effect: "
                f"{internal['layer_matched_random_ablation']['mean_effect_on_correct_margin']:.4f}.",
                f"- Original-state rescue directional success: "
                f"{internal['activation_rescue']['directional_success_rate']:.4f}.",
                f"- PDP-state suppression directional success: "
                f"{internal['activation_suppression']['directional_success_rate']:.4f}.",
                f"- Mean mediation fraction proxy: {internal['mean_mediation_fraction_proxy']}.",
            ]
        )
    else:
        lines.append("No internal causal rows were available.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The target-model prior probe operationalizes shortcut association; it is not a direct measurement of the inaccessible pretraining corpus frequency.",
            "- LLM-generated paraphrases are retained only after semantic and answer-preservation audits and target-model prior scoring.",
            "- Activation-patching mediation fractions are mechanistic proxies, not identifiable natural indirect effects.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_summary_stage(
    prepared_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    internal_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    semantic = semantic_counterfactual_summary(prepared_rows, baseline_rows, args)
    internal = internal_summary(internal_rows, args)
    summary = {
        "method": "KeyShift frequency-controlled key-selection causal validation and mitigation",
        "n_prepared_items": len(prepared_rows),
        "n_full_split_items": len(baseline_rows),
        "localization_coverage": (len(prepared_rows) / len(baseline_rows)) if baseline_rows else None,
        "semantic_counterfactual_and_mitigation": semantic,
        "internal_causal_validation": internal,
        "configuration": vars(args),
        "files": {
            "semantic_counterfactuals": str(output_dir / "semantic_counterfactuals.jsonl"),
            "baseline_coverage": str(output_dir / "baseline_coverage.jsonl"),
            "internal_causal_validation": str(output_dir / "internal_causal_validation.jsonl"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_markdown": str(output_dir / "summary.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )

    # Compact item-condition table for statistical analysis in R/Python.
    flat_rows = []
    for row in prepared_rows:
        for condition in (
            "original",
            "common_control",
            "prior_low",
            "prior_mid",
            "prior_high",
            "pdp",
            "context_link",
            "joint",
        ):
            result = condition_from_prepared(row, condition)
            if condition == "original":
                prior = row["original"]["prior_probe"]["prior_shortcut_margin"]
                text = row["shortcut"]["text"]
            elif condition in {"common_control", "prior_low", "prior_mid", "prior_high", "pdp"}:
                prior = row["selected"][condition]["prior_shortcut_margin"]
                text = row["selected"][condition]["text"]
            else:
                prior = None
                text = row["contextual_key_link"]["text"] if condition == "context_link" else "PDP + CKL"
            flat_rows.append(
                {
                    "item_index": row["item_index"],
                    "item_id": row["item_id"],
                    "condition": condition,
                    "text": text,
                    "prior_shortcut_margin": prior,
                    "correct_margin": result["correct_margin"],
                    "prediction": result["prediction"],
                    "is_correct": result["is_correct"],
                    "detector_probability": row["detector"]["hallucination_probability"],
                    "detector_trigger": row["detector"]["predicted_hallucination"],
                }
            )
    write_csv(output_dir / "condition_results.csv", flat_rows)
    return summary


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="KeyShift RealLifeQA causal validation and mitigation experiment.",
    )
    parser.add_argument("stage", choices=("prepare", "internal", "summarize", "all"))

    data = parser.add_argument_group("data")
    data.add_argument("--input", required=True, type=Path)
    data.add_argument("--detector-predictions", required=True, type=Path)
    data.add_argument("--output-dir", required=True, type=Path)
    data.add_argument("--only-split", choices=("train", "test", "all"), default="test")
    data.add_argument("--max-items", type=int, default=0)
    data.add_argument("--allow-role-fallback", action=argparse.BooleanOptionalAction, default=False)
    data.add_argument("--editor-constraint-fallback", action=argparse.BooleanOptionalAction, default=False)

    target = parser.add_argument_group("local target model")
    target.add_argument(
        "--target-model",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    target.add_argument("--device", default="cuda")
    target.add_argument("--dtype", default="bfloat16")
    target.add_argument("--trust-remote-code", action="store_true")

    editor = parser.add_argument_group("local editor LLM")
    editor.add_argument(
        "--editor-model", default=None,
        help="Local Hugging Face model/path; defaults to --target-model and reuses its weights.",
    )
    editor.add_argument("--editor-temperature", type=float, default=0.7)
    editor.add_argument("--editor-max-retries", type=int, default=3)
    editor.add_argument("--paraphrase-candidates", type=int, default=10)
    editor.add_argument("--min-naturalness", type=int, default=4)
    editor.add_argument("--max-surprisal-increase", type=float, default=2.0)

    internal = parser.add_argument_group("internal causal validation")
    internal.add_argument("--top-heads", type=int, default=8)
    internal.add_argument(
        "--head-score-mode",
        choices=("positive_difference", "original_attention", "difference_times_original"),
        default="difference_times_original",
    )
    internal.add_argument("--random-head-runs", type=int, default=5)
    internal.add_argument(
        "--ablation-scope",
        choices=("all_tokens", "final_token"),
        default="all_tokens",
        help="Zero-ablation scope. Activation patching always patches the final decision token.",
    )
    internal.add_argument(
        "--internal-subset",
        choices=("all", "triggered", "base_wrong", "repaired"),
        default="triggered",
    )
    internal.add_argument("--internal-max-items", type=int, default=100)

    misc = parser.add_argument_group("reproducibility")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--bootstrap-draws", type=int, default=5000)
    misc.add_argument("--overwrite-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_config.json", vars(args))

    items = load_items(args.input)
    items_by_index = {item.index: item for item in items}
    detections = load_detections(
        args.detector_predictions,
        allow_role_fallback=args.allow_role_fallback,
    )

    prepared_path = args.output_dir / "semantic_counterfactuals.jsonl"
    baseline_path = args.output_dir / "baseline_coverage.jsonl"
    internal_path = args.output_dir / "internal_causal_validation.jsonl"
    target_model: Optional[TargetModel] = None

    if args.stage in {"prepare", "all"}:
        target_model = TargetModel(
            model_name=args.target_model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        editor_model_name = args.editor_model or args.target_model
        if editor_model_name == args.target_model:
            editor_backend = target_model
        else:
            editor_backend = TargetModel(
                model_name=editor_model_name, device=args.device, dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
        editor = EditorLLM(
            backend=editor_backend,
            model_name=editor_model_name,
            cache=JsonCache(args.output_dir / "editor_cache.json"),
            temperature=args.editor_temperature,
            max_retries=args.editor_max_retries,
        )
        prepared_rows = run_prepare_stage(
            items,
            detections,
            editor,
            target_model,
            args,
            args.output_dir,
        )
        baseline_rows = run_baseline_coverage_stage(
            items,
            detections,
            prepared_rows,
            target_model,
            args,
            args.output_dir,
        )
    else:
        if not prepared_path.exists():
            raise FileNotFoundError(
                f"{prepared_path} does not exist. Run the prepare stage first."
            )
        prepared_rows = read_jsonl(prepared_path)
        if baseline_path.exists():
            baseline_rows = read_jsonl(baseline_path)
        elif args.stage == "summarize":
            raise FileNotFoundError(
                f"{baseline_path} does not exist. Re-run the prepare stage with this version."
            )
        else:
            baseline_rows = []

    if args.stage in {"internal", "all"}:
        if target_model is None:
            target_model = TargetModel(
                model_name=args.target_model,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
        internal_rows = run_internal_stage(
            items_by_index,
            prepared_rows,
            target_model,
            args,
            args.output_dir,
        )
    else:
        internal_rows = read_jsonl(internal_path) if internal_path.exists() else []

    if args.stage in {"summarize", "all"}:
        summary = run_summary_stage(
            prepared_rows,
            baseline_rows,
            internal_rows,
            args,
            args.output_dir,
        )
        print(json.dumps(summary["semantic_counterfactual_and_mitigation"]["mitigation"], indent=2))
        print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
