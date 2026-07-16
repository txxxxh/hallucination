#!/usr/bin/env python3
"""
Gold-blind two-stage hallucination detector for the scientist-profile dataset.

Three-way answer protocol
-------------------------
1 = the first profile
2 = the second profile
3 = the remaining description does not uniquely identify one profile

Decision rule
-------------
1. Segment only the question description between QUESTION_START_MARKER and
   QUESTION_END_MARKER.
2. Delete every span and obtain a three-way answer.
3. A span enters the negation stage when deletion either:
   a) produces answer 3 (uncertain / not uniquely identifiable), or
   b) flips from one named profile to the other named profile.
4. Negate every span selected in step 3.
5. If at least one valid negation keeps the original named-profile answer
   (negation_flip=False), predict hallucination.
6. If no deletion is informative, predict non-hallucination.
7. If all informative deletion spans are validly negated and every negation
   changes the original answer, predict non-hallucination. Otherwise abstain.

The detector never receives wrg_ans or rgt_ans. Gold answers are joined only
after all detector predictions have been completed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# These project-local modules are expected to be in the same directory as this
# script, as in the previous experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_spans import segment_scenario  # noqa: E402
import run_reallifeqa_pilot as pilot  # noqa: E402


QUESTION_START_MARKER = (
    "output the name of the person as the answer to the following question:\n"
)
NAMES_QUESTION_START_MARKER = "Question:\n"
QUESTION_END_MARKER = "Who is this person?"
NEGATOR_SYSTEM = (
    "You minimally negate one specified proposition in a sentence and return "
    "only valid JSON."
)
# Reasoning models may consume completion tokens internally before emitting the
# requested single choice token, so a limit of 2 is not sufficient.
CHOICE_MAX_COMPLETION_TOKENS = 1024

# Explicitly forbidden from entering the detector.
GOLD_FIELDS = frozenset(
    {
        "wrg_ans",
        "wrg_ans_qid",
        "rgt_ans",
        "rgt_ans_qid",
        "answer",
        "correct_option",
        "shortcut_option",
        "gold_label",
    }
)


@dataclass(frozen=True)
class GlobalSpan:
    """A span whose offsets refer to the complete binary-evaluation prompt."""

    index: int
    text: str
    start: int
    end: int
    local_start: int
    local_end: int


def _normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def _choice(value: Any) -> Optional[str]:
    """Normalize a model answer to one of the three allowed choices."""
    text = str(value).strip() if value is not None else ""
    match = re.search(r"(?<!\d)([123])(?!\d)", text)
    return match.group(1) if match else None


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON list, a common JSON container, or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL line {line_number} is not an object: {type(value).__name__}"
                )
            records.append(value)
        return records

    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError("The JSON list must contain only objects.")
        return list(obj)

    if isinstance(obj, dict):
        for key in ("data", "items", "questions", "records"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return list(value)
        # Also support a dictionary keyed by question IDs.
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())

    raise ValueError(
        "Unsupported input format. Expected a JSON list, JSONL, or a dictionary "
        "containing data/items/questions/records."
    )


def locate_question_region(prompt: str) -> Tuple[int, int, str]:
    """Return the exact intervention region requested by the experiment."""
    marker = QUESTION_START_MARKER
    marker_pos = prompt.find(marker)
    if marker_pos < 0:
        marker = NAMES_QUESTION_START_MARKER
        marker_pos = prompt.find(marker)
    if marker_pos < 0:
        raise ValueError(
            "Question start marker not found; expected either "
            + repr(QUESTION_START_MARKER) + " or "
            + repr(NAMES_QUESTION_START_MARKER)
        )
    start = marker_pos + len(marker)

    end = prompt.find(QUESTION_END_MARKER, start)
    if end < 0:
        raise ValueError(
            "Question end marker not found after start marker: "
            + repr(QUESTION_END_MARKER)
        )
    if end <= start:
        raise ValueError("The question intervention region is empty.")
    return start, end, prompt[start:end]


def extract_profile_names(prompt: str) -> Tuple[str, str]:
    """Extract the first and second profile names without using gold fields."""
    marker_pos = prompt.find(QUESTION_START_MARKER)
    if marker_pos < 0:
        marker_pos = prompt.find(NAMES_QUESTION_START_MARKER)
    prefix = prompt[:marker_pos] if marker_pos >= 0 else prompt
    names = [x.strip() for x in re.findall(r"(?m)^name:\s*(.+?)\s*$", prefix)]
    if len(names) < 2:
        numbered = {
            int(number): name.strip()
            for number, name in re.findall(r"(?m)^\s*([12])\.\s*(.+?)\s*$", prefix)
        }
        if 1 in numbered and 2 in numbered:
            names = [numbered[1], numbered[2]]
    if len(names) < 2:
        raise ValueError(
            f"Expected at least two profile names before the question; found {len(names)}."
        )
    return names[0], names[1]


def build_evaluation_prompt(
    prompt: str, profile_1: str, profile_2: str, allow_uncertain: bool
) -> str:
    """
    Append a unified 1/2/3 answer protocol to every prompt variant.

    The source dataset says "Choose exactly one profile", which would forbid the
    uncertainty response needed after deletion. Replace only that instruction;
    the question passage and the intervention markers remain unchanged.
    """
    normalized_prompt = prompt.replace(
        "Choose exactly one profile from the two, and ",
        "Follow the evaluation answer protocol below, and ",
        1,
    )
    normalized_prompt = re.sub(
        r"\AChoose one of the following two options as the answer to the question below:\s*",
        "Follow the evaluation answer protocol below.\n",
        normalized_prompt,
        count=1,
    )
    protocol = (
        "\n\nEvaluation answer protocol:\n"
        f"1 = {profile_1}\n"
        f"2 = {profile_2}\n"
    )
    if allow_uncertain:
        protocol += (
            "3 = the information in the question does not uniquely identify one "
            "of the two people\nOutput only one digit: 1, 2, or 3."
        )
    else:
        protocol += "You must choose one of the two named people. Output only 1 or 2; never output 3."
    return normalized_prompt.rstrip() + protocol


def build_ternary_prompt(prompt: str, profile_1: str, profile_2: str) -> str:
    return build_evaluation_prompt(prompt, profile_1, profile_2, True)


def make_global_spans(
    prompt: str,
    min_clause_words: int,
    min_span_words: int,
    max_span_words: int,
) -> Tuple[List[GlobalSpan], int, int, str]:
    """Segment only the requested question passage and shift offsets globally."""
    region_start, region_end, region_text = locate_question_region(prompt)
    local_spans = segment_scenario(
        region_text,
        min_clause_words,
        min_span_words,
        max_span_words,
    )

    spans: List[GlobalSpan] = []
    for new_index, span in enumerate(local_spans):
        local_start = int(span.start)
        local_end = int(span.end)
        if not (0 <= local_start < local_end <= len(region_text)):
            raise ValueError(
                f"Invalid local span offsets: {local_start}:{local_end} "
                f"for region length {len(region_text)}"
            )
        text = region_text[local_start:local_end]
        spans.append(
            GlobalSpan(
                index=new_index,
                text=text,
                start=region_start + local_start,
                end=region_start + local_end,
                local_start=local_start,
                local_end=local_end,
            )
        )
    return spans, region_start, region_end, region_text


def delete_global_span(prompt: str, span: GlobalSpan) -> str:
    """Delete exactly one span while preserving all text outside its offsets."""
    left = prompt[: span.start]
    right = prompt[span.end :]

    # Avoid joining two ordinary words directly. Apart from this boundary repair,
    # the prompt is unchanged.
    separator = ""
    if left and right:
        left_char = left[-1]
        right_char = right[0]
        if (
            not left_char.isspace()
            and not right_char.isspace()
            and right_char not in ".,;:!?)]}"
            and left_char not in "([{"
        ):
            separator = " "
    return left + separator + right


def containing_sentence(
    prompt: str,
    span: GlobalSpan,
    region_start: int,
    region_end: int,
) -> Tuple[int, int, str]:
    """Find the containing sentence without crossing the intervention region."""
    punctuation = (".", "?", "!", "\n")
    previous = [prompt.rfind(p, region_start, span.start) for p in punctuation]
    sentence_start = max(previous) + 1
    sentence_start = max(sentence_start, region_start)

    following = [
        position
        for position in (prompt.find(p, span.end, region_end) for p in punctuation)
        if position >= 0
    ]
    sentence_end = min(following) + 1 if following else region_end

    while sentence_start < sentence_end and prompt[sentence_start].isspace():
        sentence_start += 1
    return sentence_start, sentence_end, prompt[sentence_start:sentence_end]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a plain dictionary."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _completion_create(client: Any, **kwargs: Any) -> Any:
    """Call an OpenAI-compatible chat endpoint with a token-limit fallback."""
    attempt = dict(kwargs)
    for _ in range(3):
        try:
            return client.chat.completions.create(**attempt)
        except Exception as exc:
            message = str(exc)
            if "max_tokens" in attempt and (
                "max_completion_tokens" in message or isinstance(exc, TypeError)
            ):
                attempt["max_completion_tokens"] = attempt.pop("max_tokens")
                continue
            if "reasoning_effort" in attempt and (
                "reasoning_effort" in message or "Unsupported parameter" in message
                or isinstance(exc, TypeError)
            ):
                attempt.pop("reasoning_effort")
                continue
            raise
    return client.chat.completions.create(**attempt)


def _choice_probabilities(logprobs: Dict[str, float]) -> Dict[str, float]:
    maximum = max(logprobs.values())
    weights = {key: math.exp(value - maximum) for key, value in logprobs.items()}
    denominator = sum(weights.values())
    return {key: value / denominator for key, value in weights.items()}


def _three_way_logprobs(
    client: Any,
    model: str,
    prompt: str,
    cache: pilot.JsonCache,
    allow_uncertain: bool,
) -> Dict[str, Any]:
    """Measure normalized next-token probabilities for choices 1, 2, and 3."""
    cache_key = cache.make_key(
        "scientist_choice_logprobs_v5",
        {"model": model, "prompt": prompt, "allow_uncertain": allow_uncertain, "version": 5},
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    response = _completion_create(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": ("Follow the answer protocol and output only 1, 2, or 3."
                            if allow_uncertain else
                            "Follow the answer protocol and output only 1 or 2. Never output 3."),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=CHOICE_MAX_COMPLETION_TOKENS,
        reasoning_effort="low",
        logprobs=True,
        top_logprobs=20,
    )
    choices = _get(response, "choices", [])
    if not choices:
        raise RuntimeError("The chat endpoint returned no choices.")
    first = choices[0]
    message = _get(first, "message")
    raw_output = str(_get(message, "content", "") or "").strip()
    generated_choice = _choice(raw_output)

    logprobs_obj = _get(first, "logprobs")
    content_entries = _get(logprobs_obj, "content", []) if logprobs_obj else []
    if not content_entries:
        raise RuntimeError("The endpoint did not return token log-probabilities.")

    allowed = ("1", "2", "3") if allow_uncertain else ("1", "2")
    found: Dict[str, float] = {}
    # Some chat templates emit a newline before the answer digit. Search token
    # positions until one position exposes all three candidate digits.
    for token_position in content_entries:
        position_found: Dict[str, float] = {}
        candidates = list(_get(token_position, "top_logprobs", []) or [])
        candidates.append(token_position)
        for candidate in candidates:
            token = str(_get(candidate, "token", ""))
            normalized = token.strip()
            if normalized in {"1", "2", "3"}:
                value = float(_get(candidate, "logprob"))
                if normalized not in position_found or value > position_found[normalized]:
                    position_found[normalized] = value
        if set(allowed).issubset(position_found):
            found = {key: position_found[key] for key in allowed}
            break

    missing = sorted(set(allowed) - set(found))
    if missing:
        raise RuntimeError(
            "Top logprobs did not include all choices at one token position; missing "
            + ", ".join(missing)
        )

    probabilities = _choice_probabilities(found)
    prediction = max(allowed, key=lambda key: found[key])
    sorted_values = sorted(found.values(), reverse=True)
    output = {
        "method": "logprobs",
        "prediction": prediction,
        "raw_output": raw_output,
        "generated_choice": generated_choice,
        "logprob_1": found["1"],
        "logprob_2": found["2"],
        "logprob_3": found.get("3"),
        "prob_1": probabilities["1"],
        "prob_2": probabilities["2"],
        "prob_3": probabilities.get("3"),
        "choice_margin": sorted_values[0] - sorted_values[1],
    }
    cache.set(cache_key, output)
    return output


def _three_way_sampling(
    client: Any,
    model: str,
    prompt: str,
    cache: pilot.JsonCache,
    samples: int,
    allow_uncertain: bool,
    max_batch_size: int,
) -> Dict[str, Any]:
    """Fallback measurement using repeated categorical samples over 1/2/3."""
    cache_key = cache.make_key(
        "scientist_choice_sampling_v5",
        {"model": model, "prompt": prompt, "samples": samples,
         "allow_uncertain": allow_uncertain, "max_batch_size": max_batch_size,
         "version": 6},
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    raw_outputs: List[str] = []
    messages = [
        {
            "role": "system",
            "content": ("Follow the answer protocol and output only 1, 2, or 3."
                        if allow_uncertain else
                        "Follow the answer protocol and output only 1 or 2. Never output 3."),
        },
        {"role": "user", "content": prompt},
    ]
    # Providers impose different n limits (OpenAI commonly 8, DashScope 4).
    # Request larger sampling budgets in batches and merge the results.
    while len(raw_outputs) < samples:
        batch_size = min(max_batch_size, samples - len(raw_outputs))
        response = _completion_create(
            client,
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=CHOICE_MAX_COMPLETION_TOKENS,
            reasoning_effort="low",
            n=batch_size,
        )
        response_choices = list(_get(response, "choices", []) or [])
        if not response_choices:
            raise RuntimeError("The chat endpoint returned no sampling choices.")
        for response_choice in response_choices[:batch_size]:
            message = _get(response_choice, "message")
            raw_outputs.append(str(_get(message, "content", "") or "").strip())

    allowed = ("1", "2", "3") if allow_uncertain else ("1", "2")
    counts = {key: 0 for key in allowed}
    invalid_count = 0
    for raw in raw_outputs:
        parsed = _choice(raw)
        if parsed not in counts:
            invalid_count += 1
        else:
            counts[parsed] += 1

    valid_count = sum(counts.values())
    if valid_count == 0:
        raise RuntimeError("All samples were invalid for the allowed answer choices.")

    maximum = max(counts.values())
    winners = [key for key, value in counts.items() if value == maximum]
    prediction = winners[0] if len(winners) == 1 else None

    # Laplace smoothing is used only to provide finite diagnostic log-probabilities;
    # the categorical prediction itself is the unsmoothed sample mode above.
    denominator = valid_count + len(allowed)
    probabilities = {
        key: (counts[key] + 1) / denominator for key in allowed
    }
    logprobs = {key: math.log(probabilities[key]) for key in probabilities}
    sorted_values = sorted(logprobs.values(), reverse=True)
    output = {
        "method": "sampling",
        "prediction": prediction,
        "raw_output": raw_outputs[0] if raw_outputs else "",
        "raw_outputs": raw_outputs,
        "counts": counts,
        "invalid_count": invalid_count,
        "valid_count": valid_count,
        "logprob_1": logprobs["1"],
        "logprob_2": logprobs["2"],
        "logprob_3": logprobs.get("3"),
        "prob_1": probabilities["1"],
        "prob_2": probabilities["2"],
        "prob_3": probabilities.get("3"),
        "choice_margin": sorted_values[0] - sorted_values[1],
    }
    cache.set(cache_key, output)
    return output


def measure(
    client: Any,
    model: str,
    prompt: str,
    cache: pilot.JsonCache,
    samples: int,
    method: str,
    allow_uncertain: bool,
    max_batch_size: int,
) -> Dict[str, Any]:
    if method == "logprobs":
        output = _three_way_logprobs(client, model, prompt, cache, allow_uncertain)
    elif method == "sampling":
        output = _three_way_sampling(
            client, model, prompt, cache, samples, allow_uncertain, max_batch_size
        )
    else:
        raise ValueError(f"Unknown probability method: {method}")
    if output["method"] != method:
        raise RuntimeError(f"Probability method mismatch: {output['method']} != {method}")
    return output


def choose_method(
    client: Any,
    model: str,
    prompt: str,
    cache: pilot.JsonCache,
    samples: int,
    allow_uncertain: bool,
    max_batch_size: int,
) -> Tuple[str, Dict[str, Any]]:
    try:
        base = measure(client, model, prompt, cache, samples, "logprobs",
                       allow_uncertain, max_batch_size)
        return "logprobs", base
    except Exception:
        base = measure(client, model, prompt, cache, samples, "sampling",
                       allow_uncertain, max_batch_size)
        return "sampling", base


def effect(base: Dict[str, Any], changed: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Describe a three-way intervention relative to the original answer."""
    changed_prediction = _choice(changed.get("prediction"))
    base_prediction = _choice(base.get("prediction"))
    valid_pair = changed_prediction is not None and base_prediction is not None
    person_flip = bool(
        base_prediction in {"1", "2"}
        and changed_prediction in {"1", "2"}
        and changed_prediction != base_prediction
    )
    base_choice_delta: Optional[float] = None
    if base_prediction in {"1", "2", "3"}:
        base_choice_delta = (
            float(changed[f"logprob_{base_prediction}"])
            - float(base[f"logprob_{base_prediction}"])
        )
    return {
        f"{prefix}_prediction": changed_prediction,
        f"{prefix}_logprob_1": changed["logprob_1"],
        f"{prefix}_logprob_2": changed["logprob_2"],
        f"{prefix}_logprob_3": changed["logprob_3"],
        f"{prefix}_prob_1": changed["prob_1"],
        f"{prefix}_prob_2": changed["prob_2"],
        f"{prefix}_prob_3": changed["prob_3"],
        f"{prefix}_choice_margin": changed["choice_margin"],
        f"{prefix}_uncertain": changed_prediction == "3",
        f"{prefix}_person_flip": person_flip,
        f"{prefix}_same_as_original": (
            None if not valid_pair else changed_prediction == base_prediction
        ),
        f"{prefix}_flip": (
            None if not valid_pair else changed_prediction != base_prediction
        ),
        f"{prefix}_original_choice_logprob_delta": base_choice_delta,
    }


def negate_sentence(
    client: Any,
    model: str,
    question_region: str,
    sentence: str,
    span: GlobalSpan,
    cache: pilot.JsonCache,
) -> Dict[str, Any]:
    """
    Ask the negator to rewrite only the containing sentence. The full prompt is
    reconstructed locally, so the model cannot alter profiles or answer protocol.
    """
    request = (
        "Negate only the proposition expressed by the target span. Preserve all "
        "person names, awards, occupations, fields, institutions, numbers, and "
        "other facts that are not logically required to negate that proposition. "
        "Use the smallest grammatical rewrite. If the target is already negative "
        "(for example, contains 'never' or 'not'), reverse that negative proposition "
        "rather than adding a double negation. Do not answer the question. Return "
        "exactly one JSON object with keys negated_sentence, rewrite_valid, and notes.\n\n"
        f"Question passage:\n{question_region}\n\n"
        f"Target span index: {span.index}\n"
        f"Target span: {span.text}\n"
        f"Full containing sentence: {sentence}"
    )

    cache_key = cache.make_key(
        "scientist_minimal_negation_v1",
        {
            "model": model,
            "question_region": question_region,
            "sentence": sentence,
            "span_index": span.index,
            "span_text": span.text,
            "version": 1,
        },
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    raw = pilot._call_chat_text(
        client=client,
        cache=cache,
        namespace="scientist_minimal_negation_raw_v1",
        model=model,
        messages=[
            {"role": "system", "content": NEGATOR_SYSTEM},
            {"role": "user", "content": request},
        ],
        temperature=1,
        max_tokens=1000,
    )

    try:
        obj = pilot._extract_json_object(raw["content"])
        negated_sentence = obj.get("negated_sentence")
        model_valid = obj.get("rewrite_valid") is not False
        locally_valid = (
            isinstance(negated_sentence, str)
            and bool(negated_sentence.strip())
            and " ".join(negated_sentence.split()) != " ".join(sentence.split())
        )
        valid = bool(model_valid and locally_valid)
        notes = str(obj.get("notes", ""))
        if not locally_valid:
            notes = (notes + "; invalid or unchanged negated sentence").strip("; ")
        output = {
            "negated_sentence": negated_sentence,
            "rewrite_valid": valid,
            "notes": notes,
        }
    except Exception as exc:
        output = {
            "negated_sentence": None,
            "rewrite_valid": False,
            "notes": f"invalid JSON/rewrite: {exc}",
        }

    cache.set(cache_key, output)
    return output


def empty_negation_fields(reason: str) -> Dict[str, Any]:
    return {
        "selected_for_negation": False,
        "negated_sentence": None,
        "negated_prompt": None,
        "rewrite_valid": None,
        "negation_notes": reason,
        "negation_prediction": None,
        "negation_logprob_1": None,
        "negation_logprob_2": None,
        "negation_logprob_3": None,
        "negation_prob_1": None,
        "negation_prob_2": None,
        "negation_prob_3": None,
        "negation_choice_margin": None,
        "negation_uncertain": None,
        "negation_person_flip": None,
        "negation_same_as_original": None,
        "negation_flip": None,
        "negation_original_choice_logprob_delta": None,
        "method_consistent": None,
    }


def run_item_once(
    client: Any,
    detection_item: Dict[str, Any],
    args: argparse.Namespace,
    cache: pilot.JsonCache,
    forced_method: Optional[str] = None,
) -> Dict[str, Any]:
    if GOLD_FIELDS.intersection(detection_item):
        leaked = sorted(GOLD_FIELDS.intersection(detection_item))
        raise AssertionError(f"Gold fields entered detection pipeline: {leaked}")

    item_key = str(detection_item["key"])
    source_prompt = str(detection_item["prompt"])
    profile_1, profile_2 = extract_profile_names(source_prompt)
    original_prompt = build_evaluation_prompt(
        source_prompt, profile_1, profile_2, allow_uncertain=False
    )
    deletion_prompt = build_evaluation_prompt(
        source_prompt, profile_1, profile_2, allow_uncertain=True
    )

    spans, region_start, region_end, question_region = make_global_spans(
        deletion_prompt,
        args.min_clause_words,
        args.min_span_words,
        args.max_span_words,
    )
    if not spans:
        raise ValueError("No spans were produced inside the question region.")

    if forced_method is None:
        method, base = choose_method(
            client, args.target_model, original_prompt, cache, args.samples,
            allow_uncertain=False,
            max_batch_size=args.max_sampling_batch,
        )
    else:
        method = forced_method
        base = measure(
            client,
            args.target_model,
            original_prompt,
            cache,
            args.samples,
            method,
            allow_uncertain=False,
            max_batch_size=args.max_sampling_batch,
        )

    base_prediction = _choice(base.get("prediction"))
    common: Dict[str, Any] = {
        "key": item_key,
        "profile_1": profile_1,
        "profile_2": profile_2,
        "prediction_original": base_prediction,
        "predicted_name": (
            profile_1
            if base_prediction == "1"
            else profile_2
            if base_prediction == "2"
            else "UNCERTAIN"
            if base_prediction == "3"
            else None
        ),
        "original_logprob_1": base["logprob_1"],
        "original_logprob_2": base["logprob_2"],
        "original_logprob_3": base["logprob_3"],
        "original_prob_1": base["prob_1"],
        "original_prob_2": base["prob_2"],
        "original_prob_3": base["prob_3"],
        "original_choice_margin": base["choice_margin"],
        "probability_method": method,
        "question_region": question_region,
        "question_region_start": region_start,
        "question_region_end": region_end,
        "n_spans": len(spans),
    }

    # The detector's flip/non-flip logic requires an original named-profile answer.
    if base_prediction not in {"1", "2"}:
        return common | {
            "n_delete_uncertain": 0,
            "n_delete_person_flips": 0,
            "n_informative_deletions": 0,
            "n_valid_negations": 0,
            "predicted_hallucination": None,
            "decision": "original_uncertain_or_invalid",
            "candidates": [],
        }

    rows: List[Dict[str, Any]] = []
    span_by_index: Dict[int, GlobalSpan] = {}

    # Stage 1: delete every question-region span. No numerical threshold is used.
    for span in spans:
        span_by_index[span.index] = span
        deleted_prompt = delete_global_span(deletion_prompt, span)
        deleted = measure(
            client,
            args.target_model,
            deleted_prompt,
            cache,
            args.samples,
            method,
            allow_uncertain=True,
            max_batch_size=args.max_sampling_batch,
        )
        row: Dict[str, Any] = {
            "candidate_index": span.index,
            "candidate_text": span.text,
            "span_start": span.start,
            "span_end": span.end,
            "local_span_start": span.local_start,
            "local_span_end": span.local_end,
            "deleted_prompt": deleted_prompt,
        }
        row.update(effect(base, deleted, "delete"))
        row["deletion_stage_pass"] = bool(
            row["delete_uncertain"] is True
            or row["delete_person_flip"] is True
        )
        if row["deletion_stage_pass"]:
            stage_reason = (
                "delete_uncertain"
                if row["delete_uncertain"] is True
                else "delete_person_flip"
            )
        else:
            stage_reason = "deletion neither uncertain nor person flip"
        row["deletion_stage_reason"] = stage_reason
        row.update(empty_negation_fields(stage_reason))
        rows.append(row)

    informative_rows = [row for row in rows if row["deletion_stage_pass"] is True]
    negation_prompt = deletion_prompt if args.negation_allow_uncertain else original_prompt

    # Stage 2: negate exactly the deletion-uncertain or deletion-person-flip spans.
    for row in informative_rows:
        span = span_by_index[int(row["candidate_index"])]
        sentence_start, sentence_end, sentence = containing_sentence(
            negation_prompt, span, region_start, region_end
        )
        negation = negate_sentence(
            client=client,
            model=args.negator_model or args.target_model,
            question_region=question_region,
            sentence=sentence,
            span=span,
            cache=cache,
        )

        row["selected_for_negation"] = True
        row["containing_sentence"] = sentence
        row["sentence_start"] = sentence_start
        row["sentence_end"] = sentence_end
        row["negated_sentence"] = negation["negated_sentence"]
        row["rewrite_valid"] = negation["rewrite_valid"]
        row["negation_notes"] = negation["notes"]

        if negation["rewrite_valid"]:
            negated_prompt = (
                negation_prompt[:sentence_start]
                + str(negation["negated_sentence"])
                + negation_prompt[sentence_end:]
            )
            if sentence_start < region_start or sentence_end > region_end:
                raise AssertionError("Negation replacement escaped question region.")

            row["negated_prompt"] = negated_prompt
            measured = measure(
                client,
                args.target_model,
                negated_prompt,
                cache,
                args.samples,
                method,
                allow_uncertain=args.negation_allow_uncertain,
                max_batch_size=args.max_sampling_batch,
            )
            row.update(effect(base, measured, "negation"))
            row["method_consistent"] = measured["method"] == method
        else:
            invalid_fields = empty_negation_fields("invalid negation rewrite")
            invalid_fields.update(
                {
                    "selected_for_negation": True,
                    "negated_sentence": negation["negated_sentence"],
                    "rewrite_valid": False,
                    "negation_notes": negation["notes"],
                    "method_consistent": True,
                }
            )
            row.update(invalid_fields)

    valid_negation_rows = [
        row
        for row in informative_rows
        if row.get("rewrite_valid") is True
        and row.get("method_consistent") is True
        and row.get("negation_flip") in {True, False}
    ]
    has_nonflipping_negation = any(
        row["negation_flip"] is False for row in valid_negation_rows
    )
    all_informative_measured = (
        bool(informative_rows)
        and len(valid_negation_rows) == len(informative_rows)
    )

    if not informative_rows:
        predicted_hallucination: Optional[bool] = False
        decision = "no_delete_uncertain_or_person_flip"
    elif has_nonflipping_negation:
        predicted_hallucination = True
        decision = "informative_deletion_and_negation_nonflip"
    elif all_informative_measured:
        predicted_hallucination = False
        decision = "all_informative_spans_negation_flip"
    else:
        predicted_hallucination = None
        decision = "ambiguous_invalid_negation"

    return common | {
        "n_delete_uncertain": sum(row["delete_uncertain"] is True for row in rows),
        "n_delete_person_flips": sum(
            row["delete_person_flip"] is True for row in rows
        ),
        "n_informative_deletions": len(informative_rows),
        "n_valid_negations": len(valid_negation_rows),
        "predicted_hallucination": predicted_hallucination,
        "decision": decision,
        "candidates": rows,
    }


def run_item(
    client: Any,
    detection_item: Dict[str, Any],
    args: argparse.Namespace,
    cache: pilot.JsonCache,
) -> Dict[str, Any]:
    """Retry the whole item with sampling if any log-probability call fails."""
    try:
        return run_item_once(
            client, detection_item, args, cache, forced_method=None
        )
    except Exception:
        return run_item_once(
            client, detection_item, args, cache, forced_method="sampling"
        )


def confusion(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    classified = [
        record
        for record in records
        if record.get("predicted_hallucination") is not None
        and record.get("true_hallucination") is not None
    ]
    tp = sum(
        record["predicted_hallucination"] is True
        and record["true_hallucination"] is True
        for record in classified
    )
    fp = sum(
        record["predicted_hallucination"] is True
        and record["true_hallucination"] is False
        for record in classified
    )
    tn = sum(
        record["predicted_hallucination"] is False
        and record["true_hallucination"] is False
        for record in classified
    )
    fn = sum(
        record["predicted_hallucination"] is False
        and record["true_hallucination"] is True
        for record in classified
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    accuracy = (tp + tn) / len(classified) if classified else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    return {
        "n": len(classified),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def write_outputs(outdir: Path, records: Sequence[Dict[str, Any]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outdir / "cue_extraction.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat_rows: List[Dict[str, Any]] = []
    for record in records:
        parent = {key: value for key, value in record.items() if key != "candidates"}
        candidates = record.get("candidates", [])
        if candidates:
            for candidate in candidates:
                flat_rows.append(parent | candidate)
        else:
            flat_rows.append(parent)

    fieldnames: List[str] = []
    for row in flat_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    csv_path = outdir / "cue_extraction.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    successful = [record for record in records if not record.get("error")]
    metrics = confusion(successful)
    n = len(successful)
    n_ambiguous = sum(
        record.get("predicted_hallucination") is None for record in successful
    )
    n_original_uncertain = sum(
        record.get("prediction_original") == "3" for record in successful
    )
    n_delete_uncertain_items = sum(
        int(record.get("n_delete_uncertain", 0)) > 0 for record in successful
    )
    n_delete_person_flip_items = sum(
        int(record.get("n_delete_person_flips", 0)) > 0 for record in successful
    )
    n_informative_items = sum(
        int(record.get("n_informative_deletions", 0)) > 0 for record in successful
    )
    n_predicted_hallucination = sum(
        record.get("predicted_hallucination") is True for record in successful
    )

    all_candidates = [
        candidate
        for record in successful
        for candidate in record.get("candidates", [])
    ]
    deletion_available = [
        candidate
        for candidate in all_candidates
        if candidate.get("delete_prediction") in {"1", "2", "3"}
    ]
    negation_available = [
        candidate
        for candidate in all_candidates
        if candidate.get("negation_prediction") in {"1", "2", "3"}
    ]

    deletion_uncertain = sum(
        candidate.get("delete_uncertain") is True for candidate in deletion_available
    )
    deletion_person_flips = sum(
        candidate.get("delete_person_flip") is True
        for candidate in deletion_available
    )
    deletion_informative = sum(
        candidate.get("deletion_stage_pass") is True
        for candidate in deletion_available
    )
    negation_flips = sum(
        candidate.get("negation_flip") is True for candidate in negation_available
    )
    negation_nonflips = sum(
        candidate.get("negation_flip") is False for candidate in negation_available
    )
    negation_uncertain = sum(
        candidate.get("negation_uncertain") is True
        for candidate in negation_available
    )

    def fmt(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    def rate(count: int, denominator: int) -> str:
        return (
            "n/a"
            if denominator == 0
            else f"{count}/{denominator} ({count / denominator:.3f})"
        )

    lines = [
        "# Delete-uncertain/person-flip → negation-nonflip detection summary",
        "",
        f"Items processed: {n}",
        f"Original answer uncertain (3): {rate(n_original_uncertain, n)}",
        f"Items with at least one delete→uncertain span: {rate(n_delete_uncertain_items, n)}",
        f"Items with at least one delete person-flip span: {rate(n_delete_person_flip_items, n)}",
        f"Items entering negation stage: {rate(n_informative_items, n)}",
        f"Predicted hallucination: {rate(n_predicted_hallucination, n)}",
        f"Ambiguous decisions: {rate(n_ambiguous, n)}",
        "",
        "## Hallucination detection (gold joined only after prediction)",
        "",
        f"Classified coverage: {rate(metrics['n'], n)}",
        f"Precision: {fmt(metrics['precision'])}",
        f"Recall: {fmt(metrics['recall'])}",
        f"F1: {fmt(metrics['f1'])}",
        f"Accuracy: {fmt(metrics['accuracy'])}",
        (
            "Confusion matrix: "
            f"TP={metrics['tp']}, FP={metrics['fp']}, "
            f"TN={metrics['tn']}, FN={metrics['fn']}"
        ),
        "",
        "## Intervention diagnostics",
        "",
        (
            f"- deletion: uncertain={deletion_uncertain}, "
            f"person_flips={deletion_person_flips}, "
            f"informative={deletion_informative}/{len(deletion_available)}"
        ),
        (
            f"- negation among informative spans: nonflips={negation_nonflips}, "
            f"flips={negation_flips}, uncertain_outputs={negation_uncertain}, "
            f"measured={len(negation_available)}"
        ),
        "",
        "## Decision rule",
        "",
        "- Answers use 1/2 for the two profiles and 3 for not uniquely identifiable.",
        "- delete_prediction=3 OR deletion person-flip → enter negation stage.",
        "- At least one selected span with negation_flip=False → hallucination.",
        "- No informative deletion → non-hallucination.",
        "- All selected spans validly measured and negation_flip=True → non-hallucination.",
        "- Invalid/missing negation evidence without a non-flip witness → ambiguous.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_gold_option(
    raw: Dict[str, Any], profile_1: str, profile_2: str
) -> Tuple[str, str]:
    """Map rgt_ans/wrg_ans to option numbers only at the evaluation boundary."""
    right = _normalize_name(raw.get("rgt_ans"))
    wrong = _normalize_name(raw.get("wrg_ans"))
    first = _normalize_name(profile_1)
    second = _normalize_name(profile_2)

    if right == first:
        correct_option = "1"
    elif right == second:
        correct_option = "2"
    else:
        raise ValueError(
            f"rgt_ans={raw.get('rgt_ans')!r} does not match either profile name."
        )

    if wrong == first:
        wrong_option = "1"
    elif wrong == second:
        wrong_option = "2"
    else:
        raise ValueError(
            f"wrg_ans={raw.get('wrg_ans')!r} does not match either profile name."
        )

    if correct_option == wrong_option:
        raise ValueError("Correct and wrong answer map to the same option.")
    return correct_option, wrong_option


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "/Users/txh/Desktop/hallucination experiment/scientist_qa/question/"
            "shuffled_prepend_names_question.json"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of records to process; -1 means all records.",
    )
    parser.add_argument("--target-model", default=pilot.DEFAULT_MODEL)
    parser.add_argument(
        "--negator-model",
        default=pilot.DEFAULT_MODEL,
        help="Defaults to --target-model when omitted.",
    )
    parser.add_argument("--base-url", default=pilot.DEFAULT_BASE_URL)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--negation-allow-uncertain",
        action="store_true",
        help="Allow answer 3 during negation measurement (original remains binary).",
    )
    parser.add_argument(
        "--max-sampling-batch",
        type=int,
        default=8,
        help="Maximum choices requested with n in one API call.",
    )
    parser.add_argument("--min-clause-words", type=int, default=10)
    parser.add_argument("--min-span-words", type=int, default=2)
    parser.add_argument("--max-span-words", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--outdir",
        default="outputs_scientist/only_deletion_uncertain_5mini",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.max_sampling_batch < 1:
        parser.error("--max-sampling-batch must be positive")
    if args.min_span_words < 1 or args.max_span_words < args.min_span_words:
        parser.error("invalid span word limits")
    return args


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    records_raw = load_json_records(input_path)
    if args.limit >= 0:
        records_raw = records_raw[: args.limit]

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    cache = pilot.JsonCache(outdir / "cache.json")
    client = pilot._make_client(args.base_url)

    records: List[Dict[str, Any]] = []
    evaluation_gold: Dict[str, Dict[str, str]] = {}

    for index, raw in enumerate(records_raw):
        item_key = str(raw.get("key", f"question_{index:04d}"))
        try:
            prompt = raw.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Missing or empty prompt.")

            # Prepare evaluation mapping separately. It is not passed to run_item.
            profile_1, profile_2 = extract_profile_names(prompt)
            correct_option, wrong_option = resolve_gold_option(
                raw, profile_1, profile_2
            )
            evaluation_gold[item_key] = {
                "correct_option": correct_option,
                "wrong_option": wrong_option,
                "rgt_ans": str(raw.get("rgt_ans", "")),
                "wrg_ans": str(raw.get("wrg_ans", "")),
            }

            detection_item = {"key": item_key, "prompt": prompt}
            record = run_item(client, detection_item, args, cache)
            records.append(record)
        except Exception as exc:
            records.append({"key": item_key, "error": str(exc)})

        print(
            f"[{index + 1}/{len(records_raw)}] {item_key} done",
            file=sys.stderr,
            flush=True,
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Gold evaluation boundary: join only after every detector prediction is complete.
    for record in records:
        if record.get("error"):
            continue
        gold = evaluation_gold.get(str(record.get("key")))
        if gold is None:
            continue
        prediction = _choice(record.get("prediction_original"))
        record["correct_option_eval_only"] = gold["correct_option"]
        record["wrong_option_eval_only"] = gold["wrong_option"]
        record["rgt_ans_eval_only"] = gold["rgt_ans"]
        record["wrg_ans_eval_only"] = gold["wrg_ans"]
        record["true_hallucination"] = (
            None if prediction not in {"1", "2"} else prediction != gold["correct_option"]
        )

    write_outputs(outdir, records)
    print(f"Wrote outputs to {outdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
