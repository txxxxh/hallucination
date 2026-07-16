"""Shared candidate-span segmentation for RealLifeQA cue extraction.

Both the occlusion (API) and attention (local) methods must score the same
candidate spans, so segmentation lives here.

A RealLifeQA benchmark prompt has the shape:

    Scenario: <scenario text ...>
    Option1: <...>
    Option2: <...>
    Question: Which one should I choose? Answer 1 for Option1 and 2 for Option2.

Cue extraction only perturbs the scenario body; option lines and the final
answer instruction are held fixed (matching the editor-variant rules in the
pilot script).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Clause connectors worth splitting on when a sentence is long. These tend to
# separate status cues ("is marked ready, but ...") from constraints.
_CLAUSE_SPLIT = re.compile(
    r",\s+(?=(?:but|and|so|because|while|although|unless|only|before|after|until)\b)",
    flags=re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Finer cue-oriented boundaries. These split long clauses into local phrases
# while keeping meaningful multi-word cues such as "with a tablet", "from the
# port", "before releasing it", or "not a loose cover" intact enough to score.
_PHRASE_SPLIT = re.compile(
    r"(?:[,;:]\s+)|"
    r"\s+(?=(?:but|and|so|because|while|although|unless|only\s+after|after|before|until|"
    r"to|into|onto|with|from|near|under|inside|beside|during|while|not)\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Span:
    """A candidate cue span, located by character offsets in the full prompt."""

    index: int
    text: str
    start: int  # char offset in the *full* prompt
    end: int    # exclusive


def locate_scenario(prompt: str) -> Tuple[int, int]:
    """Return (start, end) char offsets of the scenario body inside prompt.

    The scenario body runs from after the leading "Scenario:" tag (if present)
    to just before the first option line. Falls back to the whole prompt if the
    expected structure is missing.
    """
    match = re.search(r"^\s*Scenario:\s*", prompt)
    start = match.end() if match else 0

    option = re.search(r"\n\s*Option1\s*:", prompt)
    end = option.start() if option else len(prompt)
    if end <= start:
        return 0, len(prompt)
    return start, end


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _is_question_scaffold(text: str) -> bool:
    stripped = text.strip()
    return bool(
        re.match(
            r"^(?:Should\s+I\b|For\s+this\s+step\b|(?:What|Where|How|Which)\s+should\b)",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def _split_by_pattern(text: str, start: int, end: int, pattern: re.Pattern[str]) -> List[Tuple[int, int]]:
    """Split ``text[start:end]`` at regex separator matches, preserving offsets."""
    pieces: List[Tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        if match.start() > cursor:
            pieces.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _split_by_word_windows(
    text: str,
    start: int,
    end: int,
    max_span_words: int,
) -> List[Tuple[int, int]]:
    """Split a still-long span into contiguous word windows."""
    words = list(re.finditer(r"\S+", text[start:end]))
    if max_span_words <= 0 or len(words) <= max_span_words:
        return [(start, end)]

    pieces: List[Tuple[int, int]] = []
    for idx in range(0, len(words), max_span_words):
        chunk = words[idx : idx + max_span_words]
        pieces.append((start + chunk[0].start(), start + chunk[-1].end()))
    return pieces


def segment_scenario(
    prompt: str,
    min_clause_words: int = 12,
    min_span_words: int = 2,
    max_span_words: int = 8,
) -> List[Span]:
    """Split the scenario body into candidate spans.

    Sentences first; long sentences are split at clause connectors, then long
    clauses are split at cue-oriented phrase boundaries. Final choice-question
    scaffolds are skipped, because they are prompt format rather than evidence.
    Remaining spans longer than ``max_span_words`` are split into word windows.
    Spans shorter than ``min_span_words`` are merged into their left neighbour
    so occlusion never deletes near-empty fragments.
    """
    scen_start, scen_end = locate_scenario(prompt)
    scenario = prompt[scen_start:scen_end]

    pieces: List[Tuple[int, int]] = []  # (start, end) relative to scenario
    cursor = 0
    for sentence in _SENTENCE_SPLIT.split(scenario):
        if not sentence.strip():
            cursor += len(sentence) + 1
            continue
        s_start = scenario.find(sentence, cursor)
        if s_start == -1:  # defensive; should not happen
            cursor += len(sentence) + 1
            continue
        s_end = s_start + len(sentence)
        cursor = s_end

        if _is_question_scaffold(sentence):
            continue

        sentence_pieces = (
            _split_by_pattern(scenario, s_start, s_end, _CLAUSE_SPLIT)
            if _word_count(sentence) > min_clause_words
            else [(s_start, s_end)]
        )
        for p_start, p_end in sentence_pieces:
            text = scenario[p_start:p_end]
            phrase_pieces = (
                _split_by_pattern(scenario, p_start, p_end, _PHRASE_SPLIT)
                if _word_count(text) > max_span_words
                else [(p_start, p_end)]
            )
            for q_start, q_end in phrase_pieces:
                pieces.extend(
                    _split_by_word_windows(
                        scenario, q_start, q_end, max_span_words=max_span_words
                    )
                )

    # Merge too-short fragments leftward.
    merged: List[Tuple[int, int]] = []
    for start, end in pieces:
        text = scenario[start:end]
        if merged and _word_count(text) < min_span_words:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    spans = []
    for idx, (start, end) in enumerate(merged):
        text = scenario[start:end].strip()
        if not text:
            continue
        # Re-anchor stripped text.
        offset = scenario.find(text, start)
        spans.append(
            Span(
                index=idx,
                text=text,
                start=scen_start + offset,
                end=scen_start + offset + len(text),
            )
        )
    return spans


def delete_span(prompt: str, span: Span) -> str:
    """Return the prompt with one span deleted and punctuation tidied.

    This is the span-level shortcut/constraint-removal operator: everything
    outside the span (options, answer instruction, other scenario spans) is
    preserved verbatim.
    """
    before = prompt[: span.start]
    after = prompt[span.end :]

    # Tidy dangling connectors/punctuation left behind by the deletion.
    before = re.sub(r"[,;:]\s*$", " ", before)
    after = re.sub(r"^\s*(?:,|;|and\b|but\b|so\b)\s*", " ", after, flags=re.IGNORECASE)
    combined = before + after
    combined = re.sub(r"[ \t]{2,}", " ", combined)
    combined = re.sub(r"\s+([.!?,])", r"\1", combined)
    combined = re.sub(r"(?m)^Scenario:\s*\.\s*", "Scenario: ", combined)
    return combined


def token_f1(pred: Optional[str], gold: Optional[str]) -> float:
    """Token-overlap F1 between a predicted span and an annotated span."""
    if not pred or not gold:
        return 0.0
    pred_tokens = re.findall(r"\w+", pred.lower())
    gold_tokens = re.findall(r"\w+", gold.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def match_span_to_gold(spans: List[Span], gold_text: Optional[str]) -> Optional[int]:
    """Index of the candidate span that best overlaps the gold substring."""
    if not gold_text:
        return None
    best_idx, best_f1 = None, 0.0
    for span in spans:
        score = token_f1(span.text, gold_text)
        if score > best_f1:
            best_idx, best_f1 = span.index, score
    return best_idx if best_f1 > 0.0 else None
