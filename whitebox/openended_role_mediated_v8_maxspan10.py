#!/usr/bin/env python3
"""Open-ended role-mediated detector v8 with a strict 10-word span cap.

This launcher leaves ``openended_role_mediated_v8.py`` intact. It replaces
only v8's span segmentation function and invokes the original CLI, so all
model, feature, intervention, training, and evaluation behavior is otherwise
identical to v8.
"""

from __future__ import annotations

import openended_role_mediated_v8 as v8


MAX_SPAN_WORDS = 10


def _split_piece_at_word_boundaries(
    source: str,
    start: int,
    end: int,
    max_words: int = MAX_SPAN_WORDS,
) -> list[tuple[int, int]]:
    """Return source-aligned chunks with no more than ``max_words`` words."""
    matches = list(v8._WORD_RE.finditer(source, start, end))
    if not matches:
        return []
    chunks: list[tuple[int, int]] = []
    for offset in range(0, len(matches), max_words):
        group = matches[offset : offset + max_words]
        chunk_start = group[0].start()
        # Keep punctuation following the last word without consuming the next
        # chunk's first word.
        chunk_end = (
            matches[offset + max_words].start()
            if offset + max_words < len(matches)
            else end
        )
        while chunk_end > chunk_start and source[chunk_end - 1].isspace():
            chunk_end -= 1
        if chunk_start < chunk_end:
            chunks.append((chunk_start, chunk_end))
    return chunks


def segment_atomic_max10(
    source: str,
    min_clause_words: int,
    min_span_words: int,
) -> list[v8.Span]:
    """Run original v8 segmentation, then enforce the strict upper bound."""
    original_spans = v8._ORIGINAL_SEGMENT_ATOMIC(
        source,
        min_clause_words,
        min_span_words,
    )
    bounded: list[v8.Span] = []
    for span in original_spans:
        for start, end in _split_piece_at_word_boundaries(
            source, span.start, span.end
        ):
            text = source[start:end].strip()
            if not text:
                continue
            real_start = source.find(text, start, end + 1)
            word_count = len(v8._WORD_RE.findall(text))
            if word_count > MAX_SPAN_WORDS:
                raise AssertionError(
                    f"span cap violated: {word_count} words in {text!r}"
                )
            bounded.append(
                v8.Span(
                    index=len(bounded),
                    text=text,
                    start=real_start,
                    end=real_start + len(text),
                )
            )
    return bounded


v8._ORIGINAL_SEGMENT_ATOMIC = v8.segment_atomic
v8.segment_atomic = segment_atomic_max10
v8.CACHE_SCHEMA_VERSION = "openended_v8_topk_attention_maxspan10_v1"


if __name__ == "__main__":
    raise SystemExit(v8.main())
