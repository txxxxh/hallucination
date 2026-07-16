#!/usr/bin/env python3
"""Fill candidate span indices in RealLifeQA gold cue JSONL.

This script maps gold cue core texts onto the exact candidate spans produced by
``cue_spans.segment_scenario``. It is intentionally conservative and auditable:
by default it only fills empty index fields, preserves existing manual indices,
and writes a review CSV with the matched span text and score for every core.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_reallifeqa_pilot as pilot  # noqa: E402
from cue_spans import Span, segment_scenario, token_f1  # noqa: E402


DEFAULT_MIN_CLAUSE_WORDS = 10
DEFAULT_MIN_SPAN_WORDS = 2
DEFAULT_MAX_SPAN_WORDS = 8


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_id(raw_item: Dict[str, Any], index: int) -> Any:
    explicit = raw_item.get("id")
    return explicit if explicit is not None else index + 1


def load_items_by_id(input_path: str) -> Dict[Any, Dict[str, Any]]:
    items = pilot.load_data(input_path)
    out: Dict[Any, Dict[str, Any]] = {}
    for index, item in enumerate(items):
        if isinstance(item, dict):
            out[item_id(item, index)] = item
    return out


def normalize_text(text: str) -> str:
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(tokens)


def content_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def contains_match(core: str, candidate: str) -> bool:
    core_norm = normalize_text(core)
    candidate_norm = normalize_text(candidate)
    if not core_norm or not candidate_norm:
        return False
    return core_norm in candidate_norm or candidate_norm in core_norm


def combined_span_text(spans: Sequence[Span], indices: Sequence[int]) -> str:
    by_index = {span.index: span.text for span in spans}
    return " ".join(by_index[index] for index in indices if index in by_index)


def contiguous_windows(spans: Sequence[Span], max_window_size: int) -> Iterable[List[int]]:
    indices = [span.index for span in spans]
    for start in range(len(indices)):
        for end in range(start + 1, min(len(indices), start + max_window_size) + 1):
            yield indices[start:end]


def overlap_span_indices(core: str, spans: Sequence[Span], min_overlap_tokens: int = 1) -> List[int]:
    core_token_set = set(content_tokens(core))
    if not core_token_set:
        return []
    out: List[int] = []
    for span in spans:
        overlap = core_token_set & set(content_tokens(span.text))
        if len(overlap) >= min_overlap_tokens:
            out.append(span.index)
    return out


def best_window_match(
    core: str,
    spans: Sequence[Span],
    min_score: float,
    max_window_size: int,
) -> Dict[str, Any]:
    """Return the best contiguous span window for a core text."""
    best: Optional[Dict[str, Any]] = None
    for indices in contiguous_windows(spans, max_window_size=max_window_size):
        text = combined_span_text(spans, indices)
        direct = contains_match(core, text)
        score = token_f1(text, core)
        overlap_count = len(set(content_tokens(core)) & set(content_tokens(text)))
        eligible = direct or score >= min_score
        if not eligible:
            continue
        row = {
            "indices": indices,
            "score": score,
            "direct": direct,
            "overlap_count": overlap_count,
            "span_text": text,
            "method": "direct" if direct else "token_f1",
        }
        if best is None:
            best = row
            continue
        # Prefer direct containment, then higher F1, then shorter windows.
        current_key = (
            int(row["direct"]),
            row["score"],
            -len(row["indices"]),
            row["overlap_count"],
        )
        best_key = (
            int(best["direct"]),
            best["score"],
            -len(best["indices"]),
            best["overlap_count"],
        )
        if current_key > best_key:
            best = row

    if best is not None:
        return best

    # No confident match. Keep the best token-overlap window as diagnostics.
    diagnostic: Optional[Dict[str, Any]] = None
    for indices in contiguous_windows(spans, max_window_size=max_window_size):
        text = combined_span_text(spans, indices)
        score = token_f1(text, core)
        row = {
            "indices": indices,
            "score": score,
            "direct": False,
            "overlap_count": len(set(content_tokens(core)) & set(content_tokens(text))),
            "span_text": text,
            "method": "below_threshold",
        }
        if diagnostic is None or (score, -len(indices)) > (
            diagnostic["score"],
            -len(diagnostic["indices"]),
        ):
            diagnostic = row
    return diagnostic or {
        "indices": [],
        "score": 0.0,
        "direct": False,
        "overlap_count": 0,
        "span_text": "",
        "method": "no_spans",
    }


def fill_role_indices(
    row: Dict[str, Any],
    spans: Sequence[Span],
    role: str,
    min_score: float,
    max_window_size: int,
    preserve_existing: bool,
    review_rows: List[Dict[str, Any]],
) -> Tuple[List[int], List[int], List[str]]:
    """Fill candidate/proposition indices for shortcut or constraint."""
    core_key = f"{role}_core_texts"
    candidate_key = f"{role}_candidate_indices"
    proposition_key = f"{role}_proposition_indices"
    cores = row.get(core_key) or []
    warnings: List[str] = []

    existing_candidate = row.get(candidate_key) or []
    existing_proposition = row.get(proposition_key) or []
    if preserve_existing and existing_candidate and existing_proposition:
        return list(existing_candidate), list(existing_proposition), warnings

    filled_candidate: List[int] = list(existing_candidate) if preserve_existing else []
    filled_proposition: List[int] = list(existing_proposition) if preserve_existing else []

    if not cores:
        return sorted(set(filled_candidate)), sorted(set(filled_proposition)), warnings

    for core in cores:
        if not isinstance(core, str) or not core.strip():
            continue
        match = best_window_match(
            core=core,
            spans=spans,
            min_score=min_score,
            max_window_size=max_window_size,
        )
        indices = list(match["indices"])
        confident = bool(match["direct"]) or match["score"] >= min_score
        if confident:
            filled_proposition.extend(indices)
            # Candidate indices are the minimal matched window. This is intentionally
            # conservative with respect to span segmentation: if the cue is split
            # across adjacent spans, every span needed to express the core cue is
            # included.
            filled_candidate.extend(indices)
        else:
            warnings.append(
                f"{role}: low-confidence core match {core!r} "
                f"(best_score={match['score']:.3f})"
            )

        review_rows.append(
            {
                "id": row.get("id"),
                "role": role,
                "core_text": core,
                "matched_indices": json.dumps(indices),
                "matched_span_text": match["span_text"],
                "score": f"{match['score']:.3f}",
                "direct": match["direct"],
                "method": match["method"],
                "confident": confident,
                "warning": "" if confident else warnings[-1],
            }
        )

    return sorted(set(filled_candidate)), sorted(set(filled_proposition)), warnings


def fill_indices(
    gold_rows: List[Dict[str, Any]],
    items_by_id: Dict[Any, Dict[str, Any]],
    min_clause_words: int,
    min_span_words: int,
    max_span_words: int,
    min_score: float,
    max_window_size: int,
    preserve_existing: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    filled_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    summary = {
        "rows": 0,
        "items_missing": 0,
        "shortcut_candidate_nonempty": 0,
        "constraint_candidate_nonempty": 0,
        "constraint_proposition_nonempty": 0,
        "rows_with_warnings": 0,
    }

    params = {
        "min_clause_words": min_clause_words,
        "min_span_words": min_span_words,
        "max_span_words": max_span_words,
        "min_score": min_score,
        "max_window_size": max_window_size,
        "preserve_existing": preserve_existing,
    }

    for row in gold_rows:
        summary["rows"] += 1
        out = dict(row)
        item = items_by_id.get(row.get("id"))
        warnings: List[str] = []
        if item is None:
            warnings.append("missing source item")
            summary["items_missing"] += 1
            out["index_fill_warnings"] = warnings
            out["index_fill_params"] = params
            filled_rows.append(out)
            continue

        spans = segment_scenario(
            item["benchmark_prompt"],
            min_clause_words=min_clause_words,
            min_span_words=min_span_words,
            max_span_words=max_span_words,
        )
        out["candidate_spans_for_index_fill"] = [
            {"span_index": span.index, "span_text": span.text} for span in spans
        ]

        shortcut_candidate, shortcut_prop, role_warnings = fill_role_indices(
            out,
            spans,
            role="shortcut",
            min_score=min_score,
            max_window_size=max_window_size,
            preserve_existing=preserve_existing,
            review_rows=review_rows,
        )
        warnings.extend(role_warnings)
        constraint_candidate, constraint_prop, role_warnings = fill_role_indices(
            out,
            spans,
            role="constraint",
            min_score=min_score,
            max_window_size=max_window_size,
            preserve_existing=preserve_existing,
            review_rows=review_rows,
        )
        warnings.extend(role_warnings)

        out["shortcut_candidate_indices"] = shortcut_candidate
        out["shortcut_proposition_indices"] = shortcut_prop
        out["constraint_candidate_indices"] = constraint_candidate
        out["constraint_proposition_indices"] = constraint_prop
        out["index_fill_method"] = "segment_core_text_match_v1"
        out["index_fill_params"] = params
        out["index_fill_warnings"] = warnings
        if warnings:
            summary["rows_with_warnings"] += 1
        if shortcut_candidate:
            summary["shortcut_candidate_nonempty"] += 1
        if constraint_candidate:
            summary["constraint_candidate_nonempty"] += 1
        if constraint_prop:
            summary["constraint_proposition_nonempty"] += 1
        filled_rows.append(out)

    return filled_rows, review_rows, summary


def write_review_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "role",
        "core_text",
        "matched_indices",
        "matched_span_text",
        "score",
        "direct",
        "method",
        "confident",
        "warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: Dict[str, int], output_path: Path, review_path: Path) -> None:
    lines = [
        "# Gold Cue Index Fill Summary",
        "",
        f"- Rows processed: {summary['rows']}",
        f"- Source items missing: {summary['items_missing']}",
        f"- Rows with warnings: {summary['rows_with_warnings']}",
        f"- Shortcut candidate indices nonempty: {summary['shortcut_candidate_nonempty']}/{summary['rows']}",
        f"- Constraint candidate indices nonempty: {summary['constraint_candidate_nonempty']}/{summary['rows']}",
        f"- Constraint proposition indices nonempty: {summary['constraint_proposition_nonempty']}/{summary['rows']}",
        "",
        f"Output JSONL: `{output_path}`",
        f"Review CSV: `{review_path}`",
        "",
        "Review the CSV rows with `confident=False` before treating the filled file as final.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="gold_cues_all_500.jsonl")
    parser.add_argument("--input", default="real_life_constrained_qa/question_and_result.json")
    parser.add_argument("--out", default="gold_cues_all_500_indexed.jsonl")
    parser.add_argument("--review", default="gold_cues_all_500_index_review.csv")
    parser.add_argument("--summary", default="gold_cues_all_500_index_summary.md")
    parser.add_argument("--min-clause-words", type=int, default=DEFAULT_MIN_CLAUSE_WORDS)
    parser.add_argument("--min-span-words", type=int, default=DEFAULT_MIN_SPAN_WORDS)
    parser.add_argument("--max-span-words", type=int, default=DEFAULT_MAX_SPAN_WORDS)
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--max-window-size", type=int, default=5)
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="recompute and overwrite nonempty manual index fields too",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    output_path = Path(args.out)
    review_path = Path(args.review)
    summary_path = Path(args.summary)

    filled, review, summary = fill_indices(
        gold_rows=read_jsonl(gold_path),
        items_by_id=load_items_by_id(args.input),
        min_clause_words=args.min_clause_words,
        min_span_words=args.min_span_words,
        max_span_words=args.max_span_words,
        min_score=args.min_score,
        max_window_size=args.max_window_size,
        preserve_existing=not args.recompute_existing,
    )
    write_jsonl(output_path, filled)
    write_review_csv(review_path, review)
    write_summary(summary_path, summary, output_path, review_path)

    print(f"Wrote filled gold cues to {output_path}")
    print(f"Wrote review CSV to {review_path}")
    print(f"Rows with warnings: {summary['rows_with_warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
