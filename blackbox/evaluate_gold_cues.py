#!/usr/bin/env python3
"""Evaluate extracted shortcut/constraint spans against candidate-level gold cues.

Strict cue recognition uses annotated candidate indices. Arbitrary one-token
intersection is intentionally not counted as a hit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def id_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def predicted_index(item: Dict[str, Any], field: str) -> Optional[int]:
    text = item.get(field)
    if text is None:
        return None
    spans = item.get("spans")
    if not isinstance(spans, list):
        return None
    matches = [
        span.get("span_index")
        for span in spans
        if isinstance(span, dict) and span.get("span_text") == text
    ]
    if len(matches) != 1 or matches[0] is None:
        return None
    return int(matches[0])


def safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def read_results_csv(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {id_key(row.get("item_id")): row for row in csv.DictReader(f)}


def evaluate(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    intervention_results: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    gold = {id_key(row.get("id")): row for row in gold_rows}
    summary = {
        "items_evaluated": 0,
        "shortcut_high_n": 0,
        "shortcut_high_hit": 0,
        "shortcut_explicit_n": 0,
        "shortcut_explicit_hit": 0,
        "constraint_n": 0,
        "constraint_strict_hit": 0,
        "constraint_proposition_hit": 0,
        "both_strict_n": 0,
        "intervention_linked_n": 0,
        "constraint_behavior_success_n": 0,
        "constraint_hit_and_behavior_success_n": 0,
        "full_pattern_success_n": 0,
    }
    output: List[Dict[str, Any]] = []

    for pred in pred_rows:
        key = id_key(pred.get("id"))
        g = gold.get(key)
        if g is None:
            continue

        sidx = predicted_index(pred, "pred_shortcut_span")
        cidx = predicted_index(pred, "pred_constraint_span")
        shortcut_indices = set(g.get("shortcut_candidate_indices") or [])
        constraint_indices = set(g.get("constraint_candidate_indices") or [])
        proposition_indices = set(g.get("constraint_proposition_indices") or [])
        confidence = str(g.get("shortcut_confidence", "none"))
        shortcut_eligible = confidence in {"high", "medium"}
        shortcut_hit = (sidx in shortcut_indices) if shortcut_eligible else None
        constraint_hit = cidx in constraint_indices
        proposition_hit = cidx in proposition_indices

        if shortcut_eligible:
            if shortcut_hit and constraint_hit:
                status = "both_strict"
            elif shortcut_hit:
                status = "shortcut_only"
            elif constraint_hit:
                status = "constraint_only"
            else:
                status = "neither"
        else:
            status = "no_explicit_shortcut"

        intervention = intervention_results.get(key, {})
        behavior_success = safe_bool(
            intervention.get("constraint_removed_activates_shortcut")
        )
        full_pattern_success = safe_bool(intervention.get("matches_expected_pattern"))

        row = {
            "id": pred.get("id"),
            "pred_shortcut_span": pred.get("pred_shortcut_span"),
            "pred_shortcut_index": sidx,
            "gold_shortcut_indices": sorted(shortcut_indices),
            "shortcut_confidence": confidence,
            "shortcut_eligible": shortcut_eligible,
            "shortcut_strict_hit": shortcut_hit,
            "pred_constraint_span": pred.get("pred_constraint_span"),
            "pred_constraint_index": cidx,
            "gold_constraint_indices": sorted(constraint_indices),
            "gold_constraint_proposition_indices": sorted(proposition_indices),
            "constraint_strict_hit": constraint_hit,
            "constraint_proposition_hit": proposition_hit,
            "candidate_level_safe": g.get("candidate_level_safe"),
            "cue_identification_status": status,
            "constraint_removed_activates_shortcut": behavior_success,
            "matches_expected_pattern": full_pattern_success,
            "notes": g.get("notes", ""),
        }
        output.append(row)

        summary["items_evaluated"] += 1
        if confidence == "high":
            summary["shortcut_high_n"] += 1
            summary["shortcut_high_hit"] += int(bool(shortcut_hit))
        if shortcut_eligible:
            summary["shortcut_explicit_n"] += 1
            summary["shortcut_explicit_hit"] += int(bool(shortcut_hit))
        summary["constraint_n"] += 1
        summary["constraint_strict_hit"] += int(constraint_hit)
        summary["constraint_proposition_hit"] += int(proposition_hit)
        summary["both_strict_n"] += int(bool(shortcut_hit) and constraint_hit)
        if intervention:
            summary["intervention_linked_n"] += 1
            summary["constraint_behavior_success_n"] += int(behavior_success is True)
            summary["constraint_hit_and_behavior_success_n"] += int(
                constraint_hit and behavior_success is True
            )
            summary["full_pattern_success_n"] += int(full_pattern_success is True)

    return output, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: List[Dict[str, Any]], s: Dict[str, int]) -> None:
    lines = [
        "# Gold Cue Evaluation",
        "",
        "Strict recognition is candidate-index membership. A one-word overlap with a "
        "multiword gold cue is not counted as a strict hit.",
        "",
        f"- Items evaluated: {s['items_evaluated']}",
        f"- Shortcut strict recall (high confidence): {s['shortcut_high_hit']}/{s['shortcut_high_n']}",
        f"- Shortcut strict recall (all explicit high+medium): {s['shortcut_explicit_hit']}/{s['shortcut_explicit_n']}",
        f"- Constraint strict recall: {s['constraint_strict_hit']}/{s['constraint_n']}",
        f"- Constraint proposition-overlap recall: {s['constraint_proposition_hit']}/{s['constraint_n']}",
        f"- Both shortcut and constraint strict: {s['both_strict_n']}/{s['shortcut_explicit_n']}",
    ]

    if any(row.get("constraint_removed_activates_shortcut") is not None for row in rows):
        lines.extend([
            "",
            "## Link to intervention behavior",
            "",
            f"- Constraint-removed variant selected the shortcut: {s['constraint_behavior_success_n']}/{s['intervention_linked_n']}",
            "- Constraint cue strict hit AND constraint-removed selected shortcut: "
            f"{s['constraint_hit_and_behavior_success_n']}/{s['intervention_linked_n']}",
            f"- Full C/C/S/C intervention pattern: {s['full_pattern_success_n']}/{s['intervention_linked_n']}",
        ])

    lines.extend([
        "",
        "## Per-item cue decisions",
        "",
        "| ID | Shortcut prediction | Shortcut strict | Constraint prediction | Constraint strict | Proposition overlap | Status |",
        "|---:|---|---:|---|---:|---:|---|",
    ])
    for row in rows:
        shortcut_value = (
            "n/a" if row["shortcut_strict_hit"] is None
            else "yes" if row["shortcut_strict_hit"] else "no"
        )
        lines.append(
            f"| {row['id']} | {row['pred_shortcut_span'] or '—'} | {shortcut_value} | "
            f"{row['pred_constraint_span'] or '—'} | "
            f"{'yes' if row['constraint_strict_hit'] else 'no'} | "
            f"{'yes' if row['constraint_proposition_hit'] else 'no'} | "
            f"{row['cue_identification_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="gold_cues_preliminary.jsonl")
    parser.add_argument("--pred", default="cue_extraction.jsonl")
    parser.add_argument(
        "--results",
        default=None,
        help="optional intervention results.csv from run_reallifeqa_pilot_fixed.py",
    )
    parser.add_argument("--outdir", default="cue_evaluation")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    results_path = Path(args.results) if args.results else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, summary = evaluate(
        read_jsonl(gold_path),
        read_jsonl(pred_path),
        read_results_csv(results_path),
    )
    write_csv(outdir / "cue_evaluation.csv", rows)
    write_jsonl(outdir / "cue_evaluation.jsonl", rows)
    write_jsonl(
        outdir / "cue_errors.jsonl",
        [
            row for row in rows
            if row["constraint_strict_hit"] is not True
            or (row["shortcut_eligible"] and row["shortcut_strict_hit"] is not True)
        ],
    )
    write_summary(outdir / "summary.md", rows, summary)

    print(
        "Shortcut strict recall (high): "
        f"{summary['shortcut_high_hit']}/{summary['shortcut_high_n']}"
    )
    print(
        "Shortcut strict recall (all explicit): "
        f"{summary['shortcut_explicit_hit']}/{summary['shortcut_explicit_n']}"
    )
    print(
        "Constraint strict recall: "
        f"{summary['constraint_strict_hit']}/{summary['constraint_n']}"
    )
    print(
        "Constraint proposition overlap: "
        f"{summary['constraint_proposition_hit']}/{summary['constraint_n']}"
    )
    if results_path:
        print(
            "Constraint removed -> shortcut: "
            f"{summary['constraint_behavior_success_n']}/{summary['intervention_linked_n']}"
        )
        print(
            "Full C/C/S/C pattern: "
            f"{summary['full_pattern_success_n']}/{summary['intervention_linked_n']}"
        )
    print(f"Wrote cue evaluation to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
