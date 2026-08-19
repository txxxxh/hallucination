#!/usr/bin/env python3
"""Final phase-1 launcher: strict cues in the actual Scientist question only."""
from __future__ import annotations

import importlib


runner = importlib.import_module("228_run_scientist_factorial_atlas")
atlas = runner.atlas
card = importlib.import_module("204_scientist_binding_override_pilot")
_PROMPTS = {x[0]: x[3] for x in importlib.import_module(
    "152_scientist_attention_pruned_current127").jobs()}


def question_only_facts(row, profiles, builder):
    prompt = _PROMPTS[str(row["key"])]
    _, question = builder.parse_item(row)
    qstart = prompt.find(question)
    if qstart < 0:
        raise RuntimeError(f"question text not found in names prompt: {row['key']}")
    words = list(atlas.WORD_RE.finditer(question))
    right = row["rgt_ans"]
    generic = {"award", "prize", "medal", "order", "university", "society",
               "college", "institute", "field", "member"}
    facts = []
    for profile in profiles:
        owner = "right" if profile["name"] == right else "wrong"
        for field in builder.FIELDS:
            for value in builder.values(profile, field):
                tokens = card.toks(value)
                if tokens:
                    facts.append({"field": field, "value": value,
                                  "tokens": tokens, "owner": owner})

    attrs, occupied = [], []
    for width in (4, 3, 2):
        for wi in range(len(words) - width + 1):
            text = question[words[wi].start():words[wi + width - 1].end()]
            span_tokens = card.toks(text)
            if not span_tokens:
                continue
            matches = []
            for fact in facts:
                overlap = span_tokens & fact["tokens"]
                abstract = fact["field"] in {
                    "occupation", "field", "position_held"}
                valid = ((abstract and fact["tokens"] <= span_tokens)
                         or len(overlap - generic) >= 2
                         or (len(overlap) >= 2 and len(overlap - generic) >= 1))
                if valid:
                    score = (len(overlap) / max(1, len(span_tokens | fact["tokens"])),
                             len(overlap - generic), len(overlap))
                    matches.append((score, fact))
            if not matches:
                continue
            score = max(x[0] for x in matches)
            best = [x[1] for x in matches if x[0] == score]
            start = qstart + words[wi].start()
            end = qstart + words[wi + width - 1].end()
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            attrs.append({
                "char_start": start, "char_end": end, "text": prompt[start:end],
                "kind": "attribute", "field": best[0]["field"],
                "value": best[0]["value"],
                "owners": sorted(set(x["owner"] for x in best)),
                "negated": False,
            })
            occupied.append((start, end))

    for rec in attrs:
        left = prompt[max(qstart, rec["char_start"] - 80):rec["char_start"]]
        rec["negated"] = bool(atlas.LOGIC_RE.search(left))
    logic = [{
        "char_start": qstart + match.start(), "char_end": qstart + match.end(),
        "text": match.group(), "kind": "logic", "field": "logic",
        "value": match.group().casefold(), "owners": [], "negated": True,
    } for match in atlas.LOGIC_RE.finditer(question)]
    return attrs + logic


atlas.question_facts = question_only_facts


if __name__ == "__main__":
    atlas.main()
