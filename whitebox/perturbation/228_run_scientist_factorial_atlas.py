#!/usr/bin/env python3
"""Run experiment 227 with strict partial-attribute grounding.

Scientist questions commonly mention only the informative suffix of a long
profile value.  This launcher installs the same strict phrase semantics used by
the binding experiments before invoking the factorial atlas.
"""
from __future__ import annotations

import importlib


atlas = importlib.import_module("227_scientist_factorial_interaction_atlas")


def strict_question_facts(row, profiles, builder):
    prompt = row["prompt"]
    qstart = prompt.rfind(atlas.QUESTION_MARKER)
    qstart = qstart + len(atlas.QUESTION_MARKER) if qstart >= 0 else 0
    question = prompt[qstart:]
    words = list(atlas.WORD_RE.finditer(question))
    qtokens = [atlas.norm(x.group()) for x in words]
    right = row["rgt_ans"]
    generic = {"award", "prize", "medal", "order", "university", "society",
               "college", "institute", "field", "member"}
    facts = []
    for profile in profiles:
        owner = "right" if profile["name"] == right else "wrong"
        for field in builder.FIELDS:
            for value in builder.values(profile, field):
                tokens = set(atlas.norm(value).split())
                if tokens:
                    facts.append({"field": field, "value": value,
                                  "tokens": tokens, "owner": owner})

    attrs, occupied = [], []
    for width in (4, 3, 2):
        for wi in range(len(words) - width + 1):
            span_tokens = set(qtokens[wi:wi + width])
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


atlas.question_facts = strict_question_facts


if __name__ == "__main__":
    atlas.main()
