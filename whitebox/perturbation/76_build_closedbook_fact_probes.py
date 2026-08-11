#!/usr/bin/env python3
"""Build closed-book atomic fact probes for all ScientistQA items."""

import json
import re
from pathlib import Path

DATA = Path("/home/tong56/whitebox/shuffled_prepend_profiles_question.json")
OUT = Path("/home/tong56/whitebox/perturbation/runs/76_closedbook_fact_probe_manifest.jsonl")
SUMMARY = Path("/home/tong56/whitebox/perturbation/runs/76_closedbook_fact_probe_manifest_summary.json")

FIELDS = {
    "award_received": "receive the award or honor '{value}'",
    "position_held": "hold the position '{value}'",
    "education": "attend or receive education at '{value}'",
    "occupation": "work as a '{value}'",
    "field": "work in the field '{value}'",
    "place_of_birth": "be born in '{value}'",
    "place_of_death": "die in '{value}'",
    "notable_work": "be known for the work '{value}'",
}


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def parse_item(row):
    prompt = row["prompt"]
    prefix = "Given two profiles of two persons:\n"
    body = prompt[len(prefix):]
    profiles_text, question = body.split(
        "\nChoose exactly one profile from the two, and output the name of the person as the answer to the following question:\n",
        1,
    )
    chunks = re.split(r"(?m)^name: ", profiles_text)[1:]
    profiles = []
    for chunk in chunks:
        lines = chunk.splitlines()
        profile = {"name": lines[0].strip()}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                profile[key.strip()] = value.strip()
        profiles.append(profile)
    assert len(profiles) == 2, (row["key"], len(profiles))
    return profiles, question


def values(profile, field):
    raw = profile.get(field, "")
    return [x.strip(" ,") for x in raw.split(";") if len(norm(x)) >= 4]


def main():
    rows = json.load(open(DATA))
    counts, total = [], 0
    with open(OUT, "w") as out:
        for row in rows:
            profiles, question = parse_item(row)
            qnorm = norm(question)
            facts = []
            for field, template in FIELDS.items():
                sets = [{norm(v): v for v in values(p, field)} for p in profiles]
                for owner in (0, 1):
                    other = 1 - owner
                    for canonical, value in sets[owner].items():
                        if canonical in sets[other] or canonical not in qnorm:
                            continue
                        facts.append({"field": field, "value": value, "owner": owner,
                                      "relation": template.format(value=value)})
            # Remove duplicates caused by repeated profile values.
            unique = {(f["field"], norm(f["value"]), f["owner"]): f for f in facts}
            facts = list(unique.values())
            probes = []
            for fact_id, fact in enumerate(facts):
                for person_id, profile in enumerate(profiles):
                    probes.append({
                        "probe_id": f"{row['key']}::f{fact_id}::p{person_id}",
                        "person": profile["name"],
                        "field": fact["field"],
                        "value": fact["value"],
                        "gold_yes": person_id == fact["owner"],
                        "prompt": (
                            "Answer the factual question from your own knowledge. "
                            "Do not infer from any supplied biography. Answer exactly Yes or No.\n"
                            f"Question: Did {profile['name']} {fact['relation']}?"
                        ),
                    })
            record = {
                "key": row["key"], "right_answer": row["rgt_ans"],
                "wrong_answer": row["wrg_ans"], "right_qid": row["rgt_ans_qid"],
                "wrong_qid": row["wrg_ans_qid"], "question": question,
                "n_discriminative_facts": len(facts), "probes": probes,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts.append(len(facts)); total += len(probes)
    summary = {
        "n_items": len(rows), "n_atomic_probes": total,
        "items_with_at_least_1_fact": sum(x >= 1 for x in counts),
        "items_with_at_least_2_facts": sum(x >= 2 for x in counts),
        "items_with_at_least_3_facts": sum(x >= 3 for x in counts),
        "mean_discriminative_facts": sum(counts) / len(counts),
        "max_discriminative_facts": max(counts),
    }
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
