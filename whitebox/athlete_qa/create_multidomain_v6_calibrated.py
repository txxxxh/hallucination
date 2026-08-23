#!/usr/bin/env python3
"""Create the probe-calibrated release slice from the v6 famous candidate set."""
from __future__ import annotations

import json
from pathlib import Path

DOMAINS = ("athlete", "musician", "building")
TARGET_TOTALS = {"athlete": 260, "musician": 265, "building": 300}
ROOT = Path(__file__).parent / "multidomain_v6_famous"
OUT = ROOT / "calibrated"


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def fame(item):
    profiles = item["profiles"].values()
    return min(profile.get("pageviews_60d", 0) for profile in profiles)


def main():
    states = {(row["domain"], row["id"]): row for row in read(ROOT / "gpt52_probe_eval/results.jsonl")}
    names = {(row["domain"], row["id"]): row for row in read(ROOT / "gpt52_eval/results.jsonl")
             if row["condition"] == "names"}
    report = {}
    selected_all = []
    for domain in DOMAINS:
        items = read(ROOT / domain / "primary_questions.jsonl")
        known = [item for item in items if states[(domain, item["id"])]["probe_state"] == "knows_both"]
        other = [item for item in items if states[(domain, item["id"])]["probe_state"] != "knows_both"]
        other.sort(key=lambda item: (-fame(item), item["id"]))
        selected = known + other[:TARGET_TOTALS[domain] - len(known)]
        selected.sort(key=lambda item: item["id"])
        for item in selected:
            item["calibration"] = {
                "model": "gpt-5.2-2025-12-11", "protocol": "closed-book binary probes",
                "probe_state": states[(domain, item["id"])]["probe_state"],
                "selection": "all knows_both plus highest-min-pageviews non-both items to target paper-level coverage",
            }
        write(OUT / domain / "primary_questions.jsonl", selected)
        write(OUT / domain / "prepend_names.jsonl", [{"id": item["id"], "prompt": item["prepend_names_prompt"],
            "rgt_ans": item["correct_answer"], "rgt_ans_qid": item["correct_answer_qid"],
            "wrg_ans": item["wrong_answer"], "wrg_ans_qid": item["wrong_answer_qid"]} for item in selected])
        write(OUT / domain / "prepend_profiles.jsonl", [{"id": item["id"], "prompt": item["prepend_profiles_prompt"],
            "rgt_ans": item["correct_answer"], "rgt_ans_qid": item["correct_answer_qid"],
            "wrg_ans": item["wrong_answer"], "wrg_ans_qid": item["wrong_answer_qid"]} for item in selected])
        write(OUT / domain / "probes.jsonl", [{"id": f"{item['id']}_probe_{index}", "parent_id": item["id"], **probe}
              for item in selected for index, probe in enumerate(item["probes"])])
        write(OUT / domain / "profiles.jsonl", sorted({profile["qid"]: profile for item in selected
              for profile in item["profiles"].values()}.values(), key=lambda profile: profile["qid"]))
        both = sum(states[(domain, item["id"])]["probe_state"] == "knows_both" for item in selected)
        main_correct = sum(names[(domain, item["id"])]["correct"] for item in selected)
        report[domain] = {"items": len(selected), "knows_both": both,
                          "knows_both_rate": both / len(selected),
                          "names_correct": main_correct, "names_accuracy": main_correct / len(selected)}
        selected_all.extend((domain, item) for item in selected)
    total = len(selected_all)
    total_both = sum(states[(domain, item["id"])]["probe_state"] == "knows_both"
                     for domain, item in selected_all)
    total_correct = sum(names[(domain, item["id"])]["correct"] for domain, item in selected_all)
    summary = {"dataset": "multidomain_v6_famous/calibrated",
               "warning": "GPT-5.2 probe-calibrated evaluation slice; not valid for unbiased GPT-5.2 knowledge evaluation",
               "selection_uses_primary_outcomes": False,
               "target_paper_knows_both_rate": 0.8554,
               "overall": {"items": total, "knows_both": total_both,
                           "knows_both_rate": total_both / total,
                           "names_correct": total_correct, "names_accuracy": total_correct / total,
                           "names_error_rate": 1 - total_correct / total},
               "by_domain": report}
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
