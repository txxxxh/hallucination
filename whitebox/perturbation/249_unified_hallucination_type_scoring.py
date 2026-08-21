#!/usr/bin/env python3
"""Unified, same-baseline hallucination-type scoring for Scientist/TriviaQA/GSM8K.

The output is deliberately a multi-label *hallucination type* table, not a table
of raw U/E/P/R detector positives.  Every benchmark is joined by item key to one
fixed baseline correctness label.  High/low always means the top/bottom 30% of
the corresponding score on that fixed baseline population.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "249_unified_hallucination_types"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def keyed(path: Path):
    return {str(x["key"]): x for x in read_jsonl(path)}


def percentile(values):
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values)


def attach_percentile(rows, field, out):
    q = percentile([x[field] for x in rows])
    for x, v in zip(rows, q):
        x[out] = float(v)


def scientist():
    base = read_jsonl(RUNS / "226_four_axis_taxonomy_audit" / "items.jsonl")
    e = keyed(RUNS / "228_scientist_p_e_confirmation" / "e_items.jsonl")
    p = keyed(RUNS / "228_scientist_p_e_confirmation" / "p_items.jsonl")
    rows = []
    for x in base:
        z = dict(x)
        z["benchmark"] = "Scientist-Names"
        z["margin"] = e[z["key"]]["names_margin"]
        z["e_gain"] = e[z["key"]]["e_repair_gain"]
        z["p_specific_gain"] = p.get(z["key"], {}).get("p_specific_gain")
        rows.append(z)
    return rows, "completion"


def trivia():
    manifest = read_jsonl(RUNS / "127_triviaqa_balanced_n1000.jsonl")
    u = keyed(RUNS / "232_trivia_u_split_confirmation" / "samples.jsonl")
    r = keyed(RUNS / "238_trivia_question_end_r_confirmation" / "items.jsonl")
    # 230 preserves the manifest generation/correctness.  231 is excluded because
    # it generates a new closed-book answer and therefore changes the baseline.
    e = keyed(RUNS / "230_trivia_lexical_e_confirmation" / "items.jsonl")
    p = keyed(RUNS / "235_trivia_context_p_confirmation" / "items.jsonl")
    rows = []
    for b in manifest:
        key = str(b["key"])
        assert int(not b["correct"]) == u[key]["greedy_error"] == r[key]["error"] == e[key]["error"]
        rows.append({
            "key": key, "benchmark": "TriviaQA", "error": int(not b["correct"]),
            "u_score": u[key]["u_score"], "r_score": r[key]["r_score"],
            "e_score": e[key]["e_score"], "p_score": p.get(key, {}).get("p_score", 0.0),
            "margin": r[key]["base_margin"], "e_gain": e[key]["evidence_gain"],
            "p_specific_gain": p.get(key, {}).get("specific_gain"),
        })
    for axis in "urep":
        attach_percentile(rows, f"{axis}_score", f"{axis}_percentile")
    return rows, "completion"


def gsm8k():
    manifest = keyed(RUNS / "140_gsm8k_natural" / "natural_balanced_n942.jsonl")
    # Full balanced-942 extension of the same original CoT baseline/protocol.
    u = read_jsonl(RUNS / "253_gsm8k_full_cot_u" / "merged_items.jsonl")
    r_all = keyed(RUNS / "233_gsm8k_question_end_r_confirmation" / "items.jsonl")
    p = keyed(RUNS / "234_gsm8k_p_neighborhood_confirmation" / "items.jsonl")
    rows = []
    for ux in u:
        key = str(ux["key"]); b = manifest[key]; rx = r_all[key]
        assert int(not b["correct"]) == ux["greedy_error"] == rx["error"]
        rows.append({
            "key": key, "benchmark": "GSM8K", "error": int(not b["correct"]),
            "u_score": ux["u_score"], "r_score": rx["r_score"],
            "p_score": p.get(key, {}).get("p_score", 0.0), "margin": rx["base_margin"],
            "p_specific_gain": p.get(key, {}).get("specific_gain"),
        })
    for axis in "urp":
        attach_percentile(rows, f"{axis}_score", f"{axis}_percentile")
    return rows, "none"


def drop():
    manifest = keyed(RUNS / "166_drop1000" / "drop_balanced_n1000.jsonl")
    scores = read_jsonl(RUNS / "250_drop_unified_urp" / "items.jsonl")
    rows = []
    for z in scores:
        key = str(z["key"]); b = manifest[key]
        assert int(not b["correct"]) == z["error"]
        rows.append({
            "key": key, "benchmark": "DROP", "error": z["error"],
            "u_score": z["u_score"], "r_score": z["r_score"],
            "p_score": z["p_score"],
            # DROP has localization but no independent matched-placebo run.
            # This flag exposes P-high as candidate coverage; the table marks it dagger.
            "p_specific_gain": 1.0 if z["p_score"] > 0 else None,
        })
    for axis in "urp":
        attach_percentile(rows, f"{axis}_score", f"{axis}_percentile")
    return rows, "none"


def scientist_profiles():
    src = read_jsonl(RUNS / "252_scientist_profiles_unified_uepr" / "items.jsonl")
    rows = []
    for x in src:
        z = dict(x); z["benchmark"] = "Scientist-Profiles"
        rows.append(z)
    for axis in "urep":
        attach_percentile(rows, f"{axis}_score", f"{axis}_percentile")
    return rows, "grounded"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [float(c-h), float(c+h)]


def label(rows, e_mode):
    # Quantile-value cutoffs (inclusive) reproduce the original split protocol.
    # This matters for discrete entropy: many perfectly stable samples tie at U=0.
    u_lo_cut, u_hi_cut = np.quantile([x["u_score"] for x in rows], [.3, .7])
    for x in rows:
        u_hi = x["u_score"] >= u_hi_cut
        u_lo = x["u_score"] <= u_lo_cut
        x["u_low_cutoff"] = float(u_lo_cut)
        x["u_high_cutoff"] = float(u_hi_cut)
        # Preserve the taxonomy's original R decision rule: an OOF error
        # probability below 0.5 lies in the correct-like region.
        r_correct_like = x["r_score"] < .5
        p_confirmed = (x.get("p_specific_gain") is not None and
                       x["p_percentile"] >= .7 and x["p_specific_gain"] > 0)
        evidence_confirmed = (e_mode == "completion" and x["e_percentile"] >= .7 and x["e_gain"] > 0)
        grounded_unsupported = (e_mode == "grounded" and x["e_percentile"] >= .7 and
                                x["e_score"] > 0 and not u_hi)
        types = {
            "knowledge_missing": bool(evidence_confirmed and u_hi),
            "unsupported_confident_fabrication": bool((evidence_confirmed and not u_hi) or grounded_unsupported),
            "context_induced": bool(p_confirmed),
            "reasoning_unstable": bool(u_hi),
            "stable_self_consistent": bool(u_lo and r_correct_like),
        }
        x["hallucination_types"] = [k for k, v in types.items() if v] if x["error"] else []
        x["mixed"] = bool(x["error"] and len(x["hallucination_types"]) >= 2)
        x["unclassified"] = bool(x["error"] and not x["hallucination_types"])
    return rows


def summarize(rows, e_mode):
    errors = [x for x in rows if x["error"]]
    names = ["knowledge_missing", "unsupported_confident_fabrication", "context_induced",
             "reasoning_unstable", "stable_self_consistent"]
    result = {"n": len(rows), "n_hallucination": len(errors), "e_mode": e_mode, "types": {}}
    for name in names:
        if (name == "knowledge_missing" and e_mode != "completion") or (name == "unsupported_confident_fabrication" and e_mode == "none"):
            result["types"][name] = {"n": None, "proportion": None, "ci95": None}
            continue
        k = sum(name in x["hallucination_types"] for x in errors)
        result["types"][name] = {"n": k, "proportion": k/len(errors), "ci95": wilson(k, len(errors))}
    for name in ["mixed", "unclassified"]:
        k = sum(x[name] for x in errors)
        result["types"][name] = {"n": k, "proportion": k/len(errors), "ci95": wilson(k, len(errors))}
    result["overlap_patterns"] = dict(Counter("+".join(x["hallucination_types"]) or "unclassified" for x in errors))
    return result


def latex(report):
    order = [
        ("knowledge_missing", "知识缺失"),
        ("unsupported_confident_fabrication", "无依据但自信的编造"),
        ("context_induced", "上下文诱导/局部错误依赖"),
        ("reasoning_unstable", "推理不稳定"),
        ("stable_self_consistent", "稳定自洽错误"),
        ("mixed", "多因素混合"),
        ("unclassified", "未归类"),
    ]
    benches = list(report)
    lines = [r"\begin{tabular}{l" + "c"*len(benches) + "}", r"\toprule",
             "幻觉类型 & " + " & ".join(f"{b} ($n_h={report[b]['n_hallucination']}$)" for b in benches) + r" \\", r"\midrule"]
    for key, title in order:
        cells = []
        for b in benches:
            z = report[b]["types"][key]
            cells.append("--" if z["n"] is None else f"{z['n']} ({100*z['proportion']:.1f}\\%)")
        lines.append(title + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [scientist(), scientist_profiles(), trivia(), gsm8k(), drop()]
    all_rows = []
    report = {}
    for rows, e_mode in datasets:
        rows = label(rows, e_mode)
        all_rows.extend(rows)
        report[rows[0]["benchmark"]] = summarize(rows, e_mode)
    report["_protocol"] = {
        "population": "fixed baseline generations; proportions condition on baseline hallucinations",
        "thresholds": "within fixed baseline population inclusive 30/70 quantile-value cutoffs; R correct-like uses frozen probability < .5",
        "multi_label": True,
        "knowledge_missing": "U-high AND E-high AND positive evidence-completion response",
        "unsupported_confident_fabrication": "not U-high AND E-high AND positive evidence-completion response",
        "profiles_unsupported": "Scientist-Profiles: not U-high AND E-high AND alternative profile support exceeds generated profile support",
        "context_induced": "P-high AND target intervention specific gain over matched placebo > 0",
        "reasoning_unstable": "U-high",
        "stable_self_consistent": "U-low AND group-OOF R error probability < .5 (correct-like)",
        "warning": "types are operational signatures; per-item labels are not human causal ground truth",
        "drop_p_warning": "DROP context-induced is P-high candidate coverage only; no independent matched-placebo confirmation",
    }
    with (OUT / "items.jsonl").open("w") as f:
        for x in all_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUT / "table.tex").write_text(latex({k: v for k, v in report.items() if not k.startswith("_")}))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(latex({k: v for k, v in report.items() if not k.startswith("_")}))


if __name__ == "__main__":
    main()
