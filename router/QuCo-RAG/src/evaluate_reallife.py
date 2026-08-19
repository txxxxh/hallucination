#!/usr/bin/env python3
"""Paired evaluation for RealLifeChoice QuCo-RAG reproduction."""
import argparse, json, re
from pathlib import Path


def parse_choice(text):
    hits = re.findall(r"\boption\s*([12])\b", str(text), re.I)
    if not hits:
        hits = re.findall(r"(?:answer\s*(?:is|:)?\s*)([12])\b", str(text), re.I)
    return int(hits[-1]) if hits else None


def load_gold(path):
    rows = json.loads(Path(path).read_text())
    return {str(r.get("key", f"reallife-{i:04d}")): int(r["answer"])
            for i, r in enumerate(rows)}


def load_arm(path, gold):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out = {}
    for row in rows:
        qid = str(row["qid"])
        pred = parse_choice(row.get("prediction", ""))
        out[qid] = {"pred": pred, "correct": pred == gold.get(qid),
                    "retrieved": int(row.get("retrieve_count", 0)) > 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="/home/tong56/whitebox/question_and_result.json")
    ap.add_argument("--wo-rag", required=True)
    ap.add_argument("--sr-rag", required=True)
    ap.add_argument("--quco-rag", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    gold = load_gold(args.gold)
    arms = {"wo_rag": load_arm(args.wo_rag, gold),
            "sr_rag": load_arm(args.sr_rag, gold),
            "quco_rag": load_arm(args.quco_rag, gold)}
    report = {"n_gold": len(gold), "arms": {}, "paired_vs_wo_rag": {}}
    for name, arm in arms.items():
        valid = [v for v in arm.values() if v["pred"] is not None]
        report["arms"][name] = {
            "n": len(arm), "parse_rate": len(valid) / max(1, len(arm)),
            "accuracy": sum(v["correct"] for v in arm.values()) / max(1, len(arm)),
            "retrieval_trigger_rate": sum(v["retrieved"] for v in arm.values()) / max(1, len(arm)),
        }
    base = arms["wo_rag"]
    for name in ("sr_rag", "quco_rag"):
        common = sorted(set(base) & set(arms[name]))
        fixed = sum(not base[q]["correct"] and arms[name][q]["correct"] for q in common)
        harmed = sum(base[q]["correct"] and not arms[name][q]["correct"] for q in common)
        report["paired_vs_wo_rag"][name] = {
            "n_paired": len(common), "wrong_to_right": fixed, "right_to_wrong": harmed,
            "net_flips": fixed - harmed,
        }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
