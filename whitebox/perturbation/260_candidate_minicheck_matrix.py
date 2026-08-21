#!/usr/bin/env python3
"""MiniCheck on the frozen Llama candidate-conditioned benchmark matrix.

The official score checks only the generated/chosen candidate.  The contrastive
extension subtracts chosen support from the support assigned to the supplied
second candidate.  The latter is oracle-conditioned when the manifest's second
candidate is gold/reference-derived.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def load_rows(dataset):
    if dataset == "trivia":
        raw = read_jsonl(RUNS / "127_triviaqa_balanced_n1000.jsonl")
        return [dict(key=x["key"], group=x["key"], correct=int(x["correct"]),
                     document=x["context"], question=x["question"],
                     chosen=x["generation"], alternative=x["other_answer"])
                for x in raw]
    if dataset == "gsm8k":
        raw = read_jsonl(RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl")
        return [dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
                     document=x["question"], question=x["question"],
                     chosen=x["generation"], alternative=x["reference_solution"])
                for x in raw]
    if dataset == "drop":
        raw = read_jsonl(RUNS / "166_drop1000/drop_balanced_n1000.jsonl")
        return [dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
                     document=x["context"], question=x["question"],
                     chosen=x["generation"], alternative=x["other_answer"])
                for x in raw]
    raise ValueError(dataset)


def sentence_claims(dataset, question, answer):
    answer = str(answer).strip()
    if dataset in {"trivia", "drop"}:
        return [f'The answer to the question "{question.strip()}" is {answer}.']
    # GSM8K contains a multi-sentence solution.  Sentence-level checking is the
    # official MiniCheck recommendation; retain every nonempty reasoning/final
    # sentence and use the minimum support as the response score.
    answer = re.sub(r"\n+", " ", answer)
    claims = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if x.strip()]
    return claims or [answer]


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["trivia", "gsm8k", "drop"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/home/tong56/.cache/minicheck"))
    ap.add_argument("--out-root", type=Path,
                    default=RUNS / "260_candidate_minicheck_matrix")
    args = ap.parse_args()

    from minicheck.minicheck import MiniCheck

    rows = load_rows(args.dataset)
    out = args.out_root / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    requests, meta = [], []
    for i, row in enumerate(rows):
        for owner in ("chosen", "alternative"):
            claims = sentence_claims(args.dataset, row["question"], row[owner])
            for claim in claims:
                requests.append((row["document"], claim))
                meta.append((i, owner))

    checker = MiniCheck(model_name="flan-t5-large",
                        cache_dir=str(args.cache_dir))
    probs = []
    for start in range(0, len(requests), args.batch):
        part = requests[start:start + args.batch]
        _, raw_prob, _, _ = checker.score(
            docs=[x[0] for x in part], claims=[x[1] for x in part])
        probs.extend(float(x) for x in raw_prob)
        print(f"[{args.dataset}] {min(start + args.batch, len(requests))}/"
              f"{len(requests)}", flush=True)

    cells = {}
    for (i, owner), prob in zip(meta, probs):
        cells.setdefault((i, owner), []).append(prob)
    items = []
    for i, row in enumerate(rows):
        chosen = cells[i, "chosen"]
        alternative = cells[i, "alternative"]
        items.append({**row, "chosen_support_min": float(min(chosen)),
                      "alternative_support_min": float(min(alternative)),
                      "chosen_sentence_support": chosen,
                      "alternative_sentence_support": alternative})
    with (out / "items.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    y = 1 - np.asarray([x["correct"] for x in items])
    official = 1 - np.asarray([x["chosen_support_min"] for x in items])
    contrastive = np.asarray([x["alternative_support_min"] -
                              x["chosen_support_min"] for x in items])
    report = {
        "dataset": args.dataset,
        "model_under_test": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "checker": "MiniCheck-Flan-T5-Large",
        "n": len(items), "errors": int(y.sum()),
        "protocol": "same frozen candidate-conditioned manifest; sentence-level MiniCheck; response support=min sentence support",
        "alternative_warning": "second candidate is manifest-supplied and may be gold/reference-derived; contrastive result is oracle candidate-conditioned",
        "metrics": {"minicheck_evidence": metrics(y, official),
                    "minicheck_contrastive": metrics(y, contrastive)},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
