#!/usr/bin/env python3
"""Official SeSE short-form method on the frozen four-benchmark matrix.

Core graph construction and structural entropy are imported directly from the
official SELGroup/SeSE checkout pinned under third_party/SeSE.  This adapter only
supplies our frozen rows/prompts, resumable generation, and AUROC/AUPRC reporting.
"""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OFFICIAL = HERE / "third_party" / "SeSE" / "sentence_structural_entropy"
N = 10

base = importlib.import_module("261_paper_baseline_matrix")

def read(path):
    p = Path(path)
    return [json.loads(line) for line in p.open() if line.strip()] if p.exists() else []

def sample(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rs = base.rows(args.dataset)
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    prior = read(path) if args.resume else []
    done = {x["key"] for x in prior}
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                        local_files_only=args.local_files_only)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).eval()
    with path.open("a" if args.resume else "w") as fh:
        for st in range(0, len(rs), args.batch):
            part = [r for r in rs[st:st+args.batch] if r["key"] not in done]
            if part:
                prompts = [tok.apply_chat_template(
                    [{"role": "user", "content": base.user_text(args.dataset, r)}],
                    tokenize=False, add_generation_prompt=True) for r in part]
                z = tok(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
                torch.manual_seed(args.seed + st)
                with torch.inference_mode():
                    g = model.generate(
                        **z, do_sample=True, temperature=1.0,
                        num_return_sequences=N,
                        max_new_tokens=192 if args.dataset == "gsm8k" else 100,
                        pad_token_id=tok.pad_token_id,
                    )
                texts = tok.batch_decode(g[:, z.input_ids.shape[1]:],
                                         skip_special_tokens=True)
                for i, r in enumerate(part):
                    rec = {
                        "key": r["key"], "correct": int(r["correct"]),
                        "question": r.get("question", ""),
                        "primary_response": str(r["pred"]),
                        "responses": texts[i*N:(i+1)*N],
                        "sampling": {"n": N, "temperature": 1.0},
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add(r["key"])
            print(args.dataset, min(st+args.batch, len(rs)), "/", len(rs), flush=True)

def score(args):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not OFFICIAL.exists():
        raise FileNotFoundError(f"Official SeSE checkout missing: {OFFICIAL}")
    if not args.skip_enhancement and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Official mode requires OPENAI_API_KEY for its GPT-4o response-enhancement step")
    # The official module constructs its OpenAI client at import time.
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
    sys.path.insert(0, str(OFFICIAL))
    from src.uncertainty_measures import construct_semantic_graph as graph
    from src.uncertainty_measures.structural_entropy import compute_se
    if args.skip_enhancement:
        graph.enhancing_answers = lambda responses, question: responses

    out = args.out / args.dataset
    records = read(out / "samples.jsonl")
    score_path = out / "scores.jsonl"
    old = read(score_path) if args.resume else []
    done = {x["key"] for x in old}
    with score_path.open("a" if args.resume else "w") as fh:
        for ix, r in enumerate(records, 1):
            if r["key"] in done:
                continue
            # These are the two unmodified official calls used by its own
            # uncertainty_quantification.py.
            matrix = graph.build_semantic_graph(r["responses"], r["question"])
            value = float(compute_se(matrix))
            rec = {"key": r["key"], "correct": int(r["correct"]),
                   "structural_entropy": value}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(r["key"])
            if ix % 10 == 0:
                print(args.dataset, ix, "/", len(records), flush=True)

    vals = read(score_path)
    y = np.asarray([1-int(x["correct"]) for x in vals])
    u = np.asarray([x["structural_entropy"] for x in vals])
    report = {
        "dataset": args.dataset, "n": len(y), "errors": int(y.sum()),
        "method": "SeSE short-form (Zhao et al., UAI 2026)",
        "official_source": "SELGroup/SeSE",
        "official_commit": args.official_commit,
        "protocol": "official graph construction and compute_se; frozen original benchmark/prompts/primary labels",
        "sampling": {"n": N, "temperature": 1.0},
        "response_enhancement": "official GPT-4o" if not args.skip_enhancement else "disabled (explicit ablation; not paper-faithful)",
        "auroc": float(roc_auc_score(y, u)),
        "auprc": float(average_precision_score(y, u)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sample", "score"])
    p.add_argument("dataset", choices=["scientist", "trivia", "gsm8k", "drop"])
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-enhancement", action="store_true")
    p.add_argument("--out", type=Path, default=RUNS/"285_sese_official")
    p.add_argument("--official-commit", default="8d4c6c5ceab61e6d973129785b82480fb0c572c3")
    a = p.parse_args()
    (sample if a.stage == "sample" else score)(a)

if __name__ == "__main__":
    main()
