#!/usr/bin/env python3
"""Official Query-Level-Uncertainty Internal Confidence on full Scientist.

The score follows tigerchen52/query_level_uncertainty: project the final query
tokens at every hidden layer onto the Yes/No LM-head rows, then aggregate the
P(Yes) matrix with the paper's positional weights.  We only materialize the
two requested vocabulary rows instead of the full vocabulary projection.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "278_scientist_internal_confidence"


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def positional(n, center=None, w=1.0):
    center = n-1 if center is None else center
    z = np.exp(-w*(center-np.arange(n, dtype=float))**2)
    return z/z.sum()


def one_token_id(tok, word):
    variants = (word, " "+word)
    for text in variants:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise RuntimeError(f"{word!r} is not a single tokenizer token")


def metrics(y, p):
    return {"n": int(len(y)), "positive": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for x in rows: union(x["right_qid"], x["wrong_qid"])
    return np.asarray([find(x["right_qid"]) for x in rows])


def collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    raw = {str(x["key"]): x for x in json.load((ROOT/"shuffled_prepend_names_question.json").open())}
    records = {str(x["key"]): x for x in read(ROOT/"tool_gate_correctness_names_llama31_8b/records.jsonl")}
    keys = sorted(k for k in raw if k in records and records[k].get("parse_valid", True))
    args.out.mkdir(parents=True, exist_ok=True)
    score_file = args.out/"scores.jsonl"
    done = {x["key"]: x for x in read(score_file)} if args.resume and score_file.exists() else {}
    keys = [k for k in keys if k not in done][:args.limit or None]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True).eval()
    yes, no = one_token_id(tok, "Yes"), one_token_id(tok, "No")
    weight = model.lm_head.weight[[yes, no]].float().detach()
    bias = None if model.lm_head.bias is None else model.lm_head.bias[[yes, no]].float().detach()
    mode = "a" if done else "w"
    with score_file.open(mode) as fh:
        for start in range(0, len(keys), args.batch):
            part = keys[start:start+args.batch]
            prompts = [tok.apply_chat_template([{"role":"user", "content":raw[k]["prompt"]}],
                       tokenize=False, add_generation_prompt=True) for k in part]
            z = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_length, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.model(input_ids=z.input_ids, attention_mask=z.attention_mask,
                                  output_hidden_states=True, use_cache=False)
            for bi, key in enumerate(part):
                valid = int(z.attention_mask[bi].sum()); begin = z.input_ids.shape[1]-valid
                take = slice(max(begin, z.input_ids.shape[1]-args.last_k), z.input_ids.shape[1])
                matrix = []
                for hidden in out.hidden_states:
                    logits = hidden[bi, take].float() @ weight.T
                    if bias is not None: logits += bias
                    matrix.append(torch.softmax(logits, -1)[:, 0].cpu().numpy())
                matrix = np.stack(matrix, 1)  # token x layer, as official code
                tw, lw = positional(len(matrix), w=args.locality_w), positional(matrix.shape[1], w=args.locality_w)
                score = float(np.sum(matrix*tw[:, None]*lw[None, :]))
                rec = {"key": key, "internal_confidence": score,
                       "p_yes_last_token_last_layer": float(matrix[-1, -1]),
                       "last_k": int(len(matrix)), "yes_token_id": yes, "no_token_id": no}
                fh.write(json.dumps(rec)+"\n"); fh.flush(); done[key] = rec
            print(f"IC {len(done)}/{len(records)}", flush=True)


def evaluate(args):
    records = {x["key"]: x for x in read(ROOT/"tool_gate_correctness_names_llama31_8b/records.jsonl")}
    manifest = {x["key"]: x for x in read(RUNS/"76_closedbook_fact_probe_manifest.jsonl")}
    probes = {x["key"]: x for x in read(RUNS/"77_closedbook_fact_probe_results.jsonl")}
    ic = {x["key"]: x for x in read(args.out/"scores.jsonl")}
    pp = {x["key"]: x for x in read(RUNS/"272_full_scientist_standard_upr_tables_rightqid/predictions.jsonl")}
    keys = sorted(set(ic)&set(pp)&set(manifest)&set(records))
    if len(keys) != 2894 and not args.allow_partial: raise RuntimeError(f"aligned {len(keys)}/2894")
    rows = [manifest[k] for k in keys]
    groups = np.asarray([x["right_qid"] for x in rows])
    y = np.asarray([int(not records[k]["correct"]) for k in keys])
    known = np.asarray([int(probes[k]["n_discriminative_facts"] >= 1 and probes[k]["binary_accuracy"] > .5 and probes[k]["pairwise_owner_accuracy"] > .5) for k in keys])
    conf = np.asarray([ic[k]["internal_confidence"] for k in keys]); pscore = np.asarray([pp[k]["p_error_probability"] for k in keys])
    raw_scores = {"IC_error": 1-conf, "P": pscore}; pred = {"P_plus_IC": np.zeros(len(y))}
    if len(keys) >= 20 and len(set(groups)) >= 2:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(conf, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=.03, class_weight="balanced", solver="liblinear", max_iter=5000))
            pred["P_plus_IC"][te] = m.fit(np.c_[pscore[tr], conf[tr]], y[tr]).predict_proba(np.c_[pscore[te], conf[te]])[:, 1]
    result = {k: metrics(y, v) for k, v in {**raw_scores, **pred}.items()}
    cells = {}
    for kval, kname in ((1,"known"),(0,"unknown")):
        mask = known == kval
        cells[kname] = {n: metrics(y[mask], s[mask]) for n, s in {**raw_scores, **pred}.items()}
    report = {"protocol":"official IC formula; full Scientist; IC standalone and grouped-OOF P+IC; no probe in detector features",
              "n":len(y), "known":int(known.sum()), "errors":int(y.sum()), "results":result, "by_knowledge":cells}
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report, indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command", choices=("collect","evaluate","all"))
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--last-k",type=int,default=6); ap.add_argument("--locality-w",type=float,default=1.0)
    ap.add_argument("--max-length",type=int,default=2048); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--resume",action="store_true"); ap.add_argument("--allow-partial",action="store_true"); ap.add_argument("--out",type=Path,default=OUT)
    a=ap.parse_args()
    if a.command in ("collect","all"): collect(a)
    if a.command in ("evaluate","all"): evaluate(a)
if __name__ == "__main__": main()
