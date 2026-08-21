#!/usr/bin/env python3
"""Same-baseline U/R/P scores for the fixed DROP balanced-1000 manifest."""
from __future__ import annotations
import argparse, json, re, string
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def norm(s):
    s = str(s).lower().strip()
    s = "".join(" " if c in string.punctuation else c for c in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def entropy(xs):
    c = np.asarray(list(Counter(xs).values()), float); p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def prompt(x):
    return ("Read the passage and answer the question. Return only the shortest direct "
            "answer, with no explanation.\n\nPassage:\n" + x["context"] +
            "\n\nQuestion: " + x["question"])


def structural_scores(rows, cache):
    hs, groups, y, pscore = [], [], [], []
    for x in rows:
        with np.load(cache / f"{x['key']}.npz") as z:
            assert int(z["correct"]) == int(x["correct"])
            hs.append(z["layer14"].astype(np.float32)); groups.append(x["group"])
            y.append(int(not x["correct"]))
            pred, other = z["stage1_pred"].astype(float), z["stage1_other"].astype(float)
            effect = (pred[0] - other[0]) - (pred[1:] - other[1:])
            pscore.append(float(max(0.0, effect.max(initial=0.0))))
    hs = np.stack(hs); y = np.asarray(y); groups = np.asarray(groups)
    probs = []
    for seed in (42, 43, 44):
        out = np.zeros(len(rows)); cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(hs, y, groups):
            sc = StandardScaler().fit(hs[train]); a, b = sc.transform(hs[train]), sc.transform(hs[test])
            pc = PCA(44, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pc.transform(a), pc.transform(b)
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed).fit(a, y[train])
            out[test] = clf.predict_proba(b)[:, 1]
        probs.append(out)
    return np.mean(probs, axis=0), np.asarray(pscore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=RUNS/"166_drop1000/drop_balanced_n1000.jsonl")
    ap.add_argument("--cache", type=Path, default=RUNS/"167_drop1000_exact")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--samples", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=RUNS/"250_drop_unified_urp")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.manifest.open() if x.strip()]
    rscore, pscore = structural_scores(rows, a.cache)
    sample_file = a.out_dir / "samples.jsonl"
    done = {x["key"]: x for x in map(json.loads, sample_file.open())} if a.resume and sample_file.exists() else {}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval(); torch.manual_seed(20260820)
    mode = "a" if done else "w"
    pending = [x for x in rows if x["key"] not in done]
    with sample_file.open(mode) as f:
        for st in range(0, len(pending), a.batch):
            part = pending[st:st+a.batch]
            texts = [tok.apply_chat_template([{"role":"user","content":prompt(x)}],
                     tokenize=False, add_generation_prompt=True) for x in part]
            z = tok(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                g = model.generate(**z, do_sample=True, temperature=.7, top_p=.95,
                    num_return_sequences=a.samples, max_new_tokens=24,
                    pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(g[:, z.input_ids.shape[1]:], skip_special_tokens=True)
            for i, x in enumerate(part):
                vals = [norm(v.strip().split("\n")[0]) for v in outs[i*a.samples:(i+1)*a.samples]]
                k = a.samples//2; gold = norm(x["other_answer"])
                mode_answer = Counter(vals[k:]).most_common(1)[0][0]
                rec = {"key":x["key"], "u_score":entropy(vals[:k]), "samples":vals,
                       "heldout_majority_correct":int(mode_answer == gold)}
                done[x["key"]] = rec; f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
            print(f"U {len(done)}/{len(rows)}", flush=True)
    out = []
    for i, x in enumerate(rows):
        out.append({"key":x["key"], "group":x["group"], "error":int(not x["correct"]),
                    "u_score":done[x["key"]]["u_score"], "r_score":float(rscore[i]),
                    "p_score":float(pscore[i]),
                    "heldout_majority_correct":done[x["key"]]["heldout_majority_correct"]})
    with (a.out_dir/"items.jsonl").open("w") as f:
        for x in out: f.write(json.dumps(x)+"\n")
    report={"protocol":"DROP fixed balanced-1000; U first3/heldout3; R group-OOF L14; P max signed positive exact neutralization effect",
            "n":len(out),"errors":sum(x["error"] for x in out),
            "u_quantiles":np.quantile([x["u_score"] for x in out],[.3,.7]).tolist(),
            "p_positive_errors":sum(x["error"] and x["p_score"]>0 for x in out)}
    (a.out_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
