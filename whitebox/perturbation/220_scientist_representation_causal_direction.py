#!/usr/bin/env python3
"""Cross-fitted causal intervention on the Scientist representation direction.

The direction is estimated as stable-error minus stable-correct on training folds
only.  It is then added at the final question token of held-out prompts.  Negative
doses test repair of TP/FN errors; positive doses test induction on TN controls.
A fold-specific random orthogonal direction of identical norm is the placebo.
"""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CACHE = RUNS / "141_scientist_all_trajectory_l8"


def layers(model):
    return model.model.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14, help="hidden-state index; patches block layer-1")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--doses", type=float, nargs="+", default=[-2., -1., -.5, 0., .5, 1., 2.])
    ap.add_argument("--out-dir", type=Path, default=RUNS / "220_scientist_representation_causal_direction")
    a = ap.parse_args()
    import torch

    a.out_dir.mkdir(parents=True, exist_ok=True)
    tax = {x["key"]: x for x in map(json.loads, (RUNS / "219_scientist_semantic_neighborhood_full" / "items.jsonl").open())}
    rep = {x["key"]: x for x in map(json.loads, (RUNS / "216_known_error_representation_trajectory_predictions.jsonl").open())}
    raw = {str(x["key"]): x for x in json.load((HERE.parent / "shuffled_prepend_names_question.json").open())}
    jobs = {x[0]: x for x in importlib.import_module("152_scientist_attention_pruned_current127").jobs()}
    rows = []
    for key, t in tax.items():
        if not (t["stable_correct"] or t["stable_systematic_error"]):
            continue
        _, group, _, prompt, pred, other = jobs[key]
        generation_correct = bool(raw[key].get("rgt_ans") == pred)
        right, wrong = (pred, other) if generation_correct else (other, pred)
        with np.load(CACHE / f"{key}.npz") as z:
            li = list(z["layers"].astype(int)).index(a.layer)
            h = z["last"][li].astype(np.float32)
        err = int(t["stable_systematic_error"])
        detected = rep[key]["delta_trajectory"] >= .5
        category = ("TP" if detected else "FN") if err else ("FP" if detected else "TN")
        rows.append(dict(key=key, group=group, prompt=prompt, right=right, wrong=wrong,
                         error=err, category=category, hidden=h))

    y = np.array([r["error"] for r in rows]); groups = np.array([r["group"] for r in rows])
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    directions = np.zeros((len(rows), len(rows[0]["hidden"])), np.float32)
    controls = np.zeros_like(directions); fold_id = np.zeros(len(rows), int)
    rng = np.random.default_rng(42)
    for fold, (train, test) in enumerate(cv.split(np.zeros(len(rows)), y, groups)):
        h = np.stack([rows[i]["hidden"] for i in train])
        d = h[y[train] == 1].mean(0) - h[y[train] == 0].mean(0)
        # Placebo is random, explicitly orthogonal to the causal direction.
        q = rng.normal(size=len(d)).astype(np.float32)
        q -= d * (np.dot(q, d) / (np.dot(d, d) + 1e-12)); q *= np.linalg.norm(d) / np.linalg.norm(q)
        directions[test] = d; controls[test] = q; fold_id[test] = fold

    loader = importlib.import_module("61_grad_span_proposal")
    model, tok = loader.load_model(a.model, "bfloat16", "cuda")
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    block = layers(model)[a.layer - 1]

    def score(kind, dose):
        result = np.zeros((len(rows), 2), np.float32)
        vecs = directions if kind == "causal" else controls
        requests = [(i, owner, r[owner]) for i, r in enumerate(rows) for owner in ("wrong", "right")]
        for start in range(0, len(requests), a.batch):
            part = requests[start:start + a.batch]
            prefixes = [tok.apply_chat_template([{"role":"user", "content":rows[i]["prompt"]}], tokenize=False, add_generation_prompt=True) for i,_,_ in part]
            pids = [tok.encode(x, add_special_tokens=False) for x in prefixes]
            aids = [tok.encode(" " + ans, add_special_tokens=False) for _,_,ans in part]
            seqs = [p + z for p,z in zip(pids,aids)]; width = max(map(len, seqs))
            ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long, device=model.device)
            mask = torch.zeros_like(ids); ends = []
            for j,(seq,p) in enumerate(zip(seqs,pids)):
                pad = width-len(seq); ids[j,pad:] = torch.tensor(seq,device=model.device); mask[j,pad:] = 1; ends.append(pad+len(p)-1)
            delta = torch.tensor(np.stack([vecs[i] for i,_,_ in part]), device=model.device, dtype=torch.bfloat16) * dose
            def hook(_m,_inp,out):
                h = out[0] if isinstance(out,tuple) else out; z=h.clone()
                for j,pos in enumerate(ends): z[j,pos] += delta[j]
                return (z,*out[1:]) if isinstance(out,tuple) else z
            handle = block.register_forward_hook(hook)
            try:
                with torch.inference_mode(): lp=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float().log_softmax(-1)
            finally: handle.remove()
            for j,(i,owner,_) in enumerate(part):
                ans_start=ends[j]+1; pos=torch.arange(ans_start-1,ans_start+len(aids[j])-1,device=model.device); target=torch.tensor(aids[j],device=model.device)
                result[i,0 if owner=="wrong" else 1]=float(lp[j,pos,target].mean().cpu())
        return result[:,0]-result[:,1]

    curves = {kind: {str(d): score(kind,d).tolist() for d in a.doses} for kind in ("causal","placebo")}
    base = np.array(curves["causal"]["0.0"])
    items=[]
    for i,r in enumerate(rows):
        items.append({"key":r["key"],"group":r["group"],"error":r["error"],"category":r["category"],"fold":int(fold_id[i]),"base_wrong_minus_right":float(base[i]),"causal":{k:v[i] for k,v in curves["causal"].items()},"placebo":{k:v[i] for k,v in curves["placebo"].items()}})
    with (a.out_dir/"items.jsonl").open("w") as f:
        for x in items:f.write(json.dumps(x)+"\n")
    def group_summary(cat):
        ix=np.array([r["category"]==cat for r in rows]); out={"n":int(ix.sum()),"base_margin":float(base[ix].mean())}
        for kind in curves:
            out[kind]={d:{"mean_margin":float(np.mean(np.array(v)[ix])),"mean_delta":float(np.mean(np.array(v)[ix]-base[ix])),"wrong_preferred_rate":float(np.mean(np.array(v)[ix]>0))}for d,v in curves[kind].items()}
        return out
    report={"protocol":"stable taxonomy; 5-fold group-cross-fitted error-minus-correct direction; held-out question-end residual intervention; equal-norm orthogonal placebo","n":len(rows),"layer_hidden_index":a.layer,"doses":a.doses,"groups":{c:group_summary(c) for c in ("TN","FP","TP","FN")}}
    (a.out_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
