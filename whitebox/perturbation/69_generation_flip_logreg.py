# -*- coding: utf-8 -*-
"""Fixed-budget single-span generation stability detector.

Features are deployment-safe: correctness is used only as the training label.
For each question we mask sampled 2/3-word spans in mean-embedding space and
record whether greedy generation leaves the original answer option.
"""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from spanattr.core import Item, SpanAttributor, set_seed


def select_balanced(records, n, seed):
    rng = np.random.default_rng(seed)
    good = [r for r in records if r.get("parse_valid") and r["correct"]]
    bad = [r for r in records if r.get("parse_valid") and not r["correct"]]
    h = min(n // 2, len(good), len(bad))
    out = ([good[i] for i in rng.choice(len(good), h, replace=False)] +
           [bad[i] for i in rng.choice(len(bad), h, replace=False)])
    rng.shuffle(out)
    return out


def collect(args):
    import torch
    data = {str(x["key"]): x for x in json.load(open(args.data))}
    records = [json.loads(x) for x in open(args.records) if x.strip()]
    chosen = select_balanced(records, args.limit, args.seed)
    done = set()
    if args.resume and Path(args.features).exists():
        done = {json.loads(x)["key"] for x in open(args.features) if x.strip()}
    elif Path(args.features).exists():
        raise FileExistsError(f"{args.features} exists; use --resume")

    load_model = importlib.import_module("61_grad_span_proposal").load_model
    parse_choice = importlib.import_module("tool_gate_correctness_stratification").parse_choice
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.batch)
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    with open(args.features, "a") as fh:
        for ni, rr in enumerate(chosen):
            key = str(rr["key"])
            if key in done: continue
            raw = data[key]; original = str(rr["parsed_answer"])
            right, wrong = str(raw["rgt_ans"]), str(raw["wrg_ans"])
            other = wrong if original == right else right
            item = Item.from_dict(dict(raw, pred=original, gold=other))
            item.pred, item.gold = original, other
            prep = att.prepare(item)
            spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
            prep.spans = spans
            rng = np.random.default_rng(args.seed + 1000003 * int(key.split("_")[-1]))
            q = min(args.queries, len(spans))
            ids = np.sort(rng.choice(len(spans), q, replace=False))
            S0 = att.S0(prep)
            u, _ = att.u_of_sets(prep, [[int(i)] for i in ids], S0=S0)

            generations = []
            for st in range(0, q, args.batch):
                sub = ids[st:st + args.batch]
                A = torch.stack([att.alpha_from_spans(prep, [int(i)]) for i in sub])
                pe = att._embeds(prep, A)
                mask = torch.ones(pe.shape[:2], device=args.device, dtype=torch.long)
                with torch.no_grad():
                    g = model.generate(inputs_embeds=pe, attention_mask=mask,
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=getattr(tok, "pad_token_id", 0) or 0)
                generations.extend(tok.batch_decode(g, skip_special_tokens=True))
            parsed = [parse_choice(g, right, wrong)[0] for g in generations]
            valid = np.asarray([x is not None for x in parsed])
            flipped = np.asarray([x is not None and x != original for x in parsed])
            kept = np.asarray([x == original for x in parsed])
            flip_u = np.abs(u[flipped])
            base = [float(S0), float(abs(S0))]
            response = [float(np.mean(u)), float(np.std(u)), float(np.min(u)),
                float(np.max(u)), float(np.mean(np.abs(u))), float(np.max(np.abs(u))),
                float(np.quantile(np.abs(u), .5)), float(np.quantile(np.abs(u), .9)),
                float(np.mean(u > 0))]
            stability = [float(np.mean(flipped)), float(np.mean(kept)),
                float(np.mean(valid)), float(flipped.any()),
                float(flip_u.mean()) if len(flip_u) else 0.0,
                float(flip_u.min()) if len(flip_u) else 0.0,
                float(flip_u.max()) if len(flip_u) else 0.0]
            row = {"key": key, "group": str(raw.get("rgt_ans_qid", key)),
                "correct": bool(rr["correct"]), "original": original,
                "n_spans": len(spans), "sampled_span_ids": ids.tolist(),
                "base_features": base, "response_features": response,
                "stability_features": stability,
                "sampled": [{"text": spans[int(i)].text, "delta_margin": float(uu),
                    "generation": gg.strip(), "parsed": pp,
                    "flipped": bool(ff)} for i, uu, gg, pp, ff in
                    zip(ids, u, generations, parsed, flipped)]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{ni+1}/{len(chosen)}] {key} y={int(rr['correct'])} "
                  f"flip={flipped.mean():.3f} valid={valid.mean():.3f}", flush=True)


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = [json.loads(x) for x in open(args.features) if x.strip()]
    y = np.asarray([int(r["correct"]) for r in rows]); groups=np.asarray([r["group"] for r in rows])
    B=np.asarray([r["base_features"] for r in rows]); R=np.asarray([r["response_features"] for r in rows]); G=np.asarray([r["stability_features"] for r in rows])
    cv=StratifiedGroupKFold(args.folds,shuffle=True,random_state=args.seed)
    sets={"likelihood_only":B,"response_only":R,"generation_stability_only":G,
          "likelihood_plus_response":np.c_[B,R],"all_features":np.c_[B,R,G]}
    out={"n":len(y),"correct":int(y.sum()),"queries_per_item":args.queries,"folds":args.folds}
    for name,X in sets.items():
        est=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=5000,class_weight="balanced",random_state=args.seed))
        p=cross_val_predict(est,X,y,groups=groups,cv=cv,method="predict_proba")[:,1]; z=p>=.5
        out[name]={"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),
          "accuracy":float(accuracy_score(y,z)),"balanced_accuracy":float(balanced_accuracy_score(y,z))}
    Path(args.report).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["collect","train","all"])
    ap.add_argument("--data",default="../shuffled_prepend_names_question.json"); ap.add_argument("--records",default="../tool_gate_correctness_names_llama31_8b/records.jsonl")
    ap.add_argument("--features",default="runs/69_generation_flip_features.jsonl"); ap.add_argument("--report",default="runs/69_generation_flip_logreg.json")
    ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--dtype",default="float32"); ap.add_argument("--device",default="cuda")
    ap.add_argument("--limit",type=int,default=128); ap.add_argument("--queries",type=int,default=16); ap.add_argument("--batch",type=int,default=4); ap.add_argument("--max_new_tokens",type=int,default=16)
    ap.add_argument("--folds",type=int,default=5); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--resume",action="store_true")
    a=ap.parse_args(); set_seed(a.seed)
    if a.stage in ("collect","all"): collect(a)
    if a.stage in ("train","all"): train(a)
if __name__=="__main__": main()
