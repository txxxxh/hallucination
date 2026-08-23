#!/usr/bin/env python3
"""100-person mirrored B/F binding dose-response experiment."""
from __future__ import annotations
import argparse, gc, json, random, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
base = __import__("216_3b_negation_binding_dose")

FIRST = ["Liora", "Marek", "Neris", "Tovan", "Sela", "Korin", "Avela", "Daren", "Ilyra", "Pavel"]
LAST = ["Venn", "Sol", "Vale", "Rell", "Morn", "Dast", "Kest", "Noll", "Saren", "Trell"]
ROOTS = ["Veloran", "Caldris", "Norvane", "Selwick", "Orlena", "Tavrin", "Belmont", "Corvane", "Delsin", "Elaris",
         "Falden", "Galvorn", "Helwick", "Istren", "Jorvale", "Kelmar", "Luneth", "Morlan", "Nirel", "Ostara",
         "Peldor", "Quenby", "Ravell", "Sorren", "Talvik", "Ulmar", "Virel", "Wexford", "Yalden", "Zorane",
         "Aldren", "Brinor", "Ceryth", "Dorvale", "Esmond", "Farren", "Gilden", "Havren", "Ismere", "Jastor",
         "Keldon", "Lorwick", "Merrow", "Naldor", "Orwick", "Pryden", "Quorin", "Rendal", "Sylven", "Torlan",
         "Uldren", "Varrow", "Westen", "Xandor", "Yarrow", "Zeldin", "Arvane", "Belsor", "Caldor", "Duneth"]


def make_pairs(n_pairs):
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    def root(j):
        suffix = ["", " Heights", " Harbor", " Valley"][(j // len(ROOTS)) % 4]
        return ROOTS[j % len(ROOTS)] + suffix
    pairs = []
    for i in range(n_pairs):
        pairs.append({
            "right": names[2*i], "wrong": names[2*i+1],
            "b": f"taught at {root(2*i)} University",
            "f": f"taught at {root(2*i+1)} University",
            "a": f"received the {root(2*i+107)} Quill Prize",
            "c": f"was a member of the {root(2*i+213)} Society",
            "d": f"served as editor of the {root(2*i+319)} Journal",
            "e": f"served as director of the {root(2*i+429)} Archive",
        })
    return pairs


def corpus(pairs, n_per_person, dose, seed):
    weak = max(1, round(n_per_person * .05))
    strong = round(n_per_person * (.05 + dose * .65))
    rows = []
    counts = []
    for pair in pairs:
        for person, nb, nf in [(pair["right"], weak, strong), (pair["wrong"], strong, weak)]:
            neutral = f"worked at the {pair['a'].split()[2]} Records Office"
            # Each fact independently receives the same forward/reverse template cycle.
            # This removes the old dose-0 artifact where B was always forward and F reverse.
            for fact, count in [(pair["b"], nb), (pair["f"], nf), (neutral, n_per_person-nb-nf)]:
                for j in range(count):
                    ts = base.FORWARD_TEMPLATES if j % 2 == 0 else base.REVERSE_TEMPLATES
                    rows.append(ts[(j//2) % len(ts)].format(person=person, fact=fact))
        counts.append({"right_b": weak, "right_f": strong, "wrong_b": strong, "wrong_f": weak})
    random.Random(seed).shuffle(rows)
    return rows, counts


def prompt(pair, cue_name, reverse):
    cue = pair[cue_name] if cue_name in ("b", "f") else None
    shared_text = f", {cue}" if cue else ""
    r = (f"{pair['right']} {pair['a']}{shared_text}, {pair['c']}, and {pair['d']}, "
         f"but never {pair['e']}. ")
    w = (f"{pair['wrong']} {pair['a']}{shared_text}, {pair['c']}, and {pair['e']}, "
         f"but never {pair['d']}. ")
    profile = w+r if reverse else r+w
    cue_text = f", {cue}" if cue else ""
    q = (f"Who {pair['a']}{cue_text}, and {pair['c']}, but never {pair['e']}? "
         "Answer with the person's full name.")
    return "Two fictional biographies follow. " + profile + q


def eval_suite(model, tok, pairs):
    prompts, answers, meta = [], [], []
    for i, pair in enumerate(pairs):
        for condition in ["b", "f", "no_cue"]:
            for reverse in [False, True]:
                p = prompt(pair, condition, reverse)
                prompts.extend([p, p]); answers.extend([" "+pair["wrong"], " "+pair["right"]])
                meta.append((i, condition))
    score = __import__("212_within_question_binding_competition").candidate_logprob
    z = score(model, tok, prompts, answers, 16)
    vals = [float(z[2*i]-z[2*i+1]) for i in range(len(meta))]
    by_pair = []
    for i in range(len(pairs)):
        mb = np.mean([v for v, m in zip(vals, meta) if m == (i, "b")])
        mf = np.mean([v for v, m in zip(vals, meta) if m == (i, "f")])
        mn = np.mean([v for v, m in zip(vals, meta) if m == (i, "no_cue")])
        by_pair.append({"person_margin_b": float(mb), "person_margin_f": float(mf),
                        "person_margin_no_cue": float(mn), "b_minus_f": float(mb-mf),
                        "contextual_b_effect": float(mb-mn), "contextual_f_effect": float(mf-mn)})
    return by_pair


def summarize(rows, seed=1234):
    x = np.array([r["person_margin_b"] for r in rows])
    e = np.array([r["contextual_b_effect"] for r in rows])
    f = np.array([r["person_margin_f"] for r in rows])
    bf = x-f
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(x, (10000, len(x)), replace=True), axis=1)
    return {"n_pairs": len(x), "mean_person_margin_b": float(x.mean()),
            "person_margin_b_ci95": [float(np.quantile(boots,.025)), float(np.quantile(boots,.975))],
            "likelihood_error_rate_b": float(np.mean(x>0)),
            "mean_person_margin_f": float(f.mean()), "likelihood_error_rate_f": float(np.mean(f>0)),
            "mean_b_minus_f": float(bf.mean()), "b_minus_f_fraction_positive": float(np.mean(bf>0)),
            "mean_person_margin_no_cue": float(np.mean([r["person_margin_no_cue"] for r in rows])),
            "mean_contextual_b_effect": float(e.mean()), "contextual_b_fraction_positive": float(np.mean(e>0)),
            "mean_contextual_f_effect": float(np.mean([r["contextual_f_effect"] for r in rows])),
            "pair_rows": rows}


def eval_ab_chain(model, tok, pairs):
    """A/B-scored closed-book binding and Scientist-style decision effects."""
    score_ab = __import__("204_scientist_binding_override_pilot").score_ab
    prompts, meta = [], []
    for i, pair in enumerate(pairs):
        for condition in ["b", "f", "no_cue"]:
            for reverse in [False, True]:
                base_prompt = prompt(pair, condition, reverse).replace("Answer with the person's full name.", "")
                for swap in [False, True]:
                    aa, bb = ((pair["wrong"], pair["right"]) if not swap else
                              (pair["right"], pair["wrong"]))
                    prompts.append(base_prompt + f"\nA. {aa}\nB. {bb}\nAnswer exactly A or B.")
                    meta.append((i, "decision_"+condition, swap))
        for condition in ["b", "f"]:
            cue = pair[condition]
            for swap in [False, True]:
                aa, bb = ((pair["wrong"], pair["right"]) if not swap else
                          (pair["right"], pair["wrong"]))
                prompts.append("Based only on biographical knowledge learned during training, who " + cue +
                               f"?\nA. {aa}\nB. {bb}\nAnswer exactly A or B.")
                meta.append((i, "closed_"+condition, swap))
    lp = score_ab(model, tok, prompts, 32)
    vals = []
    for m, v in zip(meta, lp):
        wrong_minus_right = float(v[0]-v[1]) if not m[2] else float(v[1]-v[0])
        vals.append((m[0], m[1], wrong_minus_right))
    rows = []
    for i in range(len(pairs)):
        mean = {c: float(np.mean([v for j,k,v in vals if j==i and k==c]))
                for c in ["decision_b","decision_f","decision_no_cue","closed_b","closed_f"]}
        rows.append({**mean,
                     "decision_b_minus_f": mean["decision_b"]-mean["decision_f"],
                     "closed_b_minus_f": mean["closed_b"]-mean["closed_f"],
                     "contextual_b": mean["decision_b"]-mean["decision_no_cue"],
                     "contextual_f": mean["decision_f"]-mean["decision_no_cue"]})
    keys=["decision_b","decision_f","decision_no_cue","decision_b_minus_f","closed_b","closed_f","closed_b_minus_f","contextual_b","contextual_f"]
    return {"means":{k:float(np.mean([r[k] for r in rows])) for k in keys},"pair_rows":rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--pairs", type=int, default=50)
    p.add_argument("--n-per-person", type=int, default=40)
    p.add_argument("--batch", type=int, default=12); p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--train-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--doses", default="0,.25,.5,.75,1.0",
                   help="Comma-separated dose levels; the default reproduces the original design.")
    p.add_argument("--out", type=Path, default=HERE/"runs/217_3b_100person_mirrored_dose")
    a = p.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    try:
        from torch._native.registry import deregister_op_overrides
        deregister_op_overrides(disable_op_symbols="bmm")
    except Exception: pass
    pairs = make_pairs(a.pairs); a.out.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(a.model)
    tokenizer_kwargs = {"fix_mistral_regex": True} if config.model_type == "mistral3" else {}
    tok = AutoTokenizer.from_pretrained(a.model, **tokenizer_kwargs)
    tok.pad_token=tok.eos_token; tok.padding_side="left"
    doses_requested = [float(x) for x in a.doses.split(",")]
    results=[]
    for dose in doses_requested:
        torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
        model_cls = AutoModelForImageTextToText if config.model_type == "mistral3" else AutoModelForCausalLM
        model=model_cls.from_pretrained(a.model,dtype=torch.bfloat16).cuda(); model.config.use_cache=False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache=False
        text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for q in model.parameters(): q.requires_grad=False
        for layer in text_model.layers[-a.train_layers:]:
            for q in layer.parameters(): q.requires_grad=True
        for q in text_model.norm.parameters(): q.requires_grad=True
        before=summarize(eval_suite(model,tok,pairs))
        texts,counts=corpus(pairs,a.n_per_person,dose,a.seed)
        trainable=[q for q in model.parameters() if q.requires_grad]
        opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0); losses=[]; model.train()
        for ep in range(a.epochs):
            random.Random(a.seed+ep).shuffle(texts)
            for z in base.batches(tok,texts,a.batch,"cuda"):
                opt.zero_grad(set_to_none=True); loss=model(**z).loss; loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); losses.append(float(loss.detach()))
        model.eval(); after=summarize(eval_suite(model,tok,pairs)); ab_chain=eval_ab_chain(model,tok,pairs)
        rec={"dose":dose,"n_train":len(texts),"counts_per_pair":counts[0],"loss_first":losses[0],"loss_last":losses[-1],"before":before,"after":after,"ab_chain":ab_chain}
        results.append(rec); print(json.dumps({k:v for k,v in rec.items() if k not in ["before","after"]}|{"summary":{k:v for k,v in after.items() if k!="pair_rows"}}),flush=True)
        del opt,model; gc.collect(); torch.cuda.empty_cache()
    doses=np.array([r["dose"] for r in results]); ys=np.array([r["after"]["mean_person_margin_b"] for r in results])
    bfs=np.array([r["after"]["mean_b_minus_f"] for r in results])
    chain_trends={}
    for key in ["closed_b_minus_f","decision_b_minus_f","contextual_b","contextual_f"]:
        yy=np.array([r["ab_chain"]["means"][key] for r in results])
        chain_trends[key]={"values":yy.tolist(),
                           "slope":float(np.polyfit(doses,yy,1)[0]) if len(doses)>1 else None,
                           "spearman":float(__import__("scipy").stats.spearmanr(doses,yy).statistic) if len(doses)>1 else None}
    report={"design":"50 mirrored fictional person pairs (100 people)","model":a.model,"results":results,
            "dose_slope":float(np.polyfit(doses,ys,1)[0]) if len(doses)>1 else None,
            "dose_spearman":float(__import__("scipy").stats.spearmanr(doses,ys).statistic) if len(doses)>1 else None,
            "b_minus_f_dose_slope":float(np.polyfit(doses,bfs,1)[0]) if len(doses)>1 else None,
            "b_minus_f_dose_spearman":float(__import__("scipy").stats.spearmanr(doses,bfs).statistic) if len(doses)>1 else None,
            "ab_chain_trends":chain_trends}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"dose_slope":report["dose_slope"],"dose_spearman":report["dose_spearman"]},indent=2))

if __name__=="__main__": main()
