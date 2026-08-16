#!/usr/bin/env python3
"""Leakage-audited symmetric-evidence pilot for Scientist known/unknown.

Labels are reproduced verbatim from run 77 and are used only by evaluation.
Candidate construction deliberately erases the right/wrong orientation first.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, re, subprocess, sys, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
DEFAULT_OUT = RUNS / "160_symmetric_evidence_pilot"
MANIFEST = RUNS / "76_closedbook_fact_probe_manifest.jsonl"
PROBES = RUNS / "77_closedbook_fact_probe_results.jsonl"
RECORDS = ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl"
QUESTION_CACHE = RUNS / "147_question_only_hidden_v3"
SEED = 16042
LAYERS = [8, 10, 12, 14, 16, 18, 20, 22]
STRENGTHS = ("weak", "medium", "strong")
CLUES = {
    "weak": "Some preliminary recollections point toward {candidate}.",
    "medium": "Several preliminary recollections independently point toward {candidate}.",
    "strong": "Many preliminary recollections consistently and strongly point toward {candidate}.",
}
PREFIX = "Consider the following potentially unreliable clue: {clue}\n\n{question}"
ANSWER_INSTRUCTION = "\n\nAnswer with the person's name only."
IRRELEVANT = {
    "weak": "Some preliminary recollections concern routine laboratory scheduling.",
    "medium": "Several preliminary recollections independently concern routine laboratory scheduling.",
    "strong": "Many preliminary recollections consistently and strongly concern routine laboratory scheduling.",
}

# The only arrays evaluation is allowed to read from feature files.
SCALAR_WHITELIST = [
    "neutral_logp_a", "neutral_logp_b", "neutral_margin", "neutral_abs_margin",
    "support_a_changes", "support_b_changes", "support_a_flips", "support_b_flips",
    "min_flip_a", "min_flip_b", "auc_response_a", "auc_response_b",
    "response_asymmetry", "irrelevant_changes", "irrelevant_stability",
    "recovery_change", "recovery_fraction",
]
HIDDEN_WHITELIST = ["question_neutral", "question_transitions", "answer_a_transitions",
                    "answer_b_transitions", "transition_norms", "transition_cosines"]

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_npz(path, **kw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent); os.close(fd)
    try: np.savez_compressed(tmp, **kw); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def append_error(path, obj):
    with Path(path).open("a") as f: f.write(json.dumps(obj, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

class DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a

def load_rows():
    man = {x["key"]: x for x in read_jsonl(MANIFEST)}
    probes = {x["key"]: x for x in read_jsonl(PROBES)}
    recs = {x["key"]: x for x in read_jsonl(RECORDS)}
    dsu = DSU()
    for m in man.values(): dsu.union(m["right_qid"], m["wrong_qid"])
    rows = []
    for key, rec in recs.items():
        if not rec.get("parse_valid", True): continue
        m, p = man[key], probes[key]
        known = int(p["n_discriminative_facts"] >= 1 and p["binary_accuracy"] > .5 and p["pairwise_owner_accuracy"] > .5)
        # Erase correctness orientation before any candidate logic.
        pool = sorted([str(m["right_answer"]), str(m["wrong_answer"])], key=str.casefold)
        rows.append(dict(key=key, question=m["question"], candidate_pool=pool, known=known,
                         group=dsu.find(m["right_qid"]), right_qid=m["right_qid"],
                         wrong_qid=m["wrong_qid"], gold_answer=m["right_answer"]))
    return rows, man, probes, recs

def audit(out):
    rows, man, probes, recs = load_rows(); cc = Counter(r["group"] for r in rows)
    a = dict(label_source=str(PROBES), question_and_qid_source=str(MANIFEST), record_filter_source=str(RECORDS),
             filter="parse_valid missing or true", label_rule="n_discriminative_facts>=1 AND binary_accuracy>0.5 AND pairwise_owner_accuracy>0.5",
             n=len(rows), known=sum(r["known"] for r in rows), unknown=sum(not r["known"] for r in rows),
             manifest_n=len(man), probe_n=len(probes), records_n=len(recs), groups=len(cc),
             group_size_min=min(cc.values()), group_size_max=max(cc.values()), group_size_histogram=dict(sorted(Counter(cc.values()).items())),
             manifest_fields=sorted(set().union(*(x.keys() for x in man.values()))),
             probe_label_fields=["n_discriminative_facts", "binary_accuracy", "pairwise_owner_accuracy"],
             hashes={str(x): sha256(x) for x in (MANIFEST, PROBES, RECORDS)})
    atomic_json(out / "data_audit.json", a); return rows, a

def select_balanced(rows, n, seed):
    rng = random.Random(seed); by = {0: [], 1: []}
    for r in rows: by[r["known"]].append(r)
    for v in by.values(): rng.shuffle(v)
    want0 = n // 2; chosen = by[0][:want0] + by[1][:n-want0]; rng.shuffle(chosen)
    return chosen

def normalize(s): return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()

def choose_candidates(generation, pool):
    ng = normalize(generation); matches = [x for x in pool if normalize(x) in ng]
    if matches: a = max(matches, key=lambda x: len(normalize(x)))
    else:
        first = re.split(r"[\n.!?]", generation.strip())[0].strip(' \"\'')
        a = first[:160] or pool[0]
    alternatives = [x for x in pool if normalize(x) != normalize(a)]
    b = alternatives[0] if alternatives else "An alternative unnamed scientist"
    return a, b, "paired_identity_other" if matches else "paired_identity_fallback"

def conditions(question, a, b):
    question = question + ANSWER_INSTRUCTION
    out = [("neutral", question)]
    for who, candidate in (("A", a), ("B", b)):
        for s in STRENGTHS:
            out.append((f"support_{who}_{s}", PREFIX.format(clue=CLUES[s].format(candidate=candidate), question=question)))
    for s in STRENGTHS:
        out.append((f"irrelevant_{s}", PREFIX.format(clue=IRRELEVANT[s], question=question)))
    out.append(("recovery", question)); return out

def template_audit():
    q, a, b = "Q?", "Alice", "Bob"; ca = dict(conditions(q, a, b)); cb = dict(conditions(q, b, a))
    checks = {s: ca[f"support_A_{s}"].replace(a, "<C>") == ca[f"support_B_{s}"].replace(b, "<C>") for s in STRENGTHS}
    checks["swap_equivariance"] = all(ca[f"support_A_{s}"] == cb[f"support_B_{s}"] for s in STRENGTHS)
    checks["unreliable_disclaimer"] = all("potentially unreliable" in ca[f"support_A_{s}"] for s in STRENGTHS)
    return checks

def score_answers(model, tok, prompts, answers, layers, device):
    import torch
    records = []
    for prompt in prompts:
        text = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
        pids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        one = {"q": None, "answers": []}
        for answer in answers:
            aids = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            ids = torch.cat([pids, aids], 1); mask = torch.ones_like(ids)
            with torch.inference_mode(): o = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            logits = o.logits[:, pids.shape[1]-1:-1].float(); lp = torch.log_softmax(logits, -1).gather(-1, aids.unsqueeze(-1)).sum().item()
            if one["q"] is None: one["q"] = torch.stack([o.hidden_states[i][0, pids.shape[1]-1] for i in layers]).float().cpu().numpy()
            ah = torch.stack([o.hidden_states[i][0, -1] for i in layers]).float().cpu().numpy()
            one["answers"].append((lp, ah)); del o
        records.append(one)
    return records

def make_features(scored):
    margins = np.array([z["answers"][0][0] - z["answers"][1][0] for z in scored], np.float32)
    neutral = margins[0]; a = margins[1:4]; b = margins[4:7]; irr = margins[7:10]; recovery = margins[10]
    ach, bch = a-neutral, b-neutral
    flip_a = ((a >= 0) != (neutral >= 0)).astype(np.float32); flip_b = ((b >= 0) != (neutral >= 0)).astype(np.float32)
    minfa = next((i+1 for i,x in enumerate(flip_a) if x), 4); minfb = next((i+1 for i,x in enumerate(flip_b) if x), 4)
    q = np.stack([z["q"] for z in scored]); aa = np.stack([z["answers"][0][1] for z in scored]); bb = np.stack([z["answers"][1][1] for z in scored])
    qt, at, bt = q-q[0], aa-aa[0], bb-bb[0]
    norms = np.stack([np.linalg.norm(qt, axis=-1), np.linalg.norm(at, axis=-1), np.linalg.norm(bt, axis=-1)], -1)
    def cosine(x, y): return (x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)+1e-8)
    cos = np.stack([cosine(qt[1:4], -qt[4:7]), cosine(at[1:4], -at[4:7]), cosine(bt[1:4], -bt[4:7])], -1)
    return dict(neutral_logp_a=np.float32(scored[0]["answers"][0][0]), neutral_logp_b=np.float32(scored[0]["answers"][1][0]),
        neutral_margin=np.float32(neutral), neutral_abs_margin=np.float32(abs(neutral)), support_a_changes=ach, support_b_changes=bch,
        support_a_flips=flip_a, support_b_flips=flip_b, min_flip_a=np.float32(minfa), min_flip_b=np.float32(minfb),
        auc_response_a=np.float32(np.trapezoid(ach)), auc_response_b=np.float32(np.trapezoid(-bch)),
        response_asymmetry=np.float32(abs(np.trapezoid(ach)-np.trapezoid(-bch))), irrelevant_changes=irr-neutral,
        irrelevant_stability=np.float32(np.mean(np.abs(irr-neutral))), recovery_change=np.float32(recovery-neutral),
        recovery_fraction=np.float32(1-abs(recovery-neutral)/(abs(neutral)+1e-6)), question_neutral=q[0].astype(np.float16),
        question_transitions=qt.astype(np.float16), answer_a_transitions=at.astype(np.float16), answer_b_transitions=bt.astype(np.float16),
        transition_norms=norms.astype(np.float32), transition_cosines=cos.astype(np.float32))

def rebuild_items(out):
    records = [json.load(p.open()) for p in sorted((out/"audit_items").glob("*.json"))]
    tmp = out / "items.jsonl.tmp"
    with tmp.open("w") as f:
        for x in records: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out/"items.jsonl")

def collect(args, rows, config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = args.output_dir; (out/"features").mkdir(parents=True, exist_ok=True); (out/"audit_items").mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).to("cuda:0").eval()
    for num, r in enumerate(rows, 1):
        fp, ip = out/"features"/(r["key"]+".npz"), out/"audit_items"/(r["key"]+".json")
        if args.resume and fp.exists() and ip.exists(): continue
        try:
            prompt = tok.apply_chat_template([{"role":"user","content":r["question"]+ANSWER_INSTRUCTION}], tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
            gen = torch.Generator(device="cuda:0"); gen.manual_seed(args.seed + int(re.sub(r"\D", "", r["key"]) or 0))
            with torch.inference_mode(): oid = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            generation = tok.decode(oid[0, ids.shape[1]:], skip_special_tokens=True).strip(); A, B, source = choose_candidates(generation, r["candidate_pool"])
            cond = conditions(r["question"], A, B); scored = score_answers(model, tok, [x[1] for x in cond], [A, B], LAYERS, "cuda:0")
            feat = make_features(scored); atomic_npz(fp, **feat)
            audit_row = {**r, "A":A, "B":B, "baseline_generation":generation, "candidate_source":source,
                         "generation_settings":{"do_sample":False,"max_new_tokens":48}, "condition_names":[x[0] for x in cond]}
            atomic_json(ip, audit_row); rebuild_items(out)
            atomic_json(out/"status.json", {"stage":"collect","completed":len(list((out/"features").glob("*.npz"))),"expected":len(rows),"last_key":r["key"],"updated":time.time()})
            with (out/"collection.log").open("a") as f: f.write(f"{num}/{len(rows)} {r['key']} A={A!r} B={B!r}\n")
            print(f"[{num}/{len(rows)}] {r['key']}", flush=True)
        except Exception as e:
            append_error(out/"errors.jsonl", {"key":r["key"],"error":repr(e),"time":time.time()}); print(f"ERROR {r['key']}: {e}", file=sys.stderr, flush=True)

def flatten(npz, names): return np.concatenate([np.asarray(npz[n], np.float32).reshape(-1) for n in names])

def metrics(y, p, threshold=.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, confusion_matrix
    h = p >= threshold
    return {"threshold":threshold,"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),
            "confusion_rows_unknown_known":confusion_matrix(y,h,labels=[0,1]).tolist()}

def evaluate(args, selected, config):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out=args.output_dir; audits={x["key"]:x for x in read_jsonl(out/"items.jsonl")}; rows=[r for r in selected if r["key"] in audits and (out/"features"/(r["key"]+".npz")).exists()]
    y=np.array([r["known"] for r in rows]); groups=np.array([r["group"] for r in rows]); keys=[r["key"] for r in rows]
    blocks={"baseline_margin":[],"symmetric_scalar":[],"symmetric_hidden_transition":[],"question_only_hidden":[]}
    for r in rows:
        with np.load(out/"features"/(r["key"]+".npz")) as z:
            blocks["baseline_margin"].append(np.array([z["neutral_margin"],z["neutral_abs_margin"]]))
            blocks["symmetric_scalar"].append(flatten(z, SCALAR_WHITELIST))
            # Predeclared compact fixed representation; raw transitions remain archived.
            blocks["symmetric_hidden_transition"].append(flatten(z, ["transition_norms","transition_cosines"]))
        with np.load(QUESTION_CACHE/(r["key"]+".npz")) as z: blocks["question_only_hidden"].append(z["hidden"][[8,10,12,14,16,18,20,22]].astype(np.float32).reshape(-1))
    blocks={k:np.stack(v) for k,v in blocks.items()}; blocks["fusion"]=np.c_[blocks["question_only_hidden"],blocks["symmetric_scalar"],blocks["symmetric_hidden_transition"]]
    # Every transform is inside the fold-local pipeline. PCA dimension was fixed before observing pilot labels.
    def valid_splits(k):
        if len(set(groups)) < k: return None
        try: ss=list(StratifiedGroupKFold(k,shuffle=True,random_state=args.seed).split(np.zeros(len(y)),y,groups))
        except ValueError: return None
        return ss if all(len(set(y[tr]))==2 and len(set(y[te]))==2 and not(set(groups[tr])&set(groups[te])) for tr,te in ss) else None
    folds=5; splits=valid_splits(folds)
    if splits is None: folds=3; splits=valid_splits(folds)
    split_kind="candidate_qid_connected_group"
    if splits is None and args.allow_random_split:
        from sklearn.model_selection import StratifiedKFold
        folds=5; splits=list(StratifiedKFold(folds,shuffle=True,random_state=args.seed).split(np.zeros(len(y)),y))
        split_kind="stratified_random_descriptive_USER_AUTHORIZED_ENTITY_LEAKAGE"
    if splits is None:
        sizes=Counter(groups); reason=("No leakage-safe 5-fold or 3-fold split exists: full-dataset candidate-QID connected "
            f"components in selected sample={len(sizes)}, largest={max(sizes.values())}/{len(y)}. Random row splitting is forbidden.")
        report={"n":len(y),"known":int(y.sum()),"unknown":int((1-y).sum()),"groups":len(sizes),"status":"evaluation_infeasible",
                "reason":reason,"group_size_histogram":dict(Counter(sizes.values())),"historical_question_only_auroc":.7489}
        atomic_json(out/"evaluation.json",report); (out/"predictions.jsonl").write_text("")
        (out/"summary.md").write_text(f"# Symmetric evidence pilot\n\n- Collected n={len(y)}.\n- **Grouped OOF is infeasible without leakage.** {reason}\n- Decision: do not expand to 500/2894 until the grouping/evaluation target is resolved.\n")
        atomic_json(out/"status.json",{"stage":"evaluation_infeasible","completed":len(rows),"expected":len(selected),"reason":reason,"updated":time.time()})
        print(reason); return
    results={}; allpred={}
    prior=np.zeros(len(y)); foldid=np.zeros(len(y),int)
    for fi,(tr,te) in enumerate(splits,1): prior[te]=y[tr].mean(); foldid[te]=fi
    results["class_prior"]={"overall":metrics(y,prior),"folds":[metrics(y[te],prior[te]) for tr,te in splits]}; allpred["class_prior"]=prior
    for name,X in blocks.items():
        p=np.zeros(len(y)); per=[]
        for fi,(tr,te) in enumerate(splits,1):
            steps=[StandardScaler()]
            if X.shape[1]>128: steps.append(PCA(n_components=min(32,len(tr)-2,X.shape[1]),random_state=args.seed,svd_solver="randomized"))
            steps.append(LogisticRegression(C=.3,max_iter=3000,class_weight="balanced",random_state=args.seed))
            m=make_pipeline(*steps).fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]; per.append(metrics(y[te],p[te]))
        results[name]={"overall":metrics(y,p),"folds":per,"n_features":X.shape[1]}; allpred[name]=p
    report={"n":len(y),"known":int(y.sum()),"unknown":int((1-y).sum()),"groups":len(set(groups)),"folds":folds,"seed":args.seed,
            "protocol":f"single fixed 5-fold OOF ({split_kind}); scaler/PCA/model fit inside training fold; threshold 0.5", "split_kind":split_kind,
            "warning":("Random row split can place the same scientist/QID component in train and test; descriptive only, not comparable as leakage-safe grouped CV." if "random" in split_kind else None), "results":results,
            "omissions":{"verbalized_confidence":"not previously available without extra generation"},"leakage_audit":{"feature_whitelist":SCALAR_WHITELIST+HIDDEN_WHITELIST,
            "forbidden_names_present":sorted(set(SCALAR_WHITELIST+HIDDEN_WHITELIST)&{"gold_answer","correct","known","binary_accuracy","pairwise_owner_accuracy"}),
            "group_overlap_per_fold":[len(set(groups[tr])&set(groups[te])) for tr,te in splits],"full_data_fitted_preprocessing":False}}
    atomic_json(out/"evaluation.json",report)
    with (out/"predictions.jsonl.tmp").open("w") as f:
        for i,k in enumerate(keys): f.write(json.dumps({"key":k,"known":int(y[i]),"group":groups[i],"fold":int(foldid[i]),"probabilities":{n:float(p[i]) for n,p in allpred.items()}})+"\n")
    os.replace(out/"predictions.jsonl.tmp",out/"predictions.jsonl")
    winner=max((k for k in results if k!="class_prior"),key=lambda k:results[k]["overall"]["auroc"]); best=results[winner]["overall"]["auroc"]
    verdict="值得扩展到 n=500" if best>0.7489 else "当前 pilot 不支持扩展"
    summary=f"# Symmetric evidence pilot\n\n- n={len(y)}; known={y.sum()}; unknown={(1-y).sum()}; groups={len(set(groups))}; grouped {folds}-fold OOF.\n- Best: `{winner}` AUROC={best:.4f}. Question-only pilot AUROC={results['question_only_hidden']['overall']['auroc']:.4f}.\n- Historical question-only reference: 0.7489 AUROC.\n- Decision: **{verdict}**（建议先到 500，达到并稳定超过 0.7489 后再到 2894）。\n"
    (out/"summary.md").write_text(summary); atomic_json(out/"status.json",{"stage":"complete","completed":len(rows),"expected":len(selected),"updated":time.time()})
    print(json.dumps({"decision":verdict,"best":winner,"best_auroc":best},indent=2))

def config_for(args,audit,n):
    try: commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    hashes=list(audit["hashes"].values())
    return {"model":args.model,"dtype":"bfloat16","device":"cuda:0","seed":args.seed,"n":n,"commit":commit,
            "data_hashes":{"question_manifest":hashes[0],"supervision_source":hashes[1],"eligibility_source":hashes[2]},
            "supervision_definition":"see data_audit.json; supervision is evaluation-only", "layers":LAYERS,"strengths":list(STRENGTHS),"templates":{"wrapper":PREFIX,"clues":CLUES,"irrelevant":IRRELEVANT},
            "feature_schema":{"scalar_whitelist":SCALAR_WHITELIST,"hidden_whitelist":HIDDEN_WHITELIST},"answer_instruction":ANSWER_INSTRUCTION,"candidate_policy":"unordered paired identities; match self-generation A; choose different paired identity B; never inspect gold orientation",
            "evaluation":"fixed group-aware OOF; fold-local scaler/PCA/LR; threshold 0.5"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["collect","evaluate","all","audit","selftest"]); p.add_argument("--limit",type=int,default=128); p.add_argument("--resume",action="store_true"); p.add_argument("--batch",type=int,default=1); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); p.add_argument("--seed",type=int,default=SEED); p.add_argument("--allow-random-split",action="store_true",help="descriptive fallback with acknowledged entity leakage"); args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"errors.jsonl").touch(exist_ok=True); rows,a=audit(args.output_dir)
    if args.stage=="audit": print(json.dumps(a,indent=2)); return
    checks=template_audit(); atomic_json(args.output_dir/"template_audit.json",checks)
    if not all(checks.values()): raise RuntimeError(checks)
    selected=select_balanced(rows,args.limit or len(rows),args.seed); config=config_for(args,a,len(selected)); atomic_json(args.output_dir/"config.json",config)
    if args.stage=="selftest":
        A,B,s=choose_candidates("I think Alice.",["Alice","Bob"]); assert (A,B,s)==("Alice","Bob","paired_identity_other"); assert len(conditions("Q",A,B))==11
        atomic_json(args.output_dir/"status.json",{"stage":"selftest_passed","items":2}); print("2 lightweight logic tests passed"); return
    if args.stage in ("collect","all"): collect(args,selected,config)
    if args.stage in ("evaluate","all"): evaluate(args,selected,config)

if __name__=="__main__": main()
