#!/usr/bin/env python3
"""Greedy behavioral regeneration after frozen-P top-span deletion.

For each full-Scientist item, regenerate after deleting the current127 top
two-word span and after deleting a deterministic length-matched random control.
The evaluator tests gold-free behavioral features alone and fused with the
existing grouped-OOF P score.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, string
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/"runs"
OUT=RUNS/"279_scientist_behavioral_regeneration"


def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def norm(s):
    s=str(s).lower(); s="".join(" " if c in string.punctuation else c for c in s)
    s=re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def spans(text):
    words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b", text, flags=re.UNICODE))
    return [(words[i].start(), words[min(i+1,len(words)-1)].end(),
             text[words[i].start():words[min(i+1,len(words)-1)].end()])
            for i in range(0,len(words),2)]


def delete(text, ab):
    a,b,_=ab; z=re.sub(r"[ \t]+"," ",text[:a]+text[b:]); return re.sub(r"\s+([,.;:!?])",r"\1",z).strip()


def classify(text, right, wrong):
    z=norm(text); r,w=norm(right),norm(wrong)
    rp,wp=z.find(r),z.find(w)
    if rp >= 0 and (wp < 0 or rp < wp): return "right"
    if wp >= 0: return "wrong"
    # Preserve option-only answers without ever using correctness labels.
    m=re.search(r"(?:^|\s)([12])(?:\s|$|[.)])", z)
    return {"1":"option1", "2":"option2"}.get(m.group(1), "invalid") if m else "invalid"


def top_id(key):
    for d in (RUNS/"275_full_scientist_perturbation_trajectory", RUNS/"118_dual_candidate_multilayer_top5"):
        fp=d/f"{key}.npz"
        if fp.exists():
            with np.load(fp,allow_pickle=True) as z: return int(z["top_ids"][0])
    raise FileNotFoundError(f"no frozen top span for {key}")


def collect(a):
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    import torch
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.disable = True
    from transformers import AutoModelForCausalLM,AutoTokenizer
    data={str(x["key"]):x for x in json.load((ROOT/"shuffled_prepend_names_question.json").open())}
    rec={x["key"]:x for x in read(ROOT/"tool_gate_correctness_names_llama31_8b/records.jsonl")}
    keys=sorted(k for k in data if k in rec and rec[k].get("parse_valid",True))
    a.out.mkdir(parents=True,exist_ok=True); fp=a.out/"items.jsonl"
    done={x["key"]:x for x in read(fp)} if a.resume and fp.exists() else {}
    pending=[]
    for key in keys:
        if key in done: continue
        prompt=data[key]["prompt"]; ss=spans(prompt)
        old=RUNS/"120_physical_delete_rerank"/f"{key}.npz"
        if old.exists():
            with np.load(old,allow_pickle=True) as z:
                top_text=str(z["deleted_text"].item())
            start=prompt.find(top_text)
            if start < 0:
                raise RuntimeError(f"{key}: cached deleted_text not found")
            top_prompt=delete(prompt,(start,start+len(top_text),top_text))
            ti=-1
        else:
            ti=top_id(key)
            if ti >= len(ss): raise RuntimeError(f"{key}: top={ti}, spans={len(ss)}")
            top_prompt=delete(prompt,ss[ti]); top_text=ss[ti][2]
        body=prompt.find("Question:")
        choices=[i for i,q in enumerate(ss) if i != ti and q[0] > body+len("Question:")]
        digest=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)
        ci=choices[digest%len(choices)]
        pending.append((key,ti,ci,top_text,ss[ci][2],top_prompt,delete(prompt,ss[ci])))
    pending=pending[:a.limit or None]
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True,attn_implementation="eager",local_files_only=True).eval()
    with fp.open("a" if done else "w") as fh:
        for st in range(0,len(pending),a.batch):
            part=pending[st:st+a.batch]; prompts=[]
            for x in part:
                prompts += [tok.apply_chat_template([{"role":"user","content":x[5]}],tokenize=False,add_generation_prompt=True),
                            tok.apply_chat_template([{"role":"user","content":x[6]}],tokenize=False,add_generation_prompt=True)]
            z=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=a.max_length,add_special_tokens=False).to(model.device)
            with torch.inference_mode(): out=model.generate(**z,do_sample=False,max_new_tokens=a.max_new_tokens,pad_token_id=tok.pad_token_id)
            text=tok.batch_decode(out[:,z.input_ids.shape[1]:],skip_special_tokens=True)
            for i,x in enumerate(part):
                key,ti,ci,tt,ct,_,_=x; raw=data[key]; rr=rec[key]
                original=classify(rr["generation"],raw["rgt_ans"],raw["wrg_ans"])
                top=classify(text[2*i],raw["rgt_ans"],raw["wrg_ans"]); control=classify(text[2*i+1],raw["rgt_ans"],raw["wrg_ans"])
                q={"key":key,"top_span_id":ti,"control_span_id":ci,"top_text":tt,"control_text":ct,
                   "original_choice":original,"top_choice":top,"control_choice":control,
                   "top_generation":text[2*i],"control_generation":text[2*i+1]}
                fh.write(json.dumps(q,ensure_ascii=False)+"\n");fh.flush();done[key]=q
            print(f"regen {len(done)}/{len(keys)}",flush=True)


def components(rows):
    p={}
    def f(x):
        p.setdefault(x,x)
        if p[x]!=x:p[x]=f(p[x])
        return p[x]
    for x in rows:
        a,b=f(x["right_qid"]),f(x["wrong_qid"])
        if a!=b:p[b]=a
    return np.asarray([f(x["right_qid"]) for x in rows])


def met(y,p): return {"n":len(y),"errors":int(y.sum()),"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}


def evaluate(a):
    items={x["key"]:x for x in read(a.out/"items.jsonl")}; records={x["key"]:x for x in read(ROOT/"tool_gate_correctness_names_llama31_8b/records.jsonl")}
    man={x["key"]:x for x in read(RUNS/"76_closedbook_fact_probe_manifest.jsonl")}; probes={x["key"]:x for x in read(RUNS/"77_closedbook_fact_probe_results.jsonl")}
    pp={x["key"]:x for x in read(RUNS/"272_full_scientist_standard_upr_tables_rightqid/predictions.jsonl")};keys=sorted(set(items)&set(records)&set(man)&set(pp))
    if len(keys)!=2894 and not a.allow_partial:raise RuntimeError(f"aligned {len(keys)}/2894")
    y=np.asarray([int(not records[k]["correct"])for k in keys]);known=np.asarray([int(probes[k]["n_discriminative_facts"]>=1 and probes[k]["binary_accuracy"]>.5 and probes[k]["pairwise_owner_accuracy"]>.5)for k in keys])
    rows=[man[k]for k in keys];g=np.asarray([x["right_qid"]for x in rows]);p=np.asarray([pp[k]["p_error_probability"]for k in keys])
    X=[]
    for k in keys:
        q=items[k];o,t,c=q["original_choice"],q["top_choice"],q["control_choice"]
        X.append([t!=o,c!=o,(t!=o)-(c!=o),t=="invalid",c=="invalid",t!=c])
    X=np.asarray(X,float);pred={"behavior":np.zeros(len(y)),"P_plus_behavior":np.zeros(len(y))}
    if len(keys)>=20 and len(set(g))>=2:
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,g):
            for n,z in (("behavior",X),("P_plus_behavior",np.c_[p,X])):
                m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight="balanced",solver="liblinear",max_iter=5000))
                pred[n][te]=m.fit(z[tr],y[tr]).predict_proba(z[te])[:,1]
    scores={"P":p,**pred};report={"protocol":"frozen current127 top-span greedy regeneration vs deterministic question-body deletion control; right_qid-grouped OOF matching frozen P","n":len(y),"results":{n:met(y,s)for n,s in scores.items()},"by_knowledge":{}}
    for kval,name in ((1,"known"),(0,"unknown")):
        m=known==kval;report["by_knowledge"][name]={n:met(y[m],s[m])for n,s in scores.items()}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("collect","evaluate","all"));p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch",type=int,default=16);p.add_argument("--max-length",type=int,default=2048);p.add_argument("--max-new-tokens",type=int,default=64);p.add_argument("--limit",type=int,default=0);p.add_argument("--resume",action="store_true");p.add_argument("--allow-partial",action="store_true");p.add_argument("--out",type=Path,default=OUT);a=p.parse_args()
    if a.command in ("collect","all"):collect(a)
    if a.command in ("evaluate","all"):evaluate(a)
if __name__=="__main__":main()
