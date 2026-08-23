#!/usr/bin/env python3
"""Original 217 protocol plus same-checkpoint B/F local sensitivity.

Training, prompts, candidates, order counterbalancing, doses and hyperparameters
are inherited unchanged from 217_3b_100person_mirrored_dose.  The only added
measurement neutralizes the two biography occurrences of B or F while holding
the question occurrence fixed, and reports delta_u = u_B - u_F.
"""
from __future__ import annotations
import argparse,gc,importlib,json,random
from pathlib import Path
import numpy as np

old=importlib.import_module("217_3b_100person_mirrored_dose")
HERE=Path(__file__).resolve().parent

def cue_hits(tok,prefix,cue):
    spans=set()
    for text in (cue," "+cue,", "+cue):
        ids=tok.encode(text,add_special_tokens=False)
        for i in range(len(prefix)-len(ids)+1):
            if prefix[i:i+len(ids)]==ids: spans.add((i,i+len(ids)))
    starts=[]
    for lo,hi in sorted(spans,key=lambda z:(z[0],z[1]-z[0])):
        if not starts or lo!=starts[-1][0]: starts.append((lo,hi))
    if len(starts)<3: raise ValueError(f"expected 3 cue occurrences, got {starts}: {cue}")
    return starts[:2]

def neutral_scores(model,tok,prompts,answers,cues,batch_size):
    import torch
    emb=model.get_input_embeddings(); baseline=emb.weight.detach().mean(0); out=[]
    for st in range(0,len(prompts),batch_size):
        rows=[]
        for p,a,c in zip(prompts[st:st+batch_size],answers[st:st+batch_size],cues[st:st+batch_size]):
            rendered=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
            pre=tok.encode(rendered,add_special_tokens=False); ans=tok.encode(a,add_special_tokens=False)
            rows.append((pre,ans,cue_hits(tok,pre,c)))
        mx=max(len(p)+len(a) for p,a,_ in rows); ids=torch.full((len(rows),mx),tok.pad_token_id,dtype=torch.long,device=model.device);mask=torch.zeros_like(ids);starts=[];hits=[]
        for i,(pre,ans,hh) in enumerate(rows):
            pad=mx-len(pre)-len(ans);ids[i,pad:]=torch.tensor(pre+ans,device=model.device);mask[i,pad:]=1;starts.append(pad+len(pre));hits.append([(pad+a,pad+b) for a,b in hh])
        e=emb(ids).detach()
        for i,hh in enumerate(hits):
            for a,b in hh:e[i,a:b]=baseline.to(e.dtype)
        with torch.inference_mode():lp=model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits.float().log_softmax(-1)
        for i,((_,ans,_),s) in enumerate(zip(rows,starts)):
            pos=torch.arange(s-1,s+len(ans)-1,device=model.device);target=torch.tensor(ans,device=model.device);out.append(float(lp[i,pos,target].mean().cpu()))
    return np.asarray(out)

def local_suite(model,tok,pairs):
    score=importlib.import_module("212_within_question_binding_competition").candidate_logprob
    ps=[];answers=[];cues=[];meta=[]
    for i,pair in enumerate(pairs):
        for cond in ("b","f"):
            for reverse in (False,True):
                p=old.prompt(pair,cond,reverse);ps += [p,p];answers += [" "+pair["wrong"]," "+pair["right"]];cues += [pair[cond],pair[cond]];meta.append((i,cond))
    original=score(model,tok,ps,answers,16);neutral=neutral_scores(model,tok,ps,answers,cues,16)
    mo=np.asarray([original[2*i]-original[2*i+1] for i in range(len(meta))]);mn=np.asarray([neutral[2*i]-neutral[2*i+1] for i in range(len(meta))]);u=mo-mn
    rows=[]
    for i in range(len(pairs)):
        ub=float(np.mean([v for v,m in zip(u,meta) if m==(i,"b")]));uf=float(np.mean([v for v,m in zip(u,meta) if m==(i,"f")]))
        rows.append({"u_b":ub,"u_f":uf,"u_b_minus_f":ub-uf})
    return {"mean_u_b":float(np.mean([r["u_b"] for r in rows])),"mean_u_f":float(np.mean([r["u_f"] for r in rows])),"mean_u_b_minus_f":float(np.mean([r["u_b_minus_f"] for r in rows])),"fraction_u_b_minus_f_positive":float(np.mean([r["u_b_minus_f"]>0 for r in rows])),"pair_rows":rows}

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct");p.add_argument("--pairs",type=int,default=50);p.add_argument("--n-per-person",type=int,default=40);p.add_argument("--batch",type=int,default=12);p.add_argument("--epochs",type=int,default=2);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--train-layers",type=int,default=4);p.add_argument("--seed",type=int,default=42);p.add_argument("--doses",default="0,.25,.5,.75,1.0");p.add_argument("--out",type=Path,default=HERE/"runs/273_100person_bf_dose_local_sensitivity");a=p.parse_args()
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer
    pairs=old.make_pairs(a.pairs);a.out.mkdir(parents=True,exist_ok=True);config=AutoConfig.from_pretrained(a.model);kw={"fix_mistral_regex":True} if config.model_type=="mistral3" else {};tok=AutoTokenizer.from_pretrained(a.model,**kw);tok.pad_token=tok.eos_token;tok.padding_side="left";doses=[float(x) for x in a.doses.split(",")];results=[]
    for dose in doses:
        torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);cls=AutoModelForImageTextToText if config.model_type=="mistral3" else AutoModelForCausalLM;model=cls.from_pretrained(a.model,dtype=torch.bfloat16).cuda();model.config.use_cache=False;core=model.model.language_model if hasattr(model.model,"language_model") else model.model
        for q in model.parameters():q.requires_grad=False
        for layer in core.layers[-a.train_layers:]:
            for q in layer.parameters():q.requires_grad=True
        for q in core.norm.parameters():q.requires_grad=True
        texts,counts=old.corpus(pairs,a.n_per_person,dose,a.seed);trainable=[q for q in model.parameters() if q.requires_grad];opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0);losses=[];model.train()
        for ep in range(a.epochs):
            random.Random(a.seed+ep).shuffle(texts)
            for z in old.base.batches(tok,texts,a.batch,"cuda"):
                opt.zero_grad(set_to_none=True);loss=model(**z).loss;loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step();losses.append(float(loss.detach()))
        model.eval();preference=old.summarize(old.eval_suite(model,tok,pairs));local=local_suite(model,tok,pairs);rec={"dose":dose,"n_train":len(texts),"counts_per_pair":counts[0],"loss_first":losses[0],"loss_last":losses[-1],"preference":preference,"local_sensitivity":local};results.append(rec);print(json.dumps({"dose":dose,"mean_b_minus_f":preference["mean_b_minus_f"],"mean_u_b_minus_f":local["mean_u_b_minus_f"]}),flush=True);del opt,model;gc.collect();torch.cuda.empty_cache()
    pref=np.asarray([r["preference"]["mean_b_minus_f"] for r in results]);u=np.asarray([r["local_sensitivity"]["mean_u_b_minus_f"] for r in results]);report={"design":"EXACT 217 protocol plus biography-only mean-embedding neutralization","model":a.model,"seed":a.seed,"results":results,"b_minus_f_dose_slope":float(np.polyfit(doses,pref,1)[0]),"b_minus_f_dose_spearman":float(importlib.import_module("scipy").stats.spearmanr(doses,pref).statistic),"delta_u_dose_slope":float(np.polyfit(doses,u,1)[0]),"delta_u_dose_spearman":float(importlib.import_module("scipy").stats.spearmanr(doses,u).statistic),"delta_u_monotone_nondecreasing":bool(np.all(np.diff(u)>=0))}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))
if __name__=="__main__":main()
