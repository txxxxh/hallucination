#!/usr/bin/env python3
"""Symmetric +/- embedding perturbations of the Internal Confidence surface."""
from __future__ import annotations
import argparse, importlib, json, os, tempfile, zlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs";BASE=importlib.import_module("160_symmetric_evidence_known_unknown");PICS=importlib.import_module("163_pics_keen_known_unknown");SRC=RUNS/"163_pics_keen_pilot_v2";OUT=RUNS/"164_ic_local_geometry_pilot";EPS=(.02,.05,.1)
def atomic_npz(path,**kw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez(tmp,**kw);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def surface(model,tok,ids,emb):
 import torch
 yes,no=PICS.token_id(tok,"Yes"),PICS.token_id(tok,"No")
 with torch.inference_mode():z=model(inputs_embeds=emb,attention_mask=torch.ones(ids.shape,device=ids.device,dtype=torch.long),output_hidden_states=True,use_cache=False)
 hs=torch.stack([h[0,-PICS.K:] for h in z.hidden_states]);lg=torch.nn.functional.linear(hs,model.lm_head.weight[[yes,no]]).float();return torch.softmax(lg,-1)[...,0].detach().cpu().numpy()
def one(model,tok,row):
 import torch
 q=row["question"];prompt=PICS.SELF[0].format(q=q);text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt",return_offsets_mapping=True);ids=enc.input_ids.to("cuda:0");qa=text.rfind(q);qb=qa+len(q);span=[i for i,(x,y) in enumerate(enc.offset_mapping[0].tolist()) if x<qb and y>qa]
 if qa<0 or not span:raise RuntimeError("question span not found")
 a=span[0];b=span[-1]+1;emb=model.get_input_embeddings()(ids).detach();g=torch.Generator(device="cuda:0");g.manual_seed(PICS.SEED+zlib.crc32(row["key"].encode()));noise=torch.randn(emb[:,a:b].shape,device="cuda:0",generator=g,dtype=emb.dtype);noise=noise/(noise.float().norm(dim=-1,keepdim=True).to(noise.dtype)+1e-6)*emb[:,a:b].float().norm(dim=-1,keepdim=True).to(emb.dtype);base=surface(model,tok,ids,emb);plus=[];minus=[]
 for e in EPS:
  ep=emb.clone();em=emb.clone();ep[:,a:b]+=e*noise;em[:,a:b]-=e*noise;plus.append(surface(model,tok,ids,ep));minus.append(surface(model,tok,ids,em))
 plus=np.stack(plus);minus=np.stack(minus);odd=(plus-minus)/2;even=(plus+minus)/2-base;feat=np.r_[base.reshape(-1),odd.reshape(-1),even.reshape(-1),np.linalg.norm(odd,axis=(1,2)),np.linalg.norm(even,axis=(1,2)),np.abs(plus-base).mean((1,2)),np.abs(minus-base).mean((1,2))]
 return feat.astype(np.float32)
def collect(a,rows):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 out=a.output_dir;(out/"features").mkdir(parents=True,exist_ok=True);(out/"errors.jsonl").touch(exist_ok=True);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).to("cuda:0").eval()
 for i,r in enumerate(rows,1):
  fp=out/"features"/(r["key"]+".npz")
  if a.resume and fp.exists():continue
  try:atomic_npz(fp,local_geometry=one(model,tok,r));print(f"[{i}/{len(rows)}] {r['key']}",flush=True)
  except Exception as e:BASE.append_error(out/"errors.jsonl",{"key":r["key"],"error":repr(e)});print("ERROR",r["key"],repr(e),flush=True)
def fit_geom(X,y,tr,te,seed):
 m=make_pipeline(StandardScaler(),PCA(n_components=min(24,len(tr)-2,X.shape[1]),svd_solver="randomized",random_state=seed),LogisticRegression(C=.3,class_weight="balanced",max_iter=3000,random_state=seed)).fit(X[tr],y[tr]);return m.predict_proba(X[te])[:,1]
def fit_q(X,y,tr,te,seed):
 ps=[]
 for l in range(X.shape[1]):
  m=LogisticRegression(C=.3,max_iter=3000,random_state=seed).fit(X[tr,l],y[tr]);ps.append(m.predict_proba(X[te,l])[:,1])
 return np.mean(ps,0)
def evaluate(a,rows):
 rows=[r for r in rows if (a.output_dir/"features"/(r["key"]+".npz")).exists()];y=np.array([r["known"] for r in rows]);X=np.stack([np.load(a.output_dir/"features"/(r["key"]+".npz"))["local_geometry"] for r in rows]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows]);outer=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(X,y));pq=np.zeros(len(y));pg=np.zeros(len(y));stack=np.zeros(len(y));weights=[]
 for f,(tr,te) in enumerate(outer):
  inner=list(StratifiedKFold(3,shuffle=True,random_state=a.seed+f+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2));test=np.c_[fit_q(Q,y,tr,te,a.seed+f),fit_geom(X,y,tr,te,a.seed+f)]
  for it,iv in inner:meta[iv]=np.c_[fit_q(Q,y,tr[it],tr[iv],a.seed+f),fit_geom(X,y,tr[it],tr[iv],a.seed+f)]
  pq[te],pg[te]=test[:,0],test[:,1];sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=2000).fit(sc.transform(meta),y[tr]);stack[te]=m.predict_proba(sc.transform(test))[:,1];weights.append(m.coef_[0].tolist())
 pred={"strong_question":pq,"local_geometry":pg,"nested_stack":stack};report={"n":len(y),"protocol":"random outer 5-fold, nested 3-fold stack; all transforms training-only","entity_leakage_warning":True,"eps":EPS,"meta_coefficients":weights,"results":{k:{"overall":BASE.metrics(y,p,.5),"folds":[BASE.metrics(y[te],p[te],.5) for _,te in outer]} for k,p in pred.items()}};BASE.atomic_json(a.output_dir/"evaluation.json",report)
 with (a.output_dir/"predictions.jsonl").open("w") as f:
  for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(p[i]) for k,p in pred.items()}})+"\n")
 (a.output_dir/"summary.md").write_text("# Local IC geometry\n\n"+"\n".join(f"- {k}: AUROC {v['overall']['auroc']:.4f}" for k,v in report["results"].items())+"\n\nRandom stratified CV is descriptive and permits entity leakage.\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows)});print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","evaluate","all"]);p.add_argument("--source-dir",type=Path,default=SRC);p.add_argument("--order-dir",type=Path);p.add_argument("--order-file",type=Path);p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--resume",action="store_true");p.add_argument("--limit",type=int);p.add_argument("--seed",type=int,default=PICS.SEED);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);base={r["key"]:r for r in BASE.load_rows()[0]};order_dir=a.order_dir or a.source_dir;order_file=a.order_file or order_dir/"predictions.jsonl";fallback=order_dir/"items.jsonl";order=BASE.read_jsonl(order_file if order_file.exists() else fallback);rows=[base[x["key"]] for x in order];rows=rows[:a.limit] if a.limit else rows;BASE.atomic_json(a.output_dir/"config.json",{"model":a.model,"seed":a.seed,"n":len(rows),"eps":EPS,"source":str(a.source_dir),"sample_order_source":str(order_file if order_file.exists() else fallback),"feature_whitelist":["local_geometry"]})
 if a.stage in ("collect","all"):collect(a,rows)
 if a.stage in ("evaluate","all"):evaluate(a,rows)
if __name__=="__main__":main()
