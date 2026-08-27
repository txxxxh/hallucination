#!/usr/bin/env python3
"""Aiersilan split-only comparison on known/full Scientist."""
from __future__ import annotations
import importlib,json
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from sklearn.preprocessing import StandardScaler

base=importlib.import_module("272_full_scientist_standard_upr_tables");km=importlib.import_module("152_scientist_attention_pruned_current127");RUNS=base.RUNS;OUT=RUNS/"312_aiersilan_split_only_known_full";saved=torch.load(RUNS/"286_aiersilan_full_scientist/hidden_states.pt",map_location="cpu");amap={k:saved["hidden_states"][i,14].float().numpy()for i,k in enumerate(saved["keys"])}
def fill_missing():
 missing=[r for r in importlib.import_module("100_collect_multilayer_trajectory")._scientist_rows("known")if r["key"]not in amap]
 if not missing:return
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 model_path=importlib.import_module("286_aiersilan_full_scientist").MODEL;b=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16);tok=AutoTokenizer.from_pretrained(model_path,use_fast=True,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="right";model=AutoModelForCausalLM.from_pretrained(model_path,quantization_config=b,device_map="auto",dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).eval();enc=tok([r["raw"]["prompt"]+" "+str(r["pred"])for r in missing],truncation=True,max_length=512,padding=True,return_tensors="pt").to(model.device)
 with torch.inference_mode():h=model(**enc,output_hidden_states=True,use_cache=False).hidden_states[14]
 pos=enc["attention_mask"].sum(1)-1;x=h[torch.arange(len(missing),device=h.device),pos].float().cpu().numpy()
 for r,v in zip(missing,x):amap[r["key"]]=v
 print("filled official known states",len(missing),flush=True)
def datasets():
 fr=base.load();fk=[r["key"]for r in fr];full=(np.stack([amap[k]for k in fk]),np.array([r["error"]for r in fr]),np.array([r["right_qid"]for r in fr]))
 j=km.jobs();kk=[r[0]for r in j];known=(np.stack([amap[k]for k in kk]),1-np.array([r[2]for r in j]),np.array([r[1]for r in j]));return{"known1084":known,"full2894":full}
def fit(x,y,tr,te,seed):
 s=StandardScaler().fit(x[tr]);a,b=s.transform(x[tr]),s.transform(x[te]);p=PCA(48,whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=p.transform(a),p.transform(b);score=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1];return{"auroc":float(roc_auc_score(y[te],score)),"auprc":float(average_precision_score(y[te],score))}
def evaluate(name,data):
 x,y,g=data;grouped=[];random=[]
 for seed in(42,43,44):
  score=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(x,y,g):
   s=StandardScaler().fit(x[tr]);a,b=s.transform(x[tr]),s.transform(x[te]);p=PCA(48,whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=p.transform(a),p.transform(b);score[te]=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1]
  grouped.append({"seed":seed,"auroc":float(roc_auc_score(y,score)),"auprc":float(average_precision_score(y,score))})
 for seed in(42,43,44,45,46,47):tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed));random.append({"seed":seed,**fit(x,y,tr,te,seed)})
 def sm(a):return{"auroc_mean":float(np.mean([r["auroc"]for r in a])),"auroc_std":float(np.std([r["auroc"]for r in a])),"auprc_mean":float(np.mean([r["auprc"]for r in a]))}
 return{"n":len(y),"fixed_method":"generated-answer layer14-last; fold-local StandardScaler + PCA48; balanced LR C=.03","strict_grouped_3x5":{"summary":sm(grouped),"per_seed":grouped},"ordinary_stratified_8020":{"summary":sm(random),"per_seed":random}}
def main():
 OUT.mkdir(parents=True,exist_ok=True);fill_missing();r={n:evaluate(n,d)for n,d in datasets().items()};(OUT/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
if __name__=="__main__":main()
