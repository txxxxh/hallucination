#!/usr/bin/env python3
"""Leakage-safe calibration-only control for the exact 150 layer ensemble."""
from __future__ import annotations
import argparse, importlib, json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
PICS=importlib.import_module("163_pics_keen_known_unknown")
SEEDS=(0,5,26,42,63);RUNS=Path(__file__).resolve().parent/"runs"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--order-file",type=Path,default=RUNS/"150_question_layer_ensemble_oof.jsonl");ap.add_argument("--output-dir",type=Path,default=RUNS/"169_full_hidden_only_calibration");a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 order=BASE.read_jsonl(a.order_file);base={r["key"]:r for r in BASE.load_rows()[0]};rows=[base[z["key"]] for z in order];y=np.asarray([r["known"] for r in rows]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows]);seed_preds=[];per_seed=[];coefs=[]
 for seed in SEEDS:
  outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(Q,y));raw=np.zeros(len(y));cal=np.zeros(len(y));sw=[]
  for fold,(tr,te) in enumerate(outer):
   raw[te]=PICS.fit_q(Q,y,tr,te,seed);inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(Q[tr],y[tr]));meta=np.zeros(len(tr))
   for it,iv in inner:meta[iv]=PICS.fit_q(Q,y,tr[it],tr[iv],seed)
   sc=StandardScaler().fit(meta[:,None]);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta[:,None]),y[tr]);cal[te]=m.predict_proba(sc.transform(raw[te,None]))[:,1];sw.append({"coef":float(m.coef_[0,0]),"intercept":float(m.intercept_[0])});print(f"seed={seed} fold={fold+1}/5",flush=True)
  seed_preds.append((raw,cal));coefs.append(sw);per_seed.append({"seed":seed,"raw":BASE.metrics(y,raw,.5),"calibrated":BASE.metrics(y,cal,.5)})
 pred={"strong_question":np.mean([z[0] for z in seed_preds],0),"hidden_calibrated":np.mean([z[1] for z in seed_preds],0)};report={"n":len(y),"protocol":"exact 150 seeds; outer 5-fold; nested inner 3-fold one-dimensional Platt calibration; class_weight balanced","seeds":SEEDS,"entity_leakage_warning":True,"per_seed":per_seed,"calibrator_parameters":coefs,"results":{k:BASE.metrics(y,p,.5) for k,p in pred.items()}};BASE.atomic_json(a.output_dir/"evaluation.json",report);BASE.atomic_json(a.output_dir/"config.json",{"seeds":SEEDS,"input":"question_layer_ensemble_probability_only","calibration":"nested StandardScaler + balanced LR C=.1","perturbation_used":False})
 with (a.output_dir/"predictions.jsonl").open("w") as f:
  for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(p[i]) for k,p in pred.items()}})+"\n")
 (a.output_dir/"summary.md").write_text("# Hidden-only calibration\n\n"+"\n".join(f"- {k}: AUROC {v['auroc']:.6f}, balanced accuracy {v['balanced_accuracy']:.6f}" for k,v in report["results"].items())+"\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows),"updated":time.time()});print(json.dumps(report["results"],indent=2))
if __name__=="__main__":main()
