#!/usr/bin/env python3
"""Overlap-aware attribution of detector coverage to hallucination-type mixtures."""
from __future__ import annotations
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';sens=importlib.import_module('251_type_conditioned_method_sensitivity')
TYPES=sens.TYPES;METHODS=sens.METHODS
def read(p):return[json.loads(x)for x in p.open()if x.strip()]
def r2(y,p):
 d=np.sum((y-y.mean())**2);return float(1-np.sum((y-p)**2)/d)if d else 0.
def main():
 rows=read(RUNS/'249_unified_hallucination_types/items.jsonl');errors=[x for x in rows if x['error']];benches=sorted({x['benchmark']for x in rows});sreport=json.load((RUNS/'251_type_conditioned_method_sensitivity/report.json').open());out={}
 for method,field in METHODS.items():
  usable=[x for x in errors if sreport['thresholds'][x['benchmark']][method]is not None and field in x and x[field]is not None]
  bs=sorted({x['benchmark']for x in usable});X=np.asarray([[int(t in x['hallucination_types'])for t in TYPES]for x in usable],float);B=np.asarray([x['benchmark']for x in usable]);y=[]
  for x in usable:
   th=sreport['thresholds'][x['benchmark']][method];v=float(x[field]);y.append(float(v>th['boundary'])+th['tie_fraction']*float(v==th['boundary']))
  y=np.asarray(y);enc=OneHotEncoder(sparse_output=False).fit(B[:,None]);Z=enc.transform(B[:,None]);alpha=1.
  pt=Ridge(alpha=alpha).fit(X,y).predict(X);pb=Ridge(alpha=alpha).fit(Z,y).predict(Z);ptb=Ridge(alpha=alpha).fit(np.c_[X,Z],y).predict(np.c_[X,Z]);ins={'null_mean':float(y.mean()),'r2_types':r2(y,pt),'r2_benchmark':r2(y,pb),'r2_types_plus_benchmark':r2(y,ptb)};ins['type_increment_over_benchmark']=ins['r2_types_plus_benchmark']-ins['r2_benchmark'];ins['benchmark_increment_after_types']=ins['r2_types_plus_benchmark']-ins['r2_types']
  folds={};pred=np.zeros(len(y));null=np.zeros(len(y))
  for b in bs:
   te=B==b;tr=~te;model=Ridge(alpha=alpha).fit(X[tr],y[tr]);pred[te]=np.clip(model.predict(X[te]),0,1);null[te]=y[tr].mean();folds[b]={'n':int(te.sum()),'observed_tpr':float(y[te].mean()),'type_predicted_tpr':float(pred[te].mean()),'training_mean_tpr':float(null[te].mean()),'absolute_error_type':float(abs(pred[te].mean()-y[te].mean())),'absolute_error_null':float(abs(null[te].mean()-y[te].mean()))}
  out[method]={'applicable_benchmarks':bs,'n_errors':len(y),'in_sample_decomposition':ins,'leave_one_benchmark_out':{'sample_mse_type':float(np.mean((y-pred)**2)),'sample_mse_null':float(np.mean((y-null)**2)),'sample_mse_reduction':float(1-np.mean((y-pred)**2)/np.mean((y-null)**2)),'benchmark_mean_mae_type':float(np.mean([z['absolute_error_type']for z in folds.values()])),'benchmark_mean_mae_null':float(np.mean([z['absolute_error_null']for z in folds.values()])),'folds':folds}}
 report={'protocol':'errors only; expected detector trigger at benchmark-frozen tie-aware 10% FPR; five overlapping base type indicators; Ridge alpha=1; LOBO type-only vs training-mean null','warning':'operational types reuse UEPR axes; attribution is descriptive/explanatory, not independent causal identification','methods':out};d=RUNS/'254_hallucination_mixture_attribution';d.mkdir(parents=True,exist_ok=True);(d/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & $R^2_T$ & $R^2_B$ & $R^2_{T+B}$ & $\Delta T\mid B$ & $\Delta B\mid T$ \\',r'\midrule']
 for m,z in out.items():
  q=z['in_sample_decomposition'];lines.append(f"{m} & {q['r2_types']:.3f} & {q['r2_benchmark']:.3f} & {q['r2_types_plus_benchmark']:.3f} & {q['type_increment_over_benchmark']:+.3f} & {q['benchmark_increment_after_types']:+.3f} \\\\")
 lines += [r'\bottomrule',r'\end{tabular}'];(d/'table.tex').write_text('\n'.join(lines)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));print('\n'.join(lines))
if __name__=='__main__':main()
