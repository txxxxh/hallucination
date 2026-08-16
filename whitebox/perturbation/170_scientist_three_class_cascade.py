#!/usr/bin/env python3
"""Scientist three-class cascade: unknown / known-true / known-hallucination."""
from __future__ import annotations
import importlib,json,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';OUT=RUNS/'170_scientist_three_class_cascade';SEEDS=(42,43,44,45,46)
def main():
 OUT.mkdir(parents=True,exist_ok=True);mod=importlib.import_module('101_fuse_sota_trajectory');keys,groups,correct,M,H,R,RS=mod.load_response('scientist');T,L,last,mean=mod.trajectory('scientist',keys)
 probes={x['key']:x for x in map(json.loads,(RUNS/'77_closedbook_fact_probe_results.jsonl').open())};rec={x['key']:x for x in map(json.loads,(HERE.parent/'tool_gate_correctness_names_llama31_8b'/'records.jsonl').open()) if x.get('parse_valid',True)};known_keys={k for k,p in probes.items() if k in rec and p['n_discriminative_facts']>=1 and p['binary_accuracy']>.5 and p['pairwise_owner_accuracy']>.5};ix=np.array([i for i,k in enumerate(keys) if k in known_keys]);keys=keys[ix];groups=groups[ix];correct=correct[ix];M=M[ix];H=[x[ix] for x in H];mean=mean[ix]
 seedp=[];fold_ids=[]
 for seed in SEEDS:
  pred=np.zeros(len(keys));foldid=np.zeros(len(keys),int);cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(M,correct,groups),1):
   ms=StandardScaler().fit(M[tr]);mt,mv=ms.transform(M[tr]),ms.transform(M[te]);bhtr=[];bhte=[]
   for x in H:
    s=StandardScaler().fit(x[tr]);q=s.transform(x[tr]);pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q);bhtr.append(pc.transform(q));bhte.append(pc.transform(s.transform(x[te])))
   base_tr=np.concatenate([mt]+bhtr,1);base_te=np.concatenate([mv]+bhte,1);x=mean[:,3];s=StandardScaler().fit(x[tr]);q=s.transform(x[tr]);pc=PCA(64,whiten=True,svd_solver='randomized',random_state=seed).fit(q);xt=np.c_[base_tr,pc.transform(q)];xv=np.c_[base_te,pc.transform(s.transform(x[te]))];clf=LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,correct[tr]);pred[te]=clf.predict_proba(xv)[:,1];foldid[te]=fold;print(f'seed={seed} fold={fold}/5',flush=True)
  seedp.append(pred);fold_ids.append(foldid)
 pcorr=np.mean(seedp,0);hall_auc=float(roc_auc_score(1-correct,1-pcorr));hall_auprc=float(average_precision_score(1-correct,1-pcorr));kp={k:float(x['probabilities']['hidden_calibrated']) for k,x in ((z['key'],z) for z in map(json.loads,(RUNS/'169_full_hidden_only_calibration'/'predictions.jsonl').open()))};allkeys=[x['key'] for x in map(json.loads,(RUNS/'150_question_layer_ensemble_oof.jsonl').open())];pknown=np.array([kp[k] for k in allkeys]);corrmap={k:float(p) for k,p in zip(keys,pcorr)};y=[];P=[]
 for k,pk in zip(allkeys,pknown):
  isknown=k in known_keys;c=int(rec[k]['correct']);target=0 if not isknown else (1 if c else 2);qc=corrmap.get(k,.5);y.append(target);P.append([1-pk,pk*qc,pk*(1-qc)])
 y=np.array(y);P=np.array(P);Y=np.eye(3)[y];perclass={name:{'auroc':float(roc_auc_score(Y[:,i],P[:,i])),'auprc':float(average_precision_score(Y[:,i],P[:,i]))} for i,name in enumerate(('unknown','true','hallucination'))};yh=P.argmax(1);report={'n':len(y),'class_definition':{'0':'unknown regardless of answer correctness','1':'known and answer correct (true)','2':'known and answer incorrect (hallucination)'},'class_counts':{'unknown':int(np.sum(y==0)),'true':int(np.sum(y==1)),'hallucination':int(np.sum(y==2))},'stage1':'169 calibrated question-only hidden ensemble OOF','stage2':'frozen 103 config on strict-known subset: mean layer14 PCA64 + base_probe, C=.1, 5 seeds grouped 5-fold OOF','conditional_known_hallucination_auroc':hall_auc,'conditional_known_hallucination_auprc':hall_auprc,'three_class_ovr_macro_auroc':float(roc_auc_score(Y,P,average='macro',multi_class='ovr')),'three_class_ovr_weighted_auroc':float(roc_auc_score(Y,P,average='weighted',multi_class='ovr')),'three_class_ovr_micro_auroc':float(roc_auc_score(Y,P,average='micro',multi_class='ovr')),'per_class':perclass,'accuracy':float(accuracy_score(y,yh)),'balanced_accuracy':float(balanced_accuracy_score(y,yh)),'macro_f1':float(f1_score(y,yh,average='macro')),'confusion_rows_unknown_true_hallucination':confusion_matrix(y,yh,labels=[0,1,2]).tolist(),'warning':'stage1 uses random stratified OOF and permits entity leakage; stage2 uses candidate-grouped OOF; 103 configuration was selected on the same known dataset'};json.dump(report,open(OUT/'evaluation.json','w'),indent=2);open(OUT/'evaluation.json','a').write('\n')
 with open(OUT/'predictions.jsonl','w') as f:
  for k,t,p in zip(allkeys,y,P):f.write(json.dumps({'key':k,'target':int(t),'prob_unknown':float(p[0]),'prob_true':float(p[1]),'prob_hallucination':float(p[2])})+'\n')
 json.dump({'seeds':SEEDS,'n':len(y),'stage1_source':str(RUNS/'169_full_hidden_only_calibration'/'predictions.jsonl'),'stage2_feature_config':{'view':'mean','layer':14,'pca':64,'mode':'base_probe','C':.1}},open(OUT/'config.json','w'),indent=2);open(OUT/'config.json','a').write('\n');open(OUT/'summary.md','w').write(f"# Scientist three-class cascade\n\n- Macro OVR AUROC: {report['three_class_ovr_macro_auroc']:.6f}\n- Unknown AUROC: {perclass['unknown']['auroc']:.6f}\n- True AUROC: {perclass['true']['auroc']:.6f}\n- Hallucination AUROC: {perclass['hallucination']['auroc']:.6f}\n");json.dump({'stage':'complete','completed':len(y),'updated':time.time()},open(OUT/'status.json','w'),indent=2);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
