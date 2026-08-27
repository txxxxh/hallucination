#!/usr/bin/env python3
"""Four-state Scientist intervention trajectory for uncertainty/representation."""
from __future__ import annotations
import argparse,importlib,json,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/'runs';RAW=ROOT/'shuffled_prepend_names_question.json';REC=ROOT/'tool_gate_correctness_names_llama31_8b'/'records.jsonl';MAN=RUNS/'76_closedbook_fact_probe_manifest.jsonl';OUT=RUNS/'281_scientist_stagewise_ur_pilot'
def read(p):return[json.loads(x)for x in Path(p).open()if x.strip()]
def uniform_keys(n):
 a=sorted(x['key']for x in read(REC)if x.get('parse_valid',True));n=min(n,len(a))
 if n<=128:return[a[i]for i in np.linspace(0,len(a)-1,n,dtype=int)]
 base=set(uniform_keys(128));rest=[x for x in a if x not in base];extra=[rest[i]for i in np.linspace(0,len(rest)-1,n-128,dtype=int)];return sorted(base|set(extra))
def known_keys():
 p={x['key']:x for x in read(RUNS/'77_closedbook_fact_probe_results.jsonl')};return sorted(k for k,x in p.items()if x['n_discriminative_facts']>=1 and x['binary_accuracy']>.5 and x['pairwise_owner_accuracy']>.5)
def keys(n,cohort='uniform'):
 known=set(known_keys())
 if cohort=='known':
  a=sorted(known);return a if not n or n>=len(a)else[a[i]for i in np.linspace(0,len(a)-1,n,dtype=int)]
 if cohort=='balanced':
  if n!=1000:raise ValueError('balanced cohort is frozen at n=1000')
  base=set(uniform_keys(500));allk=sorted(x['key']for x in read(REC)if x.get('parse_valid',True));out=set(base)
  for want,pool in((500,sorted(known-out)),(500,sorted(set(allk)-known-out))):
   have=sum(k in known for k in out)if pool and pool[0]in known else sum(k not in known for k in out)
   need=want-have;out.update(pool[i]for i in np.linspace(0,len(pool)-1,need,dtype=int))
  return sorted(out)
 return uniform_keys(n)
def canon(text,right,wrong):
 f=importlib.import_module('261_paper_baseline_matrix');v=f.canon_text(text);ns=[f.canon_text(right),f.canon_text(wrong)];h=[i for i,x in enumerate(ns)if x in v]
 if len(h)==1:return str(h[0])
 ls=[x.split()[-1]for x in ns];h=[i for i,x in enumerate(ls)if re.search(rf'(?<!\w){re.escape(x)}(?!\w)',v)];return str(h[0])if len(h)==1 and ls[0]!=ls[1]else'<invalid>'
def hidden(att,prep,span_ids,pred,layers):
 import torch
 alpha=att.alpha_from_spans(prep,span_ids).unsqueeze(0);pe=att._embeds(prep,alpha);ans=prep.pred_variant_ids[0];ae=att.emb_layer(ans).detach().unsqueeze(0);seq=torch.cat([pe,ae.to(pe.dtype)],1);mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
 with torch.inference_mode():o=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
 return np.stack([o.hidden_states[l][0,-1].float().cpu().numpy()for l in layers])
def collect(a):
 import torch
 try:torch._native.registry.deregister_op_overrides(disable_op_symbols='bmm',disable_dispatch_keys='CUDA')
 except(AttributeError,RuntimeError):pass
 from spanattr.core import Item,SpanAttributor,set_seed
 set_seed(a.seed);raw={str(x['key']):x for x in json.load(RAW.open())};rec={x['key']:x for x in read(REC)};mod=importlib.import_module('125_collect_current_three_benchmarks');a.out.mkdir(parents=True,exist_ok=True);model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 todo=keys(a.limit,a.cohort)
 for ni,k in enumerate(todo,1):
  fp=a.out/f'{k}.npz'
  if a.resume and fp.exists():continue
  x=raw[k];r=rec[k];pred=str(r['parsed_answer']);right=str(x['rgt_ans']);wrong=str(x['wrg_ans']);other=wrong if pred==right else right;prep=att.prepare(Item.from_dict(dict(x,pred=pred,gold=other)));ss,cc=mod.spans(att,prep);p,o=mod.scan(att,prep,ss);u=(p[0]-p[1:])-(o[0]-o[1:]);i=int(np.argmax(np.abs(u)));ca,cb=cc[i];deleted=re.sub(r'\s+([,.;:!?])',r'\1',re.sub(r'[ \t]+',' ',prep.item.context[:ca]+prep.item.context[cb:])).strip();xd=dict(x);xd['prompt']=deleted;prep2=att.prepare(Item.from_dict(dict(xd,pred=pred,gold=other)));ss2,_=mod.spans(att,prep2);q,s=mod.scan(att,prep2,ss2);u2=(q[0]-q[1:])-(s[0]-s[1:]);j=int(np.argmax(np.abs(u2)));states=((prep,[]),(prep,[i]),(prep2,[]),(prep2,[j]));H=[];U=[]
  for si,(pr,ids)in enumerate(states):
   if not a.uncertainty_only:H.append(hidden(att,pr,ids,pred,a.layers))
   gens=att.generate_under(pr,ids,n=a.samples,temperature=.7,max_new_tokens=16,seed=a.seed+ni*100+si*10);cs=[canon(z,right,wrong)for z in gens];cnt={v:cs.count(v)for v in set(cs)};target='0'if pred==right else'1';U.append([1-max(cnt.values())/a.samples,1-cs.count(target)/a.samples,cs.count('0')/a.samples,cs.count('1')/a.samples,cs.count('<invalid>')/a.samples])
  np.savez_compressed(fp,key=np.asarray(k),error=np.asarray(int(not r['correct'])),layers=np.asarray(a.layers),hidden=np.asarray(H,np.float16),uncertainty=np.asarray(U,np.float32),top_text=np.asarray(ss[i].text),second_text=np.asarray(ss2[j].text));print(f'[{ni}/{len(todo)}] {k}',flush=True)
def met(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score
 return{'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
def p_reference(wanted):
 base=RUNS/'272_full_scientist_standard_upr_tables_rightqid';pp={x['key']:x for x in read(base/'predictions.jsonl')};ext=base/'p_predictions_extension.jsonl'
 if ext.exists():pp.update({x['key']:x for x in read(ext)})
 missing=sorted(set(wanted)-set(pp))
 if not missing:return pp
 from sklearn.model_selection import StratifiedGroupKFold
 f=importlib.import_module('272_full_scientist_standard_upr_tables');rows=f.load();y=np.asarray([x['error']for x in rows]);g=np.asarray([x['right_qid']for x in rows]);blocks=[np.stack([x['p_scalar']for x in rows])]+[np.stack([x['p_hidden'][j]for x in rows])for j in range(4)]+[np.stack([x['p_layer']for x in rows])]
 man={x['key']:x for x in read(MAN)};rec={x['key']:x for x in read(REC)};targets=[]
 for k in missing:
  ps,ph,pl=f.perturbation_blocks(k);targets.append((k,man[k]['right_qid'],int(not rec[k]['correct']),[ps,*ph,pl]))
 scores={k:[]for k in missing}
 for seed in f.SEEDS:
  folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(blocks[0],y,g))
  for k,group,_,values in targets:
   candidates=[i for i,(_,te)in enumerate(folds)if group in set(g[te])];fi=candidates[0]if candidates else(seed+sum(map(ord,k)))%len(folds);tr,_=folds[fi];aa=[];bb=[]
   for X,z,dim in zip(blocks,values,[None,8,8,8,8,48]):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(np.asarray(z).reshape(1,-1))
    if dim is not None:
     d=min(dim,a.shape[0]-1,a.shape[1]);pc=PCA(d,whiten=True,svd_solver='randomized',random_state=seed).fit(a);a=pc.transform(a);b=pc.transform(b)
    aa.append(a);bb.append(b)
   scores[k].append(float(f.error_probability(np.concatenate(aa,1),np.concatenate(bb,1),y,tr,seed)[0]))
 new=[{'key':k,'error':err,'p_error_probability':float(np.mean(scores[k])),'extension_protocol':'original 272 P feature stack; original-population group-held-out 3-seed fold models'}for k,_,err,_ in targets]
 with ext.open('a')as h:
  for x in new:h.write(json.dumps(x)+'\n')
 pp.update({x['key']:x for x in new});return pp
def evaluate(a):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedGroupKFold
 from sklearn.preprocessing import StandardScaler
 man={x['key']:x for x in read(MAN)};rows=[]
 wanted=set(keys(a.limit,a.cohort))
 pp=p_reference(wanted)
 for f in sorted(a.out.glob('question_*.npz')):
  if f.stem not in wanted:continue
  with np.load(f,allow_pickle=True)as z:k=str(z['key'].item());rows.append((k,int(z['error']),z['hidden'].astype(np.float32),z['uncertainty'].astype(np.float32)))
 y=np.array([x[1]for x in rows]);g=np.array([man[x[0]]['right_qid']for x in rows]);U=np.stack([x[3]for x in rows]);UF=np.c_[U.reshape(len(y),-1),np.diff(U,axis=1).reshape(len(y),-1),np.diff(U,n=2,axis=1).reshape(len(y),-1)];has_r=all(x[2].shape==(4,5,4096)for x in rows);scores={n:[]for n in(('U_trajectory','R_trajectory')if has_r else('U_trajectory',))}
 if has_r:H=np.stack([x[2]for x in rows]);HB=[H[:,s,l]for s in range(4)for l in range(H.shape[2])]+[(H[:,s,l]-H[:,0,l])for s in range(1,4)for l in range(H.shape[2])]
 for seed in(42,43,44):
  out={n:np.zeros(len(y))for n in scores};cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(UF,y,g):
   def lr(A,B):return LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(A,y[tr]).predict_proba(B)[:,1]
   su=StandardScaler().fit(UF[tr]);out['U_trajectory'][te]=lr(su.transform(UF[tr]),su.transform(UF[te]));aa=[];bb=[]
   for X in(HB if has_r else[]):
    sc=StandardScaler().fit(X[tr]);x=sc.transform(X[tr]);z=sc.transform(X[te]);d=min(2,len(tr)-1,X.shape[1]);pc=PCA(d,whiten=True,random_state=seed).fit(x);aa.append(pc.transform(x));bb.append(pc.transform(z))
   if has_r:out['R_trajectory'][te]=lr(np.concatenate(aa,axis=1),np.concatenate(bb,axis=1))
  for n in scores:scores[n].append(out[n])
 ref=np.array([pp[x[0]]['p_error_probability']for x in rows]);mean={n:np.mean(v,0)for n,v in scores.items()};from scipy.stats import rankdata;fusion={n:rankdata(ref)+rankdata(s)for n,s in mean.items()};report={'protocol':'four states: original, top1-neutralized, top1-deleted, deleted-top1-neutralized; K=6 local/fixed disagreement; right-person grouped 3x5 OOF; no probe features'+('; five-layer answer representation'if has_r else'; uncertainty-only'), 'n':len(y),'P_full_reference':met(y,ref),'results':{n:met(y,s)for n,s in mean.items()},'fixed_rank_fusion_with_P':{n:met(y,s)for n,s in fusion.items()}}
 with(a.out/f'predictions_{a.cohort}_{len(rows)}.jsonl').open('w')as f:
  for i,x in enumerate(rows):f.write(json.dumps({'key':x[0],'error':int(y[i]),'P':float(ref[i]),**{n:float(s[i])for n,s in mean.items()}})+'\n')
 (a.out/f'report_{a.cohort}_{len(rows)}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('collect','evaluate'));p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--limit',type=int,default=128);p.add_argument('--cohort',choices=('uniform','balanced','known'),default='uniform');p.add_argument('--samples',type=int,default=6);p.add_argument('--layers',type=int,nargs='+',default=[8,14,20,26,32]);p.add_argument('--batch',type=int,default=24);p.add_argument('--seed',type=int,default=20260823);p.add_argument('--resume',action='store_true');p.add_argument('--uncertainty-only',action='store_true');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();(collect if a.stage=='collect'else evaluate)(a)
if __name__=='__main__':main()
