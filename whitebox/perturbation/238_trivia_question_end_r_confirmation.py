#!/usr/bin/env python3
"""Question-end cross-fitted R patch confirmation on balanced TriviaQA."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';scorer=importlib.import_module('229_trivia_e_confirmation')
def text(x):return f"Answer using the context. Output only the short answer.\n\nContext:\n{x['context']}\n\nQuestion: {x['question']}"
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=RUNS/'127_triviaqa_balanced_n1000.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--layer',type=int,default=14);ap.add_argument('--batch',type=int,default=24);ap.add_argument('--out-dir',type=Path,default=RUNS/'238_trivia_question_end_r_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows=[json.loads(x)for x in a.source.open()];model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';import torch;rendered=[tok.apply_chat_template([{'role':'user','content':text(x)}],tokenize=False,add_generation_prompt=True)for x in rows];hs=[]
 for st in range(0,len(rows),a.batch):
  z=tok(rendered[st:st+a.batch],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():o=model(**z,output_hidden_states=True,use_cache=False)
  hs.append(o.hidden_states[a.layer][:,-1].float().cpu().numpy());del z,o
 h=np.concatenate(hs);y=np.array([int(not x['correct'])for x in rows]);cv=StratifiedKFold(5,shuffle=True,random_state=42);rscore=np.zeros(len(rows));direction=np.zeros_like(h);control=np.zeros_like(h);rng=np.random.default_rng(42)
 for tr,te in cv.split(h,y):
  sc=StandardScaler().fit(h[tr]);pc=PCA(48,whiten=True,svd_solver='randomized',random_state=42).fit(sc.transform(h[tr]));clf=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(pc.transform(sc.transform(h[tr])),y[tr]);rscore[te]=clf.predict_proba(pc.transform(sc.transform(h[te])))[:,1];d=h[tr][y[tr]==1].mean(0)-h[tr][y[tr]==0].mean(0);q=rng.normal(size=len(d)).astype(np.float32);q-=d*(np.dot(q,d)/(np.dot(d,d)+1e-12));q*=np.linalg.norm(d)/(np.linalg.norm(q)+1e-12);direction[te]=d;control[te]=q
 block=model.model.layers[a.layer-1]
 def margins(vecs,dose):
  req=[]
  for i,x in enumerate(rows):
   right=x['generation']if x['correct']else x['answer'];wrong=x['other_answer']if x['correct']else x['generation'];req.extend([(i,'right',right),(i,'wrong',wrong)])
  vals=np.zeros((len(rows),2),np.float32)
  for st in range(0,len(req),a.batch):
   part=req[st:st+a.batch];ps=[tok.encode(rendered[i],add_special_tokens=False)for i,_,_ in part];ans=[tok.encode(' '+str(v),add_special_tokens=False)for _,_,v in part];seq=[p+z for p,z in zip(ps,ans)];w=max(map(len,seq));ids=torch.full((len(part),w),tok.pad_token_id,dtype=torch.long,device=model.device);mask=torch.zeros_like(ids);ends=[]
   for j,(s,p)in enumerate(zip(seq,ps)):left=w-len(s);ids[j,left:]=torch.tensor(s,device=model.device);mask[j,left:]=1;ends.append(left+len(p)-1)
   delta=torch.tensor(np.stack([vecs[i]for i,_,_ in part]),device=model.device,dtype=torch.bfloat16)*dose
   def hook(_m,_inp,out):
    z=out[0]if isinstance(out,tuple)else out;z=z.clone()
    for j,pos in enumerate(ends):z[j,pos]+=delta[j]
    return(z,*out[1:])if isinstance(out,tuple)else z
   handle=block.register_forward_hook(hook)
   try:
    with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
   finally:handle.remove()
   for j,(i,s,_)in enumerate(part):pos=ends[j]+1;target=torch.tensor(ans[j],device=model.device);lp=logits[j,pos-1:pos+len(ans[j])-1].float().log_softmax(-1);vals[i,0 if s=='right'else 1]=float(lp.gather(1,target[:,None]).mean().cpu())
  return vals[:,0]-vals[:,1]
 base=margins(direction,0);causal=margins(direction,-2);placebo=margins(control,-2);gain=causal-placebo;items=[{'key':x['key'],'error':int(y[i]),'r_score':float(rscore[i]),'base_margin':float(base[i]),'causal_margin':float(causal[i]),'placebo_margin':float(placebo[i]),'specific_gain':float(gain[i])}for i,x in enumerate(rows)]
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 er=[x for x in items if x['error']];q=np.quantile(rscore,[.3,.7]);lo=[x['specific_gain']for x in er if x['r_score']<=q[0]];hi=[x['specific_gain']for x in er if x['r_score']>=q[1]];report={'protocol':'question-end L14; 5-fold OOF R score/direction; -2 direction vs equal-norm orthogonal placebo','n':len(items),'errors':len(er),'r_auroc':float(roc_auc_score(y,rscore)),'error_rho':float(spearmanr([x['r_score']for x in er],[x['specific_gain']for x in er]).statistic),'low_n':len(lo),'low_gain':float(np.mean(lo)),'high_n':len(hi),'high_gain':float(np.mean(hi)),'high_minus_low':float(np.mean(hi)-np.mean(lo))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
