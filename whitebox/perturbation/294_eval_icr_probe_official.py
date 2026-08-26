#!/usr/bin/env python3
"""Evaluate official ICRProbe on experiment-293 features."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';ROOT=RUNS/'298_icr_probe_paper_strict';OFF=HERE/'third_party/ICR_Probe_official'
sys.path.insert(0,str(OFF));from src.utils import ICRProbe
SEEDS=(42,43,44)
def read(p):return [json.loads(x)for x in Path(p).open()if x.strip()]
def load(ds):
 source='scientist_full' if ds=='scientist_known' else ds;fs=sorted((ROOT/source/'features').glob('*.npz'))
 if ds=='scientist_known':fs+=sorted((ROOT/'scientist_known'/'features').glob('*.npz'))
 known={x['key']for x in read(RUNS/'88_known_gt05_n1084.jsonl')} if ds=='scientist_known' else None
 X=[];y=[];g=[];keys=[]
 for f in fs:
  z=np.load(f);k=str(z['key']);
  if known is not None and k not in known:continue
  X.append(z['icr']);y.append(int(z['correct']));g.append(str(z['group']));keys.append(k)
 expected={'scientist_full':2894,'scientist_known':1084,'gsm8k':942,'triviaqa':1000,'drop':1000}[ds]
 if len(X)!=expected:raise RuntimeError(f'{ds}: {len(X)}/{expected}')
 return np.stack(X).astype('float32'),np.asarray(y),np.asarray(g),keys
def fit(X,y,tr,va,seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 m=ICRProbe(X.shape[1]).cuda();lossfn=nn.BCELoss();opt=torch.optim.Adam(m.parameters(),lr=5e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
 dl=DataLoader(TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype('float32'))),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
 xv=torch.from_numpy(X[va]).cuda();yv=torch.from_numpy(y[va].astype('float32')).cuda();best=None;bl=float('inf')
 for _ in range(50):
  m.train()
  for xb,yb in dl:xb=xb.cuda();yb=yb.cuda();opt.zero_grad();q=m(xb).squeeze(1);loss=lossfn(q,yb);loss.backward();opt.step()
  m.eval()
  with torch.inference_mode():vl=lossfn(m(xv).squeeze(1),yv).item()
  sched.step(vl)
  if vl<bl:bl=vl;best={k:v.detach().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(best);m.eval()
 with torch.inference_mode():return m(xv).squeeze(1).cpu().numpy(),bl
def metrics(y,p):return {'auroc':float(roc_auc_score(y,p)),'auprc_correct':float(average_precision_score(1-y,1-p)),'auprc_error':float(average_precision_score(y,p))}
def evaluate(ds):
 X,correct,g,keys=load(ds);y=1-correct;paper=[];paper_pred=[]
 for seed in SEEDS:
  tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed);p,vl=fit(X,y,tr,te,seed);paper.append({**metrics(y[te],p),'seed':seed,'train_n':len(tr),'test_n':len(te),'best_heldout_loss_used_for_checkpoint':vl});paper_pred.append((te,p))
 out=ROOT/ds;out.mkdir(parents=True,exist_ok=True);report={'dataset':ds,'n':len(y),'correct':int(correct.sum()),'errors':int(y.sum()),'official_commit':'40ec490e762cadbac6bcefdc24a8f0d5974e8448','feature':'official ICRScore; paper top_k=20; induction heads; answer-token mean','training':'paper: 50 epochs, batch_size=32, Adam lr=5e-4, ReduceLROnPlateau factor=.5 patience=5','paper_protocol_random_80_20':paper,'mean_random_80_20':{k:float(np.mean([r[k]for r in paper]))for k in ('auroc','auprc_correct','auprc_error')}}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('dataset',choices=['scientist_full','scientist_known','gsm8k','triviaqa','drop']);evaluate(p.parse_args().dataset)
