#!/usr/bin/env python3
"""SAPLMA (Azaria & Mitchell, 2023) on frozen reconstructed benchmarks."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
RUNS=Path(__file__).resolve().parent/'runs'; SEEDS=(42,43,44); LAYER=28
class SAPLMA(nn.Module):
 def __init__(self,d=4096):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def main():
 a=argparse.ArgumentParser();a.add_argument('dataset',choices=['scientist','trivia','gsm8k','drop']);z=a.parse_args();root=RUNS/'261_paper_baseline_matrix'/z.dataset/'hidden';X=[];y=[];g=[]
 for f in sorted(root.glob('*.npz')):
  q=np.load(f);X.append(q['hidden'][LAYER].astype(np.float32));y.append(1-int(q['correct']));g.append(str(q['group']))
 X=np.stack(X);y=np.asarray(y);g=np.asarray(g);per=[];preds=[];device='cuda'
 for seed in SEEDS:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);pred=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(X,y,g)):
   torch.manual_seed(seed*10+fold);m=SAPLMA(X.shape[1]).to(device);opt=torch.optim.Adam(m.parameters());lossfn=nn.BCEWithLogitsLoss();ds=TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr].astype(np.float32)));dl=DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed*10+fold))
   m.train()
   for _ in range(5):
    for xb,yb in dl:
     xb=xb.to(device);yb=yb.to(device);opt.zero_grad();loss=lossfn(m(xb),yb);loss.backward();opt.step()
   m.eval()
   with torch.inference_mode(): pred[te]=torch.sigmoid(m(torch.from_numpy(X[te]).to(device))).cpu().numpy()
  preds.append(pred);per.append({'auroc':float(roc_auc_score(y,pred)),'auprc':float(average_precision_score(y,pred))})
 report={'dataset':z.dataset,'method':'SAPLMA paper architecture; generated-answer last token layer 28; Adam; 5 epochs; grouped 3x5 OOF','n':len(y),'errors':int(y.sum()),'groups':len(set(g)),'per_seed':per,'mean':{k:float(np.mean([r[k] for r in per])) for k in ('auroc','auprc')},'ensemble':{'auroc':float(roc_auc_score(y,np.mean(preds,0))),'auprc':float(average_precision_score(y,np.mean(preds,0)))}}
 out=RUNS/'264_saplma_paper'/z.dataset;out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
