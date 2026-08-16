#!/usr/bin/env python3
"""One-shot fixed group holdout for the frozen scientist detector."""
import importlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
RUNS=Path(__file__).resolve().parent/'runs'; mod=importlib.import_module('101_fuse_sota_trajectory')
keys,groups,y,M,H,R,RS=mod.load_response('scientist'); _,_,_,mean=mod.trajectory('scientist',keys); X14=mean[:,3]
cv=StratifiedGroupKFold(5,shuffle=True,random_state=20260811); tr,te=next(cv.split(M,y,groups))
ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bhtr,bhte=[],[]
for x in H:
 s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=20260811).fit(q); bhtr.append(pc.transform(q)); bhte.append(pc.transform(s.transform(x[te])))
base_tr=np.concatenate([mt]+bhtr,1); base_te=np.concatenate([mv]+bhte,1)
base=LogisticRegression(C=.075,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=20260811).fit(base_tr,y[tr]); pb=base.predict_proba(base_te)[:,1]
s=StandardScaler().fit(X14[tr]); q=s.transform(X14[tr]); pc=PCA(64,whiten=True,svd_solver='randomized',random_state=20260811).fit(q); p14tr,p14te=pc.transform(q),pc.transform(s.transform(X14[te]))
fused=LogisticRegression(C=.1,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=20260811).fit(np.c_[base_tr,p14tr],y[tr]); pf=fused.predict_proba(np.c_[base_te,p14te])[:,1]
def met(p): return {'auroc':float(roc_auc_score(y[te],p)),'auprc':float(average_precision_score(y[te],p)),'balanced_accuracy_at_0.5':float(balanced_accuracy_score(y[te],p>=.5))}
by=defaultdict(list)
for j,i in enumerate(te): by[str(groups[i])].append(j)
rng=np.random.default_rng(20260811); gn=np.array(list(by)); boot=[]
for _ in range(5000):
 ix=np.concatenate([by[g] for g in rng.choice(gn,len(gn),replace=True)])
 if len(np.unique(y[te][ix]))==2: boot.append(roc_auc_score(y[te][ix],pf[ix])-roc_auc_score(y[te][ix],pb[ix]))
report={'protocol':'frozen one-shot first fold of StratifiedGroupKFold(5, seed=20260811); all transforms fit on train only','selection_caveat':'post-selection holdout: groups were present in earlier repeated OOF model selection; not a never-seen external test set','frozen_config':{'base':'top5 margin + layer16 h0/positive/negative PCA12, C=.075','fused':'base + layer14 answer_mean PCA64, C=.1'},'train_n':len(tr),'test_n':len(te),'train_groups':len(set(groups[tr])),'test_groups':len(set(groups[te])),'test_correct':int(y[te].sum()),'test_incorrect':int(len(te)-y[te].sum()),'group_overlap':len(set(groups[tr])&set(groups[te])),'baseline':met(pb),'fused':met(pf),'auroc_lift':float(roc_auc_score(y[te],pf)-roc_auc_score(y[te],pb)),'group_bootstrap_lift_95ci':[float(x) for x in np.quantile(boot,[.025,.975])],'test_keys':keys[te].tolist()}
path=RUNS/'104_scientist_fixed_group_holdout.json'; path.write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='test_keys'},indent=2))
