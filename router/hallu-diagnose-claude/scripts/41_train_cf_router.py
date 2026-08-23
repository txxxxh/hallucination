"""41: train/evaluate a router on deployable counterfactual probe deltas."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from common import read_jsonl
from cf_probe_common import FEATURE_VERSION, TREATMENTS, vector_at_layer

def load_rows(feat_dir):
    rows = read_jsonl(Path(feat_dir) / "index.jsonl")
    first = np.load(Path(feat_dir) / f"{rows[0]['sid']}.npz")
    if int(first["feature_version"]) != FEATURE_VERSION: raise ValueError("stale CF features")
    return rows, first["states"].shape[1]

def labels_ok(y, tr, te): return set(y[te]).issubset(set(y[tr])) and len(set(y[te])) > 1

def split_rows(rows, y, mode, seed):
    groups = np.array([r["sid"] for r in rows]); rng = np.random.RandomState(seed)
    if mode == "random":
        sp = GroupShuffleSplit(n_splits=20, test_size=.2, random_state=seed)
        for tr, te in sp.split(np.zeros(len(rows)), y, groups):
            if labels_ok(y, tr, te): return "grouped-random", tr, te
    if mode == "template":
        by = defaultdict(set)
        for r in rows: by[r["label"]].add(r["template_id"])
        held = {k: sorted(v)[rng.randint(len(v))] for k, v in by.items() if len(v) > 1}
        te = np.array([i for i,r in enumerate(rows) if held.get(r["label"]) == r["template_id"]])
        tr = np.array([i for i in range(len(rows)) if i not in set(te)])
        if labels_ok(y,tr,te): return f"leave-template({held})",tr,te
    raise ValueError(f"cannot build {mode} split")

def matrix(feat_dir, rows, layer):
    X=[]
    for r in rows:
        with np.load(Path(feat_dir) / f"{r['sid']}.npz") as d: X.append(vector_at_layer(d,layer))
    return np.stack(X)

def fit_eval(X, y, tr, te, kind):
    imp=SimpleImputer(strategy="median").fit(np.where(np.isfinite(X[tr]),X[tr],np.nan))
    xtr=imp.transform(X[tr]); xte=imp.transform(np.where(np.isfinite(X[te]),X[te],np.nan))
    sc=StandardScaler().fit(xtr); clf=(LogisticRegression(max_iter=3000,C=.5) if kind=="probe"
        else HistGradientBoostingClassifier(max_iter=300))
    clf.fit(sc.transform(xtr),y[tr]); pred=clf.predict(sc.transform(xte)); proba=clf.predict_proba(sc.transform(xte))
    auc={str(c):round(roc_auc_score((y[te]==c).astype(int),proba[:,j]),3)
         for j,c in enumerate(clf.classes_) if 0<(y[te]==c).sum()<len(te)}
    return round(f1_score(y[te],pred,average="macro"),3),auc,(clf,imp,sc)

def main(feat_dir, seed):
    rows,L=load_rows(feat_dir); y=np.array([r["label"] for r in rows]); results={}
    print(f"n={len(y)} layers={L} labels={Counter(y)}")
    deploy_bundle=None
    for mode in ("random","template"):
        name,tr,te=split_rows(rows,y,mode,seed)
        # Inner grouped split selects layer without touching outer test.
        inner=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+17)
        fit_rel,val_rel=next(inner.split(np.zeros(len(tr)),y[tr],np.array([rows[i]["sid"] for i in tr])))
        fit,val=tr[fit_rel],tr[val_rel]; curve=[]
        for layer in range(0,L,max(1,L//14)):
            X=matrix(feat_dir,rows,layer); f1,_,_=fit_eval(X,y,fit,val,"probe"); curve.append((layer,f1))
        layer=max(curve,key=lambda x:x[1])[0]; X=matrix(feat_dir,rows,layer)
        pf1,pauc,bundle=fit_eval(X,y,tr,te,"probe"); gf1,gauc,_=fit_eval(X,y,tr,te,"gbdt")
        results[name]={"layer":layer,"probe_macro_f1":pf1,"probe_perclass_auroc":pauc,
                       "gbdt_macro_f1":gf1,"gbdt_perclass_auroc":gauc,"inner_curve":curve,
                       "train":len(tr),"test":len(te)}
        print(name,results[name])
        if mode=="random": deploy_bundle=(layer,bundle)
    # Deployment model is refit on all data at random-split-selected layer.
    layer,_=deploy_bundle; X=matrix(feat_dir,rows,layer)
    imp=SimpleImputer(strategy="median").fit(np.where(np.isfinite(X),X,np.nan)); xx=imp.transform(X)
    sc=StandardScaler().fit(xx); clf=LogisticRegression(max_iter=3000,C=.5).fit(sc.transform(xx),y)
    joblib.dump({"feature_version":FEATURE_VERSION,"treatments":TREATMENTS,"layer":layer,
                 "imputer":imp,"scaler":sc,"classifier":clf},Path(feat_dir)/"cf_router.joblib")
    json.dump(results,open(Path(feat_dir)/"cf_router_results.json","w"),indent=2)
    print(f"[cf41 save] {Path(feat_dir)/'cf_router.joblib'}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); main(a.features,a.seed)
