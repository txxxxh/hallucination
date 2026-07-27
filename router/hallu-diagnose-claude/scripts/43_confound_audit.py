#!/usr/bin/env python3
"""
43_confound_audit.py — router 的领域捷径审计 (无 GPU, 吃已有特征)

要回答的致命问题: leave-domain F1≈0, 而 random split 0.986 / leave-template 0.736。
后两者里有多少是 "这是不是数学题" 这个捷径?

由于 Z1/Z6 几乎全 factual、Z2/Z4 几乎全 math, 跨域评估在结构上不可行。
但**域内二分类**可行且干净:
  factual 内: Z1(未学过) vs Z6(校准失败)   —— 领域恒定, 分开必须靠 stressor 信息
  math    内: Z2(干扰)   vs Z4(budget)      —— 同上
若域内仍显著优于 chance, 则 router 不只是在读领域。这是当前数据能做的最强反驳。

同时输出:
  * stressor×domain 列联表 + Cramér's V (量化混淆程度)
  * 从特征预测 domain 的 AUROC (领域信号有多强)
  * 域内的特征族消融 (F1残差 / F2 logit-lens / F3 注意力 / F4 不确定性)
  * 置换检验 (小样本下的显著性)
  * NaN 审计 (是否集中于某个 stressor -> 伪特征风险)

用法:
  python 43_confound_audit.py --features data/features/Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import argparse, json, math, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

FAMILIES = ["f1", "f2", "f3_ent", "f3_susp", "f4"]


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def load(feat_dir: Path, layer_frac=0.66):
    idx = read_jsonl(feat_dir / "index.jsonl")
    rows, feats = [], []
    nan_report = defaultdict(lambda: {"n": 0, "any_nan": 0, "last_layer_nan": 0, "f3_nan": 0})
    for r in idx:
        p = feat_dir / f"{r['sid']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        f1 = d["f1"].astype(np.float32)
        L = f1.shape[0]
        l = min(int(L * layer_frac), L - 1)
        parts = {
            "f1": f1[l],
            "_f1_all": f1,
            "f2": d["f2"].reshape(-1).astype(np.float32),
            "f3_ent": d["f3_ent"].reshape(-1).astype(np.float32),
            "f3_susp": d["f3_susp"].astype(np.float32),
            "f4": d["f4"].astype(np.float32),
        }
        lab = r["label"]
        st = nan_report[lab]
        st["n"] += 1
        st["any_nan"] += int(any(not np.all(np.isfinite(v)) for v in parts.values()))
        st["last_layer_nan"] += int(not np.all(np.isfinite(f1[-1])))
        st["f3_nan"] += int(not np.all(np.isfinite(parts["f3_ent"])))
        rows.append(r)
        feats.append({k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in parts.items()})
    return rows, feats, {k: dict(v) for k, v in nan_report.items()}, L


def assemble(feats, families):
    families = [f for f in families if not f.startswith("_")]
    return np.stack([np.concatenate([f[k] for k in families]) for f in feats])


def cv_macro_f1(X, y, groups, seed=0, n_splits=5, select_k=128):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score, roc_auc_score
    ng = len(set(groups))
    gkf = GroupKFold(n_splits=min(n_splits, ng))
    pred = np.empty(len(y), dtype=object)
    prob = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        steps = [("sc", StandardScaler())]
        if select_k and X.shape[1] > select_k:
            steps.append(("sel", SelectKBest(f_classif, k=min(select_k, X.shape[1]))))
        steps.append(("lr", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")))
        pipe = Pipeline(steps).fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
        classes = list(pipe.classes_)
        if len(classes) == 2:
            prob[te] = pipe.predict_proba(X[te])[:, 1]
    mask = pred != None  # noqa
    f1 = f1_score(y[mask], pred[mask].astype(y.dtype), average="macro", zero_division=0)
    auc = None
    if len(set(y)) == 2:
        pos = sorted(set(y))[1]
        try:
            auc = roc_auc_score((y[mask] == pos).astype(int), prob[mask])
        except ValueError:
            auc = None
    return float(f1), (float(auc) if auc is not None else None)


def permutation_p(X, y, groups, observed, seed=0, n_perm=200):
    """按 group 打乱标签, 估计 macro-F1 的零分布。"""
    rng = np.random.RandomState(seed)
    g2y = {}
    for g, lab in zip(groups, y):
        g2y[g] = lab
    gs = np.array(sorted(g2y))
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(gs)
        mapping = {a: g2y[b] for a, b in zip(gs, perm)}
        yp = np.array([mapping[g] for g in groups])
        if len(set(yp)) < 2:
            continue
        f1, _ = cv_macro_f1(X, yp, groups, seed=0)
        null.append(f1)
    null = np.array(null)
    return float((np.sum(null >= observed) + 1) / (len(null) + 1)), \
           float(null.mean()) if len(null) else float("nan")


def cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(table)[0]
    n = table.sum()
    r, k = table.shape
    return float(math.sqrt(chi2 / (n * (min(r, k) - 1)))) if n and min(r, k) > 1 else float("nan")


def main(feat_dir, out_path, select_k, n_perm):
    feat_dir = Path(feat_dir)
    rows, feats, nan_report, L = load(feat_dir)
    labels = np.array([r["label"] for r in rows])
    domains = np.array([r.get("domain", "") for r in rows])
    groups = np.array([r["sid"].replace("__clean", "") for r in rows])
    res = {"n": len(rows), "n_layers": L, "label_counts": dict(Counter(labels)),
           "nan_audit": nan_report}

    # ---------- 1. 混淆程度 ----------
    labs = sorted(set(labels)); doms = sorted(set(domains))
    tab = np.array([[int(((labels == a) & (domains == d)).sum()) for d in doms] for a in labs])
    res["stressor_domain_table"] = {"rows": labs, "cols": doms, "counts": tab.tolist(),
                                    "cramers_v": round(cramers_v(tab), 4)}
    res["stressor_domain_table"]["note"] = (
        "Cramér's V 接近 1 = stressor 与 domain 几乎完全混淆, 跨域评估不可行, "
        "且同域外的任何结果都可能是领域捷径")

    # ---------- 2. 领域本身有多可预测 ----------
    X_all = assemble(feats, FAMILIES)
    keep = labels != "CLEAN"
    if len(set(domains[keep])) == 2:
        f1d, aucd = cv_macro_f1(X_all[keep], domains[keep], groups[keep], select_k=select_k)
        res["domain_predictability"] = {"macro_f1": round(f1d, 4),
                                        "auroc": round(aucd, 4) if aucd else None,
                                        "note": "接近 1 = 领域信号极强, 混淆下的高分需谨慎解读"}

    # ---------- 3. 域内二分类 (核心反驳) ----------
    within = {}
    for dom, pair in [("factual", ("Z1", "Z6")), ("math", ("Z2", "Z4"))]:
        m = (domains == dom) & np.isin(labels, pair)
        n_by = dict(Counter(labels[m]))
        if m.sum() < 40 or len(set(labels[m])) < 2:
            within[f"{dom}:{pair[0]}_vs_{pair[1]}"] = {"skipped": f"n={int(m.sum())}, {n_by}"}
            continue
        sub_idx = np.flatnonzero(m)
        yw, gw = labels[m], groups[m]
        # 逐层扫描 f1 残差流 (固定单层可能错过信号); 层选择在报告中明示
        layer_curve = []
        best_l, best_f1l = None, -1
        for li in range(1, L):
            Xl = np.stack([feats[i]["_f1_all"][li] for i in sub_idx])
            fl, _ = cv_macro_f1(Xl, yw, gw, select_k=select_k)
            layer_curve.append((li, round(fl, 4)))
            if fl > best_f1l:
                best_f1l, best_l = fl, li
        Xw = np.stack([np.concatenate([feats[i]["_f1_all"][best_l]] +
                                      [feats[i][k] for k in FAMILIES if k != "f1"])
                       for i in sub_idx])
        f1, auc = cv_macro_f1(Xw, yw, gw, select_k=select_k)
        maj = max(Counter(yw).values()) / len(yw)
        from sklearn.metrics import f1_score
        maj_lab = Counter(yw).most_common(1)[0][0]
        f1_maj = f1_score(yw, [maj_lab] * len(yw), average="macro", zero_division=0)
        p, null_mean = permutation_p(Xw, yw, gw, f1, n_perm=n_perm)
        # 特征族消融
        abl = {"f1@best_layer": round(best_f1l, 4)}
        for fam in [f for f in FAMILIES if f != "f1"]:
            Xf = assemble([feats[i] for i in sub_idx], [fam])
            abl[fam] = round(cv_macro_f1(Xf, yw, gw, select_k=select_k)[0], 4)
        Xnos = np.stack([np.concatenate([feats[i]["_f1_all"][best_l]] +
                                        [feats[i][k] for k in FAMILIES
                                         if k not in ("f1", "f3_susp")]) for i in sub_idx])
        abl["all_minus_f3_susp"] = round(cv_macro_f1(Xnos, yw, gw, select_k=select_k)[0], 4)
        within[f"{dom}:{pair[0]}_vs_{pair[1]}"] = {
            "n": int(m.sum()), "counts": n_by,
            "macro_f1": round(f1, 4), "auroc": round(auc, 4) if auc else None,
            "best_f1_layer": best_l, "f1_layer_curve": layer_curve,
            "majority_baseline_macro_f1": round(f1_maj, 4),
            "majority_class_rate": round(maj, 4),
            "permutation_p": p, "permutation_null_mean_f1": round(null_mean, 4),
            "family_ablation_macro_f1": abl,
        }
    res["within_domain"] = within
    res["within_domain_note"] = (
        "领域在此恒定, 因此高于多数类基线且置换 p<0.05 => router 读到的是 stressor 信息, "
        "不只是领域捷径。注意 f3_susp 使用了构造信息(干扰span位置), 部署不可得; "
        "看 no_f3_susp 一行判断可部署性能。")

    # ---------- 4. 判读 ----------
    ok = [v for v in within.values() if "macro_f1" in v]
    if ok:
        beats = [v for v in ok
                 if v["macro_f1"] > v["majority_baseline_macro_f1"] + 0.05
                 and v["permutation_p"] < 0.05]
        res["verdict"] = (
            f"{len(beats)}/{len(ok)} 个域内任务显著优于多数类 => "
            + ("router 不只是领域捷径" if len(beats) == len(ok)
               else "部分域内可分, 领域捷径风险仍需正文披露"))
    Path(out_path).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "nan_audit"},
                     indent=2, ensure_ascii=False)[:4000])
    print("\nNaN 审计:", json.dumps(nan_report, indent=1, ensure_ascii=False))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--select-k", type=int, default=128)
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()
    main(a.features, a.out or str(Path(a.features) / "confound_audit.json"),
         a.select_k, a.n_perm)
