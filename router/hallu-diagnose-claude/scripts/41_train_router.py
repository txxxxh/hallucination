"""Phase 3.2 Router 训练与评测。依赖 40_extract_features.py 的输出, 不依赖治疗矩阵。
产出论文 §5 的全部数字:
  (a) 逐层 linear probe 准确率曲线 ("模型第几层知道自己被什么困住")
  (b) 三套划分: random / leave-one-template-out / leave-one-domain-out
  (c) 任务: 4类 stressor / 5类(+CLEAN) / 多标签(Z6 叠加)
  (d) baselines: F4-only(纯不确定性) / 多数类; symptom-MAP 由 11 --stats 提供, 手动对比
用法: python scripts/41_train_router.py --features data/features/DeepSeek-R1-Distill-Llama-8B
"""
import argparse, json
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from common import DATA, read_jsonl

REQUIRED_FEATURE_VERSION = 2

def base_sid(row):
    return row["sid"].removesuffix("__clean")


def load(feat_dir, with_clean, include_privileged_f3_susp=False):
    idx = read_jsonl(Path(feat_dir) / "index.jsonl")
    if not with_clean:
        idx = [r for r in idx if r["label"] != "CLEAN"]
    X1, X2, X3, X4, y, meta = [], [], [], [], [], []
    for r in idx:
        path = Path(feat_dir) / f"{r['sid']}.npz"
        with np.load(path) as d:
            if ("feature_version" not in d
                    or int(d["feature_version"]) != REQUIRED_FEATURE_VERSION):
                raise ValueError(
                    f"{path} is stale; rerun 40_extract_features.py before 41"
                )
            X1.append(d["f1"]); X2.append(d["f2"].reshape(-1))
            f3 = [d["f3_ent"].reshape(-1)]
            if include_privileged_f3_susp:
                f3.append(d["f3_susp"])
            X3.append(np.concatenate(f3))
            X4.append(d["f4"]); y.append(r["label"]); meta.append(r)
    return (np.stack(X1), np.stack(X2), np.stack(X3), np.stack(X4)), np.array(y), meta

def splits(meta, y, mode, seed=0):
    rng = np.random.RandomState(seed)
    n = len(meta)
    groups = np.array([base_sid(m) for m in meta])
    if mode == "random":
        splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=seed)
        for tr, te in splitter.split(np.zeros(n), y, groups):
            if valid_label_support(y, tr, te):
                yield "grouped-random", tr, te
                break
        else:
            raise ValueError("could not create grouped random split with complete label support")
    elif mode == "template":
        by_label_tpl = defaultdict(set)
        for m in meta:
            if m["label"] == "CLEAN":
                continue
            by_label_tpl[m["label"]].add(m["template_id"])
        # 每个 label 留出一个 template 做测试
        held = {lab: sorted(tpls)[rng.randint(len(tpls))] for lab, tpls in by_label_tpl.items() if len(tpls) > 1}
        test_groups = {
            base_sid(m) for m in meta
            if m["label"] != "CLEAN"
            and held.get(m["label"]) == m["template_id"]
        }
        te = np.array([i for i, group in enumerate(groups) if group in test_groups])
        tr = np.array([i for i, group in enumerate(groups) if group not in test_groups])
        if valid_label_support(y, tr, te):
            yield f"leave-template({held})", tr, te
    elif mode == "domain":
        for dom in sorted({m["domain"] for m in meta}):
            te = np.array([i for i, m in enumerate(meta) if m["domain"] == dom])
            tr = np.array([i for i in range(n) if m_dom(meta, i) != dom])
            if not valid_label_support(y, tr, te):
                continue
            yield f"leave-domain({dom})", tr, te

def m_dom(meta, i):
    return meta[i]["domain"]


def valid_label_support(y, tr, te):
    train_labels, test_labels = set(y[tr]), set(y[te])
    return len(test_labels) >= 2 and test_labels.issubset(train_labels)


def inner_group_split(indices, meta, y, seed=17):
    """Split only outer-train data for layer selection."""
    groups = np.array([base_sid(meta[i]) for i in indices])
    splitter = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    for rel_fit, rel_val in splitter.split(np.zeros(len(indices)), y[indices], groups):
        fit, val = indices[rel_fit], indices[rel_val]
        if valid_label_support(y, fit, val):
            return fit, val
    raise ValueError("could not create grouped inner split with complete label support")

def eval_clf(clf, Xtr, ytr, Xte, yte):
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    Xte = np.where(np.isfinite(Xte), Xte, np.nan)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr = imp.transform(Xtr)
    Xte = imp.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    f1 = f1_score(yte, pred, average="macro")
    try:
        proba = clf.predict_proba(sc.transform(Xte))
        aucs = {c: roc_auc_score((yte == c).astype(int), proba[:, j])
                for j, c in enumerate(clf.classes_) if 0 < (yte == c).sum() < len(yte)}
    except Exception:
        aucs = {}
    return f1, aucs, pred

def main(feat_dir, with_clean, include_privileged_f3_susp=False):
    (X1, X2, X3, X4), y, meta = load(
        feat_dir, with_clean, include_privileged_f3_susp
    )
    L = X1.shape[1]
    print(f"n={len(y)}  layers={L}  labels={Counter(y)}")
    results = {}

    for mode in ("random", "template", "domain"):
        for name, tr, te in splits(meta, y, mode):
            print(f"\n===== split: {name}  (train={len(tr)} test={len(te)}) =====")
            fit, val = inner_group_split(tr, meta, y)
            # Select the layer on inner validation, never on outer test.
            val_curve = []
            for l in range(0, L, max(1, L // 16)):  # 采样层以省时间; 正式跑改步长1
                f1s, _, _ = eval_clf(LogisticRegression(max_iter=2000, C=0.5),
                                     X1[fit, l].astype(np.float32), y[fit],
                                     X1[val, l].astype(np.float32), y[val])
                val_curve.append((l, round(f1s, 3)))
            best_l, best_val_f1 = max(val_curve, key=lambda t: t[1])
            probe_test_f1, _, _ = eval_clf(
                LogisticRegression(max_iter=2000, C=0.5),
                X1[tr, best_l].astype(np.float32), y[tr],
                X1[te, best_l].astype(np.float32), y[te],
            )
            print(f"  逐层probe: selected layer={best_l} "
                  f"inner-val macroF1={best_val_f1} "
                  f"outer-test macroF1={probe_test_f1:.3f} | val曲线={val_curve}")
            # (b) 全特征 GBDT
            Xall_tr = np.concatenate([X1[tr, best_l].astype(np.float32), X2[tr], X3[tr], X4[tr]], 1)
            Xall_te = np.concatenate([X1[te, best_l].astype(np.float32), X2[te], X3[te], X4[te]], 1)
            f1g, aucs, pred = eval_clf(HistGradientBoostingClassifier(max_iter=300),
                                       Xall_tr, y[tr], Xall_te, y[te])
            print(f"  全特征GBDT: macroF1={f1g:.3f}  perclass-AUROC={ {k: round(v,3) for k,v in aucs.items()} }")
            # (c) baseline: F4-only + 多数类
            f1u, _, _ = eval_clf(LogisticRegression(max_iter=2000), X4[tr], y[tr], X4[te], y[te])
            maj = Counter(y[tr]).most_common(1)[0][0]
            f1m = f1_score(y[te], [maj] * len(te), average="macro")
            print(f"  baseline: F4-only={f1u:.3f}  多数类={f1m:.3f}")
            results[name] = dict(best_layer=best_l,
                                 probe_inner_val_f1=best_val_f1,
                                 probe_test_f1=round(probe_test_f1, 3),
                                 gbdt_f1=round(f1g, 3),
                                 f4_only=round(f1u, 3), majority=round(f1m, 3),
                                 inner_val_layer_curve=val_curve,
                                 privileged_f3_susp=include_privileged_f3_susp)
            # 混淆样例
            wrong = [(meta[te[i]]["sid"], y[te][i], pred[i]) for i in range(len(te)) if pred[i] != y[te][i]][:5]
            print("  误诊样例:", wrong)
    out = Path(feat_dir) / "router_results.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\n[save] {out}")
    print(">>> 对比线: symptom-MAP 准确率见 `11_annotate_symptom.py --stats` 输出;"
          " router 的 template/domain 划分 F1 必须显著高于它, §5 主张才成立 <<<")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--with_clean", action="store_true", help="加入 CLEAN 类 (5类设置)")
    ap.add_argument("--include_privileged_f3_susp", action="store_true",
                    help="加入构造时才知道的可疑-span特征；仅作上界/消融，默认关闭")
    a = ap.parse_args()
    main(a.features, a.with_clean, a.include_privileged_f3_susp)
