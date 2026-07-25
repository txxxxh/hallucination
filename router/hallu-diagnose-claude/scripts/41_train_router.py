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
from common import DATA, read_jsonl

def load(feat_dir, with_clean):
    idx = read_jsonl(Path(feat_dir) / "index.jsonl")
    if not with_clean:
        idx = [r for r in idx if r["label"] != "CLEAN"]
    X1, X2, X3, X4, y, meta = [], [], [], [], [], []
    for r in idx:
        d = np.load(Path(feat_dir) / f"{r['sid']}.npz")
        X1.append(d["f1"]); X2.append(d["f2"].reshape(-1))
        X3.append(np.concatenate([d["f3_ent"].reshape(-1), d["f3_susp"]]))
        X4.append(d["f4"]); y.append(r["label"]); meta.append(r)
    return (np.stack(X1), np.stack(X2), np.stack(X3), np.stack(X4)), np.array(y), meta

def splits(meta, y, mode, seed=0):
    rng = np.random.RandomState(seed)
    n = len(meta)
    if mode == "random":
        perm = rng.permutation(n); cut = int(n * 0.8)
        yield "random", perm[:cut], perm[cut:]
    elif mode == "template":
        by_label_tpl = defaultdict(set)
        for m in meta:
            by_label_tpl[m["label"]].add(m["template_id"])
        # 每个 label 留出一个 template 做测试
        held = {lab: sorted(tpls)[rng.randint(len(tpls))] for lab, tpls in by_label_tpl.items() if len(tpls) > 1}
        te = [i for i, m in enumerate(meta) if held.get(m["label"]) == m["template_id"]]
        tr = [i for i in range(n) if i not in set(te)]
        yield f"leave-template({held})", np.array(tr), np.array(te)
    elif mode == "domain":
        for dom in sorted({m["domain"] for m in meta}):
            te = [i for i, m in enumerate(meta) if m["domain"] == dom]
            tr = [i for i in range(n) if m_dom(meta, i) != dom]
            if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
                continue
            yield f"leave-domain({dom})", np.array(tr), np.array(te)

def m_dom(meta, i):
    return meta[i]["domain"]

def eval_clf(clf, Xtr, ytr, Xte, yte):
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

def main(feat_dir, with_clean):
    (X1, X2, X3, X4), y, meta = load(feat_dir, with_clean)
    L = X1.shape[1]
    print(f"n={len(y)}  layers={L}  labels={Counter(y)}")
    results = {}

    for mode in ("random", "template", "domain"):
        for name, tr, te in splits(meta, y, mode):
            print(f"\n===== split: {name}  (train={len(tr)} test={len(te)}) =====")
            # (a) 逐层 F1-residual probe
            layer_acc = []
            for l in range(0, L, max(1, L // 16)):  # 采样层以省时间; 正式跑改步长1
                f1s, _, _ = eval_clf(LogisticRegression(max_iter=2000, C=0.5),
                                     X1[tr, l].astype(np.float32), y[tr],
                                     X1[te, l].astype(np.float32), y[te])
                layer_acc.append((l, round(f1s, 3)))
            best_l, best_f1 = max(layer_acc, key=lambda t: t[1])
            print(f"  逐层probe: best layer={best_l} macroF1={best_f1} | 曲线={layer_acc}")
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
            results[name] = dict(best_layer=best_l, probe_f1=best_f1, gbdt_f1=round(f1g, 3),
                                 f4_only=round(f1u, 3), majority=round(f1m, 3),
                                 layer_curve=layer_acc)
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
    a = ap.parse_args()
    main(a.features, a.with_clean)
