"""41b: 知识边界二元 probe ("知道 vs 不知道"), T-Gate-probe 治疗的依赖。
依赖 40_extract_features.py 的输出。执行顺序: 40 -> 41b -> 21 (含 T-GateProbe)。

标签定义 (三层校准结构中的表征层检验):
  dontknow = Z1 trig 样本 (模型确实不知道) + gate 型 Z6 (template 以 gate- 开头)
  know     = CLEAN 变体 (模型答对的干净版) + Z2/Z3 trig 样本
    关键设计: Z2/Z3 归入 know —— 模型"其实知道"(clean 版答对), 只是被干扰/误导。
    若 probe 学到的真是知识边界而非"会不会出错", 它在 Z2/Z3 上应判 know,
    从而预测 T-GateProbe 对 Z2/Z3 不触发 —— 这本身是矩阵实验的一个预注册检验。

输出:
  gate_probe.joblib               probe + scaler + 层号
  gate_scores.jsonl               {sid, p_dontknow} 覆盖全部 trig 样本 (21 消费)
  逐层 AUROC 曲线 (表征层信号存在性 = 三层结构里 (a) 层的证据)
用法: python scripts/41b_train_gate_probe.py --features data/features/Qwen2.5-7B-Instruct
"""
import argparse, json
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from common import read_jsonl, DATA

def label_of(r):
    if r["label"] == "CLEAN":
        return 0                                   # know
    if r["label"] == "Z1":
        return 1                                   # dontknow
    if r["label"] == "Z6" and r["template_id"].startswith("gate-"):
        return 1                                   # dontknow (该搜没搜的底层是不知道)
    if r["label"] in ("Z2", "Z3"):
        return 0                                   # know (被干扰/误导, 但知识在)
    return None                                    # 其他 Z6 / Z4 不参与训练

def main(feat_dir, thresh):
    feat_dir = Path(feat_dir)
    idx = read_jsonl(feat_dir / "index.jsonl")
    train_rows = [(r, label_of(r)) for r in idx]
    train_rows = [(r, y) for r, y in train_rows if y is not None]
    ys = np.array([y for _, y in train_rows])
    print(f"训练样本: {len(ys)}  (dontknow={ys.sum()}, know={(ys==0).sum()})")

    # 逐层扫描找最优层 (5折 CV AUROC, 不碰测试)
    d0 = np.load(feat_dir / f"{train_rows[0][0]['sid']}.npz")
    L = d0["f1"].shape[0]
    X_by_layer = {l: [] for l in range(0, L, max(1, L // 16))}
    for r, _ in train_rows:
        f1 = np.load(feat_dir / f"{r['sid']}.npz")["f1"].astype(np.float32)
        for l in X_by_layer:
            X_by_layer[l].append(f1[l])
    best_l, best_auc, curve = None, -1, []
    for l, X in X_by_layer.items():
        X = StandardScaler().fit_transform(np.stack(X))
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, ys,
                              cv=5, method="predict_proba")[:, 1]
        auc = roc_auc_score(ys, p)
        curve.append((l, round(auc, 3)))
        if auc > best_auc:
            best_l, best_auc = l, auc
    print(f"逐层 AUROC 曲线 (表征层信号): {curve}")
    print(f"最优层 = {best_l}  CV-AUROC = {best_auc:.3f}")

    # 最优层全量训练, 给所有 trig 样本打分
    Xtr = np.stack([np.load(feat_dir / f"{r['sid']}.npz")["f1"][best_l].astype(np.float32)
                    for r, _ in train_rows])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Xtr), ys)
    joblib.dump(dict(clf=clf, scaler=sc, layer=best_l, threshold=thresh,
                     cv_auroc=best_auc, curve=curve), feat_dir / "gate_probe.joblib")

    scores = []
    for r in idx:
        if r["variant"] != "trig":
            continue
        x = np.load(feat_dir / f"{r['sid']}.npz")["f1"][best_l].astype(np.float32)
        p = float(clf.predict_proba(sc.transform(x[None]))[:, 1][0])
        scores.append(dict(sid=r["sid"], label=r["label"], p_dontknow=round(p, 4)))
    with open(feat_dir / "gate_scores.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    # 预注册检验: probe 在各 stressor 上的触发率 (p_dontknow > thresh)
    from collections import defaultdict
    fire = defaultdict(list)
    for s in scores:
        fire[s["label"]].append(s["p_dontknow"] > thresh)
    print("\n各 stressor 触发率 (p_dontknow > %.2f):" % thresh)
    for z, v in sorted(fire.items()):
        print(f"  {z}: {np.mean(v):.1%}  (n={len(v)})")
    print(">>> 预期: Z1/Z6-gate 高触发, Z2/Z3 低触发 —— 若 Z2/Z3 也高触发, "
          "说明 probe 学的是'会出错'而非'知识边界', T-GateProbe 的机制解释要修正 <<<")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    main(a.features, a.thresh)
