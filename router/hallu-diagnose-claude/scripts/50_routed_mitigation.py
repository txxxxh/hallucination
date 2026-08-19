"""Phase 4 端到端闭环: router 诊断 -> 查疗效表分发治疗 -> 对比 baselines。
依赖:
  - 40/41 的特征与 router (现在就能训)
  - 疗效查表 configs/cure_table.json (来自 30_stats 的矩阵结果; 矩阵没跑完前
    可用 DEFAULT_POLICY 占位跑通全流程, 跑完后替换)
  - symptom-routed baseline 需要 11 的 symptom 标签 (没有则自动跳过该 baseline)
用法: python scripts/50_routed_mitigation.py --model unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit \
        --features data/features/DeepSeek-R1-Distill-Llama-8B
"""
import argparse, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import read_jsonl, write_jsonl, DATA, LM, outcome

# 疗效表: stressor -> 最优治疗。矩阵跑完后由 30 的结果生成 configs/cure_table.json 覆盖。
DEFAULT_POLICY = {"Z1": "T-RAG", "Z2": "T-Clean", "Z3": "T-CF", "Z4": "T-Budget", "Z6": "T-Abstain"}
SYMPTOM_POLICY = {"S1": "T-RAG", "S2": "T-RAG", "S3": "T-Clean", "S4": "T-Budget", "S0": "T-Abstain"}

def load_policy():
    p = Path(__file__).resolve().parent.parent / "configs/cure_table.json"
    if p.exists():
        tab = json.load(open(p))   # {stressor: {treatment: cure_rate}}
        return {z: max(ts, key=ts.get) for z, ts in tab.items()}
    print("[warn] configs/cure_table.json 不存在, 用 DEFAULT_POLICY 占位")
    return DEFAULT_POLICY

def train_router(feat_dir, holdout_templates):
    """训练集排除 holdout 模板，并仅用训练集统计量填补非有限特征。"""
    idx = [r for r in read_jsonl(Path(feat_dir) / "index.jsonl") if r["label"] != "CLEAN"]
    tr = [r for r in idx if r["template_id"] not in holdout_templates]
    te = [r for r in idx if r["template_id"] in holdout_templates]
    def feats(rows, layer=None):
        X, y = [], []
        for r in rows:
            d = np.load(Path(feat_dir) / f"{r['sid']}.npz")
            l = layer if layer is not None else d["f1"].shape[0] * 2 // 3  # 默认 2/3 深度层
            X.append(np.concatenate([d["f1"][l].astype(np.float32), d["f2"].reshape(-1),
                                     d["f3_ent"].reshape(-1), d["f3_susp"], d["f4"]]))
            y.append(r["label"])
        return np.stack(X), np.array(y)
    Xtr, ytr = feats(tr)
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr = imp.transform(Xtr)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Xtr), ytr)
    return clf, imp, sc, feats, tr, te

def main(model_name, feat_dir, holdout_templates, max_per_stressor):
    policy = load_policy()
    clf, imp, sc, feats, tr_idx, te_idx = train_router(feat_dir, holdout_templates)
    if max_per_stressor:
        kept, counts = [], {}
        for r in te_idx:
            label = r["label"]
            if counts.get(label, 0) < max_per_stressor:
                kept.append(r)
                counts[label] = counts.get(label, 0) + 1
        te_idx = kept
    if not te_idx:
        raise ValueError("holdout_templates 没有命中任何特征样本")
    sid2label = {r["sid"]: r["label"] for r in te_idx}
    print(f"router 训练 {len(tr_idx)} / 闭环测试 {len(te_idx)} (holdout={holdout_templates})")

    # 测试集样本原文
    all_samples = {}
    for z in ("z1", "z2", "z3", "z4", "z6"):
        p = DATA / f"processed/{z}_final.jsonl"
        if p.exists():
            for s in read_jsonl(p):
                all_samples[s["sid"]] = s
    test = [all_samples[r["sid"]] for r in te_idx if r["sid"] in all_samples]

    # router 预测
    Xte, yte = feats(te_idx)
    Xte = np.where(np.isfinite(Xte), Xte, np.nan)
    pred = clf.predict(sc.transform(imp.transform(Xte)))
    acc = (pred == yte).mean()
    print(f"router 闭环测试诊断准确率 = {acc:.1%}")

    # 五个 arm: none / best-single / T-SC / router-routed / oracle-routed (+symptom-routed 若有)
    from importlib import import_module
    m21 = import_module("21_run_matrix") if False else None  # 复用 apply_treatment 逻辑
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from importlib.machinery import SourceFileLoader
    mat = SourceFileLoader("mat", str(Path(__file__).parent / "21_run_matrix.py")).load_module()

    lm = LM(model_name)
    arms = {}
    best_single = max(set(policy.values()), key=list(policy.values()).count)  # 最常见对因治疗当 one-size
    arms["none"] = ["none"] * len(test)
    arms["best-single"] = [best_single] * len(test)
    arms["router-routed"] = [policy.get(p, "none") for p in pred]
    arms["oracle-routed"] = [policy.get(sid2label[s["sid"]], "none") for s in test]
    if all(s.get("symptom") for s in test):
        arms["symptom-routed"] = [SYMPTOM_POLICY.get(s["symptom"][0], "none") for s in test]

    rows = []
    for arm, treats in arms.items():
        prompts = [mat.apply_treatment(t, s) for t, s in zip(treats, test)]
        gens = lm.chat(prompts, temperature=0.0, max_tokens=1024)
        n_strict = n_honest = 0
        for s, t, g in zip(test, treats, gens):
            gold = s["answer"] if s["answer"] != "UNKNOWN_ENTITY" else "UNANSWERABLE"
            o = outcome(g[0], gold, s.get("answer_aliases", []), bool(s["meta"].get("numeric")))
            n_strict += o["strict"]; n_honest += o["honest"]
            rows.append(dict(sid=s["sid"], arm=arm, treatment=t, stressor=s["stressor"], **o))
        print(f"  {arm:>16}: strict={n_strict/len(test):.1%}  honest={n_honest/len(test):.1%}")
    write_jsonl(rows, DATA / "results/routed_mitigation.jsonl")
    print("目标: router-routed > symptom-routed > best-single, 且逼近 oracle (gap<5pt)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--features", required=True)
    ap.add_argument("--holdout_templates", nargs="+",
                    default=["popqa-genre", "gsm8k-dose", "math-Algebra", "falseqa"])
    ap.add_argument("--max_per_stressor", type=int, default=100,
                    help="每个 stressor 最多保留多少个闭环测试样本；0 表示不限制")
    a = ap.parse_args()
    main(a.model, a.features, a.holdout_templates, a.max_per_stressor)
