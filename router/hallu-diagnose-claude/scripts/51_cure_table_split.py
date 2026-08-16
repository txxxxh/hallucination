#!/usr/bin/env python3
"""
51_cure_table_split.py — 修复闭环的 cure-table 泄漏 (无 GPU, 完全离线)

问题: 51.1% 的闭环数字, 其 cure table (每个 stressor 选哪个治疗) 是从**同一批**
      治疗矩阵选出的经验最优。若选表数据与评估数据重叠, 等于在训练集上评估。

关键洞察: 治疗矩阵已含 "每个样本 × 每种治疗" 的结果, 因此**任何路由策略都能离线查表求值**,
          不必重跑模型。于是修复泄漏是纯分析工作。

两路不相交划分 (按 item, 分层于 stressor):
  cure_select  : 训练 router + 挑选每个 stressor 的最优治疗 -> cure table
  eval         : 仅用于报告闭环性能

输出:
  * 泄漏版 vs 修复版闭环对比 (量化泄漏带来的虚高)
  * cure table 稳定性: 两半数据是否选出相同治疗 (策略是否稳健)
  * 五臂对比 + cluster bootstrap 95% CI
  * 若提供 features, 仅在 cure_select 上训练 router 并对 eval 生成预测
  * 若提供 router 预测文件, 使用外部的 out-of-sample 预测
  * 两者均未提供时只报告 oracle 上界

用法:
  python 51_cure_table_split.py --matrix data/results/matrix_Qwen2.5-7B-Instruct.jsonl \\
      [--router-pred data/features/.../router_test_pred.jsonl] --out data/results/closed_loop_fixed.json

router-pred 格式 (每行): {"sid": "...", "pred_stressor": "Z2"}
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

HONEST_ROWS = {"Z6"}          # 这些 stressor 用 honest 度量, 其余用 strict
EXCLUDE_TREATMENTS = {"none", "T-CleanOracle"}   # oracle 类不可部署, 不进 cure table


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def metric_of(row):
    return "honest" if row["stressor"] in HONEST_ROWS else "strict"


def build_lookup(rows):
    """(sid, treatment) -> outcome dict; sid -> stressor/domain/template"""
    lut, meta = {}, {}
    for r in rows:
        lut[(r["sid"], r["treatment"])] = r
        meta.setdefault(r["sid"], {"stressor": r["stressor"], "domain": r.get("domain", ""),
                                   "template_id": r.get("template_id", "")})
    return lut, meta


def cure_table_from(sids, lut, meta, treatments):
    """在给定 sid 子集上, 为每个 stressor 选经验最优治疗。"""
    agg = defaultdict(lambda: defaultdict(list))
    for sid in sids:
        z = meta[sid]["stressor"]
        m = "honest" if z in HONEST_ROWS else "strict"
        for t in treatments:
            r = lut.get((sid, t))
            if r is not None:
                agg[z][t].append(int(r[m]))
    table, detail = {}, {}
    for z, d in agg.items():
        rates = {t: float(np.mean(v)) for t, v in d.items() if v}
        if not rates:
            continue
        best = max(rates, key=rates.get)
        table[z] = best
        detail[z] = {"chosen": best, "rates": {t: round(v, 4) for t, v in sorted(
            rates.items(), key=lambda kv: -kv[1])}}
    return table, detail


def evaluate(sids, lut, meta, policy_fn, label):
    """policy_fn(sid) -> treatment。返回 strict/honest 与逐 stressor 分解。"""
    s_ok, h_ok, per = [], [], defaultdict(lambda: [0, 0])
    used = Counter()
    for sid in sids:
        t = policy_fn(sid)
        r = lut.get((sid, t))
        if r is None:
            continue
        used[t] += 1
        z = meta[sid]["stressor"]
        m = "honest" if z in HONEST_ROWS else "strict"
        s_ok.append(int(r["strict"])); h_ok.append(int(r["honest"]))
        per[z][0] += int(r[m]); per[z][1] += 1
    return {"arm": label, "n": len(s_ok),
            "strict": round(float(np.mean(s_ok)), 4) if s_ok else None,
            "honest": round(float(np.mean(h_ok)), 4) if h_ok else None,
            "by_stressor": {z: round(a / max(b, 1), 4) for z, (a, b) in sorted(per.items())},
            "treatment_usage": dict(used),
            "_strict_vec": s_ok}


def boot_ci(a, b, n=2000, seed=0):
    """配对 bootstrap: a-b 的 95% CI。"""
    a, b = np.array(a), np.array(b)
    if len(a) != len(b) or len(a) == 0:
        return None
    rng = np.random.RandomState(seed)
    d = [np.mean(a[i] - b[i]) for i in
         (rng.randint(0, len(a), len(a)) for _ in range(n))]
    return [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]


def train_router_predict(feat_dir, train_sids, eval_sids, meta, layer_frac=2 / 3,
                         include_f3_susp=False):
    """只在 train_sids 上拟合 router，并为同一 split 的 eval_sids 生成预测。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    feat_dir = Path(feat_dir)
    index = read_jsonl(feat_dir / "index.jsonl")
    # index 可能含重复行；按 sid 去重，并排除不在治疗矩阵中的 CLEAN paired rows。
    by_sid = {}
    for r in index:
        if r["sid"] in meta and r["label"] != "CLEAN":
            by_sid.setdefault(r["sid"], r)

    needed = set(train_sids) | set(eval_sids)
    missing_index = sorted(needed - set(by_sid))
    missing_npz = sorted(s for s in needed if not (feat_dir / f"{s}.npz").exists())
    if missing_index or missing_npz:
        raise ValueError(
            f"router 特征不完整: missing_index={len(missing_index)}, "
            f"missing_npz={len(missing_npz)}"
        )
    mismatched = sorted(
        sid for sid in needed if by_sid[sid]["label"] != meta[sid]["stressor"]
    )
    if mismatched:
        raise ValueError(f"特征标签与治疗矩阵 stressor 不一致: {mismatched[:5]}")

    def features(sids):
        X = []
        for sid in sids:
            with np.load(feat_dir / f"{sid}.npz") as d:
                f1 = d["f1"]
                layer = min(int(f1.shape[0] * layer_frac), f1.shape[0] - 1)
                parts = [
                    f1[layer].astype(np.float32),
                    d["f2"].reshape(-1).astype(np.float32),
                    d["f3_ent"].reshape(-1).astype(np.float32),
                ]
                if include_f3_susp:
                    parts.append(d["f3_susp"].reshape(-1).astype(np.float32))
                parts.append(d["f4"].reshape(-1).astype(np.float32))
                X.append(np.concatenate(parts))
        return np.stack(X)

    Xtr, Xte = features(train_sids), features(eval_sids)
    ytr = np.array([meta[sid]["stressor"] for sid in train_sids])
    yte = np.array([meta[sid]["stressor"] for sid in eval_sids])
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    Xte = np.where(np.isfinite(Xte), Xte, np.nan)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    clf.fit(sc.transform(imp.transform(Xtr)), ytr)
    pred = clf.predict(sc.transform(imp.transform(Xte)))
    return (
        {sid: str(label) for sid, label in zip(eval_sids, pred)},
        {
            "n_train": len(train_sids),
            "n_eval": len(eval_sids),
            "accuracy": round(float(np.mean(pred == yte)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
            "layer_frac": layer_frac,
            "include_f3_susp": include_f3_susp,
        },
    )


def write_router_pred(path, eval_sids, router_pred, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sid in eval_sids:
            fh.write(json.dumps({
                "sid": sid,
                "pred_stressor": router_pred[sid],
                "true_stressor": meta[sid]["stressor"],
            }, ensure_ascii=False) + "\n")


def main(args):
    rows = read_jsonl(args.matrix)
    lut, meta = build_lookup(rows)
    treatments = sorted({r["treatment"] for r in rows} - EXCLUDE_TREATMENTS)
    sids = sorted(meta)
    print(f"样本 {len(sids)} | 治疗 {treatments}")

    # ---------- 分层两路划分 ----------
    rng = np.random.RandomState(args.seed)
    by_z = defaultdict(list)
    for s in sids:
        by_z[meta[s]["stressor"]].append(s)
    cure_sids, eval_sids = [], []
    for z, lst in by_z.items():
        lst = sorted(lst); rng.shuffle(lst)
        cut = int(len(lst) * args.cure_frac)
        cure_sids += lst[:cut]; eval_sids += lst[cut:]
    print(f"cure_select {len(cure_sids)} | eval {len(eval_sids)} (不相交)")

    # ---------- cure table: 全量(泄漏) vs 仅 cure_select(修复) ----------
    tbl_leak, det_leak = cure_table_from(sids, lut, meta, treatments)
    tbl_fix, det_fix = cure_table_from(cure_sids, lut, meta, treatments)
    tbl_holdout, _ = cure_table_from(eval_sids, lut, meta, treatments)
    stability = {z: {"cure_half": tbl_fix.get(z), "eval_half": tbl_holdout.get(z),
                     "agree": tbl_fix.get(z) == tbl_holdout.get(z)} for z in by_z}

    # ---------- router 预测（必须与上面的 eval split 严格对齐） ----------
    router_pred = {}
    router_metrics = None
    if args.router_pred and Path(args.router_pred).exists():
        for r in read_jsonl(args.router_pred):
            router_pred[r["sid"]] = r.get("pred_stressor") or r.get("pred")
        print(f"载入 router 预测 {len(router_pred)} 条")
    elif args.features:
        router_pred, router_metrics = train_router_predict(
            args.features, cure_sids, eval_sids, meta,
            layer_frac=args.router_layer_frac,
            include_f3_susp=args.include_f3_susp,
        )
        pred_out = args.router_pred_out or str(
            Path(args.features) / f"router_eval_pred_seed{args.seed}.jsonl"
        )
        write_router_pred(pred_out, eval_sids, router_pred, meta)
        print(
            f"router: train={router_metrics['n_train']} eval={router_metrics['n_eval']} "
            f"macro-F1={router_metrics['macro_f1']:.4f} -> {pred_out}"
        )
    if router_pred:
        missing_pred = sorted(set(eval_sids) - set(router_pred))
        if missing_pred:
            raise ValueError(
                f"router 预测未覆盖 eval split: missing={len(missing_pred)}, "
                f"examples={missing_pred[:5]}"
            )

    # ---------- 五臂 (全部在 eval_sids 上) ----------
    best_single = max(treatments, key=lambda t: np.mean(
        [int(lut[(s, t)]["honest" if meta[s]["stressor"] in HONEST_ROWS else "strict"])
         for s in cure_sids if (s, t) in lut] or [0]))
    arms = []
    arms.append(evaluate(eval_sids, lut, meta, lambda s: "none", "none"))
    arms.append(evaluate(eval_sids, lut, meta, lambda s: best_single, f"best-single({best_single})"))
    arms.append(evaluate(eval_sids, lut, meta,
                         lambda s: tbl_leak.get(meta[s]["stressor"], "none"),
                         "oracle-routed [LEAKY cure table]"))
    arms.append(evaluate(eval_sids, lut, meta,
                         lambda s: tbl_fix.get(meta[s]["stressor"], "none"),
                         "oracle-routed [clean cure table]"))
    if router_pred:
        arms.append(evaluate(eval_sids, lut, meta,
                             lambda s: tbl_fix.get(router_pred.get(s, ""), "none"),
                             "router-routed [clean cure table]"))

    by_arm = {a["arm"]: a for a in arms}
    res = {
        "n_eval": len(eval_sids), "n_cure_select": len(cure_sids),
        "cure_frac": args.cure_frac,
        "cure_table_leaky": tbl_leak, "cure_table_clean": tbl_fix,
        "cure_table_stability_across_halves": stability,
        "cure_table_detail_clean": det_fix,
        "router_metrics": router_metrics,
        "arms": [{k: v for k, v in a.items() if not k.startswith("_")} for a in arms],
    }
    # 泄漏量化 + 关键增益 CI
    lk = by_arm.get("oracle-routed [LEAKY cure table]")
    cl = by_arm.get("oracle-routed [clean cure table]")
    if lk and cl:
        res["leakage_inflation_strict"] = round(lk["strict"] - cl["strict"], 4)
    rr = by_arm.get("router-routed [clean cure table]")
    bs = by_arm.get(f"best-single({best_single})")
    nn = by_arm.get("none")
    if rr and bs:
        res["gain_router_vs_best_single"] = {
            "delta": round(rr["strict"] - bs["strict"], 4),
            "ci95": boot_ci(rr["_strict_vec"], bs["_strict_vec"])}
    if rr and nn:
        res["gain_router_vs_none"] = {
            "delta": round(rr["strict"] - nn["strict"], 4),
            "ci95": boot_ci(rr["_strict_vec"], nn["_strict_vec"])}
    if rr and cl:
        res["router_vs_oracle_gap"] = round(cl["strict"] - rr["strict"], 4)

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False)[:3500])
    print(f"\n-> {args.out}")
    print("\n>>> 论文里报告的必须是 [clean cure table] 那一行; "
          "leakage_inflation_strict 就是原先虚高的幅度 <<<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--router-pred", default=None)
    ap.add_argument("--features", default=None,
                    help="内部训练 router；仅用 cure_select 训练并预测同一 eval split")
    ap.add_argument("--router-pred-out", default=None,
                    help="内部 router 的逐样本预测输出路径")
    ap.add_argument("--router-layer-frac", type=float, default=2 / 3)
    ap.add_argument("--include-f3-susp", action="store_true",
                    help="诊断用途；f3_susp 含构造 span 信息，部署评估默认禁用")
    ap.add_argument("--out", default="closed_loop_fixed.json")
    ap.add_argument("--cure-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
