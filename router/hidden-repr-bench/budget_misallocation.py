#!/usr/bin/env python3
"""
budget_misallocation.py — hidden state 能否看出"预算分配不合理"?

与已有工作的区别 (定位 novelty):
  已有: 预测所需长度 / 预测会不会答错  —— 这是**边缘量**, 已被做过。
  本文: **条件化于失败**, 判断这个失败是不是预算造成的, 以及错配的**方向**。
        关键难点: "预算不足" 与 "能力上限" 表面完全相同(都是难题、都答错、都用满预算),
        demand probe 对两者的预测都是"需要很多" —— 区分不了。能区分才是真正的 Z4 诊断。

标签 (逐 item × 逐 budget, 全部由 counterfactual 预算响应定义):
  under       : 在 B 错, 但存在 B' > B 使其对        -> 加预算可修 (Z4 欠分配)
  over        : 在 B 错, 但存在 B' < B 使其对        -> 减预算可修 (overthinking)
  capability  : 所有预算都错                          -> 预算无关, 能力上限
  ok          : 在 B 对                               -> 分配合理
  (under 与 over 可同时成立 -> 记为 nonmonotonic, 单独统计)

两个核心实验:
  E1 条件化诊断: 只在**失败样本**内, 区分 under vs capability。
     对照: 用 "预测需求" 单特征能否达到同样 AUROC。若 probe 显著更高,
     说明 hidden 里有超出"难度/需求"的信息 —— 即真正的错配信号。
  E2 题内供给敏感性: 同一题只改 prompt 中声明的预算, 看 probe 预测是否随之翻转。
     demand-only 表征在题内恒定, 无法翻转; 能翻转才证明读到了供给-需求比较。

依赖 budget_metacognition.py 的 Engine / 曲线数据。
用法:
  python budget_misallocation.py --stage label   --output-dir out          # 无GPU
  python budget_misallocation.py --stage paired  --output-dir out          # GPU
  python budget_misallocation.py --stage analyze --output-dir out          # 无GPU
"""
from __future__ import annotations
import argparse, gc, json, logging, math
from pathlib import Path
import numpy as np

import budget_metacognition as bm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("budget_misalloc")

# 声明预算写进 prompt (供给可见), 与 demand-only 表征区分的关键
SUPPLY_PROMPT = (
    "You have a thinking budget of approximately {budget} tokens for this problem.\n"
    "Problem: {q}\nReason step by step within the budget."
)


# ============================ 阶段 A: 标签 (无 GPU) ============================
def label_from_curve(rec, thresh: float):
    """从 acc(B) 曲线导出逐预算的错配标签。"""
    runs = sorted(rec["runs"], key=lambda r: r["budget"])
    bs = [r["budget"] for r in runs]
    ok = [r["acc"] >= thresh for r in runs]
    out = []
    any_ok = any(ok)
    for i, B in enumerate(bs):
        if ok[i]:
            lab = "ok"
        elif not any_ok:
            lab = "capability"          # 所有预算都不行
        else:
            can_more = any(ok[j] for j in range(i + 1, len(bs)))
            can_less = any(ok[j] for j in range(0, i))
            if can_more and can_less:
                lab = "nonmonotonic"
            elif can_more:
                lab = "under"           # 加预算可修 = Z4 欠分配
            elif can_less:
                lab = "over"            # 减预算可修 = overthinking
            else:
                lab = "capability"
        out.append(dict(budget=B, acc=runs[i]["acc"], label=lab,
                        mean_used=runs[i]["mean_used"], cap_rate=runs[i]["cap_rate"]))
    return out


def stage_label(args, out: Path):
    curve = bm.read_jsonl(out / "curve.jsonl")
    rows, per_item = [], []
    for rec in curve:
        labs = label_from_curve(rec, args.acc_threshold)
        best_b = next((l["budget"] for l in labs if l["label"] == "ok"), None)
        max_acc = max(l["acc"] for l in labs)
        per_item.append(dict(qid=rec["qid"], b_star=best_b, max_acc=max_acc,
                             all_fail=all(l["label"] != "ok" for l in labs),
                             level=rec.get("level", "")))
        for l in labs:
            rows.append(dict(qid=rec["qid"], level=rec.get("level", ""), b_star=best_b,
                             max_acc=max_acc, **l))
    bm_path = out / "misalloc_labels.jsonl"
    with bm_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    fails = [r for r in rows if r["label"] != "ok"]
    cf = Counter(r["label"] for r in fails)
    LOG.info("逐(题,预算)标签: %s", dict(c))
    LOG.info("失败样本内分布 (E1 的分类目标): %s", dict(cf))
    if cf.get("under", 0) < 30 or cf.get("capability", 0) < 30:
        LOG.warning("under=%d capability=%d —— 任一 <30 则 E1 功效不足, 需加题量或调预算阶梯",
                    cf.get("under", 0), cf.get("capability", 0))
    (out / "label_summary.json").write_text(json.dumps(
        {"all": dict(c), "within_failures": dict(cf), "n_items": len(per_item)},
        indent=2, ensure_ascii=False))


# ============================ 阶段 B: 配对供给采集 (GPU) ============================
def stage_paired(args, out: Path):
    """同一题 × 多个**声明**预算, 各取一次 K=0 hidden (作答前)。
    这是 E2 的数据: demand 恒定、supply 变化, 唯一能解释预测翻转的就是供给表征。"""
    import torch
    labels = bm.read_jsonl(out / "misalloc_labels.jsonl")
    by_q = {}
    for r in labels:
        by_q.setdefault(r["qid"], []).append(r)
    # 优先取有信息量的题: 至少含一个 under 或 over
    qids = [q for q, rs in by_q.items() if any(r["label"] in ("under", "over") for r in rs)]
    qids += [q for q, rs in by_q.items() if q not in qids and any(r["label"] == "capability" for r in rs)]
    qids = qids[:args.max_items]
    curve = {r["qid"]: r for r in bm.read_jsonl(out / "curve.jsonl")}
    LOG.info("配对采集 %d 题 × %d 预算", len(qids), len(bm.BUDGETS))

    eng = bm.Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                    args.quantize_4bit, args.trust_remote_code)
    hdir = out / "paired_hidden"; hdir.mkdir(parents=True, exist_ok=True)
    path = out / "paired_index.jsonl"
    done = {(json.loads(l)["qid"], json.loads(l)["stated_budget"]) for l in path.open()} \
        if (args.resume and path.exists()) else set()
    if path.exists() and not args.resume:
        path.unlink()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for qid in tqdm(qids, desc="paired"):
        rec = curve.get(qid)
        if rec is None:
            continue
        lab_by_b = {r["budget"]: r["label"] for r in by_q[qid]}
        for B in bm.BUDGETS:
            if (qid, B) in done:
                continue
            try:
                prompt = SUPPLY_PROMPT.format(budget=B, q=rec["problem"])
                enc = eng._enc(eng.fmt(prompt))
                with torch.inference_mode():
                    o = eng.model(input_ids=enc.input_ids, output_hidden_states=True)
                h = torch.stack([o.hidden_states[l][0, -1].float().cpu()
                                 for l in range(len(o.hidden_states))]).half()
                torch.save({"qid": qid, "stated_budget": B, "hidden": h},
                           hdir / f"{qid}__b{B}.pt")
                fh.write(json.dumps(dict(qid=qid, stated_budget=B,
                                         label=lab_by_b.get(B, "unknown"),
                                         b_star=by_q[qid][0]["b_star"],
                                         level=rec.get("level", ""))) + "\n")
                fh.flush()
            except Exception:
                LOG.exception("paired fail %s@%d", qid, B)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    fh.close()
    LOG.info("paired -> %s", path)


# ============================ 阶段 C: 分析 ============================
def stage_analyze(args, out: Path):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    res = {}

    # ---------- E1: 条件化于失败, under vs capability ----------
    # 分层 baseline (关键修正: 移除 oracle b*, 只用部署时可得的信息)
    #   B0 observable   : 推理结束时可直接观测 = [用量/预算, 触顶率, log2(预算)]
    #   B1 pred_demand  : 用**同样的 hidden**只预测需求 b̂*(嵌套CV, 不泄漏), 作单特征
    #                     -> 回答核心问题: 错配信号是否超出"需求/难度"信息
    #   B2 obs+pred     : B0 + B1
    #   REF oracle      : 含真实 b*, 仅作上界参考, 明确标注为泄漏, 不用于结论
    idx_path = out / "probe_index.jsonl"
    lab = {(r["qid"], r["budget"]): r for r in bm.read_jsonl(out / "misalloc_labels.jsonl")}
    if idx_path.exists():
        idx = bm.read_jsonl(idx_path)
        X, y, groups, obs, oracle_bstar, dem_target = [], [], [], [], [], []
        for r in idx:
            lr = lab.get((r["qid"], r.get("probe_budget")))
            if lr is None or lr["label"] not in ("under", "capability"):
                continue                      # 只在失败样本内, 且只比这两类
            pt = out / "hidden" / f"{r['qid']}.pt"
            if not pt.exists():
                continue
            h = torch.load(pt, map_location="cpu", weights_only=False)["hidden"]
            if 0 not in h:
                continue
            X.append(h[0].float().numpy())
            y.append(int(lr["label"] == "under"))
            groups.append(r["qid"])
            obs.append([float(lr.get("mean_used", 0)) / max(lr["budget"], 1),
                        float(lr.get("cap_rate", 0)),
                        math.log2(max(lr["budget"], 1))])
            bs = r.get("b_star")
            oracle_bstar.append(math.log2(bs) if bs else math.log2(bm.BUDGETS[-1] * 2))
            # 需求回归的监督目标 (训练侧使用; 测试侧只用预测值)
            dem_target.append(math.log2(bs) if bs else math.log2(bm.BUDGETS[-1] * 2))

        if len(X) >= 30 and len(set(y)) == 2:
            from sklearn.linear_model import Ridge
            X = np.stack(X); y = np.array(y)
            OBS = np.array(obs, float); ORC = np.array(oracle_bstar, float).reshape(-1, 1)
            DEM = np.array(dem_target, float)
            groups = np.array(groups)
            n_layers = X.shape[1]
            n_groups = len(set(groups))
            gk = GroupKFold(n_splits=min(5, n_groups))
            splits = list(gk.split(X, y, groups))

            def cv_auc(F):
                """给定特征矩阵, 用固定的 group 划分做 CV, 返回 AUROC。"""
                pr = np.zeros(len(y), float)
                for tr, te in splits:
                    sc = StandardScaler().fit(F[tr])
                    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
                    clf.fit(sc.transform(F[tr]), y[tr])
                    pr[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]
                return float(roc_auc_score(y, pr)), pr

            # --- hidden probe: 逐层扫, 层选择也在 CV 内部避免过度乐观 ---
            best = {"auroc": -1, "layer": None, "curve": []}
            for l in range(1, n_layers):
                a, _ = cv_auc(X[:, l])
                best["curve"].append((l, round(a, 3)))
                if a > best["auroc"]:
                    best.update(auroc=round(a, 4), layer=l)

            # --- B1: 嵌套CV 的需求预测 b̂* 作单特征 ---
            #     每个外层 fold 内, 仅用训练集拟合 hidden->log2(b*) 的回归
            bhat = np.zeros(len(y), float)
            dl = best["layer"] if best["layer"] is not None else n_layers // 2
            for tr, te in splits:
                sc = StandardScaler().fit(X[tr, dl])
                rg = Ridge(alpha=10.0).fit(sc.transform(X[tr, dl]), DEM[tr])
                bhat[te] = rg.predict(sc.transform(X[te, dl]))
            from scipy.stats import spearmanr as _sp
            rho_dem, _ = _sp(bhat, DEM)

            b0_auc, _ = cv_auc(OBS)
            b1_auc, _ = cv_auc(bhat.reshape(-1, 1))
            b2_auc, _ = cv_auc(np.concatenate([OBS, bhat.reshape(-1, 1)], 1))
            ref_auc, _ = cv_auc(np.concatenate([OBS, ORC], 1))

            # --- 增益的置信区间: 按 group 的 cluster bootstrap ---
            _, p_hidden = cv_auc(X[:, best["layer"]])
            _, p_b2 = cv_auc(np.concatenate([OBS, bhat.reshape(-1, 1)], 1))
            uniq = np.array(sorted(set(groups)))
            rng = np.random.RandomState(0)
            diffs = []
            for _ in range(2000):
                pick = rng.choice(uniq, len(uniq), replace=True)
                m = np.concatenate([np.flatnonzero(groups == g) for g in pick])
                if len(set(y[m])) < 2:
                    continue
                diffs.append(roc_auc_score(y[m], p_hidden[m]) - roc_auc_score(y[m], p_b2[m]))
            ci = ([round(float(np.quantile(diffs, .025)), 4),
                   round(float(np.quantile(diffs, .975)), 4)] if diffs else None)

            res["E1_under_vs_capability"] = {
                "n": int(len(y)), "n_under": int(y.sum()), "n_capability": int((y == 0).sum()),
                "n_items": int(n_groups),
                "hidden_probe_auroc": best["auroc"], "best_layer": best["layer"],
                "baselines": {
                    "B0_observable_only": round(b0_auc, 4),
                    "B1_predicted_demand_only": round(b1_auc, 4),
                    "B2_observable_plus_predicted_demand": round(b2_auc, 4),
                    "REF_oracle_bstar_LEAKY": round(ref_auc, 4),
                },
                "demand_regressor_quality_spearman": round(float(rho_dem), 4),
                "gap_vs_B2": round(best["auroc"] - b2_auc, 4),
                "gap_vs_B2_cluster_bootstrap_95CI": ci,
                "significant": bool(ci and ci[0] > 0),
                "layer_curve": best["curve"],
                "note": ("主对照是 B2(可观测 + 同一hidden预测的需求)。gap_vs_B2 的 95%CI 下界 >0 "
                         "才能主张 hidden 里有超出'需求/难度'的错配信号。REF_oracle 含真实 b*, "
                         "**存在泄漏, 仅作上界参考, 不得用于结论**。"),
            }
        else:
            res["E1_under_vs_capability"] = {"skipped": f"n={len(X)}, classes={set(y) if y else None}"}

    # ---------- E2: 题内供给敏感性 ----------
    pidx_path = out / "paired_index.jsonl"
    if pidx_path.exists():
        pidx = bm.read_jsonl(pidx_path)
        rows = []
        for r in pidx:
            pt = out / "paired_hidden" / f"{r['qid']}__b{r['stated_budget']}.pt"
            if pt.exists() and r["label"] in ("under", "ok", "over"):
                rows.append((r, torch.load(pt, map_location="cpu",
                                           weights_only=False)["hidden"].float().numpy()))
        if len(rows) >= 40:
            X = np.stack([h for _, h in rows])
            y = np.array([int(r["label"] == "under") for r, _ in rows])   # 欠分配 vs 非欠分配
            groups = np.array([r["qid"] for r, _ in rows])
            stated = np.array([r["stated_budget"] for r, _ in rows], float)
            n_layers = X.shape[1]
            gk = GroupKFold(n_splits=min(5, len(set(groups))))
            best = {"auroc": -1, "layer": None, "curve": []}
            preds_best = None
            for l in range(1, n_layers):
                Xl = StandardScaler().fit_transform(X[:, l])
                if len(set(y)) < 2:
                    break
                p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5,
                                                         class_weight="balanced"),
                                      Xl, y, cv=gk, groups=groups, method="predict_proba")[:, 1]
                a = roc_auc_score(y, p)
                best["curve"].append((l, round(a, 3)))
                if a > best["auroc"]:
                    best.update(auroc=round(a, 4), layer=l); preds_best = p
            # 题内翻转: 同一题在不同声明预算下, 预测是否随之变化
            within, spear = [], []
            from scipy.stats import spearmanr
            for q in set(groups):
                m = groups == q
                if m.sum() < 3 or len(set(y[m])) < 2:
                    continue
                within.append(roc_auc_score(y[m], preds_best[m]))
                rho, _ = spearmanr(stated[m], preds_best[m])
                if not np.isnan(rho):
                    spear.append(rho)
            res["E2_within_item_supply_sensitivity"] = {
                "n_obs": int(len(y)), "n_items": int(len(set(groups))),
                "cross_item_auroc": best["auroc"], "best_layer": best["layer"],
                "n_items_evaluable": len(within),
                "within_item_auroc_mean": round(float(np.mean(within)), 4) if within else None,
                "within_item_auroc_std": round(float(np.std(within)), 4) if within else None,
                "pred_vs_stated_budget_spearman_mean": round(float(np.mean(spear)), 4) if spear else None,
                "layer_curve": best["curve"],
                "note": ("题内 AUROC 显著 >0.5 且 pred-vs-stated 相关为负 = 表征读到了声明预算并做了"
                         "供给-需求比较; 题内≈0.5 = 只读需求, 'misallocation' 主张不成立"),
            }
            # 纯供给基线: 只用声明预算能达到多少 (必须被 hidden 超过才有意义)
            sp = cross_val_predict(LogisticRegression(max_iter=1000, class_weight="balanced"),
                                   StandardScaler().fit_transform(stated.reshape(-1, 1)), y,
                                   cv=gk, groups=groups, method="predict_proba")[:, 1]
            res["E2_within_item_supply_sensitivity"]["stated_budget_only_auroc"] = round(
                float(roc_auc_score(y, sp)), 4)
        else:
            res["E2_within_item_supply_sensitivity"] = {"skipped": f"n={len(rows)}"}

    (out / "misallocation_analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])
    LOG.info("-> %s", out / "misallocation_analysis.json")


def build_parser():
    p = argparse.ArgumentParser(description="预算错配的 hidden-state 诊断")
    p.add_argument("--stage", choices=["label", "paired", "analyze", "all"], default="label")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--acc-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    stages = ["label", "paired", "analyze"] if a.stage == "all" else [a.stage]
    for s in stages:
        LOG.info("=== stage: %s ===", s)
        {"label": stage_label, "paired": stage_paired, "analyze": stage_analyze}[s](a, out)


if __name__ == "__main__":
    main()
