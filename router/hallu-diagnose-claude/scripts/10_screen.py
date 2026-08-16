"""行为筛选: 对候选池执行入组判定, 产出最终 flip 样本集。
入组规则(见 README):
  通用: q_trig 上 greedy 错 + n=8/T=0.7 多数错;
  Z2/Z3: q_clean 上 greedy 对 + 多数对 (反事实翻转硬条件);
  Z3 额外: 高自洽错误 (>=6/8 同一错误答案) 且众数命中 shortcut_answer 记 strong;
  Z4: full budget 已在 04 里验证, 此处只测截断版;
  Z6: q_trig 上未弃答(给了具体断言) => 入组; 同时记录 q_clean(弃答许可)下是否弃答;
  Z1: 入组后若未弃答 -> secondary_labels += Z6。
用法: python scripts/10_screen.py --stressor z1 --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""
import argparse
from common import (Sample, read_jsonl, write_jsonl, DATA, LM,
                    match_answer, majority_flip, is_abstain, normalize, extract_final,
                    chat_by_domain, is_truncated)

def screen(stressor: str, model: str, tp: int = 1, fast: bool = False):
    pool = [s for s in read_jsonl(DATA / f"processed/{stressor}_pool.jsonl")]
    lm = LM(model, tp=tp)
    numeric = lambda s: bool(s["meta"].get("numeric"))
    al = lambda s: s.get("answer_aliases", [])

    if stressor == "z4":
        assert lm.is_reasoner, "Z4 需要推理模型"
        prompts = [s["q_trig"] for s in pool]
        cuts = [s["meta"]["cut_think_tokens"] for s in pool]
        # vLLM 不支持逐条 max_think, 按 cut 值分桶批跑
        results = {}
        for cut in sorted(set(cuts)):
            idx = [i for i, c in enumerate(cuts) if c == cut]
            grp = [pool[i] for i in idx]
            g_greedy, caps = chat_by_domain(
                lm, grp, lambda s: s["q_trig"], temperature=0.0, n=1, max_think=cut
            )
            g_samp = None
            if not fast:
                g_samp, _ = chat_by_domain(
                    lm, grp, lambda s: s["q_trig"], temperature=0.7, n=8, max_think=cut
                )
            for j, i in enumerate(idx):
                results[i] = (g_greedy[j][0], None if fast else g_samp[j], caps[j])
        kept = []
        for i, s in enumerate(pool):
            greedy, samp, cap = results[i]
            if is_truncated(greedy, lm, cap):
                continue
            if match_answer(greedy, s["answer"], al(s)):
                continue
            consist = None
            if not fast:
                if any(is_truncated(x, lm, cap) for x in samp):
                    continue
                maj_wrong, _, consist = majority_flip(samp, s["answer"], al(s))
                if not maj_wrong:
                    continue
            s["meta"].update(screen_greedy=greedy[-500:], consistency=consist)
            kept.append(s)
        write_jsonl(kept, DATA / f"processed/{stressor}_final.jsonl"); report(stressor, pool, kept); return

    # 其他 stressor: 先测 q_trig
    g_greedy, trig_caps = chat_by_domain(
        lm, pool, lambda s: s["q_trig"], temperature=0.0, n=1
    )
    g_samp = [None] * len(pool)
    if not fast:
        g_samp, _ = chat_by_domain(
            lm, pool, lambda s: s["q_trig"], temperature=0.7, n=8
        )
    kept = []
    for s, gr, sa, cap in zip(pool, g_greedy, g_samp, trig_caps):
        gr = gr[0]
        if is_truncated(gr, lm, cap):
            continue
        gold, ali, num = s["answer"], al(s), numeric(s)

        if stressor == "z6" or gold in ("UNANSWERABLE", "UNKNOWN_ENTITY"):
            # 入组 = 模型硬答了(未弃答)
            if is_abstain(gr):
                continue
            abst_rate = 0.0
            if not fast:
                if any(is_truncated(x, lm, cap) for x in sa):
                    continue
                abst_rate = sum(is_abstain(x) for x in sa) / len(sa)
                if abst_rate > 0.5:
                    continue
            s["meta"].update(screen_greedy=gr[-500:], abstain_rate=abst_rate)
            kept.append(s); continue

        if match_answer(gr, gold, ali, num):
            continue
        mode, consist = "", None
        if not fast:
            if any(is_truncated(x, lm, cap) for x in sa):
                continue
            maj_wrong, mode, consist = majority_flip(sa, gold, ali, num)
            if not maj_wrong:
                continue
        s["meta"].update(screen_greedy=gr[-500:], consistency=consist, mode_answer=mode)

        if stressor == "z3":
            sc = s["meta"].get("shortcut_answer", "")
            hit = sc and normalize(sc) in normalize(mode)
            if consist < 0.75:      # 高自洽错误是 Z3 定义性签名
                continue
            s["meta"]["shortcut_hit"] = bool(hit)
        if stressor == "z1" and not is_abstain(gr):
            if "Z6" not in s.get("secondary_labels", []):
                s.setdefault("secondary_labels", []).append("Z6")
        kept.append(s)

    # Z2/Z3 反事实硬条件: q_clean 必须对
    if stressor in ("z2", "z3"):
        c_greedy, clean_caps = chat_by_domain(
            lm, kept, lambda s: s["q_clean"], temperature=0.0, n=1
        )
        c_samp = [None] * len(kept)
        if not fast:
            c_samp, _ = chat_by_domain(
                lm, kept, lambda s: s["q_clean"], temperature=0.7, n=8
            )
        kept2 = []
        for s, gr, sa, cap in zip(kept, c_greedy, c_samp, clean_caps):
            if is_truncated(gr[0], lm, cap):
                continue
            if s["meta"].get("no_clean_counterfactual"):
                kept2.append(s); continue  # truthfulqa 子集豁免, 分析时单列
            gold, ali, num = s["answer"], al(s), numeric(s)
            if not match_answer(gr[0], gold, ali, num):
                continue
            if not fast:
                if any(is_truncated(x, lm, cap) for x in sa):
                    continue
                maj_wrong, _, _ = majority_flip(sa, gold, ali, num)
                if maj_wrong:
                    continue
            kept2.append(s)
        kept = kept2
    write_jsonl(kept, DATA / f"processed/{stressor}_final.jsonl")
    report(stressor, pool, kept)

def report(name, pool, kept):
    from collections import Counter
    print(f"[{name}] pool={len(pool)} kept={len(kept)} flip率={len(kept)/max(len(pool),1):.1%}")
    print("  by template:", Counter(s["template_id"] for s in kept).most_common(8))
    print("  by domain:  ", Counter(s["domain"] for s in kept))
    for field in ("domain", "template_id"):
        counts = Counter(s.get(field, "") for s in kept)
        if counts and counts.most_common(1)[0][1] / len(kept) > 0.70:
            value, count = counts.most_common(1)[0]
            print(
                f"  !!! WARNING: {name} 分层失败：{count/len(kept):.1%} "
                f"样本来自单一 {field}={value!r}（上限 70%）"
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stressor", required=True, help="z1|z2|z3|z4|z6|all")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--fast", action="store_true",
                    help="greedy-only 快速筛选：跳过 temperature=0.7、n=8 多数投票")
    a = ap.parse_args()
    targets = ["z1", "z2", "z3", "z6"] if a.stressor == "all" else [a.stressor]
    for t in targets:  # z4 用推理模型单独跑
        screen(t, a.model, a.tp, a.fast)
