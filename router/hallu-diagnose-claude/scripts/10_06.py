#!/usr/bin/env python3
"""06 real-life 候选的独立行为筛选器；不读写原 10_screen.py 的路径。

输入:
  data/processed/real_life/candidates/{z1,z2,z4,z6}_pool.jsonl

输出:
  data/processed/real_life_screened/{z1,z2,z4,z6}_final.jsonl

每类都要求与其因果开关相符:
  Z1: 无参考稳定错误，加入 gold reference 后稳定正确。
  Z2: trig 稳定错误，去干扰/反捷径 clean 稳定正确。
  Z4: full budget 稳定正确，按实测 thinking 用量截短后稳定错误。
  Z6: 信息不足版本稳定硬答，加入弃答许可后稳定弃答。

筛选结果还不是正式 zi_final；必须再经过 06_build.py resolve 的跨 Zi 互斥。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    DATA,
    LM,
    extract_final,
    is_abstain,
    is_truncated,
    match_answer,
    normalize,
    read_jsonl,
    write_jsonl,
)


DEFAULT_CANDIDATE_DIR = DATA / "processed/real_life/candidates"
DEFAULT_SCREENED_DIR = DATA / "processed/real_life_screened"
STRESSORS = ("z1", "z2", "z6", "z4")


def is_choice(sample):
    return sample.get("meta", {}).get("choice_index") in (1, 2)


def is_correct(response, sample):
    """二选一优先解析 final 中最后一个 Option1/2，避免推理正文提及选项造成误判。"""
    if sample["answer"] == "UNANSWERABLE":
        return is_abstain(response)
    if is_choice(sample):
        final = extract_final(response)
        choices = re.findall(r"\boption\s*([12])\b", final, re.I)
        if not choices:
            choices = re.findall(r"(?:^|\s)([12])(?:\s|$)", final)
        if choices:
            return int(choices[-1]) == int(sample["meta"]["choice_index"])
        gold_text = normalize(sample.get("answer_aliases", [""])[0])
        return bool(gold_text) and gold_text in normalize(final)
    return match_answer(
        response,
        sample["answer"],
        sample.get("answer_aliases", []),
        bool(sample.get("meta", {}).get("numeric")),
    )


def wrong_majority(responses, sample):
    return sum(not is_correct(response, sample) for response in responses) > len(responses) / 2


def correct_majority(responses, sample):
    return sum(is_correct(response, sample) for response in responses) > len(responses) / 2


def abstain_majority(responses):
    return sum(is_abstain(response) for response in responses) > len(responses) / 2


def run_generation(lm, samples, prompt_fn, temperature, n, max_think=None,
                   max_tokens=512):
    prompts = [prompt_fn(sample) for sample in samples]
    outputs = lm.chat(
        prompts,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens,
        max_think=max_think,
    )
    return outputs, [max_tokens] * len(samples)


def valid_outputs(outputs, lm, cap):
    return not any(is_truncated(response, lm, cap) for response in outputs)


def initial_wrong(lm, samples, fast, n_samples):
    greedy, caps = run_generation(lm, samples, lambda s: s["q_trig"], 0.0, 1)
    sampled = [None] * len(samples)
    if not fast:
        sampled, _ = run_generation(
            lm, samples, lambda s: s["q_trig"], 0.7, n_samples
        )
    passed = []
    for i, sample in enumerate(samples):
        g = greedy[i][0]
        if not valid_outputs([g], lm, caps[i]) or is_correct(g, sample):
            continue
        if not fast:
            if not valid_outputs(sampled[i], lm, caps[i]):
                continue
            if not wrong_majority(sampled[i], sample):
                continue
        passed.append((i, sample, g, None if fast else sampled[i]))
    return passed


def screen_with_rescue(lm, samples, stressor, fast, n_samples):
    failed = initial_wrong(lm, samples, fast, n_samples)
    if not failed:
        return []
    subset = [item[1] for item in failed]
    if stressor == "z1":
        def rescue_prompt(sample):
            passage = sample["meta"].get("gold_passage", "")
            return f"Reference: {passage}\n\n{sample['q_trig']}"
    elif stressor == "z2":
        rescue_prompt = lambda sample: sample["q_clean"]
    else:
        raise ValueError(stressor)

    greedy, caps = run_generation(lm, subset, rescue_prompt, 0.0, 1)
    sampled = [None] * len(subset)
    if not fast:
        sampled, _ = run_generation(lm, subset, rescue_prompt, 0.7, n_samples)

    kept = []
    for j, (_, sample, trig_greedy, trig_samples) in enumerate(failed):
        rescue_greedy = greedy[j][0]
        if not valid_outputs([rescue_greedy], lm, caps[j]):
            continue
        if not is_correct(rescue_greedy, sample):
            continue
        if not fast:
            if not valid_outputs(sampled[j], lm, caps[j]):
                continue
            if not correct_majority(sampled[j], sample):
                continue
        sample["meta"].update({
            "screen_protocol": f"10_06_{stressor}",
            "screen_trig_greedy": trig_greedy[-500:],
            "screen_rescue_greedy": rescue_greedy[-500:],
            "screen_n_samples": 1 if fast else n_samples,
        })
        kept.append(sample)
    return kept


def screen_z6(lm, samples, fast, n_samples):
    trig_greedy, caps = run_generation(lm, samples, lambda s: s["q_trig"], 0.0, 1)
    trig_sampled = [None] * len(samples)
    if not fast:
        trig_sampled, _ = run_generation(
            lm, samples, lambda s: s["q_trig"], 0.7, n_samples
        )
    hard_answered = []
    for i, sample in enumerate(samples):
        g = trig_greedy[i][0]
        if not valid_outputs([g], lm, caps[i]) or is_abstain(g):
            continue
        if not fast:
            if not valid_outputs(trig_sampled[i], lm, caps[i]):
                continue
            if abstain_majority(trig_sampled[i]):
                continue
        hard_answered.append((i, sample, g))

    if not hard_answered:
        return []
    subset = [item[1] for item in hard_answered]
    clean_greedy, clean_caps = run_generation(
        lm, subset, lambda s: s["q_clean"], 0.0, 1
    )
    clean_sampled = [None] * len(subset)
    if not fast:
        clean_sampled, _ = run_generation(
            lm, subset, lambda s: s["q_clean"], 0.7, n_samples
        )

    kept = []
    for j, (_, sample, trig_response) in enumerate(hard_answered):
        clean_response = clean_greedy[j][0]
        if not valid_outputs([clean_response], lm, clean_caps[j]):
            continue
        if not is_abstain(clean_response):
            continue
        if not fast:
            if not valid_outputs(clean_sampled[j], lm, clean_caps[j]):
                continue
            if not abstain_majority(clean_sampled[j]):
                continue
        sample["meta"].update({
            "screen_protocol": "10_06_z6",
            "screen_trig_greedy": trig_response[-500:],
            "screen_clean_greedy": clean_response[-500:],
            "screen_n_samples": 1 if fast else n_samples,
        })
        kept.append(sample)
    return kept


def thinking_tokens(response, tok):
    thinking = response.split("</think>", 1)[0]
    return len(tok.encode(thinking, add_special_tokens=False))


def screen_z4(lm, samples, fast, n_samples, full_budget, cut_ratio,
              min_think_tokens):
    if not lm.is_reasoner:
        raise ValueError("Z4 real-life 预算筛选需要 R1 类 reasoner")
    full_greedy, caps = run_generation(
        lm, samples, lambda s: s["q_trig"], 0.0, 1, max_think=full_budget
    )
    full_sampled = [None] * len(samples)
    if not fast:
        full_sampled, _ = run_generation(
            lm, samples, lambda s: s["q_trig"], 0.6, n_samples,
            max_think=full_budget,
        )

    calibrated = []
    for i, sample in enumerate(samples):
        responses = [full_greedy[i][0]] if fast else full_sampled[i]
        if not valid_outputs(responses, lm, caps[i]):
            continue
        if not is_correct(full_greedy[i][0], sample):
            continue
        if not fast and sum(is_correct(x, sample) for x in responses) < n_samples - 1:
            continue
        lengths = [thinking_tokens(x, lm.tok) for x in responses]
        avg_think = float(np.mean(lengths))
        if avg_think < min_think_tokens:
            continue
        cut = max(16, int(avg_think * cut_ratio))
        calibrated.append((sample, avg_think, cut, full_greedy[i][0]))

    kept = []
    by_cut = {}
    for item in calibrated:
        by_cut.setdefault(item[2], []).append(item)
    for cut, items in sorted(by_cut.items()):
        subset = [item[0] for item in items]
        cut_greedy, cut_caps = run_generation(
            lm, subset, lambda s: s["q_trig"], 0.0, 1, max_think=cut
        )
        cut_sampled = [None] * len(subset)
        if not fast:
            cut_sampled, _ = run_generation(
                lm, subset, lambda s: s["q_trig"], 0.7, n_samples,
                max_think=cut,
            )
        for j, (sample, avg_think, _, full_response) in enumerate(items):
            cut_response = cut_greedy[j][0]
            if not valid_outputs([cut_response], lm, cut_caps[j]):
                continue
            if is_correct(cut_response, sample):
                continue
            if not fast:
                if not valid_outputs(cut_sampled[j], lm, cut_caps[j]):
                    continue
                if not wrong_majority(cut_sampled[j], sample):
                    continue
            sample["meta"].update({
                "screen_protocol": "10_06_z4",
                "avg_think_tokens": avg_think,
                "full_think_tokens": full_budget,
                "cut_think_tokens": cut,
                "cut_ratio": cut_ratio,
                "screen_full_greedy": full_response[-500:],
                "screen_cut_greedy": cut_response[-500:],
                "screen_n_samples": 1 if fast else n_samples,
            })
            sample["meta"].pop("needs_budget_calibration", None)
            kept.append(sample)
    return kept


def load_candidates(candidate_dir, stressor, limit):
    path = Path(candidate_dir) / f"{stressor}_pool.jsonl"
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    expected = stressor.upper()
    for row in rows:
        if row.get("stressor") != expected:
            raise ValueError(f"{path}: {row.get('sid')} 的 stressor 不是 {expected}")
        if row.get("domain") != "real-life":
            raise ValueError(f"{path}: {row.get('sid')} 的 domain 不是 real-life")
        if not row.get("meta", {}).get("base_id"):
            raise ValueError(f"{path}: {row.get('sid')} 缺少 meta.base_id")
    return rows


def report(stressor, pool, kept):
    print(f"[{stressor}] pool={len(pool)} kept={len(kept)} rate={len(kept)/max(1,len(pool)):.1%}")
    print("  source:", Counter(row["meta"]["source"] for row in kept))
    print("  template:", Counter(row["template_id"] for row in kept))


def shard_output_path(screened_dir, stressor, shard_index, num_shards):
    return Path(screened_dir) / (
        f"{stressor}_final.shard-{shard_index:03d}-of-{num_shards:03d}.jsonl"
    )


def merge_shards(args):
    """校验所有分片齐全后原子汇总，避免并行任务覆盖同一 final。"""
    if args.num_shards < 2:
        raise ValueError("--merge-shards 要求 --num-shards >= 2")
    targets = STRESSORS if args.stressor == "all" else (args.stressor,)
    for z in targets:
        rows, seen = [], set()
        for index in range(args.num_shards):
            path = shard_output_path(args.screened_dir, z, index, args.num_shards)
            if not path.exists():
                raise FileNotFoundError(f"缺少分片: {path}")
            for row in read_jsonl(path):
                sid = row.get("sid")
                if not sid or sid in seen:
                    raise ValueError(f"{path}: sid 缺失或重复: {sid!r}")
                seen.add(sid)
                rows.append(row)
        out = Path(args.screened_dir) / f"{z}_final.jsonl"
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
        print(f"[merge-shards] {z}: {len(rows)} -> {out}")


def main(args):
    if args.num_shards < 1:
        raise ValueError("--num-shards 必须 >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index 必须满足 0 <= index < num_shards")
    if args.merge_shards:
        merge_shards(args)
        return
    targets = STRESSORS if args.stressor == "all" else (args.stressor,)
    if args.skip_z4:
        targets = tuple(z for z in targets if z != "z4")
    pools = {
        z: load_candidates(args.candidate_dir, z, args.limit)
        for z in targets
    }
    if args.num_shards > 1:
        pools = {
            z: pool[args.shard_index::args.num_shards]
            for z, pool in pools.items()
        }
        for z, pool in pools.items():
            print(
                f"[{z}] shard={args.shard_index}/{args.num_shards} "
                f"candidates={len(pool)}"
            )
    lm = LM(args.model, tp=args.tp)
    for z in targets:
        lm.batch_size = args.z4_batch_size if z == "z4" else args.batch_size
        print(f"[{z}] inference batch_size={lm.batch_size}")
        pool = pools[z]
        if z in ("z1", "z2"):
            kept = screen_with_rescue(lm, pool, z, args.fast, args.n_samples)
        elif z == "z4":
            kept = screen_z4(
                lm, pool, args.fast, args.n_samples,
                args.z4_full_budget, args.z4_cut_ratio,
                args.z4_min_think_tokens,
            )
        elif z == "z6":
            kept = screen_z6(lm, pool, args.fast, args.n_samples)
        else:
            raise ValueError(z)
        out = (shard_output_path(args.screened_dir, z, args.shard_index, args.num_shards)
               if args.num_shards > 1
               else Path(args.screened_dir) / f"{z}_final.jsonl")
        write_jsonl(kept, out)
        report(z, pool, kept)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stressor", required=True,
                    choices=[*STRESSORS, "all"])
    ap.add_argument("--model",
                    default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    ap.add_argument("--screened-dir", default=str(DEFAULT_SCREENED_DIR))
    ap.add_argument("--limit", type=int, default=0, help="每类候选上限；0=全部")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="候选池分片总数；默认 1 不分片")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="当前分片编号，范围 0..num_shards-1")
    ap.add_argument("--merge-shards", action="store_true",
                    help="不加载模型；校验并汇总 --num-shards 指定的全部分片")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Z1/Z2/Z6 推理 batch size")
    ap.add_argument("--z4-batch-size", type=int, default=2,
                    help="Z4 长 thinking 推理 batch size")
    ap.add_argument("--fast", action="store_true",
                    help="仅 greedy 的烟雾测试，不用于正式入组")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--z4-full-budget", type=int, default=4096)
    ap.add_argument("--z4-cut-ratio", type=float, default=0.3)
    ap.add_argument("--z4-min-think-tokens", type=int, default=64)
    ap.add_argument("--skip-z4", action="store_true",
                    help="配合 --stressor all 使用，在当前进程跳过 Z4")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
