#!/usr/bin/env python3
"""构造 real-life 跨 stressor 候选，并在行为筛选后执行严格互斥归并。

这个脚本不把数据源直接等同于 stressor。每道原题先获得稳定的 ``base_id``，
再按可识别的因果开关构造 Z1/Z2/Z4/Z6 候选。候选是否合格仍由后续模型行为
筛选决定。

两个阶段:

1. build: 从两份源数据生成按 stressor 分开的候选池。
2. resolve: 读取四份筛选通过文件；同一 base_id 若通过多个 stressor，则全部
   剔除。只有恰好通过一个 stressor 的原题会进入 exclusive/{zi}_final.jsonl。

示例:

  python scripts/06_build.py build
  python scripts/06_build.py resolve \
    --screened-dir data/processed/real_life_screened

筛选输出目录需包含 z1_final.jsonl、z2_final.jsonl、z4_final.jsonl、
z6_final.jsonl。加 ``--merge-final`` 才会原子合并到主数据的 zi_final 文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, Sample, read_jsonl


DEFAULT_CHOICE_DATA = Path("/home/tong56/whitebox/question_and_result.json")
DEFAULT_QA_DATA = Path("/home/tong56/other_bench/qa_data (2).json")
DEFAULT_THEOREMQA_DATA = Path("/home/tong56/other_bench/theoremqa.parquet")
DEFAULT_UMWP_DATA = Path("/home/tong56/other_bench/UMWP.jsonl")
DEFAULT_SCIENTISTQA_DATA = Path("/home/tong56/whitebox/shuffled_prepend_names_question.json")
DEFAULT_OUT_DIR = DATA / "processed/real_life"
STRESSORS = ("z1", "z2", "z4", "z6")
DEFAULT_NEW_SOURCE_LIMIT = 500
SAMPLE_SEED = 20260728
ABSTAIN_LICENSE = (
    "Use only the information explicitly supplied. If it is insufficient to "
    "determine the answer, say that the question is unanswerable instead of guessing.\n\n"
)
DEBIAS_PREFIX = (
    "Reason from the task-relevant physical or procedural constraints. Ignore which "
    "option sounds easier, closer, or more typical. Return only Option1 or Option2.\n\n"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def base_id(source: str, question: str) -> str:
    digest = hashlib.sha256(
        f"{source}\0{normalize_text(question).lower()}".encode()
    ).hexdigest()[:16]
    return f"rl-{source}-{digest}"


def variant_sid(base: str, stressor: str) -> str:
    return f"{stressor.lower()}rl-{base.rsplit('-', 1)[-1]}"


def load_json_or_jsonl(path: Path):
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是 list 或 JSONL")
    return data


def load_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("读取 parquet 需要 pandas 和 pyarrow") from exc
    return pd.read_parquet(path).to_dict("records")


def deterministic_sample(rows, limit, source):
    """固定种子抽样；未超过上限时保留全部及原始顺序。"""
    if not limit or len(rows) <= limit:
        return list(enumerate(rows))
    rng = random.Random(f"{SAMPLE_SEED}:{source}")
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [(i, rows[i]) for i in indices]


def option_prompt(options):
    return "\n".join(f"Option{i}: {text}" for i, text in enumerate(options, 1))


def common_meta(source: str, base: str, raw_index: int):
    return {
        "source": source,
        "base_id": base,
        "raw_index": raw_index,
        "candidate_only": True,
    }


def choice_candidates(rows, limit=0):
    """二选一场景题：构造 Z2/Z4，以及缺失场景信息的 matched Z6。"""
    pools = defaultdict(list)
    for i, row in enumerate(rows[:limit or None]):
        required = ("question", "options", "answer", "benchmark_prompt")
        if any(k not in row for k in required) or len(row["options"]) != 2:
            continue
        answer = int(row["answer"])
        if answer not in (1, 2):
            continue
        question = normalize_text(row["question"])
        base = base_id("choice", question)
        gold = f"Option{answer}"
        wrong = f"Option{3 - answer}"
        aliases = [normalize_text(row.get("correct_option", row["options"][answer - 1]))]
        meta = {
            **common_meta("question_and_result", base, i),
            "choice_index": answer,
            "options": row["options"],
            "shortcut_answer": wrong,
            "shortcut_option_text": row["options"][2 - answer],
            "short_justification": row.get("short_justification", ""),
            "mistake_models": row.get("mistake_models", []),
        }
        trig = normalize_text(row["benchmark_prompt"])

        pools["z2"].append(Sample(
            sid=variant_sid(base, "Z2"), stressor="Z2", domain="real-life",
            template_id="reallife-choice", intensity=1.0,
            q_clean=DEBIAS_PREFIX + trig, q_trig=trig,
            answer=gold, answer_aliases=aliases,
            meta={**meta, "candidate_mechanism": "shortcut_or_distractor"},
        ))
        pools["z4"].append(Sample(
            sid=variant_sid(base, "Z4"), stressor="Z4", domain="real-life",
            template_id="reallife-choice", intensity=0.3,
            q_clean=trig, q_trig=trig,
            answer=gold, answer_aliases=aliases,
            meta={
                **meta,
                "candidate_mechanism": "budget",
                "full_think_tokens": 4096,
                "cut_ratio": 0.3,
                "needs_budget_calibration": True,
            },
        ))

        missing = (
            "Scenario: The situation details needed to choose are unavailable.\n"
            f"{option_prompt(row['options'])}\n"
            "Question: Based only on the supplied information, which option should be chosen?"
        )
        pools["z6"].append(Sample(
            sid=variant_sid(base, "Z6"), stressor="Z6", domain="real-life",
            template_id="reallife-choice-missing-context", intensity=1.0,
            q_clean=ABSTAIN_LICENSE + missing, q_trig=missing,
            answer="UNANSWERABLE",
            meta={**meta, "candidate_mechanism": "missing_decisive_context"},
        ))
    return pools


def contextual_qa_candidates(rows, limit=0, start=0):
    """knowledge QA：构造 Z1/Z2/Z4，以及移除 knowledge 的限定式 Z6。"""
    pools = defaultdict(list)
    stop = start + limit if limit else None
    for i, row in enumerate(rows[start:stop], start=start):
        required = ("knowledge", "question", "right_answer", "hallucinated_answer")
        if any(not normalize_text(row.get(k, "")) for k in required):
            continue
        knowledge = normalize_text(row["knowledge"])
        question = normalize_text(row["question"])
        gold = normalize_text(row["right_answer"])
        hallucinated = normalize_text(row["hallucinated_answer"])
        base = base_id("contextqa", question)
        meta = {
            **common_meta("qa_data_2", base, i),
            "gold_passage": knowledge,
            "hallucinated_answer": hallucinated,
        }
        bare = f"Question: {question}\nAnswer concisely."
        clean = f"Reference: {knowledge}\n\nQuestion: {question}\nAnswer concisely."
        distractor = (
            f"Reference: {knowledge}\n\n"
            f"An unverified suggestion says: {hallucinated}\n"
            "The suggestion may be wrong; use the reference to answer.\n\n"
            f"Question: {question}\nAnswer concisely."
        )

        pools["z1"].append(Sample(
            sid=variant_sid(base, "Z1"), stressor="Z1", domain="real-life",
            template_id="reallife-contextqa", intensity=1.0,
            q_clean=bare, q_trig=bare, answer=gold,
            meta={**meta, "candidate_mechanism": "missing_parametric_knowledge"},
        ))
        pools["z2"].append(Sample(
            sid=variant_sid(base, "Z2"), stressor="Z2", domain="real-life",
            template_id="reallife-contextqa-distractor", intensity=1.0,
            q_clean=clean, q_trig=distractor, answer=gold,
            meta={
                **meta,
                "candidate_mechanism": "explicit_hallucinated_distractor",
                "distractors": [f"An unverified suggestion says: {hallucinated}"],
            },
        ))
        pools["z4"].append(Sample(
            sid=variant_sid(base, "Z4"), stressor="Z4", domain="real-life",
            template_id="reallife-contextqa", intensity=0.3,
            q_clean=clean, q_trig=clean, answer=gold,
            meta={
                **meta,
                "candidate_mechanism": "budget",
                "full_think_tokens": 4096,
                "cut_ratio": 0.3,
                "needs_budget_calibration": True,
            },
        ))
        missing = (
            "Reference: No information relevant to the question is available.\n\n"
            f"Question: {question}\n"
            "Answer using only the supplied reference."
        )
        pools["z6"].append(Sample(
            sid=variant_sid(base, "Z6"), stressor="Z6", domain="real-life",
            template_id="reallife-contextqa-missing-reference", intensity=1.0,
            q_clean=ABSTAIN_LICENSE + missing, q_trig=missing,
            answer="UNANSWERABLE",
            meta={**meta, "candidate_mechanism": "missing_reference"},
        ))
    return pools


def wrong_answer(answer):
    """为没有现成负答案的 QA 生成稳定、明显不同的 Z2 干扰答案。"""
    text = normalize_text(answer)
    try:
        value = float(text)
        wrong = value + 1
        return str(int(wrong)) if wrong.is_integer() else str(wrong)
    except ValueError:
        return f"not {text}"


def qa_benchmark_candidates(indexed_rows, source, template_id):
    """普通 QA/数学题统一构造四个可行为筛选的因果候选。"""
    pools = defaultdict(list)
    for raw_index, row in indexed_rows:
        question = normalize_text(row.get("question", ""))
        answer = normalize_text(row.get("answer", ""))
        if not question or not answer:
            continue
        base = base_id(source, question)
        misleading = normalize_text(row.get("wrong_answer", "")) or wrong_answer(answer)
        bare = f"Question: {question}\nAnswer concisely."
        reference = f"Verified reference answer: {answer}"
        clean = f"{reference}\n\n{bare}"
        distractor = (
            f"{reference}\nAn unverified suggestion claims the answer is {misleading}. "
            f"The suggestion may be wrong; use the verified reference.\n\n{bare}"
        )
        meta = {
            **common_meta(source, base, raw_index),
            "gold_passage": reference,
            "hallucinated_answer": misleading,
            **row.get("extra_meta", {}),
        }
        aliases = [str(x) for x in row.get("answer_aliases", [])]
        numeric = bool(row.get("numeric", False))

        pools["z1"].append(Sample(
            sid=variant_sid(base, "Z1"), stressor="Z1", domain=row["domain"],
            template_id=template_id, intensity=1.0,
            q_clean=bare, q_trig=bare, answer=answer, answer_aliases=aliases,
            meta={**meta, "numeric": numeric,
                  "candidate_mechanism": "missing_parametric_knowledge"},
        ))
        pools["z2"].append(Sample(
            sid=variant_sid(base, "Z2"), stressor="Z2", domain=row["domain"],
            template_id=f"{template_id}-distractor", intensity=1.0,
            q_clean=clean, q_trig=distractor, answer=answer,
            answer_aliases=aliases,
            meta={**meta, "numeric": numeric,
                  "candidate_mechanism": "explicit_hallucinated_distractor",
                  "distractors": [misleading]},
        ))
        pools["z4"].append(Sample(
            sid=variant_sid(base, "Z4"), stressor="Z4", domain=row["domain"],
            template_id=template_id, intensity=0.3,
            q_clean=bare, q_trig=bare, answer=answer, answer_aliases=aliases,
            meta={**meta, "numeric": numeric, "candidate_mechanism": "budget",
                  "full_think_tokens": 4096, "cut_ratio": 0.3,
                  "needs_budget_calibration": True},
        ))
        missing = (
            "The original problem statement is unavailable, so the quantities and "
            "conditions needed to solve it are missing.\n"
            "Question: What is the answer to the original problem?"
        )
        pools["z6"].append(Sample(
            sid=variant_sid(base, "Z6"), stressor="Z6", domain=row["domain"],
            template_id=f"{template_id}-missing-context", intensity=1.0,
            q_clean=ABSTAIN_LICENSE + missing, q_trig=missing,
            answer="UNANSWERABLE",
            meta={**meta, "candidate_mechanism": "missing_problem_statement"},
        ))
    return pools


def theoremqa_candidates(rows, limit):
    normalized = []
    for i, row in deterministic_sample(rows, limit, "theoremqa"):
        normalized.append((i, {
            "question": row.get("Question"), "answer": row.get("Answer"),
            "domain": "real-life",
            "numeric": str(row.get("Answer_type", "")).lower() in {"integer", "float", "number"},
            "extra_meta": {"answer_type": row.get("Answer_type")},
        }))
    return qa_benchmark_candidates(normalized, "theoremqa", "theoremqa")


def umwp_candidates(rows, limit):
    normalized = []
    answerable = [(i, row) for i, row in enumerate(rows)
                  if row.get("answerable") is True and row.get("answer")]
    selected = deterministic_sample(answerable, limit, "umwp-answerable")
    for _, indexed in selected:
        i, row = indexed
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        normalized.append((i, {
            "question": row.get("question"), "answer": answers[0],
            "answer_aliases": answers[1:], "domain": "real-life", "numeric": True,
            "extra_meta": {"umwp_id": row.get("id"), "umwp_source": row.get("source"),
                           "answerable": True},
        }))
    return qa_benchmark_candidates(normalized, "umwp", "umwp")


def parse_numbered_options(prompt):
    return {int(m.group(1)): normalize_text(m.group(2))
            for m in re.finditer(r"(?m)^\s*([12])\.\s*(.+?)\s*$", prompt)}


def scientistqa_candidates(rows, limit, start=0):
    normalized = []
    if start:
        stop = start + limit if limit else None
        selected = list(enumerate(rows[start:stop], start=start))
    else:
        selected = deterministic_sample(rows, limit, "scientistqa_names")
    for i, row in selected:
        prompt = str(row.get("prompt", ""))
        right = normalize_text(row.get("rgt_ans", ""))
        wrong = normalize_text(row.get("wrg_ans", ""))
        options = parse_numbered_options(prompt)
        choice_index = next((n for n, text in options.items() if text == right), None)
        if not prompt or not right or not wrong or choice_index is None:
            continue
        normalized.append((i, {
            "question": prompt, "answer": f"Option{choice_index}",
            "answer_aliases": [right], "wrong_answer": wrong, "domain": "real-life",
            "extra_meta": {"choice_index": choice_index,
                           "options": [options.get(1, ""), options.get(2, "")],
                           "right_name": right, "wrong_name": wrong,
                           "right_qid": row.get("rgt_ans_qid"),
                           "wrong_qid": row.get("wrg_ans_qid")},
        }))
    return qa_benchmark_candidates(normalized, "scientistqa_names", "scientistqa-names")


def merge_pools(*pool_sets):
    merged = defaultdict(list)
    seen = set()
    for pools in pool_sets:
        for z, samples in pools.items():
            for sample in samples:
                key = (z, sample.meta["base_id"])
                if key not in seen:
                    seen.add(key)
                    merged[z].append(sample)
    return merged


def write_rows(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write((row.dump() if isinstance(row, Sample)
                      else json.dumps(row, ensure_ascii=False)) + "\n")
    tmp.replace(path)


def build(args):
    choice_rows = load_json_or_jsonl(Path(args.choice_data))
    qa_rows = load_json_or_jsonl(Path(args.qa_data))
    if args.two_source_only:
        pools = merge_pools(
            contextual_qa_candidates(qa_rows, args.qa_limit, args.qa_start),
            scientistqa_candidates(
                load_json_or_jsonl(Path(args.scientistqa_data)),
                args.scientistqa_limit, args.scientistqa_start,
            ),
        )
    else:
        pool_sets = [] if args.new_only else [
            choice_candidates(choice_rows, args.choice_limit),
            contextual_qa_candidates(qa_rows, args.qa_limit, args.qa_start),
        ]
        pools = merge_pools(
            *pool_sets,
            theoremqa_candidates(load_parquet(Path(args.theoremqa_data)), args.theoremqa_limit),
            umwp_candidates(load_json_or_jsonl(Path(args.umwp_data)), args.umwp_limit),
            scientistqa_candidates(
                load_json_or_jsonl(Path(args.scientistqa_data)),
                args.scientistqa_limit, args.scientistqa_start,
            ),
        )
    out_dir = Path(args.out_dir) / "candidates"
    manifest = []
    for z in STRESSORS:
        rows = pools[z]
        write_rows(rows, out_dir / f"{z}_pool.jsonl")
        manifest.extend({
            "sid": row.sid,
            "base_id": row.meta["base_id"],
            "stressor": row.stressor,
            "source": row.meta["source"],
        } for row in rows)
        print(f"[candidate] {z}: {len(rows)} -> {out_dir / f'{z}_pool.jsonl'}")
    write_rows(manifest, Path(args.out_dir) / "candidate_manifest.jsonl")
    bases = {row["base_id"] for row in manifest}
    print(f"[build] unique base questions={len(bases)}, candidates={len(manifest)}")


def validate_screened_row(row, z, source_path):
    expected = z.upper()
    if row.get("stressor") != expected:
        raise ValueError(
            f"{source_path}: stressor={row.get('stressor')!r}, expected={expected}"
        )
    base = row.get("meta", {}).get("base_id")
    if not base:
        raise ValueError(f"{source_path}: sid={row.get('sid')} 缺少 meta.base_id")
    return base


def merge_into_main(z, rows, managed_sources, append=False):
    target = DATA / f"processed/{z}_final.jsonl"
    old = read_jsonl(target) if target.exists() else []
    if append:
        existing_sids = {row.get("sid") for row in old}
        existing_bases = {
            row.get("meta", {}).get("base_id") for row in old
            if row.get("meta", {}).get("base_id")
        }
        additions = [
            row for row in rows
            if row.get("sid") not in existing_sids
            and row.get("meta", {}).get("base_id") not in existing_bases
        ]
        write_rows(old + additions, target)
        print(
            f"[append] {z}: old={len(old)} requested={len(rows)} "
            f"deduplicated={len(rows) - len(additions)} add={len(additions)} "
            f"total={len(old) + len(additions)} -> {target}"
        )
        return
    # 整体替换本脚本上一次写入的 real-life 快照，避免已经变成
    # 多重命中的 base_id 残留在旧 Zi 中。
    preserved = [
        row for row in old
        if not (
            row.get("domain") == "real-life"
            and row.get("meta", {}).get("source") in managed_sources
            and row.get("meta", {}).get("candidate_only") is True
        )
    ]
    combined = preserved + rows
    if target.exists():
        backup = target.with_suffix(target.suffix + ".pre_reallife.bak")
        if not backup.exists():
            shutil.copy2(target, backup)
    write_rows(combined, target)
    print(
        f"[merge] {z}: old={len(old)} replaced={len(old) - len(preserved)} "
        f"add={len(rows)} total={len(combined)} -> {target}"
    )


def resolve(args):
    screened_dir = Path(args.screened_dir)
    manifest_path = Path(args.out_dir) / "candidate_manifest.jsonl"
    managed_sources = ({row.get("source") for row in read_jsonl(manifest_path)}
                       if manifest_path.exists() else
                       {"question_and_result", "qa_data_2", "theoremqa", "umwp",
                        "scientistqa_names"})
    passed = defaultdict(dict)
    counts = {}
    for z in STRESSORS:
        path = screened_dir / f"{z}_final.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        counts[z] = len(rows)
        for row in rows:
            base = validate_screened_row(row, z, path)
            if base in passed[z]:
                raise ValueError(f"{path}: base_id={base} 重复通过同一 stressor")
            passed[z][base] = row

    base_to_z = defaultdict(list)
    for z, by_base in passed.items():
        for base in by_base:
            base_to_z[base].append(z)
    exclusive = {base: zs[0] for base, zs in base_to_z.items() if len(zs) == 1}
    ambiguous = {base: sorted(zs) for base, zs in base_to_z.items() if len(zs) > 1}

    out_dir = Path(args.out_dir) / "exclusive"
    kept_counts = {}
    for z in STRESSORS:
        rows = [
            row for base, row in passed[z].items()
            if exclusive.get(base) == z
        ]
        kept_counts[z] = len(rows)
        write_rows(rows, out_dir / f"{z}_final.jsonl")
        if args.merge_final:
            merge_into_main(z, rows, managed_sources, append=args.append_final)

    audit = {
        "screened_counts": counts,
        "exclusive_counts": kept_counts,
        "n_passed_any": len(base_to_z),
        "n_exclusive": len(exclusive),
        "n_ambiguous_removed": len(ambiguous),
        "ambiguous_by_combination": dict(Counter(
            "+".join(zs) for zs in ambiguous.values()
        )),
        "ambiguous": ambiguous,
    }
    audit_path = Path(args.out_dir) / "exclusivity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in audit.items() if k != "ambiguous"},
                     indent=2, ensure_ascii=False))
    print(f"[resolve] audit -> {audit_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="构造四类候选池，不判定是否合格")
    b.add_argument("--choice-data", default=str(DEFAULT_CHOICE_DATA))
    b.add_argument("--qa-data", default=str(DEFAULT_QA_DATA))
    b.add_argument("--theoremqa-data", default=str(DEFAULT_THEOREMQA_DATA))
    b.add_argument("--umwp-data", default=str(DEFAULT_UMWP_DATA))
    b.add_argument("--scientistqa-data", default=str(DEFAULT_SCIENTISTQA_DATA))
    b.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    b.add_argument("--new-only", action="store_true",
                   help="只构造本次新增的三个数据集")
    b.add_argument("--two-source-only", action="store_true",
                   help="只构造 qa_data_2 和 scientistqa_names 两个数据源")
    b.add_argument("--choice-limit", type=int, default=250, help="默认取前 250 条；0=全部")
    b.add_argument("--qa-limit", type=int, default=250, help="默认取前 250 条；0=全部")
    b.add_argument("--qa-start", type=int, default=0,
                   help="qa_data_2 的 0-based 起始位置")
    b.add_argument("--theoremqa-limit", type=int, default=DEFAULT_NEW_SOURCE_LIMIT,
                   help="TheoremQA 最多抽样条数；0=全部")
    b.add_argument("--umwp-limit", type=int, default=DEFAULT_NEW_SOURCE_LIMIT,
                   help="UMWP 最多抽样条数；0=全部")
    b.add_argument("--scientistqa-limit", type=int, default=DEFAULT_NEW_SOURCE_LIMIT,
                   help="ScientistQA 最多抽样条数；0=全部")
    b.add_argument("--scientistqa-start", type=int, default=0,
                   help="ScientistQA 的 0-based 起始位置；非零时连续切片")
    b.set_defaults(func=build)

    r = sub.add_parser("resolve", help="筛选后强制每个 base_id 只属于一个 Zi")
    r.add_argument("--screened-dir", required=True)
    r.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    r.add_argument("--merge-final", action="store_true",
                   help="显式开启后才合并进 data/processed/zi_final.jsonl")
    r.add_argument("--append-final", action="store_true",
                   help="按 sid/base_id 去重追加，保留同来源的既有 final 数据")
    r.set_defaults(func=resolve)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
