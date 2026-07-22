"""Z6 校准失败: 本质不可答/假前提问题, 任何具体作答都是幻觉, 正确行为 = 弃答/指出前提错误。
与 Z1 的分界: Z1 有确定答案(模型不知道); Z6 无确定答案(不该答)。
来源: SelfAware unanswerable 子集 + FalseQA 假前提 + FreshQA false_premise 题。
q_clean 定义为"允许弃答"的版本(校准的开关是弃答许可, 类比 budget 之于 Z4)。
"""
import argparse, csv, json
from pathlib import Path
from common import Sample, sid_of, write_jsonl, DATA

ABSTAIN_LICENSE = 'If the question is unanswerable or based on a false premise, say so explicitly. '

def build_selfaware(path: Path):
    """github.com/yinzhangyue/SelfAware -> SelfAware.json, answerable=False 子集。"""
    if not path.exists():
        print(f"[skip] {path}"); return []
    data = json.load(open(path))
    rows = data["example"] if isinstance(data, dict) and "example" in data else data
    out = []
    for r in rows:
        if r.get("answerable") is True:
            continue
        q = r["question"]
        out.append(Sample(
            sid=sid_of(q, "z6sa"), stressor="Z6", domain="factual",
            template_id=f"selfaware-{r.get('source','x')}", intensity=1.0,
            q_trig=q, q_clean=ABSTAIN_LICENSE + q, answer="UNANSWERABLE",
            meta={"source": "selfaware"}))
    return out

def build_falseqa(path: Path):
    """github.com/thu-coai/FalseQA -> train/valid/test.csv, label==1 为假前提题。"""
    if not path.exists():
        print(f"[skip] {path}"); return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if str(r.get("label", "")).strip() not in ("1", "True", "true"):
                continue
            q = r["question"]
            out.append(Sample(
                sid=sid_of(q, "z6fp"), stressor="Z6", domain="factual",
                template_id="falseqa", intensity=1.0,
                q_trig=q, q_clean=ABSTAIN_LICENSE + q, answer="UNANSWERABLE",
                meta={"source": "falseqa", "fp_explanation": r.get("answer", "")}))
    return out

def build_freshqa_fp(path: Path):
    import csv as _csv
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            if r.get("false_premise", "").upper() != "TRUE":
                continue
            q = r["question"]
            out.append(Sample(
                sid=sid_of(q, "z6fr"), stressor="Z6", domain="factual",
                template_id="freshqa-fp", intensity=1.0,
                q_trig=q, q_clean=ABSTAIN_LICENSE + q, answer="UNANSWERABLE",
                meta={"source": "freshqa"}))
    return out

if __name__ == "__main__":
    pool = (build_selfaware(DATA / "raw/selfaware/SelfAware.json")
            + build_falseqa(DATA / "raw/falseqa/test.csv")
            + build_freshqa_fp(DATA / "raw/freshqa.csv"))
    write_jsonl(pool, DATA / "processed/z6_pool.jsonl")
