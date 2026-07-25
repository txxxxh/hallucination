# hidden_repr_bench — 关联性错误的 hidden-state 表征方式对比

自包含实验，不依赖 v2/v3 模块或任何落盘产物。当前比较 9 种原始表示与 12 种扩展实体/组合表示，
检验 transition+behavior 能否逼近 attention+behavior，并研究二候选实体相对表示。

## 配置 (全部叠加 behavior 底座)
- B0 behavior_only        基线 (置信度 + 探针分数统计)
- H1 point_lasttok        单点@答案末token (v3 复现, 预期弱)
- H2 point_entity         单点@选项实体
- H3 layer_traj           层间轨迹 h_{l+1}-h_l
- H4 cf_transition        反事实 transition h(orig)-h(删span) @末token
- H5 multiprobe           多探针 hidden mean/std (v3 的赢家)
- H6 cf_transition+multiprobe
- A1 attention(lookback)  Lookback式回看比率, 天花板对照
- A2 transition+heads     H4 + 少量 head
- E2 entity_pair          [选中实体; 未选实体]
- E3 entity_diff          选中实体 - 未选实体
- E4 entity_abs_diff      |选中实体 - 未选实体|
- T1 cf_signed_mean       多 span 反事实 transition 有符号均值
- T2 cf_mean+absmax       有符号均值 + 逐维绝对最大值
- P1 probe_diff           选中人物 - 未选人物的 probe mean/std
- C1 answer+entity_diff
- C2 entity_diff+transition
- C3 entity_diff+attention
- C4 entity_diff+transition+attention
- C5 entity_diff+probe_diff
- C6 entity_diff+transition+probe_diff

E2/E3/E4/C1-C6 依赖 prompt 中存在两个可定位候选实体，不直接适用于没有竞争候选的开放式 QA。

## 运行
```
# 采集 (GPU): 每样本约 (2+K*3) 次前向, K=max_spans
python hidden_repr_bench.py --stage collect \
  --input shuffled_prepend_names_question.json \
  --profiles shuffled_prepend_profiles_question.json \
  --right-field rgt_ans --wrong-field wrg_ans \
  --model NousResearch/Meta-Llama-3.1-8B-Instruct --quantize-4bit \
  --output-dir out --max-samples 800 --max-spans 3

# 分析 (无GPU): 21配置 x 逐层 grouped-5fold-CV, 排除embedding层
python hidden_repr_bench.py --stage analyze --output-dir out

# 方向 (轻量GPU): 存 diff-of-means 方向供 steering
python hidden_repr_bench.py --stage steer --output-dir out
```

## 字段映射
默认读 prompt/right_name/wrong_name/option1/option2; 数据集不同用 --prompt-field
--right-field --wrong-field 调整。profiles 用于构造知识探针 (make_probes)。

## 输出
- records.jsonl        逐样本 base 答案/知识状态/teacher cause
- traces/*.pt          逐样本多层 hidden / cf_deltas / lookback / probe stats
- repr_comparison.json 21 配置逐层 AUROC + 关键对比 + diff-of-means 方向 AUROC
- assoc_direction.pt   (steer阶段) 关联错误方向向量

## 读结果
comparisons 里:
- position_gain_H2_minus_H1  > 0 说明位置重要 (预期)
- transition_vs_attention_H6_over_A1  接近 1.0 说明 transition 逼近 attention (核心主张)
- multiprobe_gain_H5_minus_H1  > 0 说明多探针聚合有效 (v3 已暗示)

## 当前运行状态

`run_profiles_dual/combined/` 合并了两个 worker 的 487 条完整 records/trace。cause 分布：correct 244、contextual_interference 70、known_but_unlocalized 98、ambiguous 74、knowledge_gap 1。AUROC 分析使用 70 个关联错误正例和 343 个非关联负例；74 个 ambiguous 被排除。采集后来因磁盘配额耗尽终止，大量 `errors*.jsonl` 是失败尝试日志，不是额外有效样本。

| 模式 | 表示 | 最佳层 | grouped-5fold AUROC |
|---|---|---:|---:|
| E4 | 实体逐维绝对差分 | 18 | **0.710±0.071** |
| E2 | 选中/未选实体拼接 | 23 | 0.706±0.090 |
| C3 | 实体差分 + attention | 16 | 0.705±0.046 |
| E3 | 选中实体 - 未选实体 | 21 | 0.703±0.022 |
| C2 | 实体差分 + transition | 21 | 0.699±0.039 |
| C1 | answer + 实体差分 | 16 | 0.696±0.021 |
| A1 | attention lookback | 9 | 0.694±0.076 |
| H2 | 单个 prompt 实体 | 12 | 0.689±0.052 |
| B0 | behavior only | 0 | 0.654±0.045 |
| H1 | 答案末 token | 2 | 0.645±0.056 |

完整 21 模式与逐层曲线见 `run_profiles_dual/combined/repr_comparison.json`。E4 是当前平均 AUROC 最高候选，E3 的折间方差最低；差异尚未经过嵌套层选择或配对 bootstrap，不能声称 E4 显著优于 E2/E3/C3/A1。现有数据来自磁盘写满前保存的两个连续切片，不是随机完整样本。

## Tool-gate calibration

入口：`tool_gate_calibration.py`。模型对每题选择 `[SEARCH]`、直接回答或 `[ABSTAIN]`；construct prior（PopQA 流行度/合成实体）和独立事实 probe 提供知识状态，逐层 hidden probe 分别预测 known/unknown 和 SEARCH。

已有结果目录：`out_tool_gate_popqa_2000/`（实际是 `/tmp` 符号链接）。PopQA 严格流行度阈值只得到 1,431 条：known 1,071、unknown 360。unknown search rate=0.9444，known=0.4556，Fisher p=1.83e-70；hidden 预测 known/unknown 的最佳 AUROC=0.9978，预测 SEARCH 的最佳 AUROC=0.9791。详见 `analysis.json`。

离线补充：`known_subset_search_probe.py` 只读取上述 records/hidden，在 known 内部预测 SEARCH。最佳 5-fold AUROC=0.9745（layer 27），但 SEARCH 方向与全局 UNKNOWN 方向的 cosine 仅 0.0169（全层最大约 0.0536）。即“是否搜索”高度可读，但当前没有证据表明它就是同一个线性知识方向。详见 `known_subset_search_probe.json`。

**Artifact 状态（2026-07-25）**：符号链接目标 `/tmp/out_tool_gate_popqa_2000` 已不存在，`analysis.json` 和 `known_subset_search_probe.json` 当前均不可读取。上述数值来自此前 README 记录，应视为历史结果，重新生成 artifact 前不可独立复核或继续派生分析。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python known_subset_search_probe.py --output-dir out_tool_gate_popqa_2000
```

## Z4 early diagnosis

入口：`z4_early_diagnosis.py`。在 MATH 上构造“full budget 正确、截断后错误”的 Z4-fail，与同难度但截断后仍正确的 converged control；在生成 K=0/32/64/128/256/512 token 处取 hidden，观察逐层 probe AUROC 是否随生成推进上升。目前未发现对应输出目录，视为尚未正式运行。
