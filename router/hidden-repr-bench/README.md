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

**历史 Artifact 状态（2026-07-25，已由下方 2026-07-29 重跑更新覆盖）**：符号链接目标 `/tmp/out_tool_gate_popqa_2000` 已不存在，`analysis.json` 和 `known_subset_search_probe.json` 当前均不可读取。上述数值来自此前 README 记录，应视为历史结果，重新生成 artifact 前不可独立复核或继续派生分析。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python known_subset_search_probe.py --output-dir out_tool_gate_popqa_2000
```

## Z4 early diagnosis

入口：`z4_early_diagnosis.py`。在 MATH 上构造“full budget 正确、截断后错误”的 Z4-fail，与同难度但截断后仍正确的 converged control；在生成 K=0/32/64/128/256/512 token 处取 hidden，观察逐层 probe AUROC 是否随生成推进上升。目前未发现对应输出目录，视为尚未正式运行。


---

## 2026-07-29 全量盘点与结果更新

> 本节以当前工作区内实际可读取的 artifact 为准，并覆盖上文关于 Tool-gate artifact 已失效的旧状态说明。旧的 `/tmp` 符号链接仍失效，但带日期的重跑目录已经可用。

### 实验谱系与 `router/` 上游结果

本目录研究表征与门控；上游 `router/` 实验负责构造 teacher cause 和验证 treatment response：

| 实验 | 入口/结果目录 | 可引用结果 |
|---|---|---|
| Hallu-Diagnose matrix | `../hallu-diagnose-claude/` | 14,200 条 treatment，实际覆盖 Z1/Z2/Z6；Z3/Z4 尚未进入旧矩阵 |
| intervention teacher v1 | `../stressor_interventions_v1.py` | HaluEval 2,000 条审计仅为 `base_only`；accuracy=0.9095、181 条 hallucination，不能当 stressor 诊断结果 |
| ScientistQA router v2 | `../stressor_interventions_v2_first1500_results/` | hallucination AUROC=0.648；cause 的 probe-hidden ablation AUROC=0.754；span AUROC=0.409 |
| ScientistQA router v3 | `../stressor_interventions_v3_llama_first1500_results/` | hallucination AUROC=0.678；cause AUROC=0.855、balanced accuracy=0.673；span AUROC=0.591。span 最佳层为 layer 0，可能读取 token/replacement identity |
| hidden representation | `hidden_repr_bench.py` | 487 条 trace，70 个 contextual-interference 正例；E4 最高 `0.710±0.071` |
| Tool-gate | `tool_gate_calibration.py` | 1,431 条原始重跑和 1,000 条独立事实 probe 修正版均可复核 |
| Budget experiments | `budget_metacognition.py` 等 | 300 道 MATH×6 档预算；另有 100 对 GSM-IC clean/distracted |

### 当前 artifact 索引

| 入口 | 任务 | artifact | 状态 |
|---|---|---|---|
| `hidden_repr_bench.py` | 21 种关联错误表示 | `run_profiles_dual.tar.gz` | 已完成 487 条；当前以压缩包为可复核载体 |
| `tool_gate_calibration.py` | SEARCH/ANSWER/ABSTAIN | `out_tool_gate_popqa_2000_rerun_20260727/` | 已完成 1,431 条 |
| `probe_patch.py`, `knowledge_control.py` | 独立事实 probe、流行度控制 | `out_tool_gate_probe_patch_20260728/` | 已完成 1,000 条 |
| `known_subset_search_probe.py` | known 内预测 SEARCH | 重跑目录内 JSON | 已完成 |
| `budget_metacognition.py` | 预算曲线、gate、hidden probe | `out_budget_meta/` | 已完成 300 题 |
| `gate_scoring_fix.py` | 单 token gate 重评分 | `out_budget_meta_gate_singletoken_fix_20260728/` | 已完成，不支持供给敏感 gate |
| `budget_misallocation.py` | under/over/capability 诊断 | `out_budget_misalloc_reanalysis_20260727/` | 已完成严谨重分析 |
| `distraction_budget_reduction.py` | clean/distracted 资源税 | `out_dist_merged/` | 已完成 100 对；自动 verdict 需审计 |
| `budget_inertness.py` | 声明预算对自由生成长度的影响 | 未发现输出 | 尚无正式结果 |
| `z4_early_diagnosis.py` | 严格 Z4 的 AUROC(K) | 未发现输出 | 尚无正式结果 |
| `causal_search_steering.py` | knowledge/search direction 因果干预 | 未发现输出 | 尚无正式结果 |

`.orig` 是开发备份，不是实验。`out_budget_meta_shard0..3/`、`out_dist_shard0..2/` 是合并前分片，结论应引用合并目录。

## Tool-gate 重跑：知识边界是否驱动 SEARCH

### 1,431 条可复核重跑

`out_tool_gate_popqa_2000_rerun_20260727/records.jsonl` 实际有 1,431 条，分析也报告 `n=1431`；虽然 `config.json` 的 `max_samples` 为 1,000，样本量应以 artifact 为准。

| construct prior | n | SEARCH | ANSWER | ABSTAIN | 直接答且错 |
|---|---:|---:|---:|---:|---:|
| known | 1,071 | 45.56% | 50.79% | 3.64% | 9.71% |
| unknown | 360 | 94.44% | 2.50% | 3.06% | 2.22% |

unknown/known SEARCH odds ratio=`20.309`，Fisher `p=1.83e-70`。hidden 对 known/unknown 的最佳 AUROC=`0.9978`（layer 13），对 SEARCH 的最佳 AUROC=`0.9791`（layer 28）。

known 子集内 SEARCH=488、non-search=583，最佳 5-fold AUROC=`0.9745`（layer 27）。但该 SEARCH direction 与全局 UNKNOWN direction cosine 只有 `0.0169`，全层最大也只有 `0.0536`。所以 SEARCH 高度可读，但没有证据表明 SEARCH 和 UNKNOWN 是同一线性方向。

### 独立事实 probe 修正版

`probe_patch.py` 用同一实体的其他属性构造开放问答和真假平衡 probe，避免目标答案泄漏；`knowledge_control.py` 控制实体流行度。`out_tool_gate_probe_patch_20260728/` 有 1,000 条：

| prior | n | SEARCH | ANSWER | ABSTAIN | 直接答且错 |
|---|---:|---:|---:|---:|---:|
| known | 745 | 38.12% | 57.72% | 4.16% | 8.46% |
| unknown | 255 | 86.67% | 11.76% | 1.57% | 10.98% |

unknown/known SEARCH OR=`10.551`，Fisher `p=5.42e-44`；known/unknown AUROC=`0.9969`（layer 12），SEARCH AUROC=`0.9568`（layer 30）。

知识分数有 9 个离散值，均值 0.4974。控制 `log_pop` 后，knowledge coefficient=`-0.8606±0.1592`，LR χ²=`29.413`，`p=5.85e-8`；四个流行度层内知识分数与 SEARCH 均显著负相关。construct-unknown 内，SEARCH 组平均知识分数 0.2296，直接回答组 0.5208；加入 self-consistency 后仍有增量（LR=`97.24`，`p=6.14e-23`）。这支持搜索行为跟随实际知识，而非只跟随实体眼熟度。

旧 `analysis.json` 的 `probe_score_vs_search` 只有 20 个旧式 score 样本（Spearman=-0.1701，p=0.473），不能替代 1,000 条 `knowledge_control.json`。旧 `out_tool_gate_popqa_2000` 仍是失效 `/tmp` 链接，应引用上述带日期目录。`causal_search_steering.py` 尚无结果，所以当前仍是相关/条件控制证据，而非因果证据。

## Budget metacognition：模型是否知道自己的计算需求

### 设计与效用曲线

`budget_metacognition.py` 在 300 道 MATH 上运行 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`，thinking budgets 为 128/256/512/1024/2048/4096，每档 4 次采样，temperature=0.6；hidden 位置 K=0/16/32/64/128/256/512。`out_budget_meta/` 由四个 shard 合并。

300 题中 strict-all=57、ample-early=137、surge=106；61 题出现 overthinking（20.33%）。

| budget | 聚合准确率 |
|---:|---:|
| 128 | 0.3525 |
| 256 | 0.4050 |
| 512 | 0.4883 |
| 1024 | 0.6058 |
| 2048 | 0.6842 |
| 4096 | 0.7100 |

243 题存在最小充分预算 `b*`；中位数 256，P10=128，P90=2048，跨度 16×，说明逐题计算需求高度异质。

### Hidden probe、自报告与 gate

生成前 K=0 hidden 对失败的 AUROC=`0.7596`，需求回归 Spearman=`0.5006`。失败 AUROC 在 K=16 为 `0.7649`，到 K=512 为 `0.7130`；需求回归到 K=512 降为 0.3046。需求大体在 prompt encoding 时已经可读，而非随推理单调增强。

模型自报需求与真实 `b*` Spearman=`-0.2309`（n=468，p=4.39e-7），hidden probe 与自报的差为 0.7315：需求在表征中可读，但显式自报没有正确读出。

原 gate 格式合规率为 0，动作退化为单类，不能直接解释为可靠行为。单 token 修正版共有 1,800 行（300题×6预算）：

- margin=`logP(NEED_MORE)-logP(SOLVE)` 的题内预算斜率均值为 `+0.04136`，预期应显著小于 0；
- 只有 35% 题斜率为负；
- 不足与充分预算的 margin 差为 `-0.13884`，方向与预期相反；
- SOLVE/NEED_MORE/ABSTAIN 平均 log-prob 为 -1.215/-6.239/-6.424。

因此现有证据支持“hidden 编码需求”，不支持“模型用声明预算正确选择 NEED_MORE”。

## Budget misallocation：失败是否加预算可修

1,800 个 item-budget 标签为 capability 342、under 368、ok 1,029、over 48、nonmonotonic 13。严谨结果应引用 `out_budget_misalloc_reanalysis_20260727/misallocation_analysis.json`。

### E1：失败内区分 under 与 capability

- n=136（under 79、capability 57）；hidden AUROC=`0.7808`，layer 20。
- 可观测 B0=`0.5849`，hidden 预测需求 B1=`0.7451`，可观测+预测需求 B2=`0.7644`。
- hidden 相对 B2 增益仅 `0.0164`，item-cluster bootstrap 95% CI=`[-0.0288,0.0628]`。

hidden 能区分两类失败，但相对需求基线无显著增益，尚不能声称存在独立预算错配表征。旧 oracle `b*` baseline=1.0 使用真实答案曲线，存在标签泄漏，只能作上界。

### E2：同题改变声明预算

824 个观察、138 题；cross-item AUROC=`0.9324`（layer 16）；134 个可评估题的题内 AUROC 均值 1.0，预测与声明预算 Spearman=-1.0。但 stated-budget-only AUROC 已达 0.9037。因此更像 hidden 读取预算数字，不能单独证明正确的元认知比较；与 gate 失败共同指向“供给可读，但未被正确用于行动”。

```bash
python budget_misallocation.py --stage label --output-dir out_budget_meta
python budget_misallocation.py --stage paired --output-dir out_budget_meta
python budget_misallocation.py --stage analyze --output-dir out_budget_meta
```

## Distraction budget reduction：干扰是信息损伤还是资源税

`out_dist_merged/` 有 100 对 GSM-IC clean/distracted。`b*` 增加 22、减少 9、持平 69；mean log2 shift=0.25，median=0；sign test p=0.0147，单侧 Wilcoxon p=0.00476。distracted failures 中 60/89（67.42%）可加预算修复，29/89（32.58%）是能力上限。准确率 gap 从预算 128 的 0.1325 降至 4096 的 0.0075，闭合 94.34%。

| budget | clean acc | distracted acc | gap | clean thinking | distracted thinking |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.6225 | 0.4900 | 0.1325 | 120.4 | 123.2 |
| 256 | 0.7525 | 0.7100 | 0.0425 | 186.0 | 209.2 |
| 512 | 0.9050 | 0.8750 | 0.0300 | 294.9 | 360.6 |
| 1024 | 0.9250 | 0.9125 | 0.0125 | 424.0 | 544.7 |
| 2048 | 0.9325 | 0.9150 | 0.0175 | 482.9 | 637.3 |
| 4096 | 0.9325 | 0.9250 | 0.0075 | 525.8 | 706.6 |

子统计整体支持资源税：distracted 使用更多 tokens、触顶率更高，gap 随预算闭合。但 `distraction_analysis.json` 顶层 verdict 写 “not supported”，与自身统计矛盾。当前必须标为**子统计支持、自动 verdict 逻辑需审计**；复核 verdict 条件、右删失和 paired bootstrap 前不升级为正式结论。且仍有约 33% capability-limit，不能把所有 Z2 归约为 Z4。

## 尚无正式结果的实验

- `budget_inertness.py`：只改变声明预算、不硬截断，测自由生成长度；设计含多 seed、无预算、天花板/地板与 MDE，目前无输出。
- `z4_early_diagnosis.py`：构造严格 Z4 counterfactual 正负例，在 K=0/32/64/128/256/512 测 AUROC(K)，目前无输出。
- `causal_search_steering.py`：沿 unknown direction 做 necessity/sufficiency 和随机方向对照，目前无输出。

## 当前统一结论

1. **表征可读性较强**：实体相对表示优于末 token；知识状态、SEARCH、计算需求和失败均能从 hidden 预测。
2. **可读不等于同一方向**：known 子集 SEARCH direction 与全局 UNKNOWN direction 几乎正交。
3. **可读不等于被行为使用**：budget hidden 能读需求与声明预算，但 gate 差分不按预期变化。
4. **严格条件化会削弱强主张**：misallocation AUROC=0.7808，但相对需求基线仅增益 0.0164，CI 跨 0。
5. **干扰可能包含资源税**：100 对 GSM-IC 的 gap closure/token 使用支持，但自动 verdict 矛盾待解，且约三分之一失败不可由加预算修复。

最稳妥的主张是：**多类诊断变量在 hidden 中可解码，但它们是否为独立机制、是否被模型用于行动、以及 steering 后是否因果改变路由，仍需要严格对照。**

## Artifact 使用规范

- 优先读合并目录的 `config.json` 与 `analysis.json`，不要由目录名猜样本量。
- `out_*_shardN/` 是中间产物，不直接引用。
- `out_tool_gate_popqa_2000` 是失效 `/tmp` 链接；使用带日期重跑目录。
- `run_profiles_dual.tar.gz` 是旧 hidden-repr 当前可复核载体；error 日志不是样本。
- 最佳层结果存在逐层选择；正式比较需嵌套 CV 或独立验证集。
- 类别不均衡时同时报告 support、balanced accuracy、AUROC 和置信区间。
- 区分行为相关、hidden 可解码和因果 intervention；前两者不能替代第三者。
- artifact 内 verdict 若与子统计矛盾，应保留并显式标注，不静默修正。
