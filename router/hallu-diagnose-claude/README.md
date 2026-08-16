# Hallu-Diagnose: Phase 0 / Phase 1 代码

Symptom→Stressor 诊断研究的数据构造与疗效交互矩阵实验代码。
5 种 stressor 进主矩阵:Z1 没学过 / Z2 干扰 / Z3 捷径 / Z4 budget 不足 / Z6 校准失败。

## 数据来源(全部公开可下载)

| Stressor | 来源 | 获取方式 | 用途 |
|---|---|---|---|
| Z1 没学过 | **PopQA** | HF: `akariasai/PopQA`(含实体流行度字段 `s_pop`) | 长尾实体问答,按流行度筛低频子集 |
| Z1 没学过 | **FreshQA** | GitHub: `freshllms/freshqa`(定期更新答案的时效性 QA) | 训练截止后变化的事实 |
| Z1 没学过 | 合成传记 | 本仓库 `01_build_z1.py` 自动生成(虚构实体+虚构属性) | 真值绝对干净的"不可能学过" |
| Z2 干扰 | **GSM8K** | HF: `openai/gsm8k` | 稳定正确集的底池 |
| Z2 干扰 | **GSM-IC** | GitHub: `google-research-datasets/GSM-IC`(现成的干扰句版本) | 数学域干扰,直接复用 |
| Z2 干扰 | PopQA + 生成干扰 | `02_build_z2.py` 用 LLM 往事实问答里插无关句 | 事实域干扰(跨域要求) |
| Z3 捷径 | **TruthfulQA** | HF: `truthful_qa`(gen 子集) | 挖"流行误解型"条目:模型答出高频错误联想 |
| Z3 捷径 | 共现模板 | `03_build_z3.py` 从 Wikidata 高共现实体对生成 + 人工审核 | 反事实可控的捷径对(核心来源) |
| Z3 捷径 | EUREQA | 论文 "Deceptive Semantic Shortcuts"(arXiv:2311.09702)附带数据;若仓库不可得,用本仓库多跳链生成器替代 | 推理域捷径 |
| Z4 budget | **MATH** | HF: `EleutherAI/hendrycks_math`(或 `hendrycks/competition_math`) | thinking 截断实验底池 |
| Z6 校准 | **SelfAware** | GitHub: `yinzhangyue/SelfAware`(known/unknown 问题集) | 本质不可答问题 |
| Z6 校准 | **FalseQA** | GitHub: `thu-coai/FalseQA`(假前提问题) | 假前提:任何直接作答都是幻觉 |
| T-RAG 用 gold passage | Wikipedia dump / Wikidata | `datasets` 的 `wikipedia` 或直接用 PopQA 自带的证据字段 | 治疗材料 |

注意:各数据集的许可证以其仓库为准;GSM-IC / SelfAware / FalseQA 的具体文件名可能随仓库更新变动,`download_data.sh` 中的路径若失效请按仓库 README 调整。

## 运行流程

```bash
pip install -r requirements.txt
bash download_data.sh                      # 下载公开数据集
python scripts/01_build_z1.py              # 构造各 stressor 候选池
python scripts/02_build_z2.py
python scripts/03_build_z3.py --mine       # 先挖候选,人工审核 CSV 后 --finalize
python scripts/04_build_z4.py
python scripts/05_build_z6.py
python scripts/10_screen.py --stressor all # 行为筛选:flip 判定 + 重采样入组
python scripts/21_run_matrix.py            # 6 治疗 × 5 stressor 矩阵
python scripts/30_stats.py                 # McNemar + 交互回归 + 热图
```

## 无人值守全流程（暂不含 Z3）

默认对每个外部数据源最多取前 1000 条，使用本地 4-bit R1 模型完成 Z4，
并依次执行构建、行为筛选、symptom 标注、治疗矩阵和统计分析：

```bash
./run_full_pipeline.sh --detach
```

日志写入 `logs/`，完成标记写入 `data/run_state/`；中断后重新执行会从未完成阶段续跑。
使用 `--force` 可忽略完成标记重跑，`--dry-run` 可只打印完整命令序列。

## 当前结果状态（2026-07-25）

`data/results/matrix_Meta-Llama-3.1-8B-Instruct.jsonl` 当前有 14,200 条记录：1,775 个入组样本 × 8 个 treatment。实际 stressor 覆盖为 Z1 6,144 条、Z2 320 条、Z6 7,736 条；Z3/Z4 尚未进入该矩阵，因此不能把它称为完整的 5-stressor 矩阵。

| Treatment | 记录数 | strict | honest |
|---|---:|---:|---:|
| none | 1,775 | 0.0073 | 0.0079 |
| T-RAG | 1,775 | 0.1651 | 0.1651 |
| T-Clean | 1,775 | 0.0417 | 0.0423 |
| T-CleanOracle | 1,775 | 0.0073 | 0.0079 |
| T-CF | 1,775 | 0.0468 | 0.0485 |
| T-Budget | 1,775 | 0.0321 | 0.0383 |
| T-Abstain | 1,775 | 0.3735 | 0.7865 |
| T-SC | 1,775 | 0.0361 | 0.0361 |

这是跨 stressor 的粗汇总。T-Abstain 的 honest 提升主要反映合理弃答口径，不能等同于答对；正式因果结论仍应按 stressor、domain 和 matched sample 做配对统计。
Z3 因候选必须人工审核 `keep` 列，当前明确跳过。

## 关键设计

- **双结局度量**:`strict`(答对)与 `honest`(答对∨合理弃答)。Z6 行只在 honest 下有意义。
- **入组条件**(`10_screen.py`):Z1/Z2 保持 greedy 错 + n=8/T=0.7 多数错(排解码噪声)。Z3 使用较宽松的行为证据：触发题 greedy 错，或 8 次采样至少 2 次错；只要求干净题 greedy 正确，不再要求错误答案高度自洽或清洁题 8 次采样全对。Z6 额外要求模型给出了具体断言(未弃答)。
- **多标签**:Z1 样本若未弃答自动附加 Z6 次标签,存于 `secondary_labels`。
- 所有中间产物为 JSONL,schema 见 `scripts/common.py` 的 `Sample`。
