# Perturbation 项目实验结果梳理与论文可用性审计

> 审计日期：2026-08-20  
> 范围：`whitebox/perturbation/` 中现有脚本、JSON/CSV/Markdown 报告。本文只把已有最终报告中的数值视为“已完成结果”；仅有脚本、缓存或 pilot 的实验不视为完整证据。

## 1. 总体结论

现有实验已经形成四条相互关联的研究线，但它们的成熟度不同。

| 部分 | 目标主张 | 当前证据 | 完整性 | 正确性/口径 | 论文可用性 | 建议定位 |
|---|---|---|---|---|---|---|
| 1. 机制：关键词—人物 binding | 扰动关键词能改变错误 margin；关键词与人物存在生成绑定；绑定源于训练频率 | 单关键词属性扰动、自由生成、两个 3B 模型的合成频率干预均已有结果 | 部分完整 | 扰动效应和自由生成成立；但严格单关键词 binding 的总体检验为阴性，频率剂量效应跨 seed 不稳定 | **有限可用，必须改写主张** | 可写“属性线索与错误决策相关，且部分属性在自由生成中显现”；不能写“普遍存在强人物 binding” |
| 2. 检测：exact / attention / gradient | perturbation 特征可检测幻觉；attention/gradient 降低开销且只损失少量精度 | exact 与 attention 有 4 模型×3 benchmark 完整矩阵；gradient 主要有 Scientist 对照 | exact/attention 完整；gradient 不完整 | grouped OOF 口径较规范；gradient 的 backward 成本未折算为统一算力 | **exact/attention 可用；gradient 暂不足以支撑跨域主张** | exact 为主方法，attention 为高效近似；gradient 先作为 Scientist 消融 |
| 3. 分析：UEPR benchmark 类型 | 不同 benchmark 的错误机制组成不同，因此适合不同检测方法 | Scientist audit 与 Scientist/Trivia/GSM8K 的多项轴确认已完成 | 部分完整 | 各轴的预测有效性有证据，但尚未得到统一协议下的 benchmark 四轴比例分布 | **机制分析可用，比例结论不可直接写** | 写“轴级异质性证据”；补完统一标注后再写“类型比例解释性能差异” |
| 4. 多关键词联合 | 关键词间存在冗余、竞争、协同；联合干预优于单关键词 | 449 题全因子 atlas、18 题冻结确认、paired generation 控制均完成 | 核心结果完整但外部效度不足 | 交互符号在当前 `u` 定义下正确；确认集较小且由发现流程筛选 | **可作为强 case-study / mechanism subsection** | 强调竞争/混合占主导，纯协同比例小；联合修复结果需标注 n=18 |

一句话判断：**检测主结果（exact/attention）与多关键词联合结果最接近论文可用；单关键词 binding 的原始强叙事与 UEPR benchmark 比例叙事目前都超过了现有证据。**

## 2. 统一定义与实验口径

teacher-forced 幻觉 margin 记为：

\[
S=\log p(\hat y)-\log p(y^*),
\]

其中 \(\hat y\) 是模型错误答案，\(y^*\) 是正确答案。对关键词集合 \(\mathcal S\) 做保长度 neutralization 后，定义修复增益：

\[
u(\mathcal S)=S(\varnothing)-S(\mathcal S).
\]

因此 \(u>0\) 表示去除该线索后错误 margin 降低，即该线索支持错误答案。两关键词交互为：

\[
I_{ij}=u(\{i,j\})-u(\{i\})-u(\{j\}).
\]

| 交互 | 判定 | 解释 |
|---|---|---|
| 冗余 | \(I_{ij}<0\)，且两个单体均有修复作用 | 两个线索提供可替代的重复错误证据 |
| 协同 | \(I_{ij}>0\) | 联合效果超过单体效果之和 |
| 竞争 | 单独移除有益，但联合移除的行为受相反线索抵消/重排 | 两个线索对答案的支持方向或强度发生竞争 |
| mixed | 不满足单一纯关系 | 局部关系随背景集合变化 |
| pure combination | 单体均不能修复，联合才能修复 | 最严格的“必须联合”情形 |

注意：以上符号只对当前增益函数 \(u\) 成立，不能直接套用以原始 margin \(S\) 为目标的交互符号。

## 3. 第一部分：关键词—人物 binding 机制

### 3.1 层次一：关键词扰动改变错误 margin

严格属性匹配数据覆盖 453 个错误题，其中 421 题存在可归属的人物属性关键词。被选中的属性关键词本身具有正的错误支持效应，整体平均 `mean_perturb_u=0.513`；不同属性的平均扰动效应如下。

| 属性类型 | n | 平均扰动修复增益 `u` | binding effect 均值 | binding effect 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| award received | 209 | 0.461 | 0.001 | [-0.085, 0.085] | 扰动影响 margin，但无人物特异 binding |
| education | 16 | 0.582 | 0.320 | [0.047, 0.633] | 探索性正效应，样本小；BH 校正后不显著 |
| field | 71 | 0.617 | -0.077 | [-0.248, 0.096] | 强扰动效应，但不支持 binding |
| occupation | 114 | 0.542 | -0.007 | [-0.140, 0.127] | 强扰动效应，但不支持 binding |
| position held | 8 | 0.400 | 0.359 | [-0.094, 0.766] | 探索性趋势，样本极小；BH 校正后不显著 |
| notable work | 2 | 0.477 | -0.062 | [-0.812, 0.688] | 不可推断 |
| place of birth | 1 | 0.392 | 0.250 | 单样本 | 不可推断 |
| **总体** | **421** | **0.513** | **0.005** | **[-0.059, 0.070]** | 扰动效应存在；严格人物特异 binding 总体不成立 |

论文可用表述：

- 可写：人物属性类关键词（education、occupation、field、position 等）被 neutralize 后能够显著改变错误答案 margin，是错误决策的局部支持线索。
- 不可直接写：这些类别都与对应人物形成了普遍、显著的 binding。
- 当前报告没有直接汇总“margin 从正变负”的总翻转率，因此“会使 margin 翻转”应改成“降低/改变 margin”，或者补做统一 flip-rate 表。

严格 binding 指标总体为 `mean=0.0048`、正值比例 `49.4%`、单侧 Wilcoxon `p=0.549`；binding 与扰动强度相关性 `rho=-0.043, p=0.375`。这是明确的总体阴性结果，不能隐藏。

### 3.2 层次二：自由生成中的人物—属性共现

Llama-3.1-8B-Instruct 对 812 个关键词、421 个问题、每个人物—关键词 20 次采样。下表中的数值是错误人物相对正确人物的属性生成频率差。

| 属性 | 关键词数 | 问题数 | exact 频率差 | exact 95% CI | loose 频率差 | loose 95% CI | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| award received | 357 | 269 | 0.060 | [0.025, 0.098] | 0.062 | [0.024, 0.100] | 正向且区间不跨 0 |
| education | 32 | 32 | 0.133 | [0.041, 0.228] | 0.164 | [0.063, 0.266] | 最强、但样本较小 |
| field | 156 | 114 | 0.033 | [0.013, 0.056] | 0.126 | [0.080, 0.174] | loose 匹配下明显 |
| occupation | 250 | 164 | 0.005 | [-0.017, 0.026] | 0.027 | [-0.011, 0.065] | 不显著 |
| position held | 14 | 11 | 0.120 | [0.030, 0.241] | 0.125 | [0.027, 0.248] | 正向但样本很小 |
| **总体** | **812** | **421** | **0.039** | **[0.021, 0.058]** | **0.071** | **[0.047, 0.096]** | 自由生成共现总体成立 |

但是，“被 perturbation 选中的关键词比未选中关键词绑定更强”并未得到支持：359 个匹配组中，exact 差值 `0.0149`，95% CI `[-0.0163, 0.0458]`，`p=0.279`；loose 差值 `0.0176`，95% CI `[-0.0153, 0.0508]`，`p=0.317`。因此自由生成结果支持“人物—属性共现”，不支持“扰动算法专门找到了绑定更强的属性”。

### 3.3 层次三：训练出现频率的因果干预

实验使用 50 组镜像虚构人物对（100 人），在 Qwen2.5-3B-Instruct 和 Llama-3.2-3B-Instruct 上控制属性 B 相对匹配属性 F 的训练频率 dose。核心指标 `B-F margin` 越大，表示 B 的训练频率优势越强地进入人物选择。

| 模型 | dose | 训练后平均 B-F margin | B-F>0 比例 | 结果 |
|---|---:|---:|---:|---|
| Qwen2.5-3B（seed 42） | 0 / .25 / .5 / .75 / 1 | .022 / .028 / .171 / .346 / .286 | .58 / .58 / .74 / .80 / .86 | 大体随 dose 增强，但最高 dose 非单调 |
| Qwen2.5-3B（seed 43） | 0 / .25 / .5 / .75 / 1 | .024 / .050 / .073 / .096 / .199 | .52 / .60 / .70 / .72 / .88 | 清晰单调趋势 |
| Qwen2.5-3B（seed 44） | 0 / .25 / .5 / .75 / 1 | .039 / .086 / .149 / .156 / .210 | .54 / .66 / .78 / .80 / .72 | margin 单调，正比例最高 dose 回落 |
| Llama-3.2-3B | 0 / .25 / .5 / .75 / 1 | -.005 / .008 / .009 / .009 / .047 | .48 / .58 / .56 / .66 / .80 | 方向一致但效应远小于 Qwen，仅单 seed |

这一实验支持“训练共现频率可以因果改变人物—属性偏好”，但当前不能严谨地称为“预训练频率证明”：它是**受控合成 fine-tuning**，最多证明模型能够通过训练频率学得此类关联，不能证明真实大模型中的 binding 就来自自然预训练语料频率。论文中应使用“causal analogue / controlled training intervention”措辞，并把真实预训练语料频率关联作为单独证据补齐。

### 3.4 第一部分结论

| 原计划主张 | 审计判断 | 可发表版本 |
|---|---|---|
| 某些属性关键词 perturb 后会使 margin 翻转 | 部分支持；有 `u`，缺统一翻转率 | “属性关键词对错误 margin 有可测的因果影响” |
| 关键词与人物的 binding 普遍存在 | 严格总体检验不支持 | “自由生成中存在属性共现；education/position 等呈探索性人物特异效应” |
| binding 是预训练高频导致 | 合成 fine-tune 支持可学习性，不证明自然预训练来源 | “受控训练频率能够诱导/增强该关联” |

## 4. 第二部分：Perturbation 幻觉检测

### 4.1 多模型、多 benchmark 主结果

统一矩阵采用固定 `current127 scalar47 + four hidden PCA8 + layer14 PCA48` 特征、逻辑回归 `C=.03`，使用 grouped `3×5` OOF；该矩阵没有在各格单独调参。exact 为完整枚举，attention 为候选剪枝。

| 模型 | 方法 | Scientist AUROC | Trivia AUROC | GSM8K AUROC | 平均 AUROC | 查询减少 |
|---|---|---:|---:|---:|---:|---:|
| Llama | exact | .894 | .948 | .949 | .930 | 0% |
| Llama | attention | .888 | .947 | .951 | .929 | 35.8–41.4% |
| Qwen | exact | .858 | .910 | .784 | .851 | 0% |
| Qwen | attention | .834 | .905 | .789 | .843 | 35.6–40.9% |
| Mistral | exact | .809 | .964 | .794 | .856 | 0% |
| Mistral | attention | .805 | .962 | .792 | .853 | 34.5–40.6% |
| Falcon3 | exact | .645 | .897 | .872 | .805 | 0% |
| Falcon3 | attention | .650 | .898 | .872 | .807 | 35.5–41.5% |

attention 在 12 个格子中相对 exact 的平均 AUROC 差约 `-0.003`，同时减少约 35–41% 的前向查询；这足以支持“attention 粗筛以很小精度代价降低计算开销”。

### 4.2 Scientist 上三种方法的同口径比较

| 方法 | n | AUROC | AUPRC | Balanced Acc. | 计算/查询 | 相对 exact AUROC |
|---|---:|---:|---:|---:|---|---:|
| exact enumeration | 1084 | .902 | .920 | .817 | 74.20 次前向/样本 | — |
| max-head attention pruning | 1084 | .884 | .906 | .794 | 49.79 次；减少 32.9% | -.018 |
| first-order gate gradient | 1084 | .878 | .903 | .790 | 42.49 次前向扰动；减少 42.7%，另有 2 次 backward | -.024 |
| class-separated gradient | 1084 | .878 | .901 | .782 | 40.52 次前向扰动；减少 45.4%，另有 2 个 backward graph/stage | -.023 |

这里的“gradient 降低开销”只能指减少了 perturbation forward queries。由于报告明确没有把 backward pass 换算成 forward-equivalent FLOPs、时延或显存，不能直接写“总计算成本降低 42.7%”。


#### `.894` 与 `.902` 的口径差异

这两个 Llama Scientist exact AUROC **不是同一份特征上的重复评估**，不能用随机 seed 波动解释。

| 数值 | 来源 | 样本 | 特征来源 | CV/分类器 |
|---:|---|---:|---|---|
| `.894` | `paper4_self_matrix_v2/evaluation/combined/evaluation.json` | 1077 题（628 correct） | paper4 重新采集的 exact `scalar47 + four hidden PCA8 + layer14 PCA48`；`layer14` 是错误答案最后一个 token 的 hidden state | grouped 3×5 OOF，LR `C=.03` |
| `.902` | `153_exact_current127_samecv_report.json` | 1084 题 | 复用 `120_physical_delete_rerank`、`116_dual_candidate_hidden_top5` 和 `100_scientist_trajectory_l8` 旧缓存；最后的 PCA48 特征是 layer 14 上**答案所有 token 的平均** hidden state | grouped 3×5 OOF，LR `C=.03` |

paper4 矩阵相对 1084 题集合缺少 `question_0129`、`question_0361`、`question_0491`、`question_0912`、`question_1003`、`question_2041`、`question_2831` 七题。因此差异同时包含**样本子集差异**和**特征构造差异**；现有报告无法将 `.008` 差值唯一归因给其中一项。论文主表应以统一 paper4 矩阵的 `.894` 为准；三方法对照表可保留 `.902`，但应明确标为“早期 1084 题旧缓存对照”，而不是与 paper4 主表完全同口径。若要比较 exact/attention/gradient 的绝对差值，需将三种方法在同一份冻结 manifest 和同一 layer-14 定义上重评。

### 4.3 跨域冻结迁移

Scientist 上拟合 scaler、PCA 和 LR 后全部冻结，目标域不做任何拟合。每个单元格为 `AUROC / AUPRC / Balanced Accuracy`；各模型的目标样本数不同，是因为使用了该模型自身的可用生成与标签。项目内部域名 `building` 对应建筑/architecture 领域。

| 模型 | 方法 | Athlete（n） | Architecture/building（n） | Musician（n） | 合并 AUROC |
|---|---|---:|---:|---:|---:|
| Llama | exact | `.953/.987/.777` (154) | `.931/.976/.793` (184) | `.871/.925/.712` (139) | `.917` |
| Llama | attention | `.944/.984/.785` (154) | `.907/.965/.764` (184) | `.844/.919/.696` (139) | `.895` |
| Qwen | exact | `.839/.926/.699` (100) | `.904/.970/.754` (157) | `.712/.785/.596` (93) | `.840` |
| Qwen | attention | `.755/.870/.621` (100) | `.808/.934/.681` (157) | `.624/.691/.584` (93) | `.748` |
| Mistral | exact | `.886/.966/.798` (128) | `.857/.950/.739` (176) | `.854/.924/.737` (119) | `.854` |
| Mistral | attention | `.862/.958/.779` (128) | `.779/.915/.696` (176) | `.805/.889/.697` (119) | `.800` |
| Falcon3 | exact | `.729/.839/.544` (92) | `.615/.812/.587` (115) | `.552/.736/.516` (90) | `.626` |
| Falcon3 | attention | `.644/.795/.593` (92) | `.626/.786/.572` (115) | `.539/.735/.506` (90) | `.606` |

跨域结果显示明显的模型和目标域依赖：Llama 的 exact 在三域都较强；Qwen 的 musician 明显弱于 athlete/building；Falcon3 的 building 和 musician 接近随机，其 pooled `.626` 不应被解读为稳健迁移。attention 在 Llama 上的跨域损失较小，但在 Qwen 和 Mistral 的 building/musician 上损失较大，因此“小精度代价”目前只能稳健地用于域内 4×3 矩阵，不能无条件推广到冻结跨域设置。

### 4.4 第二部分结论

| 问题 | 判断 |
|---|---|
| exact 准确率较高但开销大 | 支持；4 模型×3 benchmark 完整 |
| attention 降低开销且精度下降少 | 强支持；同一完整矩阵中成立 |
| gradient 降低开销且精度下降少 | Scientist 上支持；跨模型跨 benchmark 不完整，总计算口径不完整 |
| 多模型多 benchmark 均表现好 | 不能笼统写；Falcon3 Scientist `.645`、Qwen/Mistral GSM8K 约 `.78-.79` 明显较弱 |

## 5. 第三部分：UEPR 四轴与 benchmark 异质性

四个轴的现有定义为：U（semantic uncertainty）、R（representation/trajectory）、E（external evidence gap）、P（prompt/context-local misleading support）。

### 5.1 Scientist 统一审计

Scientist/Llama common-key audit 共 1,076 题、453 个错误。

| 轴 | 错误检测 AUROC | 错误均值 | 正确均值 | 错误中的 dominant 数 | 占 453 错误比例 |
|---|---:|---:|---:|---:|---:|
| U | .604 | .676 | .587 | 84 | 18.5% |
| R | .816 | .663 | .391 | 147 | 32.5% |
| E | .978 | .038 | -.038 | 194 | 42.8% |
| P | .281 | .488 | 1.114 | 28 | 6.2% |

这些 dominant 比例是按探索性 30/70 全局 rank threshold 强制分配的**假设标签**，不是人工 ground truth。P 的 AUROC 小于 0.5 并不等价于 P 无效，而是当前 P 分数方向/定义与“所有错误”不一致：P 更适合识别局部可删除错误子类，而非统一错误检测器。

### 5.2 跨 benchmark 的轴级确认结果

| Benchmark | 轴 | 样本 | 关键结果 | 是否符合预测 | 解释 |
|---|---|---:|---|---|---|
| Scientist | U | 1076 | pooled entropy AUROC .593；stable systematic error 70 | 弱支持 | 不确定性只能解释一部分错误 |
| Scientist | R | 1076 | 初始 R AUROC .816；独立方向干预效果依组别而异 | 部分支持 | R 有强检测性，但因果干预不对所有错误统一有效 |
| Scientist | E | 1076 | held-out evidence completion high-low 仅 .092；总体 rho -.007 | 不支持/很弱 | 初始 E 高 AUROC 可能部分依赖定义或同源信号 |
| Scientist | P | 226 个可定位错误 | target-placebo gain .082；rho .065 | 小幅支持 | 有特定可删除子群，但排序相关弱 |
| Trivia | E | 1000 | closed-book E-positive 错误修复增益 2.529，E-nonpositive -.109；rho .409 | 强支持 | 外部证据缺失是重要类型 |
| Trivia | P | 313 | target-placebo gain 1.687；high-low 2.576；rho .419 | 强支持 | 上下文中的误导句是重要类型 |
| Trivia | U | 1000 | high-U 错误率更高，但 held-out majority repair 更低 | 支持风险分层，不支持“采样即可修复” | U 适合 selective prediction |
| Trivia | R | 1000 | R AUROC .805，但方向干预 high-low 仅 .089 | 检测支持、因果弱 | representation score 可预测，不代表统一可修复方向 |
| GSM8K | R | 942 | AUROC .697；high-low specific gain -.005 | 检测中等、因果不支持 | question-end R 不是主要可干预机制 |
| GSM8K | P | 115 | target-placebo gain -.037；high-low -.209 | 不支持 | 简单删除 question span 会破坏推理语义 |
| GSM8K | U | 300 CoT | high-U 错误修复 .079，low-U .328；风险随 coverage 降低 | 支持风险分层 | U 能标出难题，但重采样本身不一定修复 |

### 5.3 第三部分结论

现有结果确实显示 benchmark 之间存在机制异质性：Trivia 的 E/P 干预强，GSM8K 的 P/R 干预失败，Scientist 的 R 检测较强。但目前没有在**相同模型、相同样本抽样、相同阈值、相同四轴计算方式**下产出每个 benchmark 的 UEPR 比例表。因此可以写“不同 benchmark 对不同轴干预的响应不同”，暂时不能把某个检测方法优劣定量归因于“UEPR 类型比例分布”。

## 6. 第四部分：多关键词联合——协同、冗余与竞争

### 6.1 全因子交互图谱

Scientist grounded-cue full-factorial atlas 覆盖 449 题，其中 245 题为 likelihood error；候选关键词数主要为 2–4，共分析 1,781 个 pair。

| 关系 | pair 数 | 比例 | 论文解释 |
|---|---:|---:|---|
| mixed | 776 | 43.6% | 交互依赖背景集合，不能用固定二阶标签完全描述 |
| competition | 662 | 37.2% | 不同线索相互竞争，是最主要的纯关系 |
| redundancy | 295 | 16.6% | 多个线索可替代，单独删除已能产生修复 |
| synergy | 45 | 2.5% | 联合增益超过加性预期，但较少见 |
| pure combination | 3 | 0.17% | 单体不够、必须联合的严格协同极少 |

平均 local pair effect 为 `0.0687`，95% CI `[0.0438, 0.0936]`；平均 Banzhaf pair effect 为 `0.0794`，95% CI `[0.0634, 0.0969]`。候选集合整体必要率为 `24.5%`；265 题存在某个 repair set，但只有 24 题的最小修复集合包含多个 cue。

按属性对看，样本量较充分且区间不跨 0 的正交互包括：award×award `0.094 [0.070,0.118], n=881`，award×field `0.111 [0.051,0.177], n=165`，award×logic `0.166 [0.090,0.251], n=64`，award×position `0.108 [0.031,0.194], n=48`。其余许多属性对样本过小，不宜解释。

### 6.2 冻结确认与自由生成验证

| 验证 | n | 指标 | 结果 | 判断 |
|---|---:|---|---:|---|
| neutralization target vs matched random | 18 | target-random repair gain | .359 [.158,.585] | 强正结果，但 cohort 由 neutralization 发现，存在选择偏差 |
| physical deletion | 18 | target-random repair gain | .073 [-.176,.325] | 不显著 |
| paired free generation | 18×20/condition | joint-base correction | .397 [.314,.475] | 强正结果 |
| paired free generation | 18×20/condition | joint-preselected random | .189 [.006,.361] | 正向，区间刚好不跨 0 |
| paired free generation | 18×20/condition | joint-best single | .308 [.211,.408] | 强正结果，94.4% 题为正 |

paired-seed 条件下，正确率从 base `.128` 提升到 joint `.525`，高于 best single `.217` 和预先选择的 random set `.336`。这为“多关键词联合具有超越最佳单关键词的生成层效应”提供了直接证据。

不过，多关键词机制的 headline 应是“竞争和背景依赖普遍，严格纯协同罕见；在筛出的 multi-cue repair cohort 上，联合干预显著优于最佳单关键词”，而不是“多数错误由协同导致”。

## 7. 缺失、错误或分量不足的实验清单

### P0：投稿前必须补

| 编号 | 缺口/问题 | 为什么当前不可用 | 建议 setting | 最低交付物 |
|---|---|---|---|---|
| P0-1 | **统一的关键词 margin flip-rate** | 当前只有平均 `u`，没有按属性报告正→负翻转率；原主张用了“翻转” | 在冻结的 421/453 题上，按同一 neutralize 算子报告 base margin>0 中的 flip rate；配 matched random span 与长度/位置控制 | 总体及各属性 `n、mean Δmargin、flip%、random flip%、paired CI/p` |
| P0-2 | **重写或补强严格 binding** | 总体 binding `p=.549`，selected-vs-nonselected 也不显著；现有强结论错误 | 预注册人物交换/属性交换的 2×2 factorial；按人物分组 holdout；扩大 education/position 至每类至少 100–200 个独立人物 | person-specific interaction、cluster bootstrap CI、FDR 后结果；若仍阴性则保留阴性并降级主张 |
| P0-3 | **自然预训练频率证据** | 两个 3B 合成 fine-tune 只证明“可学得”，不能证明真实 binding 来源于预训练 | 从可检索预训练代理语料统计 person-attribute 共现/PMI；在 held-out 人物上预测 perturb/free-generation effect，控制人物流行度、属性基频和字符串频率 | 相关/回归系数、partial R²、人物级 bootstrap；与合成干预形成 observational+causal 链 |
| P0-4 | **gradient 的完整 4×3 检测矩阵** | 目前 gradient 主要只有 Scientist；不能支持“多个模型多个 benchmark 三方法” | 完全复用 paper4 的模型、manifest、fold、特征和冻结超参，补齐 4 模型×3 benchmark | exact/attention/gradient AUROC/AUPRC/BAcc 均值±seed/CI |
| P0-5 | **统一真实计算成本** | gradient 查询减少未计 backward，attention/exact 也缺 wall-clock/FLOPs/显存 | 同硬件、同 batch、预热后测端到端 latency、峰值显存、forward/backward 次数；至少 3 次重复 | accuracy–cost Pareto 表；秒/样本、tokens/s、峰值显存、forward-equivalent FLOPs |
| P0-6 | **统一 UEPR benchmark 比例矩阵** | 当前是分散的轴验证，不是同协议下的比例分布 | 对 Scientist/Trivia/GSM8K（最好再加 HaluEval/DROP/HotpotQA）使用同一模型、同一冻结 scorer、同一阈值；人工标注一个分层子集校准轴语义 | 每 benchmark 的 U/E/P/R 单标签与多标签比例、CI、inter-annotator agreement、阈值敏感性 |
| P0-7 | **UEPR 比例与检测性能的定量连接** | “类型不同导致方法表现不同”尚是推断 | 在 item-level 用 UEPR score/label 预测 exact-attention-gradient 的正确/错误与性能差；benchmark-level 仅作辅助 | 分轴 AUROC、method×axis interaction、回归/混合效应模型、held-out benchmark 验证 |
| P0-8 | **多关键词独立大样本确认** | strongest generation 结果只有筛选出的 18 题，外部效度不足 | 冻结选择规则后在新的 ≥100 个 multi-cue 候选题上确认；发现集与确认集人物不重叠 | joint vs best-single vs preselected-random 的 paired generation CI；同时报告筛选通过率 |

### P1：强烈建议补，决定论文说服力

| 编号 | 缺口/问题 | 建议 |
|---|---|---|
| P1-1 | Qwen 频率剂量跨 seed 有非单调，Llama 只有单 seed | Llama 至少补 3 seeds；对两模型用 dose 连续回归/有序趋势检验，而不是挑单点；报告 seed×dose interaction |
| P1-2 | 合成 fine-tune 的 dose 可能与总训练分布或 loss 难度混淆 | 保持总 token 数、人物出现次数、属性 B/F 总频率、句式模板完全平衡，只改变 person-attribute 配对；增加 label permutation 和 unrelated-attribute placebo |
| P1-3 | 自由生成共现可能受词面匹配和基础属性频率影响 | 加入匿名人物、同义改写、稀有/常见属性匹配、随机人物对照；报告 exact 与语义判定的一致性及人工抽检准确率 |
| P1-4 | 多模型检测矩阵缺传统 baseline | 在完全相同 fold 上加入 raw margin/entropy、max-softmax、semantic entropy、P(True)、hidden-state probe 等；报告 perturbation 相对最佳 baseline 的增量和 paired CI |
| P1-5 | 检测结果只有点估计，主表缺显著性 | 对 item/group 做 paired bootstrap，报告 attention/gradient 相对 exact 及相对 baseline 的 ΔAUROC 95% CI；跨 seed 不是样本不确定性的替代 |
| P1-6 | Scientist→multidomain 迁移在 Falcon3 很弱 | 分析 tokenizer、生成正确率、可用样本数和特征尺度；如无法修复，明确作为模型依赖性 limitation |
| P1-7 | UEPR 的 E 在 Scientist 初始 AUROC 极高但独立确认接近零 | 审查信息泄漏/同源标签；用严格 held-out evidence-completion 操作重新定义 E，避免 chosen/alternative scorer 直接编码标签 |
| P1-8 | GSM8K 的 P 删除干预方向错误 | 不应删除数学条件；改为保语义的数值/关系局部 counterfactual、reasoning-step intervention，或明确 P 不适用于该 benchmark |
| P1-9 | 多关键词分类中 mixed 占 43.6%，类别定义可能不稳定 | 报告 local 与 Banzhaf 标签一致率、阈值敏感性、null interaction 阈值、bootstrap 分类稳定性 |
| P1-10 | interaction atlas 只有 Scientist/Llama | 至少在另一模型和一个不同 benchmark 上做冻结复现；否则将结论限定为 Scientist grounded profile setting |

### P2：补充材料或稳健性实验

| 编号 | 建议 |
|---|---|
| P2-1 | neutralize、physical deletion、同义替换三种算子的效应相关与 flip 一致率 |
| P2-2 | span 长度、位置、候选数、top-k、层选择和 PCA 维数的敏感性 |
| P2-3 | generation 验证增加温度、采样数和判分器变化；报告 paired seed 结果 |
| P2-4 | 属性子类小样本表移入附录，主文只展示 n 足够且预先定义的类别 |
| P2-5 | 发布每个主表对应的冻结 config、manifest hash、fold assignment 和一键复现命令 |

## 8. 建议的论文主表结构

| 主表/图 | 内容 | 当前状态 |
|---|---|---|
| Table 1 | 数据集、模型、正确/错误样本数、平均长度 | 需从 manifests 统一汇总 |
| Table 2 | exact/attention/gradient × 4 models × 3 benchmarks | exact/attention 已有；gradient 待补 |
| Table 3 | 方法准确率—成本 Pareto | 查询数已有；真实成本待补 |
| Table 4 | 单关键词属性扰动与自由生成 binding | 数值已有；需补 flip rate，并如实显示严格 binding 阴性 |
| Table 5 | 两个 3B 模型的频率剂量效应 | 基本已有；Llama seeds 与趋势统计待补 |
| Table 6 | UEPR benchmark 比例及分轴检测性能 | 尚未形成统一矩阵 |
| Table 7 | 多关键词关系比例与联合修复 | atlas 与 n=18 确认已有；独立大样本确认待补 |
| Figure 1 | 方法流程：关键词定位→扰动轨迹→检测 | 可制作 |
| Figure 2 | accuracy–cost Pareto | 待真实成本测量 |
| Figure 3 | dose-response（模型×seed） | 数据已有大半 |
| Figure 4 | UEPR benchmark composition + method suitability | 等 P0-6/P0-7 后制作 |
| Figure 5 | interaction atlas 与 joint-vs-single paired plot | 已有结果可制作 |

## 9. 主要结果来源

- 检测统一矩阵：`runs/paper4_self_matrix_v2/evaluation/combined/summary.md`
- exact/attention/gradient 同口径：`runs/153_exact_current127_samecv_report.json`、`155_...report.json`、`156_...report.json`、`159_...report.json`
- 严格单关键词 binding：`runs/209_strict_attribute_binding_full/analysis.json`
- 自由生成：`runs/221_free_generation_person_attribute_binding/report.json`
- selected vs nonselected：`runs/224_selected_vs_nonselected_free_binding/report.json`
- 合成频率干预：`runs/217e_llama32_3b_100person_b_vs_f/report.json`、`runs/225_qwen3b_binding_mediation_chain*/report.json`
- UEPR：`runs/226_four_axis_taxonomy_audit/report.json` 及 `runs/227_*`–`238_*` confirmation reports
- 多关键词 atlas 与确认：`runs/230_scientist_factorial_interaction_atlas/report.json`、`233_confirm_multicue_repairs/report.json`、`234_paired_generation_multicue_controls/report.json`

## 10. 第一部分结果（更新版；替代第 3 节）

本节按照论文当前的证据链重新组织第一部分结果。旧第 3 节保留为实验审计记录，但不再作为正文写法。更新后的主张分为三个层次：首先验证问题中的人物属性短语是否对候选偏好具有方向性影响；随后用自由生成频率检验人物与属性之间是否存在可观察的记忆关联；最后通过受控训练频率干预验证这种关联能否因曝光频率而增强。卡片式 binding effect、loose/lexical-semantic 匹配和其他辅助指标均不进入正文主表。

### 10.1 层次一：signed perturbation 揭示候选所依赖的属性线索

对问题中的人物属性短语进行保语义、保宽度 neutralization，并以附近同长度非属性短语作为 matched control。所有效应均基于 wrong-minus-right teacher-forced margin。定义 matched-control-adjusted perturbation effect 为

\[
\Delta_{\mathrm{pert}}
=u_{\mathrm{target}}-u_{\mathrm{control}}.
\]

在当前符号约定下，\(\Delta_{\mathrm{pert}}>0\) 表示相对于普通局部改写，目标属性短语更偏向支持错误候选；\(\Delta_{\mathrm{pert}}<0\) 表示该短语更偏向支持正确候选。结果同时覆盖模型最终答错和答对的题目。

| 模型结果 | 属性所属人物 | 关键词数 | 题目数 | \(\Delta_{\mathrm{pert}}\) | 95% person-group CI |
|---|---|---:|---:|---:|---:|
| 错误 | 错误人物 | 841 | 461 | **0.368** | **[0.280, 0.401]** |
| 错误 | 正确人物 | 158 | 125 | **-0.137** | **[-0.173, -0.017]** |
| 正确 | 错误人物 | 1189 | 623 | **0.305** | **[0.235, 0.341]** |
| 正确 | 正确人物 | 221 | 187 | **-0.096** | **[-0.165, -0.030]** |

效应符号由属性的 owner 决定，而不是由模型最终是否答错决定：错误人物的属性短语在错误题和正确题中都相对支持错误候选，正确人物的属性短语在两组中都相对支持正确候选。这说明 signed perturbation 测量的不是“错误标签本身”，而是问题中局部属性线索对两个候选的方向性支持。错误题上的效应更强，但正确题同样存在，因此正文不应把统计总体限制为 erroneous responses。

这张表只支持“模型的候选偏好依赖于特定人物属性线索”。它不再承担证明人物—关键词记忆 binding 的任务；后者由自由生成频率直接检验。

### 10.2 层次二：自由生成频率直接显示人物—属性 binding

冻结 10.1 中的属性短语后，分别以属性实际所属人物（owner）和配对的另一人物为条件进行自由生成。每个人物使用 4 个 prompts、每个 prompt 采样 5 次。正文只采用严格字符串召回的 exact frequency difference：

\[
\Delta_{\mathrm{freq}}
=f(\text{attribute}\mid\text{owner})
-f(\text{attribute}\mid\text{other}).
\]

| 模型结果 | 题目数 | 关键词数 | Exact \(\Delta_{\mathrm{freq}}\) | 95% person-group CI |
|---|---:|---:|---:|---:|
| 错误 | 455 | 929 | **0.335** | **[0.297, 0.374]** |
| 正确 | 614 | 1288 | **0.319** | **[0.288, 0.352]** |

无论模型最终答对还是答错，同一属性都更容易由其 owner 触发生成，且两组区间均远离零。这为人物—属性 binding 提供了直接的生成层证据。错误题的频率差略高于正确题（0.335 vs. 0.319），但核心发现不是两组之间的差异，而是两组内部都存在稳定、显著的 owner advantage。因此，binding 是模型记忆中的普遍人物—属性关联；是否最终产生 hallucination 还取决于该关联在具体问题中与其他证据如何竞争。

10.1 与 10.2 的 overall \(n\) 不完全相同不是统计问题。10.1 要求目标短语及其 matched control 都能完成 teacher-forced neutralization；10.2 要求同一冻结短语通过自由生成的 owner/other 配对、生成完成和 exact-scoring 有效性检查。两张表回答不同问题，使用各自冻结后的可评估集合，不应为了形式统一而删除有效样本。正文可以连续呈现这两张小表，但不能把行级样本假定为完全一一对应。

### 10.3 层次三：训练频率差因果增强人物—属性偏好

为了检验 binding 是否会随训练曝光频率增强，我们构造 50 组镜像虚构人物对（100 人），控制属性 B 与匹配属性 F 的人物级训练频率差。五个 dose 为 \(0,.25,.5,.75,1\)，每个人物总训练条数固定为 40；每个模型运行 seeds 42、43、44。核心指标为

\[
\Delta_{B-F}=M_B-M_F,
\]

其中 \(M_B\) 和 \(M_F\) 分别是在 B、F 线索条件下的 wrong-minus-right candidate margin。\(\Delta_{B-F}\) 越大，表示 B 相对 F 的频率优势越强地进入候选偏好。

| 模型 | Seed | \(d=0\) | \(d=.25\) | \(d=.5\) | \(d=.75\) | \(d=1\) | Slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 42 | .022 | .028 | .171 | .346 | .286 | .338 |
|  | 43 | .024 | .050 | .073 | .096 | .199 | .159 |
|  | 44 | .039 | .086 | .149 | .156 | .210 | .165 |
|  | **Mean** | **.028** | **.055** | **.131** | **.200** | **.232** | **.221** |
| Llama-3.2-3B | 42 | -.005 | .008 | .009 | .009 | .047 | .042 |
|  | 43 | .000 | .029 | .016 | .029 | .124 | .099 |
|  | 44 | .011 | .003 | .018 | .033 | .021 | .020 |
|  | **Mean** | **.002** | **.013** | **.014** | **.024** | **.064** | **.054** |
| Ministral-3-3B | 42 | -.031 | .074 | .143 | .374 | .589 | .616 |
|  | 43 | .020 | .062 | .121 | .202 | .398 | .359 |
|  | 44 | .050 | .042 | .021 | .279 | .704 | .618 |
|  | **Mean** | **.013** | **.059** | **.095** | **.285** | **.564** | **.531** |

九个 model-seed runs 的 dose slope 全部为正。三模型的平均 slope 分别为 Qwen 0.221、Llama 0.054 和 Ministral 0.531；平均曲线均从近零的 dose-0 基线向正方向增长。效应强度存在明显模型差异：Ministral 最强，Qwen 居中，Llama 较弱。Llama 的三个 seed 虽然方向一致，但 seed-level slope 检验未达到显著（\(p=0.149\)），因此应表述为稳定的正向趋势而非强效应；Ministral 的三个 slope 均为正且平均为 \(0.531\pm0.149\)，Qwen 也在三个 seed 上复现正 slope。

该实验建立的是受控训练频率的因果证据：当人物—属性配对的相对曝光频率增加时，相同属性在线索条件下更强地改变人物候选偏好。它是自然预训练频率假说的 causal analogue，而不是对真实预训练语料频率的直接测量；论文中应避免把 synthetic fine-tuning 直接写成“证明真实预训练频率导致 hallucination”。

### 10.4 第一部分的正文结论

三个层次形成一条紧凑证据链：

1. signed perturbation 表明问题中的人物属性短语对候选偏好具有 owner-aligned 的方向性作用，而且这种作用同时存在于正确题和错误题；
2. 自由生成表明模型确实把属性更高频地与其所属人物共同召回，从而直接支持人物—属性 binding；
3. 合成 dose-response 表明增加人物—属性的相对训练频率会因果增强该属性对人物选择的影响，并在三个约 3B 模型、每模型三个 seeds 上复现正 slope。

因此，第一部分最稳妥的主张是：**模型会学习人物—属性的频率依赖关联；这些关联在问题中表现为可由 signed perturbation 测量的候选支持方向，并可能在局部证据竞争中推动系统性错误。** 当前结果不要求把所有 hallucination 都归因于 binding，也不声称自然预训练语料频率已经被直接测量。

本节新增或更新的结果来源：`runs/241_paper_keyword_reliance/report.json`、`runs/243_paper_owner_aligned_free_generation/report.json`、`runs/225_qwen3b_binding_mediation_chain*/report.json`、`runs/217e_llama32_3b_100person_b_vs_f/report.json`、`runs/246_llama32_3b_100person_b_vs_f_seed4*/report.json`、`runs/245_ministral3_3b_100person_b_vs_f_seed4*/report.json`。
## 统一 UEPR 后的 hallucination 类型比例（2026-08-20）

本节统计的是由 UEPR 签名定义的 **hallucination 类型**，不是 U/E/P/R detector 阳性率。三套数据均固定为 `NousResearch/Meta-Llama-3.1-8B-Instruct`，并在每个 benchmark 内按 key 对齐到同一条 baseline generation 和同一个 correctness label；会重新生成 baseline answer 的 TriviaQA closed-book 结果不参与合并。Scientist 使用全部 1,076 条（453 个错误），TriviaQA 使用固定 balanced-1,000（500 个错误），GSM8K 使用原 CoT baseline 上同时具有有效 U score 的固定 balanced-300（150 个错误）。分母是 baseline hallucinations，而不是 UEPR 缓存覆盖数。

统一操作定义如下。`U-high/U-low` 使用固定 baseline 全体样本 U score 的 70%/30% inclusive quantile；`R correct-like` 固定为 group-OOF error probability `<0.5`。知识缺失为 `U-high + E-high + 补证据后正向响应`；无依据但自信的编造为 `非 U-high + E-high + 补证据后正向响应`；上下文诱导/局部错误依赖为 `P-high + target intervention 相对 matched placebo 的 specific gain >0`；推理不稳定为 `U-high`；稳定自洽错误为 `U-low + R correct-like`。类型是多标签，不能横向求和为 100%。

\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
幻觉类型 & Scientist ($n_h=453$) & TriviaQA ($n_h=500$) & GSM8K ($n_h=150$) \\
\midrule
知识缺失 & 93 (20.5\%) & 50 (10.0\%) & -- \\
无依据但自信的编造 & 146 (32.2\%) & 25 (5.0\%) & -- \\
上下文诱导/局部错误依赖 & 38 (8.4\%) & 217 (43.4\%) & 21 (14.0\%)$^{\dagger}$ \\
推理不稳定 & 159 (35.1\%) & 310 (62.0\%) & 89 (59.3\%) \\
稳定自洽错误 & 29 (6.4\%) & 85 (17.0\%) & 27 (18.0\%) \\
多因素混合 & 121 (26.7\%) & 220 (44.0\%) & 20 (13.3\%) \\
未归类 & 122 (26.9\%) & 40 (8.0\%) & 33 (22.0\%) \\
\bottomrule
\end{tabular}
\caption{统一 scoring 下、以 baseline hallucinations 为分母的多标签类型比例。$^{\dagger}$GSM8K 的 P-target 删除相对 placebo 在总体上未通过选择性验证，因此 14.0\% 只能称为逐样本 operational candidate coverage，不能称为已确认的上下文诱导机制比例。}
\label{tab:unified-hallucination-types}
\end{table}

该表支持的最直接结论是：Scientist 的 evidence-responsive factual-error signatures 明显多于 TriviaQA；TriviaQA 的局部上下文依赖候选明显多于 Scientist；GSM8K 中 59.3\% 的错误为 U-high，同时有 18.0\% 属于 U-low、R correct-like 的稳定自洽候选。它**不支持**“GSM8K 的推理不稳定比例高于 TriviaQA”，因为 TriviaQA 的离散 semantic entropy 在 70% cutoff 上存在大量并列，U-high 覆盖 62.0% 的错误。它也尚不能把检测 AUROC 差异定量归因于这些比例：类型为多标签、E 对 GSM8K 不适用，而且 GSM8K 的 P 机制验证失败。下一步若要做性能归因，应在这个同源 manifest 上比较 detector 在各类型内的 AUROC/recall，并使用 overlap-aware regression，而不能回到 `247/248` 那种跨 generation 拼接。

可复现文件：`249_unified_hallucination_type_scoring.py`；逐样本标签、Wilson 95% CI、overlap pattern 和表格分别位于 `runs/249_unified_hallucination_types/items.jsonl`、`report.json` 与 `table.tex`。`runs/247_uepr_conditioned_detector_audit` 和 `runs/248_uepr_mixture_standardization` 混用了不同 baseline generations/labels，结果作废，不得用于论文。
