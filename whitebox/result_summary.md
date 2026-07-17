# White-box Hallucination Detection：实验原理与结果汇总

> 本文档汇总 Qwen/Qwen2.5-7B-Instruct 在二选一 shortcut-hallucination benchmark 上的白盒检测实验。除特别说明外，最终测试指标均来自固定随机种子 42 的分层训练/测试划分：训练集 375 条，测试集 125 条，其中 hallucination 39 条、正确回答 86 条。

## 1. 研究问题与核心结论

实验研究两个问题：

1. 能否利用模型内部的 logit、attention、spectral 和 gradient 信号检测错误回答？
2. 能否定位模型依赖的 constraint/shortcut span，并让角色机制真正参与检测，而不只是事后解释？

核心结果：

- 单一 constraint attribution 基线区分力弱：前 100 条 AUROC 仅 0.557。
- 多特征监督 detector v1/v2 的 held-out AUROC 为 0.761/0.767。
- 训练集干预产生 span 角色弱标签后，weakly supervised 版本达到 AUROC 0.852、F1 0.622。
- role-mediated v3 combined AUROC 达到 **0.867**，F1 达到 **0.701**；39 个 hallucination 中检出 34 个。
- v3 role-only AUROC 为 0.778；combined 比 residual-only 的 AUROC 高 0.040。
- 当前 constraint 机制仍未学好：beta_constraint=0.0028，测试集 constraint explanation coverage 为 0%。目前最强证据是“预测 shortcut evidence 与 hallucination 显著相关并实质性驱动检测器预测”，尚不能声称具体 shortcut span 已被因果验证为基座模型答错的原因。

## 2. 数据、标签与评价协议

### 2.1 数据和模型

- 数据：question_and_result.json，共 500 条。
- 基座模型：Qwen/Qwen2.5-7B-Instruct。
- 决策：比较下一 token 为 1 和 2 的 log probability。
- 操作性标签：chosen != gold 时记为 hallucination。

这里把“答错”作为 benchmark 中的 hallucination 标签。若研究定义需要区分事实性幻觉、推理错误和指令违背，必须进一步细化标签。

### 2.2 划分与防泄漏

- test_size=0.25，seed=42，按 hallucination 标签分层划分。
- span 干预只在训练集执行。
- 测试预测不使用干预，也不使用 gold answer。
- 角色模型用于训练下游 detector 的训练集角色概率由 item-level OOF 产生，避免同一题的 span 同时用于角色模型拟合和下游训练特征生成。
- 分类阈值仅在训练集 OOF 预测上选择，再固定应用于测试集。

### 2.3 指标

- **AUROC**：检测分数对正负样本的整体排序能力；0.5 约等于随机。
- **AUPRC**：precision-recall 曲线下面积，对类别不平衡更敏感。测试集正类率 0.312，随机基线约 0.312。
- **Precision**：预测为 hallucination 的样本中实际答错的比例。
- **Recall**：实际 hallucination 中被检出的比例。
- **F1**：precision 与 recall 的调和平均。
- **Accuracy**：全部样本上的分类正确率，受类别比例与阈值影响。

## 3. 方法演进与实验原理

### 3.1 单一 attribution：whitebox_run.py

先用词汇规则定位 constraint sentence，再对模型选择 1/2 的 contrastive logit 做 gradient × input attribution：

\[
\text{constraint_share}=
\frac{\sum_{t\in C}\lVert\nabla_{e_t}\Delta\ell\odot e_t\rVert}
{\sum_{t\in scenario}\lVert\nabla_{e_t}\Delta\ell\odot e_t\rVert}.
\]

模型置信 margin 足够高但 constraint share 很低时判为 hallucination。前 100 条结果：

- hallucination：31/100；
- AUROC（-constraint_share）：0.557；
- confident subset AUROC：0.548；
- 默认阈值 TP=0、FP=0、FN=31、F1=0。

结论：单一 attribution share 和人工固定阈值不足以稳定检测。

### 3.2 Multi-feature v1：whitebox_detector.py

v1 提取多组特征，再训练 StandardScaler + L1 LogisticRegressionCV：

- logit：答案 log probability、perplexity、entropy、top-2 margin；
- attention：lookback ratio、constraint share、prompt attention entropy；
- spectral：attention Laplacian top-k eigenvalues；
- gradient：最后若干 transformer 层的 answer-NLL gradient norm。

分类器只在训练 split 拟合，测试 split 只预测。

### 3.3 Token-indexed v2：whitebox_detector_v2.py

v2 显式比较 constraint 与 shortcut：

- ROLE-LAP：分别池化 constraint/shortcut token 的未排序 Laplacian diagonal；
- TOKENFLOW：答案 token 对 constraint 和 shortcut 的相对 attention：

\[
\rho_t=\frac{A(t\rightarrow C)}
{A(t\rightarrow C)+A(t\rightarrow S)};
\]

- LAPEIG：保留 sorted top-k spectral baseline；
- LOGIT：答案置信和不确定性特征。

v2 取消 gradient backward，速度更快并支持逐答案 token attribution。但本 benchmark 的答案只是单字符 1/2，TOKENFLOW 优势未充分发挥；shortcut span 仍由启发式规则指定。

完整数据中，正常与 hallucination 的 rho_mean 无显著差异：正常 0.622，hallucination 0.631；Mann–Whitney p=0.287，rank-biserial effect size 约 0.059。因此 v2 的检测力主要来自 Laplacian/logit，而不是直接的 constraint-vs-shortcut attention ratio。

### 3.4 Weakly supervised detector

该版本不再人工指定 span 角色，而由训练集行为干预产生弱监督标签。

#### A. 候选 span

默认按 clause 切分 question。该步骤只提出候选区域，不分配 constraint、shortcut 或 irrelevant 角色。

#### B. 原始白盒特征

每个 span 提取 decision attention mass、长度归一化 density、最大 head attention、token-indexed Laplacian、contrastive gradient × input，以及 span 长度和相对位置。每个 item 还提取 choice margin、choice entropy、选项位置、prompt 长度、候选 span 数、decision-row attention entropy 和 sorted top-10 LapEig。

#### C. 干预产生角色伪标签

定义 gold margin：

\[
g(x)=\log P(y^*\mid x)-\log P(y^{other}\mid x).
\]

对 span s 执行 delete、neutralize、mask：

\[
\Delta(s,I)=g(x)-g(I_s(x)).
\]

- Δ>0：移除 span 后 gold margin 降低，偏 constraint-like；
- Δ<0：移除后 gold margin 提高，偏 shortcut-like；
- Δ≈0：偏 irrelevant-like。

三种干预用 median 聚合，并根据方向一致性、MAD 与贡献强度产生 soft role distribution 和 reliability。干预不是最终预测特征，而是角色模型的弱监督目标；测试时不做干预。

#### D. 角色模型与 hallucination head

角色模型学习：

\[
f_{role}(\text{original span features})
\rightarrow P(C),P(S),P(I).
\]

角色概率被聚合为 constraint/shortcut max、mean、count、role entropy，以及角色加权 attention/gradient。最终 hallucination head 使用“全局特征 + 角色汇总特征”。

### 3.5 Role-mediated v3

普通 weakly supervised head 可能绕过角色特征。v3 将角色机制写成显式、单调、可分解的组成：

\[
E^S_i=P(S_i)U_i/N,\qquad E^C_i=P(C_i)U_i/N,
\]

\[
L_{role}=b+\beta_S\sum_iE^S_i-\beta_C\sum_iE^C_i,
\qquad \beta_S,\beta_C\ge0.
\]

每个 span 因而有精确 signed contribution：shortcut 只能增加风险，constraint 只能降低风险。

全局 logit/attention/LapEig 进入独立 residual channel：

\[
L_{raw}=L_{role}+c\tanh(L_{residual}).
\]

当前 residual_cap c=1.0，全局通道不能无限覆盖角色机制。最后只做 bias/temperature calibration。

## 4. 实际使用的特征

### Logit / decision

- choice_margin_abs = |log P(1)-log P(2)|
- choice_entropy
- chosen_is_a

### Attention

- decision_attn_entropy_*
- attn_mass_*：决策 attention 落在 span 的总量
- attn_density_*：按 span token 数归一化
- attn_headmax_*：最关注该 span 的 head

星号表示 early/mid/late/mean/max/std 等跨层汇总。

### Spectral

- lap_token_*：保留 token 身份后对 span 池化的 Laplacian diagonal
- lapeig0_* 至 lapeig9_*：每层 sorted top-10 LapEig 的跨层汇总

### Gradient

- grad_norm_sum
- grad_norm_density
- grad_signed_sum
- grad_signed_density

gradient 目标是模型自身选择相对另一选项的 contrastive logit，不需要测试 gold。

### 结构与角色汇总

- span words/tokens/characters、相对起止位置和长度；
- constraint/shortcut/irrelevant 概率的 max/mean/count；
- shortcut-minus-constraint；
- role-weighted attention/gradient；
- role entropy。

## 5. 干预与角色预测质量

| 伪标签角色 | 数量 | 平均 reliability | 中位 reliability | reliability≥0.2 | 三干预一致率 |
|---|---:|---:|---:|---:|---:|
| Constraint | 470 | 0.367 | 0.264 | 58.3% | 94.5% |
| Shortcut | 380 | 0.392 | 0.368 | 63.9% | 91.3% |
| Irrelevant | 6 | 0.300 | 0.311 | 50.0% | 50.0% |

平均绝对干预贡献：constraint 6.83，shortcut 5.94。shortcut 伪标签的中位可靠度和通过率略高，但 constraint 的干预一致率与贡献幅度并不更差。

对 reliability≥0.2 的 520 个训练 span，角色 OOF 硬预测与干预 hard pseudo-label 的一致率为 48.3%，按 reliability 加权为 50.2%：

| 干预伪标签 | 预测 Constraint | 预测 Shortcut | 预测 Irrelevant |
|---|---:|---:|---:|
| Constraint | 130 | 76 | 68 |
| Shortcut | 69 | 121 | 53 |
| Irrelevant | 2 | 1 | 0 |

这不是真实角色 accuracy：训练目标是 soft label，hard pseudo-label 本身有噪声。但它提示单 span 定位仍有限，尤其 irrelevant 类极少。高 detector AUROC 不等于角色定位准确。

## 6. 统一测试结果

### 6.1 同一 held-out protocol

| 方法 | 有效总数 | Test n | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Multi-feature v1 | 500 | 125 | 0.761 | 0.513 | 0.712 | 0.548 | 0.436 | 0.486 | [[72,14],[22,17]] |
| Token-indexed v2 | 499 | 125 | 0.767 | 0.528 | 0.680 | 0.481 | 0.333 | 0.394 | [[72,14],[26,13]] |
| Weakly supervised | 500 | 125 | 0.852 | **0.757** | **0.776** | **0.657** | 0.590 | 0.622 | [[74,12],[16,23]] |
| Role-mediated v3 combined | 500 | 125 | **0.867** | 0.734 | 0.768 | 0.586 | **0.872** | **0.701** | [[62,24],[5,34]] |

- v3 的 AUROC、recall、F1 最佳，仅漏检 5/39 个 hallucination。
- weakly supervised 的 AUPRC、accuracy、precision 更高，高分样本纯度更好。
- v3 并非所有指标全面最好；版本选择取决于漏检与误报成本。
- v2 有 1 条无法构造合法 shortcut span，因此有效总数为 499。

### 6.2 v3 通道消融

| 通道 | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---|
| Role only | 0.778 | 0.647 | 0.664 | 0.475 | 0.718 | 0.571 | [[55,31],[11,28]] |
| Residual only | 0.827 | 0.622 | 0.448 | 0.361 | 1.000 | 0.531 | [[17,69],[0,39]] |
| Combined | **0.867** | **0.734** | **0.768** | **0.586** | 0.872 | **0.701** | [[62,24],[5,34]] |

\[
\Delta AUROC=0.867-0.827=0.040.
\]

已有 paired bootstrap 分析的 95% CI 约为 [0.008,0.079]，约 0.6% 重采样差值不大于 0，支持角色通道提供 residual 之外的增量排序信息。

## 7. v3 Shortcut 机制分析

### 7.1 组间差异

| 指标 | 正确回答 (n=86) | Hallucination (n=39) |
|---|---:|---:|
| Shortcut evidence 均值 | 0.150 | **0.240** |
| Shortcut evidence 中位数 | 0.131 | **0.223** |
| 最大 shortcut probability | 0.408 | **0.570** |
| 最大 shortcut logit contribution | 0.383 | **0.600** |

- 均值差 0.090；
- bootstrap 95% CI 约 [0.057,0.125]；
- Cohen's d 约 1.13；
- Mann–Whitney p≈7.1×10⁻⁷；
- shortcut evidence 单独作为分数的 AUROC 为 0.778，与 role-only AUROC 一致。

严格结论：hallucination 与更高的**预测 shortcut reliance**显著相关。

### 7.2 Dose–response

| Shortcut evidence 四分位 | n | Hallucination rate |
|---|---:|---:|
| 最低 25% | 32 | 6.3% |
| 第二组 | 31 | 25.8% |
| 第三组 | 31 | 29.0% |
| 最高 25% | 31 | 64.5% |

hallucination prevalence 随 shortcut evidence 单调上升。

### 7.3 Shortcut explanation 风险分层

| | 检出 shortcut | 未检出 shortcut |
|---|---:|---:|
| Hallucination | 29 | 10 |
| 正确回答 | 29 | 57 |

- hallucination 中检出率：29/39=74.4%；
- 正确回答中未检出率：57/86=66.3%；
- P(H|detected)=50.0%，P(H|not detected)=14.9%；
- odds ratio 5.70，95% CI [2.44,13.29]；
- Fisher exact p≈3.6×10⁻⁵。

Shortcut detection 是强风险标志，但既非充分条件也非必要条件。

### 7.4 显式角色公式

v3 学到：

\[
L_{role}=-0.744+3.634S-0.00281C,
\]

\[
L_{final}=\frac{L_{role}+R_{capped}+0.0267}{0.9004}.
\]

因此：

\[
\frac{\partial L_{final}}{\partial S}
=\frac{3.634}{0.9004}\approx4.04.
\]

shortcut evidence 每增加 0.1，hallucination log-odds 增加约 0.404，odds 约乘以 1.50。这是结构保证的单调贡献，不是事后可视化。

constraint 系数几乎为零。解释覆盖率：

- shortcut resolved：46.4%；
- constraint resolved：0%；
- distinct pair resolved：0%。

constraint coverage 为 0 不代表角色头从不输出 constraint probability；主要原因是 beta_constraint 太小，constraint signed contribution 无法通过解释阈值。

### 7.5 Shortcut 置零反事实消融

保持 residual、bias、temperature 不变，仅令 S=0：

| 模型 | AUROC | AUPRC | Recall |
|---|---:|---:|---:|
| Combined | **0.867** | **0.734** | **0.872** |
| Shortcut evidence 置零 | 0.827 | 0.622 | 0.256 |

- 完整模型 flag 58 条，置零后只 flag 17 条；
- 41 条因 shortcut contribution 才跨过阈值，其中 24 条是真 hallucination；
- 58 个 positive predictions 中 70.7% 为 shortcut-decision-critical；
- 34 个 true positives 中 24 个（70.6%）依靠 shortcut evidence 跨过阈值。

### 7.6 控制 residual 后的增量预测价值

补充逻辑回归：

\[
H\sim\text{residual logit}+\text{shortcut evidence}.
\]

已有分析显示，控制 residual 后：

- shortcut evidence 每增加 0.1，hallucination odds 约乘以 2.41；
- 95% CI [1.32,4.41]，p=0.004；
- 相对 residual-only 的 likelihood-ratio test：p=0.0017。

## 8. 为什么 weak supervision / v3 更好

1. **行为监督取代角色猜测**：通过移除 span 后 gold margin 的变化定义功能角色。
2. **多干预更稳健**：delete、neutralize、mask 的 median 和一致性降低单一操作偏差。
3. **Soft label + reliability**：不确定 span 不被强制硬分类，低可靠样本被降权或过滤。
4. **局部与全局结合**：角色信号回答“用了什么”，global signal 描述整体决策状态。
5. **OOF 防泄漏**：下游模型看到角色模型对未参与该折训练 item 的预测。
6. **训练集阈值校准**：避免固定 0.5 或人工阈值失配。
7. **v3 强制角色中介**：residual 被 capped，每个 span 有可加和 signed contribution。

但高 detector AUROC 不自动证明角色定位准确。当前 hard pseudo-label OOF 一致率有限，irrelevant 类极少，性能可能来自连续角色聚合、global residual 和数据结构共同作用。

## 9. 可声称与不可声称的结论

### 当前结果支持

- White-box internal signals 能有效预测该 benchmark 上的模型错误。
- 弱监督角色信息显著改善 held-out 检测性能。
- 预测 shortcut evidence 与 hallucination 强相关并呈 dose–response。
- shortcut evidence 对 v3 最终 logit 有显式、单调、可分解的贡献。
- 控制 residual 后，shortcut evidence 仍有增量预测信息。

### 当前结果尚不支持

- 不能声称真实 span 角色定位 accuracy 很高；尚无人工 span gold annotation。
- 不能声称 constraint protection 已成功建模；beta_constraint≈0、coverage=0。
- 不能声称具体 shortcut 因果导致基座模型答错；本次 behavioral_explanation_audit=null。
- 不能直接声称跨数据集泛化；当前训练测试来自同一数据集随机划分。
- 不能把所有答错都无条件解释为事实性 hallucination。

适合汇报的严格表述：

> Predicted shortcut evidence statistically predicts hallucination and materially mediates the detector's prediction. Behavioral and cross-domain validation are still required before claiming that the localized shortcut causally drives the base model's erroneous answer.

## 10. 下一步实验

1. 启用 --intervene-test：审计删除预测 shortcut 是否提高 gold margin、删除预测 constraint 是否降低 gold margin，并与随机 span 对照。
2. 人工标注 100–200 个 span，报告角色模型 macro-F1 和逐类 precision/recall。
3. 调节 deadzone 或加入无关 span，解决 irrelevant 类只有 6 条的问题。
4. 对 residual cap、role/usage/contribution thresholds 做训练集内敏感性分析。
5. 固定训练模型，在 shuffled_prepend_names_question.json 和 shuffled_prepend_profiles_question.json 上做不重新训练的 zero-shot evaluation。
6. 报告多随机种子或交叉验证均值与置信区间，降低单次 split 偶然性。

## 11. 结果与复现文件

- 单一 attribution：whitebox_results_2.5_7b_rl.jsonl
- Multi-feature v1：whitebox_detector_results.jsonl
- Token-indexed v2：whitebox_detector_v2_2.5_7b_rl.jsonl
- Weakly supervised：weakbox_output/summary.json、predictions.jsonl
- Role-mediated v3：role_mediated_output/summary.json、predictions.jsonl
- v3 部署模型：role_mediated_output/role_mediated_bundle.joblib

summary 明确记录：测试预测不使用 intervention 或 gold；训练 span role 来自 intervention soft labels；v3 span contribution 可精确加和到 role logit。
