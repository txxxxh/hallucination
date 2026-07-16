# SKID-Fixed：基于双边事实 Probe 与反事实干预的幻觉检测

## 1. 实验目标

本实验研究如何在不知道正确答案（gold answer）的情况下，仅通过黑盒 API 调用，检测语言模型在 ScientistQA 双候选题目中是否出现了由人物先验或其他 shortcut 导致的错误。

这里需要区分两个概念：

- **答案错误标签**：模型的原始答案是否与数据集 gold 不一致，只在实验结束后用于评价。
- **SKID 检测目标**：模型是否表现出“自身事实知识与选择矛盾”或“没有按照决定性约束进行推理”的行为。

因此，SKID 检测的是一种具体的错误机制，而不保证覆盖所有类型的错误。模型即使答错，也可能确实对约束敏感，只是理解或推理方向错误。

## 2. 数据与模型

- 数据集：ScientistQA
- 样本数：500
- 受测模型：`qwen3.7-max`
- 干预生成模型：`qwen3.7-max`
- API：DashScope OpenAI-compatible Chat Completions
- Thinking：关闭
- 每个变体采样次数：1
- 运行错误：0
- 实现文件：`claude_method/skid_fixed.py`
- 结果文件：`claude_method/qwen_skid_fixed_results_3.7.jsonl`

Gold answer 不进入提示词和检测规则，只在所有预测完成后用于计算 TP、FP、TN 和 FN。

## 3. 方法概览

每道题先让受测模型回答原题，再围绕题目中的决定性约束生成多种干预版本，并观察答案是否发生变化。

核心假设是：

> 如果模型真正依据决定性约束作答，那么答案应当对约束含义的变化敏感，而不应主要受人物名气、选项位置或表面表达影响。

流程可以概括为：

```text
原题回答
   │
   ├── 双边闭卷事实 Probe
   ├── 约束极性翻转（negation）
   ├── 删除约束（ablation）
   ├── 强调约束（emphasis）
   ├── 等义改写（paraphrase）
   └── 交换选项顺序（swap）
            │
            ▼
       组合行为证据
            │
            ▼
     hallucination / negative
```

## 4. 干预信号

### 4.1 `probe_violation`

把原题的决定性人物事实改写成独立的闭卷 Yes/No 问题，并询问模型所选择的候选人。

如果模型的 Probe 回答表明该候选人违反原题约束，则：

```text
probe_violation = True
```

例如，原题要求寻找“从未获得某奖的人”，模型选择 A，但在独立 Probe 中又回答 A 获得过该奖。这说明模型的选择与其自身可访问的事实知识发生冲突。

### 4.2 `probe_two_sided`

只询问被选择候选人容易受到 Yes bias 影响。例如，模型可能对两个著名科学家都回答 Yes，此时单个 Yes 并不能区分候选人。

因此，SKID-Fixed 对两个候选人分别进行同一个 Probe。只有满足以下条件时，Probe 才具有区分力：

1. 所选候选人的 Probe 答案表明其违反约束；
2. 另一个候选人的 Probe 答案不表明其违反约束。

此时：

```text
probe_two_sided = True
```

Yes/Yes、No/No 或无法解析的 Probe 不作为双边知识冲突证据。

### 4.3 `neg_invariant`

把决定性约束的极性翻转，例如：

```text
从未获得 X  →  获得过 X
```

如果模型仍选择原候选人，则：

```text
neg_invariant = True
```

这说明答案可能没有随约束的语义方向协变，即约束对答案呈现因果惰性。

### 4.4 `abl_invariant`

从问题中删除或中和决定性约束。如果模型仍选择原候选人，则：

```text
abl_invariant = True
```

Ablation 单独使用时噪声较大，因为模型可能在缺少约束后依靠先验碰巧选对。因此，本版本不允许 `abl_invariant` 单独触发检测，只把它作为强双边 Probe 之后的补充证据。

### 4.5 `emph_rescue`

保持约束含义不变，只把约束改写得更加醒目。如果模型改变答案，则：

```text
emph_rescue = True
```

它表示模型原本可能受到人物先验或显著性不足的影响，而强调约束后才重新注意到约束。

### 4.6 辅助诊断信号

- `para_flip`：等义改写后答案变化，反映模型对表面表达敏感。
- `swap_flip`：交换选项顺序并映射回原候选人后，模型选择的实体发生变化，反映位置偏差。

这两个信号保留在结果中用于诊断，但不直接参与当前最终检测规则。

## 5. 最终检测规则

SKID-Fixed 使用两个互补分支：

```text
knowledge_contradiction =
    probe_two_sided
    AND (neg_invariant OR abl_invariant)

salience_rescue =
    neg_invariant
    AND emph_rescue

predicted_hallucination =
    knowledge_contradiction OR salience_rescue
```

完整表达式为：

```text
(probe_two_sided AND (neg_invariant OR abl_invariant))
OR
(neg_invariant AND emph_rescue)
```

### 分支一：知识矛盾

双边 Probe 表明模型自身知识可以区分两个候选人，而且其选择与该知识矛盾；与此同时，negation 或 ablation 没有影响原选择。这是“模型知道相关事实，却没有在原题中使用”的证据。

### 分支二：显著性救援

模型对约束极性翻转不敏感，却会因为同一约束被强调而改变答案。这说明模型更可能对约束的显著性敏感，而不是对约束的语义内容敏感。该分支不依赖人物事实 Probe。

原有加权分数仍以 `weighted_score` 保存在结果中，便于审计，但不参与最终 flag。当前 `score` 是最终规则产生的 0/1 值，阈值参数不影响判定。

## 6. 主要实验结果

受测模型在 500 道题中答错 117 道：

```text
原始错误率：117/500 = 23.4%
原始正确率：383/500 = 76.6%
```

SKID-Fixed 的混淆矩阵为：

| | 实际错误 | 实际正确 |
|---|---:|---:|
| 检测为 hallucination | TP = 82 | FP = 10 |
| 检测为 negative | FN = 35 | TN = 373 |

总体指标：

| 指标 | 数值 |
|---|---:|
| Precision | 89.1% |
| Recall | 70.1% |
| F1 | 78.5% |
| Accuracy | 91.0% |
| Specificity | 97.4% |

这意味着：被系统标记为 hallucination 的 92 道题中，有 82 道确实答错；全部 117 道错误中，有 82 道被检出。

## 7. 两个分支的贡献

### 7.1 只使用知识矛盾分支

```text
TP = 71
FP = 4
FN = 46
TN = 379
Precision = 94.7%
Recall = 60.7%
F1 = 74.0%
```

该分支 Precision 很高，是新版本减少 FP 的主要原因。

### 7.2 加入显著性救援分支

显著性分支在知识矛盾分支之外额外检出：

```text
新增 TP = 11
新增 FP = 6
```

最终 Recall 从 60.7% 提升到 70.1%，F1 从 74.0% 提升到 78.5%，但 Precision 从 94.7% 降到 89.1%。这体现了覆盖率和误报率之间的取舍。

### 7.3 Ablation 的受控贡献

如果知识分支只允许 `probe_two_sided AND neg_invariant`，结果为：

```text
TP = 67
FP = 3
```

允许强双边 Probe 与 `abl_invariant` 配合后，知识分支变为：

```text
TP = 71
FP = 4
```

即受控使用 ablation 增加 4 个 TP、1 个 FP。Ablation 在这里是补充证据，而不是独立检测器。

## 8. 与旧规则的公平比较

旧版本使用：

```text
probe_violation AND neg_invariant
```

旧结果文件直接报告：

```text
TP = 88, FP = 35, TN = 337, FN = 40
Precision = 71.5%
Recall = 68.8%
F1 = 70.1%
```

但两次 API 运行并非完全确定：两次运行有 99/500 道原始答案不同，旧运行答错 128 道，新运行答错 117 道。因此，不能只比较两个结果文件来评价规则。

在新运行的同一批 500 条 evidence trace 上重新应用两个规则，得到：

| 同一批 trace | Precision | Recall | F1 |
|---|---:|---:|---:|
| 旧规则 | 81.3% | 63.2% | 71.2% |
| SKID-Fixed | 89.1% | 70.1% | 78.5% |

因此，在控制模型回答和干预结果后，新规则仍然同时改善 Precision、Recall 和 F1。主要改善来自双边 Probe 对 Yes bias 和非区分性 Probe 的过滤。

## 9. False Negative 分析

SKID-Fixed 共有 35 个 FN。Probe 分布如下：

| Probe 形态 | FN 数量 |
|---|---:|
| No/No | 25 |
| Yes/Yes | 5 |
| 两个答案不同 | 5 |

30/35（85.7%）的 FN 来自两个候选人的 Probe 相同。相同 Probe 无法确认哪一个候选人违反约束，因此知识分支主动拒绝使用这类证据。

FN 中同时有：

- 28/35 的 `neg_invariant=True`；
- 29/35 的 `abl_invariant=True`；
- 32/35 的 `emph_rescue=False`。

这说明许多漏检样本虽然表现出约束不敏感，但 Probe 无法区分候选人，而且简单强调也不能使模型改变答案。可能原因包括模型缺少相关人物知识、事实记忆错误、人物先验过强，以及 Probe 只覆盖了复合约束中的非区分性子条件。

另外，少量 FN 的答案虽然错误，却会随 negation 和 ablation 改变。这类模型可能使用了约束，但错误理解了约束或使用了错误的推理方向，不属于典型的“约束完全惰性”错误。

## 10. False Positive 分析

最终只有 10 个 FP。主要来源不是双边知识分支，而是显著性救援分支：在知识分支之外新增的 6 个 FP 表明，一部分正确答案也会因为强调而发生不稳定变化。

因此：

- 如果目标是高可信报警，可以只使用知识矛盾分支，Precision 为 94.7%；
- 如果目标是提高错误覆盖率，可以使用当前双分支规则，Recall 为 70.1%，Precision 为 89.1%。

## 11. 方法限制

1. **检测机制不等于所有答案错误。** 当前 gold 标签把所有错误都视为 hallucination，但 SKID-Fixed 主要检测知识矛盾、约束惰性和显著性 shortcut，无法覆盖所有错误推理。
2. **依赖可区分的事实 Probe。** 双 Yes、双 No 和无法解析的 Probe 会降低 Recall。
3. **Probe 可能本身出错。** 模型在独立事实问题上的回答不一定正确，双边设计只能降低而不能消除这种误差。
4. **干预生成质量会影响结果。** Negation 必须真正翻转约束含义，paraphrase 和 emphasis 必须保持原语义，否则测到的可能只是提示扰动。
5. **单次采样存在不稳定性。** 即使 temperature 为 0，不同 API 运行仍可能产生不同答案。正式比较规则时应固定同一批 evidence trace，或增加重复采样。
6. **当前 score 是二值。** 因此不应把基于 `score` 的 AUROC 描述为连续分数 AUROC；主要报告应使用 Precision、Recall、F1 和混淆矩阵。

## 12. 对 Real-LifeQA 的适用性

ScientistQA 的双边 Probe 依赖可脱离上下文核验的人物事实，不能直接迁移到 Real-LifeQA。Real-LifeQA 更多依赖物理、空间、程序和媒介前置条件，这些条件通常只有结合完整场景才有意义。

可以迁移的是反事实协变思想，例如 negation、ablation 和 emphasis；不能直接迁移的是当前的双边人物事实 Probe。Real-LifeQA 需要先把场景拆成可靠的原子条件，再验证答案是否沿逻辑预期方向变化。无法生成可靠反事实的样本应输出 `abstain`，而不是强制判断。

## 13. 汇报结论

SKID-Fixed 的核心改进是把“单候选 Probe 冲突”改为“可区分的双边 Probe 冲突”，并增加不依赖 Probe 的显著性救援分支。双边 Probe 显著减少了由 Yes bias 和相同 Probe 答案造成的 FP；显著性分支则补回了一部分 Probe 无法覆盖的错误。

在 500 道 ScientistQA 上，SKID-Fixed 达到 89.1% Precision、70.1% Recall 和 78.5% F1。该结果表明，黑盒事实 Probe 与反事实干预的组合能够以较高精度识别一部分 shortcut 型错误，但当前方法仍依赖 Probe 的区分能力，并不等同于通用的答案正确性检测器。
