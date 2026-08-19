# 跨 benchmark 错误类型与方法适用性

> 日期：2026-08-19
> 说明：这是 `8_18日汇总.md` 的跨 benchmark 补充。类型采用多标签；只有 detector 表型的称为候选，对应干预优于 placebo 且具有选择性响应的才称为机制确认。

## 类型口径

当前先不讨论缺少明确实际情境对应的“内部异常”。保留的主要类型为：证据缺失、无依据编造、上下文诱导/干扰、推理不稳定和稳定自洽错误。U-high 应称为“推理不稳定型”，不能直接称为“推理步骤错误”；后者要求定位并修正具体错误步骤。

## 跨 benchmark 结果

Scientist 主实验为 names-only。453 个 names 错误中，E-positive 428（94.5%），U-high 且 E-positive 154（34.0%），P-high 67（14.8%），U-low、R correct-like 且 E-positive 28（6.2%）。这些比例可重叠。names 相比 profiles 出现更多证据缺失和稳定自洽候选在方向上合理，但 Scientist 的 E 近乎饱和，且 profile completion 会破坏约32.7%的原正确候选偏好，所以不能把全部错误归为证据缺失。

TriviaQA closed-book 的402个错误中，E-positive／证据响应候选为134（33.3%）；在 E-positive 且原 margin 错误的109条中，补证据修复37条（33.9%），E-nonpositive 对应约10.1%。Context-generated 的500个错误中，P-strong／上下文诱导为124（24.8%）；目标删除修复率11.7%，placebo 3.2%。U-low 且 R correct-like 的稳定自洽候选为85/500（17.0%），但 R patch 没有确认共同修复方向。

GSM8K balanced CoT pilot 的150个错误中，U-high／推理不稳定型为89（59.3%），U-low 为61（40.7%），U-low 且 R correct-like 的稳定自洽候选为27（18.0%）。P detector 曾定位115/471（24.4%）个 span 候选，但独立删除没有通过特异性验证；R 可解码错误但共享方向 patch 不修复。

DROP 的证据已位于 passage 中，因此“证据选择错误”归入更宽泛的上下文诱导/干扰型。DROP-1000 的 perturbation/hidden detector grouped OOF AUROC 为0.922。在500个错误中，365（73.0%）至少存在一个 mean-neutralization 后相对削弱错误答案的正向 passage span；正确样本为166/500（33.2%），差异39.8 pp，bootstrap 95% CI `[34.0,45.6]`。若要求 top span 占全部正效应至少50%，错误为22.4%，正确为9.8%。73.0%仍是同族扰动得到的候选覆盖率，不是独立物理删除确认的 prevalence。

RealLifeQA 更像决策不稳定与语境捷径混合。在原 detector 固定 test 的29个错误中，absolute choice-margin U-high 为18/29（62.1%）。严格 revised KeyShift 口径的36个错误中，detector 触发22（61.1%），可定位并生成合格语义反事实的为17（47.2%）；定向改写的独立修复样本仍少，因此局部语境机制属于较强候选而非强确认。29与36来自不同推理/打分实现，不能合并分母。

## GSM8K uncertainty 方法比较

| 方法 | AUROC | AUPRC |
|---|---:|---:|
| Answer entropy，3 samples | 0.705 | 0.640 |
| Answer entropy，6 samples | 0.742 | 0.687 |
| Variation ratio，6 samples | 0.753 | 0.694 |
| Greedy-answer disagreement，3 samples | 0.800 | 0.721 |
| Greedy-answer disagreement，6 samples | 0.821 | 0.746 |
| Kadavath-style P(True) | 0.776 | 0.789 |
| Existing OOF perturbation/hidden detector | 0.771 | 0.788 |
| Greedy disagreement + existing，rank fusion | 0.860 | 0.859 |
| Greedy disagreement + P(True) | 0.865 | 0.860 |
| 三者 rank fusion | **0.877** | **0.884** |

Greedy-answer disagreement 定义为 `1 - 原 greedy 最终答案在独立采样中的频率`。六采样版本相对 existing detector 的 AUROC 差为+0.049，paired bootstrap 95% CI `[-0.015,+0.113]`；二者 rank fusion 相对较优单项的增益95% CI为`[+0.0067,+0.0670]`。因此 GSM8K 上 uncertainty 更贴近主要错误表型，但与 perturbation/hidden 信号互补，而非替代关系。

## 统一解释

结果支持 benchmark-specific mechanism mixture：Scientist names 更突出事实支持不足与稳定自洽；TriviaQA closed-book 有证据响应亚群，context 设置有上下文诱导；GSM8K 以推理不稳定为主；DROP 以上下文干扰/证据选择为主；RealLifeQA 是决策不稳定与语境捷径混合。相应地，关键词/扰动方法在 P 型占比高的 DROP 上最强，在 GSM8K 上较弱，而联合 uncertainty 后明显改善。

实验索引：`whitebox/perturbation/227_scientist_*`、`228_scientist_p_e_confirmation.py`、`231_trivia_closedbook_e_confirmation.py`、`232_trivia_u_split_confirmation.py`、`233_gsm8k_question_end_r_confirmation.py`、`234_gsm8k_p_neighborhood_confirmation.py`、`235_trivia_context_p_confirmation.py`、`237_gsm8k_cot_u_split_confirmation.py`、`238_trivia_question_end_r_confirmation.py`、`239_gsm8k_uncertainty_methods.py` 与 `240_gsm8k_ptrue_uncertainty.py`；逐样本结果和报告位于同编号 `runs/` 目录。
