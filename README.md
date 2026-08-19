# 白盒幻觉检测与 KeyShift 因果验证

---

## 0. 汇报摘要

本项目研究一种特定但重要的幻觉机制：**key-selection hallucination（关键证据选择错误）**。其特点不是模型完全不知道答案，而是提示中已经存在足够的正确证据，模型却将回答建立在一个统计上显著、语义上相关、但对题目约束无效的片段上。项目把真正决定答案的片段称为 **constraint key**，把诱导错误答案的片段称为 **shortcut key**。

围绕这一问题，工作区中的研究逐步回答四个问题：

1. 能否仅利用模型内部信号检测当前回答是否可能错误？
2. 能否进一步定位模型依赖的是 constraint 还是 shortcut，而不是只输出一个黑盒风险分数？
3. 如果定位到 shortcut，改变其表面先验是否会按理论预测改变答案 margin？
4. 这种文本层面的效应，能否通过模型内部 attention-head 状态的双向 patching 得到部分机制验证？

## 1. 工作区结构与研究演进

### 1.1 三个目录的职责

| 目录 | 主要内容 | 在研究中的作用 |
|---|---|---|
| `whitebox` | 检测器、多代 role-mediated 方法、RealLifeQA/ScientistQA、KeyShift v9、环境与结果 | 方法开发与核心实验 |
| `other_bench` | HaluEval、GSM8K、TriviaQA 数据，以及 HaluEval v7/v8/v10 结果 | 跨数据集外部验证 |
| `theory` | `hallucination_detection_7_20_theory.pdf` | 形式化定义、单调性命题、KeyShift 与 patching 理论 |

### 1.2 方法演进顺序

1. `whitebox_run.py`：单一 constraint attribution 基线。
2. `whitebox_detector.py`：logit、attention、spectral、gradient 多特征监督检测器 v1。
3. `whitebox_detector_v2.py`：显式 constraint/shortcut token-indexed 特征 v2。
4. `weakly_supervised_whitebox.py`：用文本干预产生 span 角色伪标签。
5. `role_mediated_whitebox_v3.py` / `v4.py`：将 shortcut/constraint 角色写入显式单调风险公式，并加入 post-prediction causal audit。
6. ScientistQA v4/v5：从简单名称拼接扩展到 atomic span 与完整 profile。
7. `role_mediated_whitebox_v7_interventional_multimodal.py`：测试时使用行为、attention、gradient、spectral 多模态 span 信号并做特征消融。
8. `openended_role_mediated_v8.py`：从二选一 margin 扩展到开放式生成和 reference-mediated correctness。
9. `reallifeqa_keyshift_experiment_v9.py` / revised：做 frequency-controlled 语义反事实、detector-gated mitigation 和内部 activation patching。
10. `keyshift_halueval_open_v10.py`：把 KeyShift 推广到 HaluEval 开放式、paired correct/hallucinated answer 设置。
