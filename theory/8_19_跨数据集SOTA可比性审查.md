# Perturbation detector 跨数据集 SOTA 与可比性审查

> 审查日期：2026-08-19  
> 本地正式结果：`paper4_self_matrix_v2/evaluation/combined/evaluation.json`  
> 主指标：response-level error detection AUROC

## 1. 先给结论

当前公开文献不足以构造一个严格的“统一 SOTA 排名”。Scientist、Athlete、Building、Musician 是本项目自建数据和筛选协议，没有外部论文在相同样本、相同生成答案与相同标签上报告结果。TriviaQA 和 GSM8K 虽有大量同名实验，但不同论文通常重新生成答案，且在生成模型、模型版本、解码温度、答案格式、正确性判定、训练/测试划分、监督强度和推理调用次数上不同。

因此可支持两级结论：

1. **严格结论：** 目前不能仅凭论文表格声称本方法超过 SOTA；需要在本项目固定 generations/labels/splits 上复跑各 baseline。
2. **描述性结论：** 本方法的公开数字位置很强。Exact 方法在 4 模型 × 3 数据集上的宏平均 AUROC 为 **0.8604**；仅公开共有的 TriviaQA 与 GSM8K，8 个模型—数据集单元宏平均为 **0.8899**。Llama-3.1-8B-Instruct 上两者平均为 **0.9488**，高于所找到的同模型家族 attention detector 的论文数字，但这些仍不是 paired comparison。

## 2. 本方法的正式口径

- 每个模型检测自己的 generation 和 correctness label，不混用其他模型答案。
- Exact/current127：逐 span perturbation + hidden/score features；固定 LR `C=.03`。
- Attention-pruned：用 attention 预筛 span，减少约 35%–41% perturbation query。
- In-domain：3 seeds × 5-fold OOF；Scientist/GSM8K grouped，TriviaQA stratified。
- TriviaQA：1,000 条带检索 context 的短答案，greedy generation，alias-normalized correctness。
- GSM8K：942 条 train/test 固定池，自由生成完整解题过程，greedy；只按最终 `#### number` 判断 correctness；parse failure 也作为错误。
- Scientist：各模型仅保留 parse-valid 且通过独立 closed-book probes 的 known 子集，因此样本量随模型变化（Llama 1,077；Qwen 1,204；Mistral 621；Falcon3 1,099）。

### 2.1 AUROC

| 模型 | Scientist | TriviaQA | GSM8K | 三数据集平均 |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.8938 | 0.9484 | 0.9491 | **0.9305** |
| Qwen2.5-7B-Instruct | 0.8584 | 0.9104 | 0.7841 | **0.8510** |
| Mistral-7B-Instruct-v0.3 | 0.8091 | 0.9639 | 0.7943 | **0.8558** |
| Falcon3-7B-Instruct | 0.6448 | 0.8968 | 0.8718 | **0.8045** |
| **宏平均** | **0.8015** | **0.9299** | **0.8499** | **0.8604** |

Attention-pruned 的对应宏平均为 0.8577，和 Exact 相差 -0.0027，同时减少约 35%–41% span queries。

## 3. TriviaQA：公开前列结果与本方法位置

下表列的是截至审查日期找到的高结果，不表示严格排行榜。

| 方法/论文 | 生成/检测模型 | 论文 AUROC | 本项目相近单元 | 可比性 |
|---|---|---:|---:|---|
| MultiHaluDet | Mistral-7B-Instruct | 0.9830 | Mistral 0.9639 | 弱：论文构造与本项目答案池、hard-negative/label protocol 不同 |
| Multiple Testing | Llama-3.1-8B | 0.9478 | Llama 0.9484 | 弱到中：同模型家族与数据集；对方每题 20 次 `T=1.0` 采样、聚合多种 UQ score，本项目一次 greedy + supervised perturbation probe |
| ARS + supervised probing | Qwen3-8B | 0.9162 | Qwen2.5-7B 0.9104 | 弱：模型代际不同；ARS 使用 reasoning-trace counterfactual shaping |
| Spectral LapEigvals | Llama-3.1-8B | 0.889 | Llama 0.9484 | 中等偏弱：同模型家族/benchmark/response AUROC；对方 `T=1.0`、不同生成池与 split |
| SinkProbe | Llama-3.1-8B | 0.883 ± 0.012 | Llama 0.9484 | 中等偏弱：同模型家族和 5-fold probe；对方约 9.6K valid generations、`T=0.1`、GPT-4.1 judge，本项目 1K greedy、alias match |
| PEP | Qwen3-32B | 0.868 | Qwen2.5-7B 0.9104 | 弱：模型大小/代际、样本划分和 probe training 均不同 |

描述性位置：

- 本方法 TriviaQA 四模型平均 **0.9299**。
- Llama 单元 **0.9484**，与 Multiple Testing 的 0.9478 数值相当，高于 Spectral/SinkProbe 的 0.889/0.883。
- Mistral 单元 **0.9639**，比 MultiHaluDet 的 0.9830 低 0.0191；但两者不在同一答案池，不能判定排名第二。
- 其余模型没有找到完全相同的 Qwen2.5-7B-Instruct 或 Falcon3-7B-Instruct 公布单元。

## 4. GSM8K：公开前列结果与本方法位置

| 方法/论文 | 生成/检测模型 | 论文 AUROC | 本项目相近单元 | 可比性 |
|---|---|---:|---:|---|
| Spectral LapEigvals | Mistral-Small-24B | 0.925 | 无相同模型 | 很弱：模型规模不同 |
| ARS + CCS | Qwen3-8B | 0.9037 | Qwen2.5-7B 0.7841 | 弱：模型代际与 reasoning-trace shaping 不同 |
| PEP | Qwen3-0.6B | 0.879 | 无相同模型 | 弱：模型与 split 不同；论文使用完整 GSM8K 官方划分 |
| Spectral LapEigvals | Llama-3.1-8B | 0.872 | Llama 0.9491 | 中等偏弱：同模型家族/benchmark；不同温度、generation pool、label/split |
| SinkProbe | Llama-3.1-8B | 0.824 ± 0.025 | Llama 0.9491 | 中等偏弱：对方 1,297 valid test generations、`T=0.1`；本项目 942 train/test pool、greedy 且 parse failure 入负类 |
| Noise-enhanced answer entropy | Mistral-7B-Instruct-v0.3 | 0.7850 | Mistral 0.7943 | 中等：模型版本相同、数据集相同；对方每题 10 次采样并调 noise，本项目单次回答的监督 probe |

描述性位置：

- 本方法 GSM8K 四模型平均 **0.8499**。
- Llama 单元 **0.9491** 高于所找到的公开 Llama-3.1-8B Spectral/SinkProbe 数字。
- Mistral 单元 **0.7943** 与同版本 noise-enhanced sampling 的 **0.7850** 接近。
- Qwen 单元相对 ARS 看似偏低，但 Qwen2.5 与 Qwen3、自然一次生成与 trace shaping 不是同一 setting。

## 5. Scientist 与跨域 transfer

### 5.1 Scientist

没有公开论文使用完全相同的 Scientist names benchmark、known probe 筛选、同一生成答案和 grouped split。因此这里不存在可引用的外部 top-5。

本项目内部可比较结果：

| 方法 | Llama Scientist-known AUROC | 证据条件 |
|---|---:|---|
| MiniCheck whole contrastive | 0.978 | 使用两位候选的完整外部 profiles；不是纯内部 detector |
| Perturbation exact | 0.894 | 白盒、监督、无 gold/external checker at test |
| Perturbation attention-pruned | 0.888 | 白盒、监督；减少 41.4% query |
| Representation + uncertainty | 0.823 | 白盒、监督 |
| Representation delta | 0.816 | 白盒、监督 |
| Likelihood NLL | 0.674 | 单次输出 uncertainty，无训练 |

MiniCheck 不能列入同类 SOTA 排名，因为它在测试时读取完整人物 profile，相当于外部证据重新核验；perturbation detector 则研究目标模型自身的错误信号。

### 5.2 Athlete/Building/Musician frozen transfer

这些也是本项目自建 target。正式意义只能报告本项目内部结果，不能与外部论文平均：

- Llama Scientist→all exact AUROC 0.917；
- Qwen 0.840；Mistral 0.854；Falcon3 0.626；
- 四模型宏平均 0.8093。

## 6. 可以计算的“平均值”与不能计算的平均值

### 6.1 本方法统一矩阵平均

| 范围 | Exact AUROC 宏平均 | Attention-pruned AUROC 宏平均 |
|---|---:|---:|
| 4 模型 × Scientist/TriviaQA/GSM8K（12 cells） | **0.8604** | **0.8577** |
| 4 模型 × TriviaQA/GSM8K（8 cells） | **0.8899** | **0.8896** |
| TriviaQA（4 models） | **0.9299** | **0.9280** |
| GSM8K（4 models） | **0.8499** | **0.8511** |

### 6.2 最接近模型—数据集单元的描述性 paired-cell 平均

这些只是同名 cell 的描述性对照，不是同一 test examples 上的 paired test：

| 对照论文 | 共同近似 cells | 论文平均 | 本方法平均 | 差值 |
|---|---:|---:|---:|---:|
| Spectral Features, Llama-3.1, TriviaQA+GSM8K | 2 | 0.8805 | **0.9488** | +0.0683 |
| SinkProbe, Llama-3.1, TriviaQA+GSM8K | 2 | 0.8535 | **0.9488** | +0.0953 |
| Noise-enhanced entropy, Mistral-v0.3, TriviaQA+GSM8K | 2 | 0.7813 | **0.8791** | +0.0978 |
| Multiple Testing, Llama-3.1 TriviaQA | 1 | 0.9478 | **0.9484** | +0.0006 |
| MultiHaluDet, Mistral TriviaQA | 1 | **0.9830** | 0.9639 | -0.0191 |

不能把上述论文的所有表格数字直接混在一起再求一个“竞争者平均”，原因是每篇覆盖的模型/数据集数量不同；这样会让覆盖容易数据集更多的论文得到更大权重，并混合单采样、多采样、外部 judge、监督 probe 等不同任务。

## 7. 当前最稳妥的论文表述

可以写：

> Across four 7–8B model families, our exact detector achieves a macro-average AUROC of 0.930 on TriviaQA and 0.850 on GSM8K. On the Llama-3.1-8B cells, its descriptive AUROC (0.948 on TriviaQA and 0.949 on GSM8K) is higher than previously reported attention-based Spectral and SinkProbe results. However, because prior work uses independently generated responses, different decoding and labeling protocols, these numbers are contextual rather than a controlled SOTA comparison.

暂时不要写：

> Our method is the new SOTA and outperforms all previous methods on TriviaQA and GSM8K.

## 8. 得到严格 SOTA 结论所需实验

固定本项目现有 generations、labels 和 3×5 OOF splits，在同一批样本上复跑：

1. NLL / entropy / P(True)；
2. last-token hidden linear probe；
3. AttnLogDet；
4. LapEigvals；
5. SinkProbe；
6. 若算力允许，再跑 semantic entropy（10–20 samples）和 PEP/ARS。

然后对每个模型—数据集 cell 报告同一 split 的 AUROC、AUPRC、balanced accuracy，并用 paired bootstrap 或 DeLong/OOF-aware bootstrap 比较。本方法是否超过 SOTA，应以这张同答案池 benchmark 表为准。

## 9. 主要来源

- Spectral Features: <https://aclanthology.org/2025.emnlp-main.1239/>
- SinkProbe: <https://arxiv.org/abs/2604.10697>
- Multiple Testing: <https://arxiv.org/abs/2508.18473>
- ARS: <https://arxiv.org/abs/2601.17467>
- PEP: <https://arxiv.org/abs/2608.08024>
- MultiHaluDet: <https://aclanthology.org/2026.mellm-1.6/>
- Noise-enhanced sampling: <https://openreview.net/forum?id=WnM3sluiVn>

