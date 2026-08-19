# spanattr — 干预驱动的细粒度 span 归因

用**扰动效应**而非 attention 质量来定义关键词。四阶段流水线，接入现有 `hallu-diagnose/` 编号体系（61–64）。

---

## 0. 当前状态（2026-08-11）

当前 whitebox 环境已经可以加载 torch 和 Llama-3.1-8B；61–89 的主要脚本均已完成
语法检查，61–64 的纯逻辑测试也已跑通。首次换模型、数据或 tokenizer 时仍应先跑 smoke，
不要直接复用旧 span/token 索引。

| 测试 | 状态 | 覆盖 |
|---|---|---|
| `python -m spanattr.selftest` | ✅ **25 项全部实测通过** | 符号约定、二阶目标、贪心/穷举、冗余聚类、NMS、统计工具 |
| `python tests/test_contracts.py` | ✅ **28 项全部实测通过** | 61→62→63→64 的 JSONL schema、索引映射、控制组构造、磁盘往返 |
| `python -m spanattr.core --smoke` | 可运行 | 随机权重 toy Llama 上的前向/梯度/IG completeness |
| `61–64 --smoke` | 可运行 | 各阶段端到端 |
| `81_active_subspace_diagnosis.py --smoke` | 已产生 smoke 产物 | active basis 的拟合、保存和报告 |

**首次在新环境跑，请先按顺序执行 `--smoke`**。`runs/*.pt`、`runs/**/*.npz`
是体积较大的中间张量，不纳入 Git；JSON/JSONL 报告才是可追踪结果。

编写测试的过程抓出了三个真实问题，都已修掉并固化为断言：

1. **贪心在超模（协同）实例上必然失败** — 它先拿单体增益最大的 span，永远发现不了协同对。所以 `63_` 在 C(m,k) 可承受时一律用穷举，只在超限时退化为贪心并告警。
2. **冗余/协同的符号解读预设 $u_i>0$。** 对 $u_i<0$ 的 span（支持 gold 的证据），"任一即可"不再蕴含 $I_{ij}<0$。$I$ 的算术不受影响，但簇级叙事必须限制在 $u_i>0$ 子集上。`61_` 因此报告 $u$ 的符号分布。
3. 候选 span **必须 token 互斥**（NMS 保证），否则 union 语义下 $I_{ij}$ 不可解释。

---

## 1. 框架

一切都在**门控空间** $\alpha\in[0,1]^P$ 里做：

$$E(\alpha)_t = E_t + \alpha_t(\bar e_t - E_t)$$

$\bar e$ 是中性化基线（**保长度**，不删 token，因此不引入位置移动）。一阶梯度、IG、有限差分交互量都是同一个量的不同阶，量纲与符号天然一致。

**目标（teacher-forced margin）**

$$S(\alpha)=\underbrace{\text{logsumexp}_v\,\text{lp}(\hat y_v)}_{\text{幻觉答案语义类}}-\underbrace{\text{logsumexp}_v\,\text{lp}(y^*_v)}_{\text{gold 语义类}}$$

**增益函数（下游只用这个，$S$ 只出现在定义式里）**

$$u(\mathcal S)=S(\mathbf 0)-S(\mathbf 1_{\mathcal S})$$

**交互量与符号约定**

$$I_{ij}=u(\{i,j\})-u(\{i\})-u(\{j\})$$

| | 含义 | 结构 |
|---|---|---|
| $I_{ij}<0$ | **冗余**（互为替代，证据重复） | 次模，贪心有 $1-1/e$ 保证 |
| $I_{ij}>0$ | **协同**（多 token 单元，top-k 结构上找不到） | 超模，贪心失效 |

$$\text{选择准则：}\quad \max_{|\mathcal S|\le k}\ \sum_{i\in\mathcal S}u_i+\sum_{i<j\in\mathcal S}I_{ij}$$

> 上一轮讨论中我把冗余/协同的符号写反了，根因是把 Shapley 交互指数文献里"增益函数"的惯例直接搬到了 $S$ 上。**结构性修法是让 $S$ 完全不出现在任何下游表达式里**，而不是去逐处改符号。代码与文档现在只用 $u$，`selftest.py` 用带已知真值的合成集合函数把这条钉死。

---

## 2. 流水线

```
61_grad_span_proposal.py   2/3-word 滑窗 → 一阶梯度 / IG / 实测 u / attention 基线
                            → σ_null 噪声底 → NMS 得 m 个互斥候选
62_interaction_matrix.py   候选内全 pairwise 有限差分 → I 矩阵 → 冗余簇 / 协同对 / 主特征向量
63_subset_select.py        五策略头对头 + 位置匹配 null + 两层验证
64a_vocab_decode.py        梯度方向 → 词表投影 → 指数级联合穷举与真实 margin
64b_vocab_recovery_generation.py 恢复词写回 prompt → 3 次采样生成 → gold/pred 判定

80–89 是后续两条支线：

80/87/89                 固定 detector 配置，在不同 knowledge 子集做 grouped OOF
81_active_subspace_diagnosis.py  校准集 span gradient SVD → 冻结 active basis
82_zo_active_keywords.py  held-out span 在 random/vocab/active 子空间内做 forward-only ZO
83_compare_zo_subspaces.py 汇总子空间搜索与 mean span 的排序/效应差异
84–86                   span selection × direction construction → 离散替换 → generation
87_projection_aware_decode.py 三种投影 + projection-aware 离散搜索
88_tokenwise_active_projection.py span 内每个 token 使用独立 active 系数并投影
```

```bash
python -m spanattr.selftest          # 先跑，无需 torch
python tests/test_contracts.py
python 61_grad_span_proposal.py --smoke   # 再跑，需 torch，CPU 几秒
bash run_all.sh                       # 真跑（MODEL=... ITEMS=... 可覆盖）
```

### 61 的关键产出
**校准 ρ**：廉价一阶代理（$\hat u$、IG）与**实测** $u$ 的 Spearman。脚本内置决策规则：

- IG ρ **> 0.90** → 一阶已经解释掉效应，62/63 的二阶机器是过度工程，砍掉写进 limitation。
- IG ρ **≤ 0.90** → 存在实质未解释方差，继续。

IG 用中性化算子作为 baseline，因此满足 completeness：$\sum_t \mathrm{IG}_t = u(\text{all})$，脚本运行时校验并报告 `completeness_rel_err`。这也让"top-k 覆盖了多少总效应"成为有意义的量。

### 62 的关键产出
$I$ 矩阵**只需前向传播**（$m$ 个单体 + $\binom{m}{2}$ 个配对 + 1 个基线；$m{=}12$ → 79 行）。这意味着同一套代码可以原封不动地跑 DeepSeek 或任何 API 模型，跨模型对比是干净的。

噪声阈值直接由随机、互斥且宽度匹配的 span pair 的经验交互量
$I_{\mathrm{null}}$ 估计，默认取 $|I_{\mathrm{null}}|$ 的 95% 分位数。单 span
效应在位置间的方差包含真实语义信号，不能当作测量噪声再按
$\sqrt{3}$ 传播。**若 `frac_sig < 5%`，这是负结果：交互实质不存在，加性模型足够，不要去聚类噪声。**

谱分析保留**特征向量**而非排序后的特征值——span 身份得以保留，正是排序谱会摧毁的东西。

### 63 的关键产出
五个策略在同一候选池、同一预算 $k$ 下对比：

`attention_topk`（现方案）· `first_order` · `second_order` · `greedy` · `random_matched`（**位置与长度匹配的随机对照，必须有**）

**两层测量，职责严格分离：**

- **Tier 1**（廉价、稠密、teacher-forced）：$u(\mathcal S)$。**只用于搜索。** 有偏的搜索启发式代价是统计效力，不是效度——因为每个选出的集合都会被 Tier 2 复核。
- **Tier 2**（昂贵、稀疏、generation）：采样生成下 $P(\hat y)$ 的下降与 $P(y^*)$ 的上升。**所有对外声明都建立在 Tier 2 上。**

脚本报告两层的 Spearman ρ 及 bootstrap CI，这个数决定论文框架：ρ ≥ 0.6 → teacher-forced 归因可作为廉价代理直接汇报；ρ 低 → Tier 1 降格为纯搜索启发式，所有 headline 数字退到 Tier 2。**两种结果都站得住，但是不同的论文，所以必须报。** 这也对你另一个 66.5% 一致率的发现构成直接补充证据——注意 66.5% 是**阈值化标签**的一致率，连续量上的相关可能高得多。

### 64 的关键产出
**算子隔离**：选择用中性化（61/62/63），验证用离散词替换（64）。两者不共享任何机制，所以这里的正结果不可能是"同一算子既造伪标签又造特征"的循环产物——这正是 v8 缺的那一环。

词表投影用**归一化方向匹配** $\langle w_v-e_t,\,-g_t\rangle/\|w_v-e_t\|$。不归一化的话 argmax 会被 embedding 范数主导，也就是被词频主导，那就又变成一个频率检测器了。

### 65 的关键产出
65 不改变 64 的联合穷举，只读取其 `joint_top` 恢复结果并把 token id
真实写回原 prompt。默认对原 prompt 和恢复后 prompt 各做 3 次 temperature
sampling，保存生成文本、gold/pred 命中、`p_gold`、`rise_p_gold`、
`drop_p_pred` 和配对的 `correction_rate_paired`。


### 80–89：active subspace、离散投影与 detector

这部分有两个名字接近但目的不同的 `81`：

- `81_zo_span_keywords.py`：每个 span 直接在完整 embedding 空间搜索一个共享方向，用于早期 ZO-vs-mean 对照。
- `81_active_subspace_diagnosis.py`：在校准题的逐 span gradient 上做 SVD，保存全局 active basis。basis 必须在 held-out 评估前冻结；`82_zo_active_keywords.py` 只做前向查询。

active basis 把 4096 维搜索限制到 rank-$r$ 子空间，是为了降低 ZO 查询方差和查询数，并非理论硬限制：

| calibration | gradient 数 | 90%/95% effective rank | rank 16/32 能量 |
|---|---:|---:|---:|
| `ex1` | 89 | 17 / 23 | 89.6% / 98.4% |
| `question_0000` | 185 | 29 / 41 | 78.4% / 91.6% |

因此 rank 16/32 是计算预算与覆盖率的折中。跨题时优先用 rank 32，并把 rank 16/64 作为敏感性分析。

#### 三个比较轴

1. **span selection**：mean neutralization 按实测 $u$ 排 span；active/ZO 按子空间内可找到的连续效应排 span。
2. **direction construction**：逐位置 gradient direction 是每个 token 的局部一阶下降方向；shared active direction 是整个 span 共用一个低秩方向；token-wise active 允许每个 token 在同一个 basis 内有独立系数。
3. **projection/validation**：连续 embedding 的 margin 改善不等于离散 token 替换后的改善。词表是稀疏点集，nearest token 不会复制目标向量；最终必须用真实 teacher-forced margin 重排，并用自由生成验证。

所以“当前 shared-active direction 不如逐位置 gradient direction”只描述 direction construction，**不能推出 active span selection 不如 mean span selection**。判断后者必须在同一投影器、同一编辑预算和同一 held-out 题集上做交叉实验。

#### 84–88 的离散化实验

- `84_active_vocab_decode.py` 做 `mean/active span × gradient/active direction` 四格交叉，再用真实 margin 穷举候选组合。
- `85_active_word_generation.py` 将候选写回 prompt 并重新生成；`86_...` 汇总 `rise_p_gold`、`drop_p_pred` 和 paired correction。
- `87_projection_aware_decode.py` 比较方向匹配、目标 embedding 最近邻、候选并集的真实 margin 重排；projection-aware 路径只搜索可实现 token displacement。
- `88_tokenwise_active_projection.py` 让 span 内各 token 使用独立 rank-$r$ 系数，但共同受 span 级 Frobenius budget 约束。它解决 shared direction 表达力不足，不自动解决词表稀疏。

`question_1400` 的诊断中，shared continuous active 只改善约 0.04 margin 且未跨界；token-wise continuous active 改善约 11.51 并跨界，但当前离散候选反而让 margin 变差。这说明 projection gap 是当前主要瓶颈，也说明增加连续自由度本身不能保证离散成功。它是单题诊断，不能作为 active-vs-mean 的总体结论。

```bash
cd /home/tong56/whitebox/perturbation
source ../activate_whitebox.sh

# 校准题与 held-out 题必须分开
python 81_active_subspace_diagnosis.py \
  --items data/items_n128_generation_flip.json --item_id question_0000 \
  --basis_out runs/81_q0000_active_basis.pt \
  --report runs/81_q0000_active_report.json

# 30 题 active/mean span 搜索 + token-wise 投影
bash run_82_88_n30.sh

# 小规模四格离散替换与 generation 验证
bash run_84_86_active_words.sh
```

#### detector 的冻结评估

固定 `top11 / layer16 / PCA8 / C=0.5`，使用 `StratifiedGroupKFold(5)` 产生严格 OOF 预测；PCA、标准化和逻辑回归均只在每折训练集拟合。

| 子集 | n | full AUROC | margin-only AUROC | lift | group-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| probe-perfect (`87`) | 343 | 0.837 | 0.808 | +0.029 | [+0.001, +0.054] |
| both knowledge scores > 0.5 (`89`) | 1084 | 0.851 | 0.789 | +0.063 | [+0.041, +0.084] |

这支持 hidden-delta/perturbation 特征在 margin 之外提供增量信号；它不直接回答 active span 是否优于 mean span。

---

## 3. 已知取舍

- `length_norm=True` 时 `_class_logprob` 是对各变体**平均** logprob 的 logsumexp，是个让不同长度变体可比的启发式，不是严格混合分布。要严格形式设 `--length_norm 0`。
- 默认候选是原始文本上的 2/3-word 滑窗，再通过 tokenizer offset mapping
  映射回 token gate；可用 `--span_unit tokens` 恢复旧的 2/3-token 模式。
- 细粒度下**默认算子必须是保长度的 neutralize**；span 越短，delete 的语法破坏相对信息量越大，且位置移动会把位置混淆重新引进来。`delete` 只作为 robustness check。
- $m$ 太大时 `--exh_cap` 会让 `second_order` 退化为贪心，而贪心在协同实例上必然失败（见测试 §4）。$m\le 16,k\le 3$ 时穷举只有 560 种组合，无成本。

## 4. 下一步实验判据

把 61 的校准 ρ、`topk_null_ratio`/`topk_min_null_ratio`、62 的
`frac_sig`/`synergy_share`、63 的头对头表贴给我。Stage 1 用 NMS 后
top-k 的平均及末位 `|u|` 相对随机 span `|u|` 的 95% 分位数判断头部信号，
不再用全体 span 的平均 SNR 决定是否继续。


对 active-vs-mean 扩展实验，至少同时报告：连续 `u_realized`、连续跨界率、离散 `u_realized`、离散跨界率、generation correction rate，以及按题配对的 active-minus-mean 差值和 bootstrap CI。没有离散与 generation 两层验证时，只能称为方向搜索诊断。
