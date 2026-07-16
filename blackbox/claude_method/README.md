# SKID — Shortcut-Key Intervention Detector

Black-box, API-only (no GPU) detection of **shortcut-key hallucinations** as
formalized in TRAPQA (*"Understanding Why Language Models Hallucinate:
Testing Reasoning Against Priors"*, arXiv:2607.00447). Works on both TRAPQA
components: **ScientistQA** (`prepend_names`) and **Real-Life Constrained QA**.

## The idea in one paragraph

TRAPQA models an answer as a mixture over two latent inference paths: a
constraint-sensitive path `(k*, t*)` and a pretraining-frequent shortcut path
`(ks, ts)`. Hallucination = the shortcut posterior dominating (Thm 3.4),
which implies positive inference loss (Thm 3.6). The decisive constraint `C`
enters the computation **only through the constraint path**; under shortcut
dominance (Assumption 3.1(ii) + 3.3) the answer is causally **inert to C**.
SKID exploits this asymmetry with do()-style prompt interventions:

> **A faithful answer is covariant with the constraint and invariant to
> salience/surface form. A shortcut answer is invariant to the constraint and
> covariant with salience/surface form.**

So the diagnostic event is not "any flip" — it is a *mismatch between the
observed sensitivity profile and the profile a faithful reasoner must have*.

## 中文实验记录

### ScientistQA：事实 Probe 与最小反事实不变性

#### 方法

ScientistQA 的关键约束通常是可以独立核验的人物事实，例如“从未获得某奖”、
“没有就读某校”或“没有担任某职位”。因此可以把约束改写成闭卷 Yes/No
问题，并分别询问两个候选人。

当前实验使用显式规则：

```text
检测为 hallucination = probe_violation AND neg_invariant
```

- `probe_violation`：模型在原题选择候选人 A，但在独立 probe 中又认为 A
  违反原题的决定性约束。这只检查模型答案与其自身可访问知识是否冲突，实际
  detection 不需要 gold label。
- `neg_invariant`：把决定性约束的极性翻转后，模型仍选择原来的候选人，说明
  答案可能没有随约束协变。
- Gold 只在所有预测完成后加入，用来计算 TP、FP、FN、TN；不参与 detection。

实验文件为 `qwen_skid_probe_and_neg_full_3.7.jsonl`。虽然文件名包含
`full`，当前实际记录数是 500，且无运行错误。受测模型在这 500 道题上答错
128 道，原始 hallucination rate 为 25.6%。

#### 不剔除同答案 Probe

所有 500 道题都进入统计时：

| 指标 | 数值 |
|---|---:|
| TP / FP / FN / TN | 88 / 35 / 40 / 337 |
| Precision | 71.5% |
| Recall | 68.8% |
| F1 | 70.1% |
| Accuracy | 85.0% |

主要误差来自两个候选人的 probe 得到相同答案：66 道为 Yes/Yes，75 道为
No/No，合计 141/500（28.2%）。其中 Yes/Yes 很容易令正确选择错误触发
`probe_violation`，造成 FP；No/No 则令错误选择无法触发 `probe_violation`，
造成 FN。相同答案不一定证明模型没有知识，也可能说明两人客观上共享该属性、
probe 只覆盖了复合约束中的非区分性子条件，或模型在强制 Yes/No 格式下猜测。

#### 剔除 Yes/Yes 与 No/No

如果只保留两个候选人的 probe 答案不同的项目，筛选过程不使用 gold；这应被
理解为一个允许 abstain 的 selective detector，而不是把被拒绝样本当作不存
在。共保留 359/500 道，覆盖率为 71.8%。条件结果为：

| 指标 | 数值 |
|---|---:|
| TP / FP / FN / TN | 70 / 5 / 6 / 278 |
| Precision | 93.3% |
| Recall（在可判定子集上） | 92.1% |
| F1（在可判定子集上） | 92.7% |
| Accuracy（在可判定子集上） | 96.9% |
| Coverage | 71.8% |

被拒绝的 141 道题中包含 52 个真实 hallucination。若把 abstain 都视作“未检
出”并放回完整 500 道题，结果为 TP=70、FP=5、FN=58、TN=367，Precision
为 93.3%，overall Recall 为 54.7%，F1 为 69.0%。因此正式报告必须同时给出
条件性能与 coverage，不能只报告剔除后的 92.7% F1。

推荐把输出分成 `positive / negative / abstain`：当两个 probe 相同或无法解析
时，K 路径返回 abstain，再交给不依赖人物事实的原子反事实干预处理。

### Real-LifeQA：为什么不采用人物事实 Probe

Real-LifeQA 的决定因素通常不是“某人是否获得某奖”一类可脱离上下文核验的
外部事实，而是场景内部的物理、空间、程序和媒介前置条件。例如检查车架 VIN
是否要求车辆在场、某一步骤是否要求原件、某操作是否必须使用特定媒介。这些
条件常常：

- 依赖完整场景，不能可靠压缩成单个闭卷 Yes/No 事实；
- 由多个子条件共同决定，单独 probe 任一条件都不能推出选项；
- 可能随流程、地点、权限和例外情况变化；
- 在简化 probe 中会直接提示模型关键推理，测到的是题目被简化后的能力，而非
  原场景中是否使用约束。

因此，ScientistQA 的 `probe_violation` 不能直接作为 Real-LifeQA 的核心信号。
Real-LifeQA 更适合测试“选择是否随场景前置条件协变”。

#### `extract_cues_occlusion_reallife.py` 使用的方法

该脚本实现 deletion → minimal negation 的无 gold detection 流程：

1. 把 scenario 分割成多个候选 span。
2. 逐个删除 span，并重新询问模型。删除题允许输出 `3`，表示剩余场景无法决定
   Option1 或 Option2。
3. 如果删除某个 span 后模型输出 `3`，或者在 Option1/Option2 之间翻转，则该
   span 被视为 informative deletion，说明它可能承载必要条件或决策线索。
4. 只对 informative span 生成最小 negation：保留人物、对象、数字、选项及无
   关事实，只翻转该 span 表达的命题。
5. 对 negated prompt 再次询问模型。若至少一个有效 negation 没有引起答案翻
   转，则判为 `negation_nonflip` hallucination evidence；如果没有 informative
   deletion，则判为 negative；如果所有有效 negation 都引起翻转，则判为
   negative；生成失败或证据不完整则返回 ambiguous。
6. Detection 全程不允许 gold 字段进入；所有预测结束后才连接 gold，计算
   precision、recall、F1 和混淆矩阵。

这个方法检测的不是外部知识冲突，而是：删除一个可能必要的场景条件能使决策
不确定/改变，但把该条件反向改写后，模型却仍坚持原答案——即对关键场景命题
呈现不合理的反事实不敏感性。

#### 当前结果状态

# Cue intervention and hallucination detection summary

Items processed: 100
At least one valid candidate key: 71/100 (0.710)
Shortcut key detected: 68/100 (0.680)
Constraint key detected: 33/100 (0.330)
Ambiguous decisions: 29/100 (0.290)

## Hallucination detection (gold used only here)

Precision: 0.162
Recall: 0.917
F1: 0.275
Confusion matrix: TP=11, FP=57, TN=2, FN=1

## Intervention ablations

- delete: flips=65/201, mean |delta|=7.535
- emphasis: flips=31/201, mean |delta|=4.433
- negation: flips=49/201, mean |delta|=6.299
- paraphrase: flips=16/201, mean |delta|=2.850


## Signals

Per item, SKID re-queries **the same subject model** with targeted edits
(edits are produced by a generator LLM; any cheap model works):

| # | Signal | Intervention | Hallucination evidence when… | Weight |
|---|--------|--------------|------------------------------|--------|
| K | `probe_violation` | Closed-book Yes/No probe of the decisive fact for the **chosen** answer (separate conversation) | the model's own isolated knowledge contradicts its choice | 0.35 |
| K₂| `probe_two_sided` | Same probe for the **other** option | …and the other option is confirmed to satisfy `C` | +0.10 |
| N | `neg_invariant` | Flip the polarity of every decisive clause (`never received X → received X`) | answer **doesn't** flip → `C` is causally inert | 0.25 |
| A | `abl_invariant` | Delete `C` entirely | answer unchanged → it equals the zero-constraint **prior** default | 0.15 |
| E | `emph_rescue` | Restate `C` maximally salient ("Decisive fact: …") | answer **changes** → original answer was salience-limited | 0.15 |
| P | `para_flip` | Meaning-preserving paraphrase | answer changes → keyed to surface statistics | 0.05 |
| S | `swap_flip` | Swap option order | chosen entity/action changes → positional shortcut | 0.05 |

Score = weighted sum (∈ [0,1]); flag if ≥ threshold (default **0.30**). Off-option/abstain responses are flagged at
1.0, matching the paper's scoring. Every item gets a JSONL evidence trace
(constraint text, every perturbed answer, per-signal values) for auditing.

Why signal K works so well here: the paper's Table 2 shows models answer the
decisive fact correctly **in isolation** 76–97% of the time, and 36–80% of
hallucinations occur *despite* both probes being correct ("known-fact
hallucinations"). K simply re-derives the paper's probe for the chosen
answer and checks self-consistency — turning the paper's diagnostic finding
into a detector. The intervention signals (N/A/E) then cover the residual
where the model *also* lacks the fact: those answers are maximally
prior-driven, i.e. exactly the ones predicted to be negation/ablation-inert.

### Projected recall of signal K alone (from the paper's own tables)

Combining Table 2 (known-fact fraction) with Table 5 (eliminative-probe
asymmetry) for the names-only condition:

| Subject (low thinking) | Hallucinations | K-catchable (eliminative probe known) |
|---|---|---|
| Claude Sonnet 4.6 | 699 | ≈ 486 (**69.5%**) |
| DeepSeek V3.2 Chat | 1089 | ≈ 929 (**85.3%**) |
| GPT-5.2 | 344 | ≈ 274 (**79.7%**) |
| Gemini 3.1 Pro | 73 | ≈ 62 (**85%**) |

…before adding N/A/E/P/S. (Directional projection: probe answers vary
run-to-run; treat as an expected floor, not a guarantee.)

## Usage

```bash
# keys as needed for the models you audit / generate with:
export DASHSCOPE_API_KEY=...

# ScientistQA (prepend_names), auditing Qwen 3.5 Flash:
python skid.py --benchmark scientist --data shuffled_prepend_names_question.json \
  --subject qwen:qwen3.5-flash \
  --generator qwen:qwen3.5-flash \
  --out sci_results.jsonl --workers 8

# Real-Life Constrained QA, auditing Qwen 3.5 Flash:
python skid.py --benchmark reallife --data question_and_result.json \
  --subject qwen:qwen3.5-flash \
  --generator qwen:qwen3.5-flash \
  --out rl_results.jsonl

# Quick pilot on a random subsample:
  ... --limit 200 --seed 0

# Audit an EXISTING run instead of re-answering (JSON {key: option_index|text}).
# IMPORTANT: --subject must be the same model+settings that produced those
# answers, or the sensitivity signals are meaningless:
  ... --answers-file my_model_answers.json

# Robust mode: majority vote of 3 samples per variant (3x subject cost):
  ... --samples 3

# Offline pipeline check with a simulated subject (no network, no keys):
python skid.py --benchmark reallife --data question_and_result.json \
  --subject mock:subject,shortcut=0.30,seed=7 --generator mock:generator
```

Provider specs: `qwen:MODEL`, `anthropic:MODEL`, `openai:MODEL`, `deepseek:MODEL`,
`gemini:MODEL` (OpenAI-compat endpoint), or `openai-compat:MODEL@BASE_URL#ENV_VAR`.
`qwen:` uses DashScope's OpenAI-compatible endpoint, reads
`DASHSCOPE_API_KEY` (falling back to `QWEN_API_KEY`), and disables thinking so
the detector receives concise, directly parseable answers. Override the endpoint
with `DASHSCOPE_BASE_URL` when using another region/workspace.
Runs are cached in `skid_cache.jsonl` → fully resumable; re-running is free.

The report prints: subject hallucination rate, detector precision/recall/F1
at the threshold, AUROC of the continuous score, per-signal fire rates on
hallucinated vs. correct items, and (Real-Life) overlap of your reproduced
errors with the benchmark's recorded `mistake_models`.

## Cost

Per item ≈ 1 generator call + 8–9 short subject calls (~9 for ScientistQA,
~8 for Real-Life). Full suite (2,925 + 500 items) ≈ **30k short calls**, all
parallelized, cached, and resumable. `max_tokens` are tiny (single-name /
single-digit / yes-no answers), so cost is dominated by prompt tokens.

## Operating points & known failure modes

- **Balanced (default)**: threshold 0.30. **High precision**: threshold 0.45
  (requires probe+two-sided, or multiple interventions to agree).
- **Correct-for-wrong-reasons** answers (prior happens to point at the gold
  answer; cf. paper §H.6) fire N+A and count as false positives under
  answer-correctness labels — though for reliability audits, surfacing
  unfaithful-but-lucky inferences is arguably a feature. TRAPQA is built so
  the salient association targets the distractor, so this mode is rare here.
- **Probe noise**: a wrong probe answer about a correctly chosen candidate
  can fire K spuriously; bounded by (1 − probe accuracy) ≈ 3–24% and cut by
  the two-sided gate / higher threshold.
- **Negation subtlety**: for a hallucinating model, flipping `C` makes the
  constraint *agree* with the shortcut — so invariance under negation is a
  robust hallucination marker in both partial- and full-shortcut regimes,
  while a faithful model must flip. This is what makes N the cleanest single
  intervention.
- The **generator can be any model** (even the subject itself): perturbation
  writing is mechanical editing, not subject to the shortcut being tested,
  and outputs are validated (constraint must be absent from the ablation,
  probe must contain the `{NAME}` slot, etc.) with a structural fallback for
  ScientistQA (constraint = final sentence(s) — holds by construction).

## Files

- `skid.py` — everything (stdlib-only; providers, perturbation generation,
  detection, scoring, metrics, cache, CLI, offline mocks).
- Output JSONL — one evidence record per item:
  `{key, a0, gold_idx, hallucinated, signals{...}, score, flag,
    answers{original, negate, ablate, …, probe::NAME}, evidence{constraint, …}}`
