## 第四部分：competition 方向性冻结确认

Scientist-Names 全量 factorial 对照显示，错误题的强 competition 比正确题高 5.5 个百分点（95% question-bootstrap CI [2.3, 8.7]），而 redundancy、synergy 与高阶 Harsanyi mass 没有相应增加。为验证该相关性是否具有方向性行为意义，按 SHA256(person/group) 奇偶固定 50% confirmation groups，在其中选择两个 signed effects 方向相反且 $|u|>0.1$ 的错误题。每题冻结最大化较弱侧效应的一个 pair。92 道入选题中，1 道无法构造严格等宽、位置匹配且不重叠的随机对照，最终确认 $n=91$。七个条件共享逐题 sampling seed，每条件采样10次。

\input{runs/257_competition_direction_confirmation/table.tex}

删除错误侧关键词使正确答案生成率增加14.5个百分点，95% CI [11.0, 18.1]；删除正确侧关键词使其下降22.5个百分点，95% CI [-28.4, -16.9]。两种干预的方向性 contrast 为+37.0个百分点，95% CI [30.3, 43.7]。错误侧关键词干预相对等宽、位置匹配随机片段仍多提高正确率14.4个百分点，95% CI [9.8, 19.5]。错误答案生成率给出镜像结果：错误侧干预降低16.2点，正确侧干预增加22.6点，方向性 contrast 为38.8点，95% CI [32.1, 45.8]。

联合删除的正确率为25.9%，明显低于单独删除错误侧关键词的52.4%，符合 competition 的抵消结构，而不是 multi-cue synergy。因此，全量正确题对照首先说明 competition 与 hallucination 富集相关；冻结自由生成确认进一步说明特定 competition pair 具有预测一致、超出普通随机删除的方向性作用。
