# Decisive-constraint paraphrase experiment

This intervention tests whether a model's biographical identification errors
are partly driven by memorized co-occurrence between a candidate name and the
proper name of a decisive fact (an award, university, office, or technical
contribution).

## Files

- `question/pretraining_debiased_first100_question.json`: the first 100 items.
- `build_pretraining_debiased_questions.py`: deterministic dataset builder.
- `run_pretraining_debiased_experiment.py`: paired model runner and evaluator.

The current dataset is rewrite version 2. Questions 16, 75, and 76 were
revised after the first pilot because their entity descriptions were not
sufficiently identifiable. Version 1 results are retained separately and must
not be resumed against the version 2 questions.

Each revised item retains the original option order, background clues, answer,
QIDs, and key. Only the final decisive constraint is replaced. Additional
fields record both versions, the intervention type, and the source index.

## Run

From the repository root:

```bash
python -m pip install openai
export OPENAI_API_KEY="your-key"
python scientist_qa/run_pretraining_debiased_experiment.py \
  --model gpt-5-mini \
  --reasoning-effort low \
  --condition both \
  --repetitions 1
```

Use the exact model and reasoning effort from the original experiment when
making a direct comparison. `--condition both` sends both the untouched and
paraphrased version, in seeded shuffled order. The output is checkpointed after
every call, so rerunning the same command resumes incomplete work.

For a small smoke test before spending on all 200 paired calls:

```bash
python scientist_qa/run_pretraining_debiased_experiment.py \
  --model gpt-5-mini --reasoning-effort low --limit 3 \
  --output scientist_qa/benchmark_result/gpt/pretraining_debiased_smoke.json
```

The final `summary` reports accuracy by condition and paired transitions:
`wrong_to_right`, `right_to_wrong`, `unchanged_right`, and `unchanged_wrong`.

## Interpretation note

This is a lexical intervention, not a perfect isolation of pretraining
frequency: the paraphrases are longer and contain extra identifying facts.
The paired original condition controls model/version/run differences, while
the `right_to_wrong` count helps reveal paraphrases that accidentally made a
constraint harder. A stronger follow-up is to add a length-matched control
paraphrase that keeps the original proper name.
