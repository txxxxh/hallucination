# Real-Life Constrained QA

Real-Life Constrained QA is a diagnostic question set for testing whether large
language models follow prompt-grounded constraints in everyday action-choice
scenarios. Each item presents a short realistic situation and two candidate
options. One option is a salient but incorrect shortcut, while the other option
satisfies a physical, spatial, procedural, or medium-specific constraint stated
in the prompt.

This dataset is part of **TrapQA**, a benchmark for studying hallucination as
inference misalignment: models may follow statistically salient shortcuts even
when the prompt contains the decisive constraint.

## Dataset Contents

The dataset contains **500 two-option questions** covering **13 aspects of daily
life**. Each example includes:

- a natural-language scenario;
- two candidate options;
- the gold answer;
- metadata describing the underlying constraint pattern;
- model-result annotations used in our paper.

The main data file is:

`question_and_result.json`


## Task Format

Each question asks the model to choose between two options. For example, a
scenario may describe a task where the tempting option is normally reasonable,
but violates a constraint in the prompt. The model is correct only if it selects
the option that satisfies the full scenario constraint.

## Result Counts
| Model | Thinking level | Hallucinations |
| :-- | ---: |----------------|
| claude Sonnet 4.6 | low | 81             |
| Deepseek  V3.2 | chat(non-reasoning) | 182            |
| gemini-3.1-pro-preview | low | 18             |
| gpt-5.2 | low | 44             |

## Intended Use

This dataset is intended for research on:

* LLM hallucination;
* constraint-sensitive reasoning;
* shortcut-driven failures;
* action-choice reliability;
* diagnostic evaluation of language models.

It should **not** be used as a general safety certificate, deployment-readiness
benchmark, or comprehensive measure of model reliability.

## Construction Summary

We use human word-association cues from the Small World of Words (SWOW) resource
to identify salient shortcut associations. These cues are used only for seed
selection. The final released dataset contains synthetic QA items and does not
redistribute raw SWOW participant data or raw cue--response tables.

Items are generated, augmented, and filtered to ensure that each question is
realistic, self-contained, and has a single intended correct answer.

## License and Source Notice

Real-Life Constrained QA is a derived diagnostic QA asset. Please follow the
license and usage terms provided with the released dataset package.

This dataset uses SWOW-derived cues for seed selection. Raw SWOW data are not
redistributed in this release. Please cite the original SWOW resource when using
this dataset.