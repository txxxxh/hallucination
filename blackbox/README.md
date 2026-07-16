# TrapQA Submission Artifact

This directory contains the supplementary data artifact for **TrapQA**, a diagnostic benchmark for studying hallucination as inference misalignment. TrapQA contains two complementary components:

- **Scientist QA**: entity-disambiguation questions over highly similar scientist profiles, with names-only and profiles-in-context prompt variants plus closed-book factual probes.
- **Real-Life Constrained QA**: everyday two-option scenarios where a salient shortcut conflicts with a physical, spatial, procedural, or medium-specific constraint stated in the prompt.

This artifact is intended for research and review. It contains benchmark items, labels, saved model responses, and incorrect-response subsets. It is **not** a full standalone code release for regenerating the datasets or rerunning every API call.

## Directory Structure

```text
submission/
  README.md

  real_life_constrained_qa/
    question_and_result.json
    README.md

  scientist_qa/
    README.md
    question/
      shuffled_prepend_names_question.json
      shuffled_prepend_profiles_question.json
      probe_question.json
      probe_question_answer.json
    benchmark_result/
      claude/
        prepend_names/
        prepend_profiles/
        probe/
      deepseek/
        prepend_names/
        prepend_profiles/
        probe/
      gemini/
        prepend_names/
        prepend_profiles/
        probe/
      gpt/
        prepend_names/
        prepend_profiles/
        probe/

  wiki_data/
    all_scientist_profile.json
    README.md
```

## Scientist QA

`scientist_qa/question/` contains the Scientist QA question files:

| File                                      | Description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| `shuffled_prepend_names_question.json`    | Names-only prompt variant. Each item provides two candidate names and a disambiguating question. |
| `shuffled_prepend_profiles_question.json` | Profiles-in-context prompt variant. Items are aligned by index with the names-only file. |
| `probe_question.json`                     | Supplementary closed-book probe questions. Each Scientist QA item has two probes. |
| `probe_question_answer.json`              | Gold binary labels for the probe questions, aligned by index with `probe_question.json`. |

The benchmark uses **2,925** Scientist QA base questions and **5,850** probe questions. For base question index `i`, the corresponding probes are located at indices `2*i` and `2*i + 1` in the probe files.

`scientist_qa/benchmark_result/` contains saved responses and incorrect-response subsets for the evaluated model families. Each model directory is organized by prompt condition:

```text
benchmark_result/<model>/
  prepend_names/
    low_tk_rsp.json
    high_tk_rsp.json
    low_tk_hallucination.json
    high_tk_hallucination.json
  prepend_profiles/
    low_tk_rsp.json
    high_tk_rsp.json
    low_tk_hallucination.json
    high_tk_hallucination.json
  probe/
    probe_low_tk_rsp.json
    probe_high_tk_rsp.json
```

Response files are aligned by list index with the corresponding question file. Hallucination files contain only incorrect or unmatched responses.

## Real-Life Constrained QA

`real_life_constrained_qa/question_and_result.json` contains **500** everyday two-option scenarios. Each item includes a scenario, two options, the gold answer, a short justification, and model-error annotations used in the paper.

Typical fields include:

| Field                 | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `question`            | The natural-language scenario.                               |
| `options`             | Two candidate actions or choices.                            |
| `answer`              | The gold option index.                                       |
| `correct_option`      | The text of the correct option.                              |
| `mistake_models`      | Models that selected the shortcut/wrong option in our evaluation. |
| `short_justification` | Brief explanation of the decisive constraint.                |

Real-Life Constrained QA is synthetic. It is intended to test whether models follow prompt-grounded constraints when a salient shortcut option is tempting.

## Wiki Data

`wiki_data/all_scientist_profile.json` contains public scientist-profile metadata used for Scientist QA construction. The file contains structured public attributes and identifiers used for profile matching, bookkeeping, and candidate linking.

Scientist QA necessarily contains real scientist names because the task is entity disambiguation. The release is limited to public benchmark-relevant attributes and does not include private contact information, images, surveillance data, or other private personal data.

## Source and License Notes

- Scientist QA is constructed from Wikipedia-linked and Wikidata-linked public scientist metadata. Wikipedia text is generally available under CC BY-SA 4.0 unless otherwise noted, while Wikidata structured data is available under CC0.
- Real-Life Constrained QA uses SWOW-derived lexical associations only for seed selection. This release does **not** redistribute raw SWOW participant records or raw cue-response tables.
- Model outputs are saved for research and verification of the reported results. No model weights, trained models, or tool-using agents are released.
- Please follow the license and usage terms included with the final public release of this artifact.

## Intended Use

This artifact is intended for research on:

- LLM hallucination;
- inference misalignment;
- knowledge deployment;
- constraint-sensitive reasoning;
- diagnostic evaluation of language models.

It should **not** be used as a general model-safety certificate, deployment-readiness benchmark, individual assessment tool, or comprehensive measurement of model reliability across all domains.

## Citation

If you use this artifact, please cite the TrapQA paper and the original resources used in construction, including SWOW, Wikipedia/Wikidata, and the evaluated model providers where appropriate.