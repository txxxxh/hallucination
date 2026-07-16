# ScientistQA Data

This directory contains the ScientistQA question set and benchmark results used for submission. The dataset has 2,925 base questions.

The question keys are numbered sequentially from `question_0000` to `question_2924`.

For probe questions, each base question has two corresponding probe items. If a base question has index `i`, its probe question/answer indices are:

```text
i * 2
i * 2 + 1
```

The probe count is 5,850.

## Question Files

Files are under `question/`.

| File | Count | Description |
| --- | ---: | --- |
| `shuffled_prepend_names_question.json` | 2,925 | Base questions where only candidate names are prepended. |
| `shuffled_prepend_profiles_question.json` | 2,925 | Base questions where full candidate profiles are prepended. Items with the same index are variants of the same base question as `shuffled_prepend_names_question.json`. |
| `probe_question.json` | 5,850 | Probe questions. Every two consecutive probe questions correspond to one base question. |
| `probe_question_answer.json` | 5,850 | Gold binary answers for `probe_question.json`, aligned by index. |

## Benchmark Results

Benchmark results are under `benchmark_result/<model>/`, where `<model>` is one of:

```text
claude deepseek gemini gpt
```

Each model directory uses this layout:

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

`low_tk` and `high_tk` denote low thinking effort and high thinking effort, respectively.

Response files (`*_rsp.json`) are aligned by list index with the corresponding question file:

- `prepend_names/*_rsp.json` has 2,925 items aligned with `question/shuffled_prepend_names_question.json`.
- `prepend_profiles/*_rsp.json` has 2,925 items aligned with `question/shuffled_prepend_profiles_question.json`.
- `probe/probe_*_rsp.json` has 5,850 items aligned with `question/probe_question.json`.

Hallucination files (`*_hallucination.json`) contain only incorrect responses. Each item includes:

- `question_id`: the base-question index.
- `question`: the question object at that index.
- `response`: the model response that was judged incorrect.

Raw response and batch request files are not included in this submission directory.

## Result Counts

| Model | Split | Low responses | High responses | Low hallucinations | High hallucinations |
| --- | --- | ---: | ---: | ---: | ---: |
| claude | prepend_names | 2,925 | 2,925 | 699 | 182 |
| claude | prepend_profiles | 2,925 | 2,925 | 5 | 0 |
| deepseek | prepend_names | 2,925 | 2,925 | 1,089 | 309 |
| deepseek | prepend_profiles | 2,925 | 2,925 | 99 | 0 |
| gemini | prepend_names | 2,925 | 2,925 | 73 | 92 |
| gemini | prepend_profiles | 2,925 | 2,925 | 0 | 0 |
| gpt | prepend_names | 2,925 | 2,925 | 344 | 300 |
| gpt | prepend_profiles | 2,925 | 2,925 | 0 | 0 |

All models have 5,850 probe responses for both `probe_low_tk_rsp.json` and `probe_high_tk_rsp.json`.

## Benchmark Details
To benchmark models using **ScientistQA**, use the following system prompt:
```python
LLM_BENCHMARK_SYSTEM_PROMPT = """
## Task: Answer the Biographical Identification Question

You will be given a short biographical paragraph describing a person.
Your task is to:

1. Identify the full name of the person being described. You MUST provide an answer.
2. Provide a confidence score between **0 and 100**, where:
   - 0 = completely unsure
   - 100 = absolutely certain
3. Output json style string as illustrated below. Make your answer short.

### Requirements

- Output only the final answer.
- Do not include explanations.
- Do not include reasoning steps.
- Do not include additional commentary.
- The confidence score must reflect your internal certainty.

### Output Format
{
    "answer": "...", # The full name of the person you believe is being described.
    "confidence": ... # Your confidence score (0-100) in your answer.
}

""".strip()
```
Note that sometimes the model outputs a name with the middle name omitted. In our benchmark setting such answers are manually verified and normalized to the original name if they point to the same person.
For convenience, you can modify the system prompt to ask the model to output a 0/1-style label.

Meanwhile, the system prompt for the probe questions is:
```python
LLM_BENCHMARK_PROBE_SYSTEM_PROMPT = """
You will be presented with a question asking about some fact about a person. The answer to the questions is either yes or no. 
Answer the question based only on your parametric knowledge, without using any external sources or tools.
After answering, provide a short justification.
Output format:
{
    "answer": "...", # The answer to the question. Use 0 for no and 1 for yes.
    "justification": "...", # A short justification for your answer.
}
""".strip()
```