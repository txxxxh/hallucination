# Local white-box reproduction

This experiment applies the paired-label hidden-state method from
*Hallucination Is Linearly Decodable from Mid-Layer Hidden States in
Quantized LLMs* to:

- `../shuffled_prepend_names_question.json`
- `../question_and_result.json`

## Exact setup

- Official repository commit: `ea0b96781809ef35d205dbb01142ad23986f8bb6`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Loading: bitsandbytes 4-bit NF4, double quantization, bfloat16 compute
- Representation: all 29 hidden-state tensors (embedding plus 28 blocks),
  pooled at the last candidate-answer token
- Probe: the official single affine PyTorch probe, AdamW, learning rate
  `1e-3`, weight decay `1e-4`, 30 epochs, batch size 128
- Split: 70/10/20, three seeds (42, 43, 44)
- Input length cap: 512 tokens. No local sample was truncated (maximum
  lengths: 211 and 110 tokens).

Each original question yields two candidate-conditioned examples: the correct
answer (label 1) and incorrect answer (label 0). This is the paper's
paired-label protocol; it detects whether a supplied candidate is true, rather
than detecting a free generation without seeing a candidate.

## Results

| Dataset / transfer | Evaluation | Layer selection | Layer | AUROC |
|---|---|---:|---:|---:|
| shuffled_prepend_names | paper item-random | test oracle (paper style) | 17 | 0.7960 ± 0.0078 |
| shuffled_prepend_names | question-grouped | test oracle (paper style) | 17 | 0.8099 ± 0.0122 |
| shuffled_prepend_names | question-grouped | validation selected | 18 | 0.8097 |
| question_and_result | paper item-random | test oracle (paper style) | 23 | 0.9995 ± 0.0004 |
| question_and_result | question-grouped | test oracle (paper style) | 25 | 0.9992 ± 0.0009 |
| question_and_result | question-grouped | validation selected | 18 | 0.9985 |
| names → question_and_result | cross-dataset | source-validation selected | 18 | 0.4110 ± 0.0046 |
| question_and_result → names | cross-dataset | source-validation selected | 18 | 0.4741 ± 0.0053 |

The names benchmark therefore does **not** retain the paper's reported
0.904–1.000 AUROC range. The second benchmark is almost perfectly separable
even with question-grouped splitting, but this direction does not transfer to
the names benchmark. Conversely, the names-trained direction does not transfer
to the second benchmark. The strong second-dataset score should therefore be
interpreted as dataset-specific linear decodability, not evidence for one
universal hallucination direction.

## Run

```bash
/home/tong56/venvs/whitebox/bin/python scripts/run_custom_whitebox.py \
  --stage extract --batch-size 8 --force

/home/tong56/venvs/whitebox/bin/python scripts/run_custom_whitebox.py \
  --stage evaluate

/home/tong56/venvs/whitebox/bin/python scripts/plot_custom_whitebox_results.py
```

Raw layer-by-layer, seed-level metrics are under `results/custom_metrics/`.
The normalized paired examples are under `data/custom/`, and extracted
representations are under `results/custom_hidden_states/`.
