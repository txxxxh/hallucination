# Span-centered v3 with Llama-3.1-8B: first 1500 ScientistQA records

Run date: 2026-07-24

## Run configuration and integrity

- Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`
- Quantization: 4-bit NF4, double quantization, bfloat16 compute
- Dataset: first 1500 rows of `shuffled_prepend_names_question.json`
- Profiles: aligned `shuffled_prepend_profiles_question.json`
- Span readout: enabled
- Maximum intervention spans per routed item: 5
- Intervention operators: delete, neutralize, mask
- Hidden states: all 33 outputs, including embedding output 0
- Collection: two concurrent 750-item workers on one A100 40GB
- GPU collection time: about 52 minutes
- Collected records: 1500
- Unique keys: 1500
- Coverage: `question_0000` through `question_1499`, in source order
- Collection errors: 0
- Missing tensor traces: 0

The two collection shards were merged before any classifier training. All
layer selection used validation/train-only data; test metrics were not used to
choose a layer.

## Teacher-label distribution

| Cause | Count |
|---|---:|
| correct | 785 |
| knowledge_gap | 652 |
| contextual_interference | 23 |
| known_but_unlocalized | 40 |

There were 1485 parseable base answers. Interventions ran for 63 routed items,
yielding 315 candidate-span rows and 42 positive culprit-span labels.

## Held-out results

| Head | Selected layer | Test support | Balanced accuracy | Macro F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Hallucination | 14 | 297 | 0.618 | 0.618 | 0.678 |
| Cause: contextual interference vs knowledge gap | 25 | 135 | 0.673 | 0.623 | 0.855 |
| Culprit span | 0 | 65 | 0.502 | 0.501 | 0.591 |

### Hallucination head

- Test classes: 157 correct, 140 hallucination
- Confusion matrix, rows/columns = correct, hallucination:
  `[[101, 56], [57, 83]]`

The global last-answer-token hidden probe is moderately above chance, but much
weaker than the near-perfect dataset-specific result previously observed on
`question_and_result.json`.

### Cause head

- Features: base-answer hidden plus option-ordered, per-person probe hidden
  mean/std
- Explicitly excluded: probe accuracy, knowledge-state value, intervention
  presence, intervention recovery, and answer correctness
- Test classes: 5 contextual interference, 130 knowledge gap
- Confusion matrix, rows/columns = contextual interference, knowledge gap:
  `[[2, 3], [7, 123]]`

The AUROC of 0.855 is encouraging and does not have the v2 zero-vs-nonzero
intervention gating leak. However, only five contextual-interference examples
are present in the test split, so the estimate has high variance. At the
selected threshold the model detects 2/5 contextual cases and produces seven
false positives.

Probe prompts are constructed from the aligned profile teacher. The
classifier does not consume probe correctness, but this is still a
profile-supervised rather than label-free detector.

### Culprit-span head

- Test classes: 58 non-culprit, 7 culprit
- Confusion matrix, rows/columns = non-culprit, culprit:
  `[[50, 8], [6, 1]]`
- Only 1/7 positive spans is recovered at the selected threshold.

The AUROC improves numerically over the earlier Qwen v2 span result (0.409 to
0.591), but it is not evidence that contextual span hidden states have been
successfully decoded. Validation selected layer 0, the embedding output.
Feature inspection found:

| Selected feature family | Coordinates |
|---|---:|
| Original span token pools | 74 |
| Span/answer scalar relations | 0 |
| Full-context span readout | 0 |
| Fixed-answer transition | 0 |
| Regenerated-answer transition | 49 |
| Aligned span delta mean | 76 |
| Aligned span delta max-abs | 57 |

Among the 20 largest classifier coefficients, 11 are regenerated-answer
transition coordinates and six are aligned max-absolute span deltas. At layer
0 these mainly encode generated/replacement token identity rather than
mid-layer contextual semantics. The span score is therefore weak and largely
lexical.

## Main conclusion

The strongest scientifically useful result is the no-gating cause head:
base-answer plus isolated per-person probe hidden states rank contextual
interference above knowledge gaps with AUROC 0.855, albeit with only five
positive test examples.

The new span-centered representation improves the raw span AUROC, but does not
yet solve localization. A stricter next experiment should exclude embedding
layer 0, evaluate feature-family ablations, increase the number of contextual
interference training items, and report repeated grouped splits or grouped
cross-validation.

## Artifacts

- `training_summary.json`: complete layer scans and held-out metrics
- `heldout_reason_predictions.jsonl`: held-out cause predictions
- `span_feature_audit.json`: selected span feature-family audit
- `teacher_records.jsonl`: teacher labels and intervention metadata
- `models/*.joblib`: trained classifier bundles
- `collection_summary_part_a.json`, `collection_summary_part_b.json`: shard summaries
- `collection_config_part_a.json`, `collection_config_part_b.json`: exact collection configs

Raw tensor traces remain in:

- `/tmp/stressor_interventions_v3_llama_first1500_a`
- `/tmp/stressor_interventions_v3_llama_first1500_b`

They were not copied into the nearly-full home workspace.
