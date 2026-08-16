# Stressor interventions v2: first 1500 ScientistQA records

Run date: 2026-07-24

## Coverage

- Input keys: `question_0000` through `question_1499`
- Collected records: 1500
- Unique keys: 1500
- Missing tensor traces: 0
- Model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4
- Hidden states: all 29 outputs, last generated-answer token

The collection was split into two 750-record GPU workers and merged before
training. Generation used temperature 0. Raw tensor traces remain under
`/tmp/stressor_v2_first1500_{a,b}`; this directory contains the compact
records, trained classifiers, held-out predictions, and metrics.

## Teacher-label distribution

| Cause | Count |
|---|---:|
| correct | 676 |
| knowledge_gap | 687 |
| contextual_interference | 51 |
| known_but_unlocalized | 86 |

Of 1500 generations, 1340 were parseable. Interventions ran for 137 records
and produced 108 positive culprit-span labels among 685 candidate spans.

## Held-out results

| Head / ablation | Selected layer | Balanced accuracy | Macro F1 | Macro AUROC |
|---|---:|---:|---:|---:|
| Hallucination head | 16 | 0.630 | 0.630 | 0.648 |
| Cause head, full v2 features | 0 | 1.000 | 1.000 | 1.000 |
| Cause, base hidden only | 11 | 0.553 | 0.541 | 0.572 |
| Cause, base + probe hidden, no scores/gating | 6 | 0.624 | 0.563 | 0.754 |
| Span culprit head | 21 | 0.423 | 0.430 | 0.409 |

The full cause-head score is not evidence of perfect hidden-state
decodability. In v2, interventions run only for wrong answers whose knowledge
probe is already classified as known. Knowledge-gap rows therefore have zero
transition blocks while contextual-interference rows have nonzero transition
blocks. Feature inspection confirmed that the highest-scoring selected
features were intervention-transition coordinates. Treat 1.000 as routing
policy leakage.

The stricter `base + probe hidden` ablation is the most relevant current
estimate for distinguishing knowledge gaps from interference without explicit
probe scores or intervention-presence gating. Its held-out AUROC is 0.754, but
the contextual-interference test support is only 10 records, so uncertainty is
large.

The span classifier did not generalize: it found only 2 of 22 positive culprit
spans in the held-out set. Its raw accuracy of 0.650 is misleading because of
the 118-negative / 22-positive test imbalance.

## Artifacts

- `training_summary.json`: full layer scans and held-out metrics
- `cause_ablation_summary.json`: strict hidden-state cause ablations
- `heldout_reason_predictions.jsonl`: held-out full cause-head predictions
- `teacher_records.jsonl`: compact teacher labels and intervention metadata
- `models/*.joblib`: trained full-v2 classifier bundles
- `collection_summary_part_{a,b}.json`: collection summaries for both shards
