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
