# Cue intervention and hallucination detection summary

Items processed: 84
At least one valid candidate key: 69/84 (0.821)
Shortcut key detected: 3/84 (0.036)
Constraint key detected: 67/84 (0.798)
Ambiguous decisions: 15/84 (0.179)

## Hallucination detection (gold used only here)

Precision: 0.333
Recall: 0.077
F1: 0.125
Confusion matrix: TP=1, FP=2, TN=54, FN=12

## Intervention ablations

- delete: flips=68/204, mean |delta|=7.900
- emphasis: flips=37/204, mean |delta|=4.913
- negation: flips=49/204, mean |delta|=6.622
