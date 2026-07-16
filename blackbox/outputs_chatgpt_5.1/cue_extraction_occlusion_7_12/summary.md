# Cue intervention and hallucination detection summary

Items processed: 100
At least one valid candidate key: 72/100 (0.720)
Shortcut key detected: 69/100 (0.690)
Constraint key detected: 31/100 (0.310)
Ambiguous decisions: 28/100 (0.280)

## Hallucination detection (gold used only here)

Precision: 0.159
Recall: 0.846
F1: 0.268
Confusion matrix: TP=11, FP=58, TN=1, FN=2

## Intervention ablations

- delete: flips=72/208, mean |delta|=7.162
- emphasis: flips=31/208, mean |delta|=4.142
- negation: flips=45/208, mean |delta|=5.662
- paraphrase: flips=21/208, mean |delta|=2.867
