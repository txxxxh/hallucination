# Span-occlusion cue extraction summary
Items processed: 30 (valid: 30)
Mean candidate spans per item: 4.6
Shortcut span identified (above threshold): 13/30
Constraint span identified (above threshold): 20/30
## Hallucination Detection
Primary rule: detected hallucination iff deleting the predicted shortcut cue flips the answer and deleting the predicted constraint cue does not flip the answer.
- True hallucinations: 3/30
- Detected hallucinations: 2/30
- Observable cue-deletion evidence: 4/30
- Accuracy: 0.967
- Precision: 1.000
- Recall: 0.667
- F1: 0.800
- Confusion matrix: TP=2, FP=0, TN=27, FN=1
- Shortcut-cue deletion flips: 2/13 (0.154) (shortcut_to_correct=2, correct_to_shortcut=0)
- Constraint-cue deletion no-flips: 13/20 (0.650)
Directional rule: original answer is the shortcut, shortcut-cue deletion changes it to the correct answer, and constraint-cue deletion keeps the shortcut answer.
- Directional accuracy: 0.967
- Directional precision: 1.000
- Directional recall: 0.667
- Directional F1: 0.800
- Directional confusion matrix: TP=2, FP=0, TN=27, FN=1
## Candidate-Level Gold Cue Evaluation
Strict recognition uses candidate span indices from the gold file; one-token overlap alone is not counted as a hit.
Gold file: `gold_cues_preliminary.jsonl`
- Items evaluated: 30
- Shortcut strict recall (high confidence): 2/10
- Shortcut strict recall (all explicit high+medium): 3/28
- Constraint strict recall: 14/30
- Constraint proposition-overlap recall: 16/30
- Both shortcut and constraint strict: 1/28
## Per-Item Gold Decisions
| ID | Pred shortcut index | Shortcut strict | Pred constraint index | Constraint strict | Proposition overlap | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | None | no | 2 | yes | yes | constraint_only |
| 2 | None | no | None | no | no | neither |
| 3 | 3 | no | 4 | no | yes | neither |
| 4 | 1 | no | 3 | yes | yes | constraint_only |
| 5 | 3 | no | None | no | no | neither |
| 6 | None | no | 2 | yes | yes | constraint_only |
| 7 | None | no | 1 | no | no | neither |
| 8 | 0 | n/a | None | no | no | no_explicit_shortcut |
| 9 | None | no | 2 | yes | yes | constraint_only |
| 10 | None | no | 0 | no | no | neither |
| 11 | None | no | 2 | yes | yes | constraint_only |
| 12 | 2 | no | None | no | no | neither |
| 13 | None | no | 1 | yes | yes | constraint_only |
| 14 | None | no | 2 | yes | yes | constraint_only |
| 15 | 2 | no | None | no | no | neither |
| 16 | None | no | 3 | yes | yes | constraint_only |
| 17 | None | no | 2 | yes | yes | constraint_only |
| 18 | 6 | yes | 4 | yes | yes | both_strict |
| 19 | None | no | 3 | yes | yes | constraint_only |
| 20 | 6 | yes | 3 | no | yes | shortcut_only |
| 21 | 1 | no | None | no | no | neither |
| 22 | 3 | no | None | no | no | neither |
| 23 | None | no | 1 | no | no | neither |
| 24 | None | no | 0 | no | no | neither |
| 25 | 3 | yes | None | no | no | shortcut_only |
| 26 | 1 | n/a | None | no | no | no_explicit_shortcut |
| 27 | 3 | no | None | no | no | neither |
| 28 | None | no | 2 | yes | yes | constraint_only |
| 29 | None | no | 2 | yes | yes | constraint_only |
| 30 | None | no | 2 | yes | yes | constraint_only |
