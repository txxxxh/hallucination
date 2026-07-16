# Delete-uncertain/person-flip → negation-nonflip detection summary

Items processed: 50
Original answer uncertain (3): 0/50 (0.000)
Items with at least one delete→uncertain span: 38/50 (0.760)
Items with at least one delete person-flip span: 22/50 (0.440)
Items entering negation stage: 41/50 (0.820)
Predicted hallucination: 31/50 (0.620)
Ambiguous decisions: 7/50 (0.140)

## Hallucination detection (gold joined only after prediction)

Classified coverage: 43/50 (0.860)
Precision: 0.387
Recall: 0.750
F1: 0.511
Accuracy: 0.465
Confusion matrix: TP=12, FP=19, TN=8, FN=4

## Intervention diagnostics

- deletion: uncertain=186, person_flips=81, informative=267/631
- negation among informative spans: nonflips=62, flips=167, uncertain_outputs=135, measured=229

## Decision rule

- Answers use 1/2 for the two profiles and 3 for not uniquely identifiable.
- delete_prediction=3 OR deletion person-flip → enter negation stage.
- At least one selected span with negation_flip=False → hallucination.
- No informative deletion → non-hallucination.
- All selected spans validly measured and negation_flip=True → non-hallucination.
- Invalid/missing negation evidence without a non-flip witness → ambiguous.
