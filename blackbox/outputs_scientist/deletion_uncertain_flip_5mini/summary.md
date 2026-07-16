# Delete-uncertain/person-flip → negation-nonflip detection summary

Items processed: 50
Original answer uncertain (3): 13/50 (0.260)
Items with at least one delete→uncertain span: 20/50 (0.400)
Items with at least one delete person-flip span: 4/50 (0.080)
Items entering negation stage: 23/50 (0.460)
Predicted hallucination: 1/50 (0.020)
Ambiguous decisions: 16/50 (0.320)

## Hallucination detection (gold joined only after prediction)

Classified coverage: 34/50 (0.680)
Precision: 1.000
Recall: 0.333
F1: 0.500
Accuracy: 0.941
Confusion matrix: TP=1, FP=0, TN=31, FN=2

## Intervention diagnostics

- deletion: uncertain=25, person_flips=4, informative=29/498
- negation among informative spans: nonflips=1, flips=25, uncertain_outputs=7, measured=26

## Decision rule

- Answers use 1/2 for the two profiles and 3 for not uniquely identifiable.
- delete_prediction=3 OR deletion person-flip → enter negation stage.
- At least one selected span with negation_flip=False → hallucination.
- No informative deletion → non-hallucination.
- All selected spans validly measured and negation_flip=True → non-hallucination.
- Invalid/missing negation evidence without a non-flip witness → ambiguous.
