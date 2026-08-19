#!/usr/bin/env python3
"""Post-hoc taxonomy joining uncertainty mechanisms with representation OOF."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
A = [json.loads(x) for x in
     (RUNS / "217_scientist_uncertainty_mechanism_items.jsonl").open()]
P = {x["key"]: x for x in map(json.loads,
     (RUNS / "216_known_error_representation_trajectory_predictions.jsonl").open())}
y = np.array([x["error"] for x in A])
nll = np.array([x["answer_nll"] for x in A])
gap = np.array([x["chosen_full_gap"] for x in A])
flip = np.array([x["swap_choice_flip"] for x in A])
rep = np.array([P[x["key"]]["delta_trajectory"] for x in A])

def summary(ix):
    return {"n": int(ix.sum()), "mean_nll": float(nll[ix].mean()),
            "mean_chosen_gap": float(gap[ix].mean()),
            "swap_flip_rate": float(flip[ix].mean()),
            "mean_representation_error_score": float(rep[ix].mean()),
            "representation_detect_rate_at_0.5": float(np.mean(rep[ix] >= .5))}

report = {"definition": ("systematic error = generated wrong candidate also has higher "
                         "teacher-forced mean sequence likelihood than the alternative"),
          "categories": {
              "correct": summary(y == 0),
              "systematic_error": summary((y == 1) & (gap > 0)),
              "generation_likelihood_inconsistent_error": summary((y == 1) & (gap <= 0)),
          }, "certainty_prefixes": {}}
for q in (.25, .5, .75, 1.0):
    selected = nll <= np.quantile(nll, q)
    err = selected & (y == 1)
    report["certainty_prefixes"][str(q)] = {
        "n": int(selected.sum()), "errors": int(err.sum()),
        "systematic_error_fraction": float(np.mean(gap[err] > 0)),
        "strong_wrong_gap_gt_0.5_fraction": float(np.mean(gap[err] > .5)),
        "swap_flip_rate": float(np.mean(flip[err])),
        "representation_detect_rate_at_0.5": float(np.mean(rep[err] >= .5))}
median_false_negative = (y == 1) & (nll < np.median(nll))
both = median_false_negative & (rep < .5)
report["median_nll_false_negatives"] = {
    "n": int(median_false_negative.sum()),
    "systematic_fraction": float(np.mean(gap[median_false_negative] > 0)),
    "strong_wrong_gap_gt_0.5_fraction": float(np.mean(gap[median_false_negative] > .5)),
    "representation_detect_rate": float(np.mean(rep[median_false_negative] >= .5)),
    "also_representation_false_negative_n": int(both.sum()),
    "both_false_negative_systematic_fraction": float(np.mean(gap[both] > 0)),
}
(RUNS / "218_scientist_uncertainty_failure_taxonomy.json").write_text(
    json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
