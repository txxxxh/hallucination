# KeyShift v10 — HaluEval open-answer summary

- Input items: 250
- Shortcut localized: 244 (97.6%)
- Semantic counterfactual complete: 219 (87.6%)

## Sequence-level semantic conditions

| Condition | n | Pair accuracy | Mean correct margin | Margin change | W→C | C→W |
|---|---:|---:|---:|---:|---:|---:|
| original | 219 | 0.406 | -1.285 | +0.000 | 0 | 0 |
| common_control | 219 | 0.393 | -1.438 | -0.152 | 4 | 7 |
| prior_low | 219 | 0.411 | -1.408 | -0.122 | 5 | 4 |
| prior_mid | 219 | 0.379 | -1.452 | -0.167 | 3 | 9 |
| prior_high | 219 | 0.370 | -1.529 | -0.244 | 3 | 11 |
| pdp | 219 | 0.411 | -1.383 | -0.098 | 5 | 4 |
| context_link | 11 | 0.182 | -2.560 | +0.020 | 0 | 0 |
| joint | 11 | 0.182 | -2.443 | +0.137 | 0 | 0 |

## Within-item relationship

- Candidate points: 992
- Spearman(prior shortcut margin, full correct margin): -0.8717757891385988
- Fixed-effect slope: -0.3094676641021567
- Fixed-effect bootstrap 95% CI: [-0.42348730917301936, -0.1792258509156498]

## Internal causal validation

- Eligible items: 87
- Cross-fit runs: 12
- Forward target-minus-random: {'mean_difference': -0.007260826675371192, 'bootstrap_95_ci': [-0.01781694453966115, 0.001553579394730568]}
- Reverse target-minus-random: {'mean_difference': -0.0052968725162447616, 'bootstrap_95_ci': [-0.017171409185476227, 0.004876214863582572]}
- Mean mediation proxy: 0.05554653457453247
