# KeyShift RealLifeQA Experiment Summary

- Prepared items: 82
- Internal causal items: 28

## Frequency-controlled semantic counterfactuals

| Condition | Accuracy | Mean correct margin | Margin change | W→C | C→W |
|---|---:|---:|---:|---:|---:|
| original | 0.6707 | 1.4939 | 0.0000 | 0 | 0 |
| common_control | 0.6707 | 1.3171 | -0.1768 | 2 | 2 |
| prior_low | 0.6951 | 1.4512 | -0.0427 | 4 | 2 |
| prior_mid | 0.6098 | 1.2515 | -0.2424 | 1 | 6 |
| prior_high | 0.6220 | 1.1189 | -0.3750 | 1 | 5 |
| pdp | 0.7073 | 1.4543 | -0.0396 | 5 | 2 |
| context_link | 0.6951 | 1.7561 | 0.2622 | 6 | 4 |
| joint | 0.7073 | 1.7104 | 0.2165 | 7 | 4 |

## Frequency relationship

- Within-item Spearman(prior shortcut margin, full correct margin): -0.2668 (naive p=3.019e-13; expected negative).
- Item-fixed-effect slope: -0.19972145575771197 with cluster-bootstrap CI [-0.2844429780377079, -0.11264540860162461].

## Detector-gated mitigation

| Method | Accuracy | Net corrections | Trigger rate |
|---|---:|---:|---:|
| pdp | 0.7360 | 3 | 0.3040 |
| context_link | 0.7280 | 2 | 0.3040 |
| joint | 0.7360 | 3 | 0.3040 |

## Internal causal validation

- Target-head ablation mean effect: 0.1429.
- Layer-matched random ablation mean effect: 0.1219.
- Original-state rescue directional success: 0.3214.
- PDP-state suppression directional success: 0.2143.
- Mean mediation fraction proxy: 0.12913672865595943.

## Interpretation guardrails

- The target-model prior probe operationalizes shortcut association; it is not a direct measurement of the inaccessible pretraining corpus frequency.
- LLM-generated paraphrases are retained only after semantic and answer-preservation audits and target-model prior scoring.
- Activation-patching mediation fractions are mechanistic proxies, not identifiable natural indirect effects.
