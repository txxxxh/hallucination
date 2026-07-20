# KeyShift RealLifeQA Experiment Summary

- Prepared items: 27
- Internal causal items: 8

## Frequency-controlled semantic counterfactuals

| Condition | Accuracy | Mean correct margin | Margin change | W→C | C→W |
|---|---:|---:|---:|---:|---:|
| original | 0.3704 | -1.0093 | 0.0000 | 0 | 0 |
| common_control | 0.3704 | -0.9491 | 0.0602 | 1 | 1 |
| prior_low | 0.4444 | -0.8519 | 0.1574 | 2 | 0 |
| prior_mid | 0.3333 | -1.0139 | -0.0046 | 0 | 1 |
| prior_high | 0.3704 | -1.1481 | -0.1389 | 0 | 0 |
| pdp | 0.4444 | -0.8241 | 0.1852 | 2 | 0 |
| context_link | 0.5000 | -0.1042 | 0.5625 | 2 | 1 |
| joint | 0.6667 | 0.1667 | 0.8333 | 3 | 1 |

## Frequency relationship

- Within-item Spearman(prior shortcut margin, full correct margin): -0.3031 (naive p=6.648e-06; expected negative).
- Item-fixed-effect slope: -0.1918079992422269 with cluster-bootstrap CI [-0.3906852795199523, -0.03707267135088105].

## Detector-gated mitigation

| Method | Accuracy | Net corrections | Trigger rate |
|---|---:|---:|---:|
| pdp | 0.7280 | 2 | 0.3040 |
| context_link | 0.7280 | 2 | 0.3040 |
| joint | 0.7360 | 3 | 0.3040 |

## Internal causal validation

- Held-out target-head ablation mean effect: 0.0781.
- Attention-matched random ablation mean effect: -0.0531.
- Held-out forward-patch directional success: 0.7500.
- Held-out reverse-rescue directional success: 0.7500.
- Both patch directions succeed: 0.6250.
- Mean mediation fraction proxy: 0.2938391265597148.

## Interpretation guardrails

- The target-model prior probe operationalizes shortcut association; it is not a direct measurement of the inaccessible pretraining corpus frequency.
- LLM-generated paraphrases are retained only after semantic and answer-preservation audits and target-model prior scoring.
- Activation-patching mediation fractions are mechanistic proxies, not identifiable natural indirect effects.
