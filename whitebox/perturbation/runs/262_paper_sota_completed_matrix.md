# Completed paper-method baseline matrix

Metric: error-positive AUROC. Model: `NousResearch/Meta-Llama-3.1-8B-Instruct`.

| Dataset | Perturbation (ours) | Repr. Aiersilan (2026) | Repr. ICR Probe (2025) | Repr. SAPLMA (2023) | Uncert. K=6 disagreement | Uncert. Semantic Entropy (2024) | MiniCheck unilateral | MiniCheck contrastive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scientist | **0.902** | **0.770** | **0.512** | **0.671** | **0.577** | **0.518** | **0.717** | **0.926** |
| TriviaQA | **0.965** | **0.873** | **0.798** | **0.840** | **0.786** | **0.708** | **0.658** | **0.823** |
| GSM8K | **0.743** | **0.806** | **0.736** | **0.719** | **0.821** | **0.503** | **0.558** | **0.506** |
| DROP | **0.922** | **0.826** | **0.808** | **0.869** | **0.760** | **0.752** | **0.604** | **0.526** |

## Protocol and provenance

- Perturbation Scientist is the exact-enumeration current127 result, 3 seeds x grouped 5-fold CV, n=1084: `runs/153_exact_current127_samecv_report.json`. This is the 0.902 version; the later unified audit's 0.886 is not used.
- Perturbation TriviaQA/GSM8K: `runs/paper4_matrix/evaluation/evaluation.json`; DROP: `runs/168_drop1000_fast_stage1_report.json`.
- Representation uses the leakage-free 0.770 protocol: fixed layer 14, last-token plus mean-token blocks, fold-local scaling and PCA8 per block, LR C=0.03, and grouped 3x5 OOF. Scientist source: `runs/216_known_error_representation_trajectory.json` (parse-valid n=1076). TriviaQA/GSM8K/DROP sources: `runs/263_representation_0770_protocol/{trivia,gsm8k,drop}/report.json` (n=1000/942/1000).
- ICR Probe follows [Zhang et al. (2025)](https://arxiv.org/abs/2507.16488), Algorithm 1: per-layer information-content-rate features, top-k=20 attended tokens, sample-standardized softmax/JSD, answer-token mean over all 32 layers, and the paper's 32-128-64-32-1 probe (BatchNorm, LeakyReLU, dropout 0.3, Adam 5e-4, 50 epochs). To prevent repeated-question leakage in the reconstructed two-candidate data, evaluation is grouped 3-seed x 5-fold OOF rather than the paper's ordinary train/test split. Sources: `runs/265_icr_probe_paper/{scientist,trivia,gsm8k,drop}/report.json` (n=1084/1000/942/1000); values are the mean of the three OOF AUROCs.
- SAPLMA follows [Azaria and Mitchell (2023)](https://arxiv.org/abs/2304.13734): generated-answer final-token activation at layer 28 and the paper's 4096-256-128-64-1 ReLU MLP, Adam, 5 epochs. Evaluation is likewise grouped 3-seed x 5-fold OOF to prevent pair/group leakage. Sources: `runs/264_saplma_paper/{scientist,trivia,gsm8k,drop}/report.json` (n=1084/1000/942/1000); values are the mean of the three OOF AUROCs, not the post-hoc probability ensemble.
- Uncertainty is K=6 stochastic-answer disagreement with the fixed greedy answer (temperature 0.7, top-p 0.95), after dataset-specific answer canonicalization. Scientist/TriviaQA/DROP are new full-matrix runs in `runs/261_paper_baseline_matrix/*/samples.jsonl` and `report.json`. GSM8K is the previously completed formal n=300 run, whose exact 0.820844 result is in `runs/239_gsm8k_uncertainty_methods/report.json`.
- Semantic Entropy follows [Farquhar et al. (2024)](https://www.nature.com/articles/s41586-024-07421-0): M=10 samples, temperature 1, top-p 0.9, top-k 50, length-normalized sequence log probabilities, and bidirectional semantic-equivalence clustering. The paper's formally evaluated `microsoft/deberta-large-mnli` entailment variant is used in place of its API-based GPT-3.5 judge. Sources: `runs/266_semantic_entropy_paper/{scientist,trivia,gsm8k,drop}/report.json` (n=1084/1000/942/1000). Main-table values are probability-weighted semantic entropy; the reports also retain discrete semantic entropy AUROCs (0.518/0.711/0.503/0.755).
- MiniCheck unilateral is the paper's one-candidate evidence-support score (one minus minimum sentence support). MiniCheck contrastive is the project-added two-candidate contrastive extension, not a method claimed by the MiniCheck paper. Scientist source: `runs/221_scientist_minicheck_flan/report.json`; other datasets: `runs/260_candidate_minicheck_matrix/{trivia,gsm8k,drop}/report.json`.
- Do not use the `uncertainty: 0.5` field inside the GSM8K `runs/261.../report.json`: it is an explicit placeholder used only to let the shared evaluator write the representation report. The table's GSM8K uncertainty cell is exclusively sourced from run 239.
