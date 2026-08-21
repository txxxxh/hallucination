# Llama-3.1 reconstructed two-candidate detection matrix

Metric: error-positive AUROC. Values for supervised probes are the mean of three
group-aware 5-fold OOF runs. Model under test is
`NousResearch/Meta-Llama-3.1-8B-Instruct`.

| Dataset | Perturbation Detector（Perturbation） | Candidate Likelihood Gap（Uncertainty） | Paired Hidden-State Probe（Representation） | MiniCheck（Evidence） | MiniCheck（Contrastive） |
|---|---:|---:|---:|---:|---:|
| Scientist | 0.886 | 0.666 | 0.882 | 0.717 | 0.926 |
| TriviaQA | 0.965 | 0.785 | 0.973 | 0.658 | 0.823 |
| GSM8K | 0.743 | 0.524 | 0.739 | 0.558 | 0.506 |
| DROP | 0.922 | 0.229 | 0.865 | 0.604 | 0.526 |

## Protocol notes

- The primary generated answer and manifest-supplied alternative are fixed before
  detector evaluation. Error labels always refer to the primary generated answer.
- Scientist has 1,084 rows for perturbation/likelihood/representation. The official
  sentence-level MiniCheck run uses the 1,076 parse-valid common-key subset (453
  errors). TriviaQA has 1,000 rows (500 errors), GSM8K 942 (471 errors), and DROP
  1,000 (500 errors).
- MiniCheck Evidence is the original one-candidate evidence-support method: error
  score is one minus the minimum sentence support. MiniCheck Contrastive subtracts
  the chosen support from the alternative support. The latter is a project-added
  oracle candidate-conditioned feature, not a method proposed by the MiniCheck
  paper.
- For Scientist, the whole-response MiniCheck alternatives are Evidence 0.781 and
  Contrastive 0.978. They are not the primary cells because MiniCheck is documented
  as a sentence-level checker; the table uses atomic-min aggregation consistently.
- The alternative answer is often gold/reference-derived. Contrastive results are
  therefore oracle-conditioned and are not deployment-style hallucination
  detection results.
- Candidate Likelihood Gap is the paired uncertainty baseline required by the
  strict “every method reads both candidates” rule. It is not Semantic Entropy or
  the published sampling-based uncertainty SOTA and must not be labeled as such.
- The old GSM8K perturbation AUROC 0.949 used 614 formatting/parse failures among
  942 examples. On the reconstructed natural-error balanced set, the comparable
  perturbation AUROC is 0.743 (an independent current127 rerun gives 0.747 mean and
  0.753 ensemble).

## Exact result sources

- Perturbation Scientist/TriviaQA/GSM8K:
  `runs/paper4_matrix/evaluation/evaluation.json`
- Perturbation DROP: `runs/168_drop1000_fast_stage1_report.json`
- Candidate likelihood and paired representation:
  `runs/260_candidate_ur_matrix/report.json`
- MiniCheck Scientist: `runs/221_scientist_minicheck_flan/report.json`
- MiniCheck other datasets:
  `runs/260_candidate_minicheck_matrix/{trivia,gsm8k,drop}/report.json`
