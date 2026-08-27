# Full Scientist P+R extension summary

All experiments use 2,894 parse-valid rows, stratified outer 80/20 splits for
seeds 42--47, development-only inner selection, and untouched outer tests.

| Method | Mean AUROC | Std | Mean AUPRC | Conclusion |
|---|---:|---:|---:|---|
| Existing chosen-only early fusion | 0.84279 | - | 0.82509 | Reference (pooled reports 291/292) |
| Paired contrastive R (`alternative - chosen`) | **0.86491** | 0.01641 | **0.84498** | Clear improvement; selected in all 6 seeds |
| R predicts cross-fitted P residual | 0.83855 | 0.01499 | 0.81973 | Small +0.00209 over its matched P-only baseline (0.83645) |
| Low-rank P x R interaction | 0.83678 | 0.01650 | 0.81647 | No stable improvement; 2/6 seeds fell back to no interaction |

The paired representation is the only extension with a material and stable
gain. Its mean AUROC improvement over the existing chosen-only early fusion is
approximately +0.02212. Every seed selected the same representation family,
PCA dimension (96), and LR C (0.01).
