# DROP evaluation

3 seeds x 5-fold grouped OOF; same feature/PCA/LR settings as 159_evaluate_paper4_matrix.py

| Model | Method | N | Positive | AUROC | AUPRC | Accuracy | Bal. Acc. | Macro-F1 | Query reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama | exact | 1000 | 500 | 0.919 | 0.911 | 0.836 | 0.836 | 0.836 | 0.0% |
| llama | attention | 1000 | 500 | 0.915 | 0.901 | 0.833 | 0.833 | 0.832 | 46.6% |
| llama | gradient | 1000 | 500 | 0.915 | 0.903 | 0.835 | 0.835 | 0.834 | 43.5% |
| mistral | exact | 1000 | 500 | 0.968 | 0.959 | 0.912 | 0.912 | 0.912 | 0.0% |
| mistral | attention | 1000 | 500 | 0.969 | 0.961 | 0.910 | 0.910 | 0.910 | 46.3% |
| mistral | gradient | 1000 | 500 | 0.965 | 0.955 | 0.908 | 0.908 | 0.908 | 42.9% |
| qwen | exact | 1000 | 500 | 0.907 | 0.909 | 0.819 | 0.819 | 0.819 | 0.0% |
| qwen | attention | 1000 | 500 | 0.902 | 0.902 | 0.812 | 0.812 | 0.812 | 46.7% |

Qwen gradient is absent because generation was intentionally skipped.
