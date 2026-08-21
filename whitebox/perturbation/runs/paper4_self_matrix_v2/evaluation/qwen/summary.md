# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| qwen | exact | scientist | 0.858 | 0.866 | 0.776 | 0.0% |
| qwen | exact | trivia | 0.910 | 0.902 | 0.831 | 0.0% |
| qwen | exact | gsm8k | 0.784 | 0.912 | 0.720 | 0.0% |
| qwen | attention | scientist | 0.834 | 0.835 | 0.754 | 40.9% |
| qwen | attention | trivia | 0.905 | 0.896 | 0.829 | 40.1% |
| qwen | attention | gsm8k | 0.789 | 0.914 | 0.707 | 35.6% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| qwen | exact | all | 0.840 | 0.926 | 0.687 | 0.0% |
| qwen | exact | athlete | 0.839 | 0.926 | 0.699 | 0.0% |
| qwen | exact | building | 0.904 | 0.970 | 0.754 | 0.0% |
| qwen | exact | musician | 0.712 | 0.785 | 0.596 | 0.0% |
| qwen | attention | all | 0.748 | 0.874 | 0.635 | 30.8% |
| qwen | attention | athlete | 0.755 | 0.870 | 0.621 | 30.8% |
| qwen | attention | building | 0.808 | 0.934 | 0.681 | 30.8% |
| qwen | attention | musician | 0.624 | 0.691 | 0.584 | 30.8% |
