# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| falcon3 | exact | scientist | 0.645 | 0.652 | 0.609 | 0.0% |
| falcon3 | exact | trivia | 0.897 | 0.915 | 0.810 | 0.0% |
| falcon3 | exact | gsm8k | 0.872 | 0.925 | 0.787 | 0.0% |
| falcon3 | attention | scientist | 0.650 | 0.655 | 0.612 | 41.5% |
| falcon3 | attention | trivia | 0.898 | 0.916 | 0.813 | 41.2% |
| falcon3 | attention | gsm8k | 0.872 | 0.925 | 0.790 | 35.5% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| falcon3 | exact | all | 0.626 | 0.784 | 0.552 | 0.0% |
| falcon3 | exact | athlete | 0.729 | 0.839 | 0.544 | 0.0% |
| falcon3 | exact | building | 0.615 | 0.812 | 0.587 | 0.0% |
| falcon3 | exact | musician | 0.552 | 0.736 | 0.516 | 0.0% |
| falcon3 | attention | all | 0.606 | 0.767 | 0.559 | 30.5% |
| falcon3 | attention | athlete | 0.644 | 0.795 | 0.593 | 30.5% |
| falcon3 | attention | building | 0.626 | 0.786 | 0.572 | 30.5% |
| falcon3 | attention | musician | 0.539 | 0.735 | 0.506 | 30.5% |
