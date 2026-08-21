# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| mistral | exact | scientist | 0.809 | 0.836 | 0.730 | 0.0% |
| mistral | exact | trivia | 0.964 | 0.949 | 0.902 | 0.0% |
| mistral | exact | gsm8k | 0.794 | 0.629 | 0.725 | 0.0% |
| mistral | attention | scientist | 0.805 | 0.831 | 0.721 | 40.6% |
| mistral | attention | trivia | 0.962 | 0.948 | 0.898 | 40.0% |
| mistral | attention | gsm8k | 0.792 | 0.604 | 0.729 | 34.5% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| mistral | exact | all | 0.854 | 0.944 | 0.748 | 0.0% |
| mistral | exact | athlete | 0.886 | 0.966 | 0.798 | 0.0% |
| mistral | exact | building | 0.857 | 0.950 | 0.739 | 0.0% |
| mistral | exact | musician | 0.854 | 0.924 | 0.737 | 0.0% |
| mistral | attention | all | 0.800 | 0.918 | 0.709 | 30.0% |
| mistral | attention | athlete | 0.862 | 0.958 | 0.779 | 30.0% |
| mistral | attention | building | 0.779 | 0.915 | 0.696 | 30.0% |
| mistral | attention | musician | 0.805 | 0.889 | 0.697 | 30.0% |
