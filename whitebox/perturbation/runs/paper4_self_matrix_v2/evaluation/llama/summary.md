# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | scientist | 0.894 | 0.919 | 0.803 | 0.0% |
| llama | exact | trivia | 0.948 | 0.956 | 0.876 | 0.0% |
| llama | exact | gsm8k | 0.949 | 0.788 | 0.906 | 0.0% |
| llama | attention | scientist | 0.888 | 0.912 | 0.802 | 41.4% |
| llama | attention | trivia | 0.947 | 0.954 | 0.874 | 40.6% |
| llama | attention | gsm8k | 0.951 | 0.781 | 0.908 | 35.8% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | all | 0.917 | 0.968 | 0.761 | 0.0% |
| llama | exact | athlete | 0.953 | 0.987 | 0.777 | 0.0% |
| llama | exact | building | 0.931 | 0.976 | 0.793 | 0.0% |
| llama | exact | musician | 0.871 | 0.925 | 0.712 | 0.0% |
| llama | attention | all | 0.895 | 0.959 | 0.745 | 31.7% |
| llama | attention | athlete | 0.944 | 0.984 | 0.785 | 31.7% |
| llama | attention | building | 0.907 | 0.965 | 0.764 | 31.7% |
| llama | attention | musician | 0.844 | 0.919 | 0.696 | 31.7% |
