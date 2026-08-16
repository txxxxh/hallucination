# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | scientist | 0.886 | 0.910 | 0.790 | 0.0% |
| llama | exact | trivia | 0.965 | 0.970 | 0.905 | 0.0% |
| llama | exact | gsm8k | 0.743 | 0.712 | 0.683 | 0.0% |
| llama | attention | scientist | 0.881 | 0.903 | 0.795 | 41.4% |
| llama | attention | trivia | 0.966 | 0.970 | 0.906 | 40.5% |
| llama | attention | gsm8k | 0.742 | 0.718 | 0.678 | 35.9% |
| mistral | exact | scientist | 0.778 | 0.818 | 0.713 | 0.0% |
| mistral | exact | trivia | 0.958 | 0.963 | 0.895 | 0.0% |
| mistral | exact | gsm8k | 0.744 | 0.701 | 0.694 | 0.0% |
| mistral | attention | scientist | 0.773 | 0.811 | 0.712 | 40.7% |
| mistral | attention | trivia | 0.954 | 0.961 | 0.885 | 39.9% |
| mistral | attention | gsm8k | 0.737 | 0.696 | 0.689 | 34.5% |
| qwen | exact | scientist | 0.787 | 0.820 | 0.713 | 0.0% |
| qwen | exact | trivia | 0.943 | 0.943 | 0.870 | 0.0% |
| qwen | exact | gsm8k | 0.757 | 0.724 | 0.698 | 0.0% |
| qwen | attention | scientist | 0.754 | 0.788 | 0.687 | 41.0% |
| qwen | attention | trivia | 0.939 | 0.933 | 0.866 | 39.8% |
| qwen | attention | gsm8k | 0.757 | 0.738 | 0.695 | 35.5% |
| falcon3 | exact | scientist | 0.648 | 0.698 | 0.610 | 0.0% |
| falcon3 | exact | trivia | 0.945 | 0.950 | 0.866 | 0.0% |
| falcon3 | exact | gsm8k | 0.809 | 0.804 | 0.736 | 0.0% |
| falcon3 | attention | scientist | 0.643 | 0.685 | 0.609 | 41.4% |
| falcon3 | attention | trivia | 0.947 | 0.952 | 0.873 | 40.7% |
| falcon3 | attention | gsm8k | 0.808 | 0.794 | 0.742 | 35.5% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | all | 0.917 | 0.968 | 0.760 | 0.0% |
| llama | exact | athlete | 0.952 | 0.986 | 0.777 | 0.0% |
| llama | exact | building | 0.929 | 0.976 | 0.790 | 0.0% |
| llama | exact | musician | 0.878 | 0.929 | 0.712 | 0.0% |
| llama | attention | all | 0.889 | 0.957 | 0.751 | 31.7% |
| llama | attention | athlete | 0.938 | 0.983 | 0.792 | 31.7% |
| llama | attention | building | 0.906 | 0.964 | 0.782 | 31.7% |
| llama | attention | musician | 0.840 | 0.917 | 0.691 | 31.7% |
| mistral | exact | all | 0.768 | 0.909 | 0.668 | 0.0% |
| mistral | exact | athlete | 0.750 | 0.924 | 0.629 | 0.0% |
| mistral | exact | building | 0.843 | 0.950 | 0.755 | 0.0% |
| mistral | exact | musician | 0.683 | 0.824 | 0.598 | 0.0% |
| mistral | attention | all | 0.719 | 0.882 | 0.644 | 29.7% |
| mistral | attention | athlete | 0.747 | 0.924 | 0.628 | 29.7% |
| mistral | attention | building | 0.781 | 0.925 | 0.708 | 29.7% |
| mistral | attention | musician | 0.629 | 0.776 | 0.588 | 29.7% |
| qwen | exact | all | 0.759 | 0.907 | 0.677 | 0.0% |
| qwen | exact | athlete | 0.718 | 0.914 | 0.640 | 0.0% |
| qwen | exact | building | 0.828 | 0.943 | 0.697 | 0.0% |
| qwen | exact | musician | 0.695 | 0.823 | 0.671 | 0.0% |
| qwen | attention | all | 0.687 | 0.869 | 0.654 | 30.9% |
| qwen | attention | athlete | 0.654 | 0.886 | 0.627 | 30.9% |
| qwen | attention | building | 0.751 | 0.910 | 0.698 | 30.9% |
| qwen | attention | musician | 0.624 | 0.761 | 0.606 | 30.9% |
| falcon3 | exact | all | 0.614 | 0.826 | 0.565 | 0.0% |
| falcon3 | exact | athlete | 0.633 | 0.856 | 0.602 | 0.0% |
| falcon3 | exact | building | 0.613 | 0.849 | 0.541 | 0.0% |
| falcon3 | exact | musician | 0.596 | 0.754 | 0.549 | 0.0% |
| falcon3 | attention | all | 0.622 | 0.827 | 0.589 | 30.2% |
| falcon3 | attention | athlete | 0.630 | 0.877 | 0.582 | 30.2% |
| falcon3 | attention | building | 0.661 | 0.865 | 0.641 | 30.2% |
| falcon3 | attention | musician | 0.566 | 0.726 | 0.522 | 30.2% |
