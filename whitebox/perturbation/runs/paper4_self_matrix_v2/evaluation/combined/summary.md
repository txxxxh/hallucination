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
| qwen | exact | scientist | 0.858 | 0.866 | 0.776 | 0.0% |
| qwen | exact | trivia | 0.910 | 0.902 | 0.831 | 0.0% |
| qwen | exact | gsm8k | 0.784 | 0.912 | 0.720 | 0.0% |
| qwen | attention | scientist | 0.834 | 0.835 | 0.754 | 40.9% |
| qwen | attention | trivia | 0.905 | 0.896 | 0.829 | 40.1% |
| qwen | attention | gsm8k | 0.789 | 0.914 | 0.707 | 35.6% |
| mistral | exact | scientist | 0.809 | 0.836 | 0.730 | 0.0% |
| mistral | exact | trivia | 0.964 | 0.949 | 0.902 | 0.0% |
| mistral | exact | gsm8k | 0.794 | 0.629 | 0.725 | 0.0% |
| mistral | attention | scientist | 0.805 | 0.831 | 0.721 | 40.6% |
| mistral | attention | trivia | 0.962 | 0.948 | 0.898 | 40.0% |
| mistral | attention | gsm8k | 0.792 | 0.604 | 0.729 | 34.5% |
| falcon3 | exact | scientist | 0.645 | 0.652 | 0.609 | 0.0% |
| falcon3 | exact | trivia | 0.897 | 0.915 | 0.810 | 0.0% |
| falcon3 | exact | gsm8k | 0.872 | 0.925 | 0.787 | 0.0% |
| falcon3 | attention | scientist | 0.650 | 0.655 | 0.612 | 41.5% |
| falcon3 | attention | trivia | 0.898 | 0.916 | 0.813 | 41.2% |
| falcon3 | attention | gsm8k | 0.872 | 0.925 | 0.790 | 35.5% |

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
| qwen | exact | all | 0.840 | 0.926 | 0.687 | 0.0% |
| qwen | exact | athlete | 0.839 | 0.926 | 0.699 | 0.0% |
| qwen | exact | building | 0.904 | 0.970 | 0.754 | 0.0% |
| qwen | exact | musician | 0.712 | 0.785 | 0.596 | 0.0% |
| qwen | attention | all | 0.748 | 0.874 | 0.635 | 30.8% |
| qwen | attention | athlete | 0.755 | 0.870 | 0.621 | 30.8% |
| qwen | attention | building | 0.808 | 0.934 | 0.681 | 30.8% |
| qwen | attention | musician | 0.624 | 0.691 | 0.584 | 30.8% |
| mistral | exact | all | 0.854 | 0.944 | 0.748 | 0.0% |
| mistral | exact | athlete | 0.886 | 0.966 | 0.798 | 0.0% |
| mistral | exact | building | 0.857 | 0.950 | 0.739 | 0.0% |
| mistral | exact | musician | 0.854 | 0.924 | 0.737 | 0.0% |
| mistral | attention | all | 0.800 | 0.918 | 0.709 | 30.0% |
| mistral | attention | athlete | 0.862 | 0.958 | 0.779 | 30.0% |
| mistral | attention | building | 0.779 | 0.915 | 0.696 | 30.0% |
| mistral | attention | musician | 0.805 | 0.889 | 0.697 | 30.0% |
| falcon3 | exact | all | 0.626 | 0.784 | 0.552 | 0.0% |
| falcon3 | exact | athlete | 0.729 | 0.839 | 0.544 | 0.0% |
| falcon3 | exact | building | 0.615 | 0.812 | 0.587 | 0.0% |
| falcon3 | exact | musician | 0.552 | 0.736 | 0.516 | 0.0% |
| falcon3 | attention | all | 0.606 | 0.767 | 0.559 | 30.5% |
| falcon3 | attention | athlete | 0.644 | 0.795 | 0.593 | 30.5% |
| falcon3 | attention | building | 0.626 | 0.786 | 0.572 | 30.5% |
| falcon3 | attention | musician | 0.539 | 0.735 | 0.506 | 30.5% |
