# Paper4 unified evaluation

fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; LR C=.03; no hyperparameter tuning on this matrix

## In-domain 3x5-fold OOF

| Model | Method | Dataset | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | scientist | 0.910 | 0.930 | 0.825 | 0.0% |
| llama | exact | trivia | 0.949 | 0.956 | 0.876 | 0.0% |
| llama | exact | gsm8k | 0.949 | 0.805 | 0.894 | 0.0% |
| llama | attention | scientist | 0.906 | 0.924 | 0.825 | 41.4% |
| llama | attention | trivia | 0.946 | 0.952 | 0.877 | 40.6% |
| llama | attention | gsm8k | 0.953 | 0.804 | 0.905 | 35.8% |
| llama | gradient | scientist | 0.905 | 0.923 | 0.817 | 45.4% |
| llama | gradient | trivia | 0.943 | 0.950 | 0.873 | 42.7% |
| llama | gradient | gsm8k | 0.950 | 0.802 | 0.896 | 44.3% |
| mistral | exact | scientist | 0.822 | 0.844 | 0.739 | 0.0% |
| mistral | exact | trivia | 0.964 | 0.948 | 0.896 | 0.0% |
| mistral | exact | gsm8k | 0.842 | 0.720 | 0.760 | 0.0% |
| mistral | attention | scientist | 0.817 | 0.838 | 0.730 | 40.6% |
| mistral | attention | trivia | 0.964 | 0.950 | 0.897 | 40.0% |
| mistral | attention | gsm8k | 0.841 | 0.713 | 0.773 | 34.5% |
| mistral | gradient | scientist | 0.819 | 0.835 | 0.729 | 46.6% |
| mistral | gradient | trivia | 0.966 | 0.951 | 0.901 | 41.5% |
| mistral | gradient | gsm8k | 0.836 | 0.708 | 0.753 | 44.0% |
| qwen | exact | scientist | 0.861 | 0.869 | 0.778 | 0.0% |
| qwen | exact | trivia | 0.909 | 0.904 | 0.829 | 0.0% |
| qwen | exact | gsm8k | 0.804 | 0.929 | 0.717 | 0.0% |
| qwen | attention | scientist | 0.835 | 0.837 | 0.757 | 40.9% |
| qwen | attention | trivia | 0.902 | 0.893 | 0.825 | 40.1% |
| qwen | attention | gsm8k | 0.802 | 0.929 | 0.716 | 35.6% |
| qwen | gradient | scientist | 0.830 | 0.832 | 0.750 | 46.6% |
| qwen | gradient | trivia | 0.898 | 0.894 | 0.820 | 43.6% |
| qwen | gradient | gsm8k | 0.793 | 0.924 | 0.703 | 45.1% |

## Frozen Scientist to multidomain

| Model | Method | Target | AUROC | AUPRC | Bal. Acc. | Query reduction |
|---|---|---|---:|---:|---:|---:|
| llama | exact | all | 0.931 | 0.972 | 0.791 | 0.0% |
| llama | exact | athlete | 0.955 | 0.988 | 0.789 | 0.0% |
| llama | exact | building | 0.945 | 0.981 | 0.852 | 0.0% |
| llama | exact | musician | 0.897 | 0.935 | 0.728 | 0.0% |
| llama | attention | all | 0.922 | 0.967 | 0.785 | 31.7% |
| llama | attention | athlete | 0.949 | 0.985 | 0.785 | 31.7% |
| llama | attention | building | 0.934 | 0.974 | 0.849 | 31.7% |
| llama | attention | musician | 0.889 | 0.930 | 0.723 | 31.7% |
| llama | gradient | all | 0.928 | 0.970 | 0.767 | 45.4% |
| llama | gradient | athlete | 0.947 | 0.985 | 0.757 | 45.4% |
| llama | gradient | building | 0.940 | 0.976 | 0.815 | 45.4% |
| llama | gradient | musician | 0.896 | 0.936 | 0.723 | 45.4% |
| mistral | exact | all | 0.850 | 0.942 | 0.679 | 0.0% |
| mistral | exact | athlete | 0.872 | 0.963 | 0.764 | 0.0% |
| mistral | exact | building | 0.856 | 0.947 | 0.672 | 0.0% |
| mistral | exact | musician | 0.860 | 0.928 | 0.635 | 0.0% |
| mistral | attention | all | 0.803 | 0.920 | 0.659 | 30.0% |
| mistral | attention | athlete | 0.829 | 0.947 | 0.712 | 30.0% |
| mistral | attention | building | 0.793 | 0.923 | 0.644 | 30.0% |
| mistral | attention | musician | 0.832 | 0.909 | 0.647 | 30.0% |
| mistral | gradient | all | 0.809 | 0.919 | 0.691 | 43.8% |
| mistral | gradient | athlete | 0.806 | 0.933 | 0.711 | 43.8% |
| mistral | gradient | building | 0.842 | 0.945 | 0.696 | 43.8% |
| mistral | gradient | musician | 0.826 | 0.890 | 0.688 | 43.8% |
| qwen | exact | all | 0.848 | 0.929 | 0.671 | 0.0% |
| qwen | exact | athlete | 0.837 | 0.924 | 0.675 | 0.0% |
| qwen | exact | building | 0.911 | 0.971 | 0.738 | 0.0% |
| qwen | exact | musician | 0.724 | 0.794 | 0.589 | 0.0% |
| qwen | attention | all | 0.761 | 0.876 | 0.614 | 30.8% |
| qwen | attention | athlete | 0.754 | 0.867 | 0.605 | 30.8% |
| qwen | attention | building | 0.815 | 0.933 | 0.660 | 30.8% |
| qwen | attention | musician | 0.658 | 0.720 | 0.564 | 30.8% |
| qwen | gradient | all | 0.811 | 0.906 | 0.652 | 44.5% |
| qwen | gradient | athlete | 0.791 | 0.896 | 0.619 | 44.5% |
| qwen | gradient | building | 0.872 | 0.954 | 0.743 | 44.5% |
| qwen | gradient | musician | 0.700 | 0.766 | 0.593 | 44.5% |
