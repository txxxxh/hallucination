# Whitebox experiment environment

This file records the environment used for the reported experiments.

## Hardware and operating system

- OS: Ubuntu 22.04.3 LTS (Jammy)
- Kernel: Linux 6.8.0-134-generic x86_64
- GPU: NVIDIA A100-SXM4-40GB
- GPU memory: 40960 MiB
- NVIDIA driver: 580.159.03
- GCC: 11.4.0

## Python and ML stack

- Python: 3.10.12
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- CUDA available: true
- Transformers: 5.14.1
- Accelerate: 1.14.0
- Triton: 3.7.1
- NumPy: 2.2.6
- SciPy: 1.15.3
- scikit-learn: 1.7.2

All Python packages are pinned in requirements-lock.txt.

## Important no-sudo workaround

Triton compiles a CUDA helper on first use and requires Python.h. This server
did not expose Python 3.10 development headers and the user had no sudo access.
The Ubuntu Jammy package was extracted under:

    ~/.local/python310-dev

Required compiler path:

    export CPATH="$HOME/.local/python310-dev/usr/include/python3.10:$HOME/.local/python310-dev/usr/include"

Use source activate_whitebox.sh to activate the environment and set CPATH.

## Rebuild

From the cloned repository's whitebox directory:

    cd hallucination/whitebox
    bash setup_whitebox_env.sh
    source activate_whitebox.sh

Then verify:

    python -c 'import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'

Model weights are intentionally not stored in Git. The first experiment run
will download Qwen/Qwen2.5-7B-Instruct from Hugging Face.

## Reproducibility caveats

- The NVIDIA driver and host CUDA compatibility must support CUDA 13 wheels.
- Exact wheels may become unavailable from the default package index.
- Hugging Face model revisions can change. Pin the model commit hash before
  the final publication run for strict reproducibility.
