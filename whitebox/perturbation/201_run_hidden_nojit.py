#!/usr/bin/env python3
"""Run the hidden collector with runtime compilation paths disabled."""
import os, runpy
os.environ["PYTORCH_JIT"]="0"
os.environ["TORCHDYNAMO_DISABLE"]="1"
os.environ["TORCH_COMPILE_DISABLE"]="1"
os.environ["TRITON_CACHE_DIR"]="/tmp/triton_mistral"
os.environ["TORCHINDUCTOR_CACHE_DIR"]="/tmp/torchinductor_mistral"
import torch
from torch._native import registry
registry.deregister_op_overrides(disable_op_symbols="bmm")
runpy.run_path("147_collect_question_only_hidden.py",run_name="__main__")
