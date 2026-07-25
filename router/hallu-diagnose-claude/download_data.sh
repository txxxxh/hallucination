#!/usr/bin/env bash
# 公开数据下载。HF 数据集(PopQA/GSM8K/TruthfulQA/MATH)由 datasets 库在脚本内自动拉取。
set -e
mkdir -p data/raw && cd data/raw

# GSM-IC (Z2)
git clone --depth 1 https://github.com/google-research-datasets/GSM-IC gsm_ic || true

# SelfAware (Z6)
mkdir -p selfaware
# 仓库: https://github.com/yinzhangyue/SelfAware  (数据文件路径以仓库 README 为准)
git clone --depth 1 https://github.com/yinzhangyue/SelfAware selfaware_repo || true
cp selfaware_repo/data/SelfAware.json selfaware/ 2>/dev/null || \
  echo ">> 请按 SelfAware 仓库 README 手动定位 json 并放到 data/raw/selfaware/SelfAware.json"

# FalseQA (Z6)
mkdir -p falseqa
curl -fL --retry 3 \
  https://raw.githubusercontent.com/thunlp/FalseQA/refs/heads/main/dataset/test.csv \
  -o falseqa/test.csv

# FreshQA (Z1 时效 + Z6 假前提): 官方表格的 CSV 导出地址。
curl -fL --retry 3 \
  'https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/gviz/tq?tqx=out:csv' \
  -o freshqa.csv

echo "done."
