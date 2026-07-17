# White-box 项目环境恢复指南

本文档用于在临时服务器释放后，在一台新的 Linux GPU 服务器上恢复代码、
Python 环境、无 sudo 编译头文件和实验运行方式。仓库地址为：

```text
https://github.com/txxxxh/hallucination.git
```

## 1. 新服务器最低要求

- Linux x86_64；原实验系统为 Ubuntu 22.04.3 LTS。
- NVIDIA GPU；原实验使用 A100-SXM4-40GB。
- NVIDIA 驱动能够运行 PyTorch 使用的 CUDA 13 runtime。原驱动版本为
  580.159.03。
- Python 3.10；原版本为 3.10.12。
- `git`、`curl`、`gcc`、`dpkg-deb` 和 `python3.10 -m venv` 可用。
- 不要求 sudo。`Python.h` 会通过解压 Ubuntu deb 到用户目录解决。
- 建议至少预留 80 GB 磁盘空间。Git 结果约占数百 MB，Qwen/Llama 模型
  缓存还需要数十 GB。

先检查机器：

```bash
nvidia-smi
python3.10 --version
git --version
gcc --version
curl --version
dpkg-deb --version
```

如果新服务器没有 Python 3.10，需要先使用服务器提供的 conda/mamba 模块
安装 Python 3.10，或联系管理员安装；当前自动脚本明确要求 `python3.10`。

## 2. 克隆仓库

推荐克隆到 home 目录下的独立目录，不要再把整个 home 目录作为工作树：

```bash
cd ~
git clone https://github.com/txxxxh/hallucination.git
cd ~/hallucination/whitebox
```

目录职责：

```text
hallucination/
├── blackbox/     # black-box 项目，保持独立
└── whitebox/     # 本项目全部代码、数据、模型头和实验输出
```

确认下载完整：

```bash
git status
ls question_and_result.json shuffled_prepend_names_question.json \
   shuffled_prepend_profiles_question.json
```

`base_features.jsonl` 中有接近 GitHub 100 MB 上限的文件。普通 `git clone`
会直接下载它们，本仓库当前没有使用 Git LFS。

## 3. 一键重建 Python 环境（推荐）

在 `~/hallucination/whitebox` 下执行：

```bash
bash setup_whitebox_env.sh
source activate_whitebox.sh
```

脚本会完成：

1. 创建 `~/venvs/whitebox`；
2. 按 `requirements-lock.txt` 安装精确版本；
3. 下载 Ubuntu Jammy 的 `libpython3.10-dev` deb；
4. 在没有 sudo 的情况下解压到 `~/.local/python310-dev`；
5. 由激活脚本设置 Triton 编译需要的 `CPATH`。

每次重新登录服务器后都要执行：

```bash
cd ~/hallucination/whitebox
source activate_whitebox.sh
```

不要只执行虚拟环境自己的 `activate`，否则 `CPATH` 可能未设置，Triton 会报：

```text
fatal error: Python.h: No such file or directory
```

## 4. 自动脚本失败时的手工恢复方法

### 4.1 创建虚拟环境

```bash
python3.10 -m venv ~/venvs/whitebox
source ~/venvs/whitebox/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/hallucination/whitebox/requirements-lock.txt
```

### 4.2 无 sudo 安装 Python.h

不要使用 `apt download libpython3.10-dev`：新服务器的软件源可能没有启用该
包，之前出现过 `Unable to locate package`。直接下载已验证的 Ubuntu Jammy
deb：

```bash
mkdir -p ~/.local/python310-dev
curl -fLo /tmp/libpython3.10-dev.deb \
  https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/libpython3.10-dev_3.10.12-1~22.04.16_amd64.deb
dpkg-deb -x /tmp/libpython3.10-dev.deb ~/.local/python310-dev
test -f ~/.local/python310-dev/usr/include/python3.10/Python.h
```

设置编译路径：

```bash
export CPATH="$HOME/.local/python310-dev/usr/include/python3.10:$HOME/.local/python310-dev/usr/include${CPATH:+:$CPATH}"
```

如果固定 deb URL 将来失效，可在 Ubuntu Packages 中寻找与服务器 Python
3.10 ABI 匹配的 Jammy `libpython3.10-dev` amd64 deb，然后仍使用
`dpkg-deb -x` 解压到同一目录。不要混用 Python 3.11/3.12 的头文件。

## 5. 环境验证

```bash
cd ~/hallucination/whitebox
source activate_whitebox.sh

python -c 'import torch, transformers, accelerate, triton; \
print("torch", torch.__version__); \
print("transformers", transformers.__version__); \
print("accelerate", accelerate.__version__); \
print("triton", triton.__version__); \
print("cuda", torch.cuda.is_available(), torch.version.cuda); \
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

python -c 'import torch; x=torch.ones(2,device="cuda"); print((x*2).tolist())'
```

原始关键版本记录在 `ENVIRONMENT.md`，完整 Python 包版本在
`requirements-lock.txt`。

## 6. Hugging Face 模型访问

Qwen 模型可以匿名下载，但未认证请求限速较低。建议创建 Hugging Face
read token，并在 shell 中设置（不要把 token 提交到 Git）：

```bash
export HF_TOKEN='hf_your_read_token'
```

也可以交互登录：

```bash
huggingface-cli login
```

官方 `meta-llama/Llama-3.1-8B-Instruct` 是 gated repo。必须先在模型页面
接受 Meta 许可，并使用已获批准账号的 token，否则会收到 401。当前已完成的
Llama 实验使用可匿名访问的镜像：

```text
NousResearch/Meta-Llama-3.1-8B-Instruct
```

模型权重没有存入 Git；首次运行会下载到 `~/.cache/huggingface/hub`。

## 7. 核心实验复现命令

所有命令均从 `~/hallucination/whitebox` 运行，并先激活环境：

```bash
cd ~/hallucination/whitebox
source activate_whitebox.sh
```

### 7.1 Real-life QA：Qwen v4

```bash
python role_mediated_whitebox_v4.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data question_and_result.json \
  --out-dir role_mediated_output_v4 \
  --limit 0 --resume
```

### 7.2 Real-life QA：Llama 3.1 8B v4

```bash
python role_mediated_whitebox_v4.py \
  --model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data question_and_result.json \
  --out-dir role_mediated_reallife_nous_llama31_8b_v4_output \
  --limit 0 --resume
```

如果官方权限已经开通，可将 `--model` 改成：

```text
meta-llama/Llama-3.1-8B-Instruct
```

### 7.3 ScientistQA names：atomic v5

```bash
python role_mediated_whitebox_v5_scientistqa_atomic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data shuffled_prepend_names_question.json \
  --out-dir role_mediated_scientistqa_atomic_v5_output \
  --limit 0 --resume
```

### 7.4 ScientistQA profiles：atomic v5

```bash
python role_mediated_whitebox_v5_scientistqa_profiles_atomic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data shuffled_prepend_profiles_question.json \
  --out-dir role_mediated_scientistqa_profiles_atomic_v5_output \
  --limit 0 --resume
```

### 7.5 Weakly supervised 基线

```bash
python weakly_supervised_whitebox.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data question_and_result.json \
  --out-dir weakbox_output \
  --limit 0 --resume
```

## 8. 断点续跑与输出文件

长实验务必使用 `--resume`。每条样本处理后会增量写缓存；服务器断开后，在
相同输出目录重新运行完全相同的命令即可继续。

典型输出目录包含：

- `base_features.jsonl`：原始 prompt 的 white-box 特征；
- `intervention_labels.jsonl`：训练集干预产生的弱角色标签；
- `predictions.jsonl`：held-out 逐条预测和解释；
- `causal_audit.jsonl`：严格 post-prediction 因果审计；
- `role_mediated_bundle.joblib`：轻量检测头与校准器；
- `summary.json`：AUROC、AUPRC、阈值、解释覆盖率和审计统计。

查看运行进度和结果：

```bash
wc -l OUTPUT_DIR/base_features.jsonl OUTPUT_DIR/intervention_labels.jsonl
python -m json.tool OUTPUT_DIR/summary.json | less
```

不要在参数、数据或模型已经改变时复用旧输出目录，否则 `--resume` 会混合不兼容
缓存。不同基础模型必须使用不同输出目录。

## 9. Git 日常保存流程

```bash
cd ~/hallucination
git status --short
git add whitebox
git commit -m "Describe experiment or code change"
git pull --rebase origin main
git push origin main
```

推送前检查大文件：

```bash
find whitebox -type f -size +90M -printf '%s %p\n' | sort -nr
```

GitHub 单文件强制上限约 100 MB。当前若生成更大的 JSONL，不要直接提交；应
拆分、压缩，或先为仓库配置 Git LFS。虚拟环境、模型缓存、HF token、CUDA
缓存均不应加入 Git。

## 10. 常见错误排查

### `Python.h: No such file or directory`

重新执行：

```bash
source ~/hallucination/whitebox/activate_whitebox.sh
test -f ~/.local/python310-dev/usr/include/python3.10/Python.h
echo "$CPATH"
```

### Hugging Face `401 Unauthorized` / `GatedRepoError`

模型需要许可或当前 shell 没有 token。检查：

```bash
python -c 'import os; print(bool(os.environ.get("HF_TOKEN")))'
```

### CUDA OOM

- 确认 GPU 没有被其他进程占用：`nvidia-smi`；
- 使用 `--resume` 重启，已缓存样本不会重算；
- profiles 数据含很长的双 profile，少量超长样本可能被脚本跳过；
- 不要同时运行两个 7B/8B white-box 提取任务。

### 找不到数据文件

确认当前目录是 `~/hallucination/whitebox`。代码的相对默认路径以当前工作目录
为基准；也可以始终传入绝对 `--data` 和 `--out-dir`。

### 精确复现实验仍有差异

模型仓库默认分支未来可能变化。正式论文复现时应把 Hugging Face 模型固定到
具体 commit revision，并记录 GPU、驱动、随机种子和实际成功提取样本数。
