#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs outputs .cache/huggingface .cache/modelscope

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/crawler_qwen2p5_3b_lora.yaml}"
PACKAGE_PATH="packages/llamafactory-0.9.5-py3-none-any.whl"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required for llamafactory==0.9.5.")
PY

if [ ! -d ".venv" ]; then
  "${PYTHON_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install --prefer-binary \
  "torch>=2.4.0,<2.8.0" \
  "torchaudio>=2.4.0,<2.8.0" \
  "torchvision>=0.19.0,<0.23.0" \
  "accelerate>=1.3.0,<=1.11.0" \
  "datasets>=2.16.0,<=4.0.0" \
  "transformers==4.56.2" \
  "peft>=0.18.0,<=0.18.1" \
  "trl>=0.18.0,<=0.24.0" \
  "sentencepiece==0.2.0" \
  "tiktoken" \
  "protobuf" \
  "safetensors" \
  "pyyaml" \
  "omegaconf" \
  "tyro<0.9.0" \
  "fire" \
  "einops" \
  "numpy" \
  "pandas" \
  "scipy" \
  "matplotlib>=3.7.0" \
  "hf-transfer" \
  "modelscope"

if [ -f "${PACKAGE_PATH}" ]; then
  python -m pip install --no-deps "${PACKAGE_PATH}"
else
  python -m pip install --no-deps "llamafactory==0.9.5"
fi

python scripts/prepare_crawler_data.py \
  --source data/raw/sft_crawler_train.jsonl \
  --out-dir data \
  --eval-size "${EVAL_SIZE:-500}"

python scripts/check_env.py

export PYTHONUTF8=1
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_DIR}/.cache/modelscope}"
export USE_MODELSCOPE_HUB="${USE_MODELSCOPE_HUB:-1}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "No CUDA GPU is available; aborting training." >&2
  exit 4
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(python - <<'PY'
import torch
chosen = []
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    used = total - free
    if used < 1024 ** 3:
        chosen.append(str(i))
print(",".join(chosen))
PY
)"
  export CUDA_VISIBLE_DEVICES
fi

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  echo "No GPU with less than 1GB used memory is available; aborting training." >&2
  exit 5
fi

VISIBLE_GPU_COUNT="$(python - <<'PY'
import os
print(len([item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]))
PY
)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-${VISIBLE_GPU_COUNT}}"
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training config: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "Log file: ${LOG_FILE}"

python -m llamafactory.cli train "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"
