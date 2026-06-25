#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/data/k8s/anaconda3/envs/py-train/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" scripts/train_sequence_classifier.py --config configs/gate_biobert_remote.json
"$PYTHON_BIN" scripts/train_sequence_classifier.py --config configs/intent_biobert_remote.json
