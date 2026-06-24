#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/data/k8s/anaconda3/envs/py-train/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" scripts/train_sequence_classifier.py --config configs/gate_bge_m3_remote.json
"$PYTHON_BIN" scripts/train_sequence_classifier.py --config configs/intent_bge_m3_remote.json
