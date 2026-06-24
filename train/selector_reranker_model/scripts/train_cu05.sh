#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/home/model_train/selector_reranker_model}"
PASA_DATA_DIR="${2:-/home/model_train/pasa/data}"
VENV_DIR="${VENV_DIR:-/home/model_train/py-train2}"
BASE_PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$BASE_PYTHON_BIN" >/dev/null 2>&1; then
  BASE_PYTHON_BIN="python"
fi

cd "$ROOT_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$BASE_PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/framework/sentence-transformers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" -m selector_reranker.data_builder \
  --pasa-data-dir "$PASA_DATA_DIR" \
  --output-dir data/processed \
  --max-abstract-chars 3500

"$PYTHON_BIN" -m selector_reranker.train_cross_encoder \
  --train-file data/processed/train.jsonl \
  --dev-file data/processed/dev.jsonl \
  --output-dir outputs/bge-reranker-base-pasa \
  --model-name BAAI/bge-reranker-base \
  --max-length 512 \
  --epochs 2 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 100 \
  --fp16 \
  --no-bf16
