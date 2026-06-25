#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

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

"$PYTHON_BIN" -m pip install -r requirements.txt

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/framework/sentence-transformers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" -m selector_reranker.data_builder \
  --pasa-data-dir "$PASA_DATA_DIR" \
  --output-dir data/processed \
  --max-abstract-chars 3500

run_variant() {
  local run_name="$1"
  shift
  echo "===== RUN ${run_name} ====="
  rm -rf "outputs/${run_name}"
  "$PYTHON_BIN" -m selector_reranker.train_cross_encoder \
    --train-file data/processed/train.jsonl \
    --dev-file data/processed/dev.jsonl \
    --output-dir "outputs/${run_name}" \
    "$@"
}

run_variant bge-base-continue-ep3-lr5e-6-bs8 \
  --model-name outputs/bge-reranker-base-pasa/final \
  --max-length 512 \
  --epochs 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --warmup-ratio 0.05 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 100 \
  --dataloader-num-workers 0 \
  --fp16 \
  --no-bf16

run_variant bge-base-fresh-ep4-lr1e-5-bs8 \
  --model-name BAAI/bge-reranker-base \
  --max-length 512 \
  --epochs 4 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 100 \
  --dataloader-num-workers 0 \
  --fp16 \
  --no-bf16

run_variant minilm-msmarco-ep4-lr2e-5-bs32 \
  --model-name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --max-length 512 \
  --epochs 4 \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 100 \
  --dataloader-num-workers 0 \
  --fp16 \
  --no-bf16

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

rows = []
for metrics_path in sorted(Path("outputs").glob("*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    rows.append({
        "run": metrics_path.parent.name,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "f1_threshold": metrics["f1_threshold"],
    })
rows.sort(key=lambda item: (item["f1"], item["accuracy"], item["average_precision"]), reverse=True)
Path("outputs/tuning_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
