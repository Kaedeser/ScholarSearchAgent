#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(python - <<'PY'
import torch
best = None
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    used = total - free
    if best is None or used < best[1]:
        best = (index, used)
print("" if best is None else best[0])
PY
)"
  export CUDA_VISIBLE_DEVICES
fi

ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen2p5-3b-crawler-lora-r16-e3}"
LOCAL_BASE_MODEL=".cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct"
if [ -z "${BASE_MODEL:-}" ] && [ -d "${LOCAL_BASE_MODEL}" ]; then
  BASE_MODEL="${LOCAL_BASE_MODEL}"
fi
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
EVAL_FILE="${EVAL_FILE:-data/crawler_sft_eval.jsonl}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-${ADAPTER_DIR}/eval_predictions.jsonl}"
METRICS_FILE="${METRICS_FILE:-${ADAPTER_DIR}/action_metrics.json}"
if [ -n "${MAX_SAMPLES:-}" ]; then
  python scripts/generate_eval_predictions.py \
    --base-model "${BASE_MODEL}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --eval-file "${EVAL_FILE}" \
    --output-file "${PREDICTIONS_FILE}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --max-samples "${MAX_SAMPLES}"
else
  python scripts/generate_eval_predictions.py \
    --base-model "${BASE_MODEL}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --eval-file "${EVAL_FILE}" \
    --output-file "${PREDICTIONS_FILE}" \
    --batch-size "${BATCH_SIZE:-4}"
fi

python scripts/evaluate_actions.py "${PREDICTIONS_FILE}" | tee "${METRICS_FILE}"
