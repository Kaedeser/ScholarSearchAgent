#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs
LOG_FILE="logs/start_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/train.pid"

nohup bash scripts/train_cu05.sh > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"
echo "pid=$(cat "${PID_FILE}")"
echo "log=${PROJECT_DIR}/${LOG_FILE}"
