#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_sequence_classifier.py --config configs/gate_deberta_v3_base.json
python scripts/train_sequence_classifier.py --config configs/intent_deberta_v3_base.json
