#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_sequence_classifier.py --config configs/gate_deberta_v3_base.json
python scripts/train_sequence_classifier.py --config configs/intent_deberta_v3_base.json
