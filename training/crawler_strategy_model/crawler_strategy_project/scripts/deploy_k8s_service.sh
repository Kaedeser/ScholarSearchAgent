#!/usr/bin/env bash
# 中文功能说明：项目 shell 脚本，封装训练、评估、部署或远端执行命令。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KUBECTL="${KUBECTL:-/opt/kube/bin/kubectl}"

"${KUBECTL}" apply -f "${PROJECT_DIR}/k8s/crawler-strategy-service.yaml"
"${KUBECTL}" rollout status deployment/crawler-strategy-service -n default --timeout=900s
"${KUBECTL}" get svc crawler-strategy-service -n default -o wide
"${KUBECTL}" get pods -n default -l app=crawler-strategy-service -o wide
