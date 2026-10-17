#!/usr/bin/env bash
# 노드마다 생성 서버 컨테이너를 시작한다. 가중치와 HF 토큰은 각 노드의 HF 캐시를 쓴다.
# 사용법: up.sh [노드...]
source "$(dirname "$0")/lib.sh"

guardrail_flag=""
if [[ "${GUARDRAILS}" != "1" ]]; then
  guardrail_flag="--no-guardrails"
fi

nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  echo "Starting ${CONTAINER_NAME} on ${node}."
  run_on "${node}" <<REMOTE
set -euo pipefail
docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
docker run -d --name ${CONTAINER_NAME} \
  --runtime nvidia --gpus all --ipc host --restart unless-stopped \
  -p ${PORT}:8000 \
  -v "\${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  ${IMAGE} \
  vllm serve ${MODEL} --omni \
    --revision ${MODEL_REVISION} \
    ${guardrail_flag} \
    --init-timeout 1800 \
    --host 0.0.0.0 --port 8000 >/dev/null
REMOTE
done
echo "Started. Model load takes about 4 minutes. Check readiness with scripts/verify.sh."
