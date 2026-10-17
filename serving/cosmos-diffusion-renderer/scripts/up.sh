#!/usr/bin/env bash
# 노드마다 DiffusionRenderer 서버 컨테이너를 시작한다. 가중치는 각 노드의 HF 캐시를 쓴다.
# 사용법: up.sh [노드...]
# GB10은 CPU와 GPU가 메모리를 공유한다. GPU 메모리는 프로세스 RSS에 잡히지 않는다.
# 그래서 메모리가 부족할 때 OOM killer가 이 컨테이너를 먼저 종료하도록 oom-score-adj를 최대로 둔다.
source "$(dirname "$0")/lib.sh"

nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  echo "Starting ${CONTAINER_NAME} on ${node}."
  run_on "${node}" <<REMOTE
set -euo pipefail
docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
docker run -d --name ${CONTAINER_NAME} \
  --runtime nvidia --gpus all --ipc host --restart unless-stopped --oom-score-adj 1000 \
  -p ${PORT}:8000 \
  -e INVERSE_MODEL=${INVERSE_MODEL} -e INVERSE_REVISION=${INVERSE_REVISION} \
  -e FORWARD_MODEL=${FORWARD_MODEL} -e FORWARD_REVISION=${FORWARD_REVISION} \
  -e TOKENIZER_MODEL=${TOKENIZER_MODEL} -e TOKENIZER_REVISION=${TOKENIZER_REVISION} \
  -v "\${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  ${IMAGE} >/dev/null
REMOTE
done
echo "Started. The first start on a node downloads about 60 GB. Check readiness with scripts/verify.sh."
