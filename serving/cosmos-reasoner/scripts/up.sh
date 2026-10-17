#!/usr/bin/env bash
# 노드마다 서버 컨테이너를 시작한다. 가중치는 각 노드의 HF 캐시를 쓴다.
# 사용법: up.sh [노드...]
source "$(dirname "$0")/lib.sh"

HF_TOKEN=""
env_file="${SERVING_DIR}/.env"
if [[ -f "${env_file}" ]]; then
  # shellcheck source=/dev/null
  source "${env_file}"
fi
if [[ "${HF_HUB_OFFLINE}" != "1" && -z "${HF_TOKEN}" ]]; then
  echo "FAIL: HF_HUB_OFFLINE is not 1. Set HF_TOKEN in ${env_file}." >&2
  exit 1
fi

nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  echo "Starting ${CONTAINER_NAME} on ${node}."
  # 토큰이 명령줄과 로그에 남지 않도록 stdin 스크립트 안에서만 쓴다.
  run_on "${node}" <<REMOTE
set -euo pipefail
docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
docker run -d --name ${CONTAINER_NAME} \
  --runtime nvidia --gpus all --ipc host --restart unless-stopped \
  -p ${PORT}:8000 \
  -e HF_TOKEN='${HF_TOKEN}' -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
  -v "\${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  ${IMAGE} \
  vllm serve ${MODEL} \
    --revision ${MODEL_REVISION} --tokenizer-revision ${MODEL_REVISION} \
    --hf-overrides '{"architectures":["Cosmos3ReasonerForConditionalGeneration"]}' \
    --tensor-parallel-size 1 \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --host 0.0.0.0 --port 8000 >/dev/null
REMOTE
done
echo "Started. Check readiness with scripts/verify.sh."
