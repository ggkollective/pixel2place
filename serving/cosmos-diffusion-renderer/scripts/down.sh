#!/usr/bin/env bash
# 노드마다 서버 컨테이너를 중지하고 삭제한다. 가중치는 HF 캐시에 남는다.
# 사용법: down.sh [노드...]
source "$(dirname "$0")/lib.sh"

nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  echo "Stopping ${CONTAINER_NAME} on ${node}."
  run_on "${node}" <<<"docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true"
done
