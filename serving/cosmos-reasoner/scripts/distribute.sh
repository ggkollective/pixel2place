#!/usr/bin/env bash
# 이 노드의 이미지를 다른 노드로 복사한다. 모든 노드가 같은 이미지 ID를 쓴다.
# 사용법: distribute.sh [노드...]
source "$(dirname "$0")/lib.sh"

local_id="$(docker image inspect -f '{{.Id}}' "${IMAGE}")"
nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  [[ "${node}" == "$(hostname)" ]] && continue
  echo "Copying ${IMAGE} to ${node}."
  docker save "${IMAGE}" | ssh -o BatchMode=yes "${node}" docker load
  remote_id="$(ssh -o BatchMode=yes "${node}" docker image inspect -f '{{.Id}}' "${IMAGE}")"
  if [[ "${remote_id}" != "${local_id}" ]]; then
    echo "FAIL: image ID on ${node} is ${remote_id}. Expected ${local_id}." >&2
    exit 1
  fi
done
echo "All nodes have image ${local_id}."
