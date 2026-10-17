#!/usr/bin/env bash
# 이 노드에서 서버 운영 이미지를 빌드한다. megatron-core와 nvdiffrast를 소스에서 빌드한다.
source "$(dirname "$0")/lib.sh"

docker build -t "${IMAGE}" "${SERVING_DIR}"
echo "Built ${IMAGE}."
