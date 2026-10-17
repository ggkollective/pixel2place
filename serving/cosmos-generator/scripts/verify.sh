#!/usr/bin/env bash
# 노드마다 /v1/models와 작은 text-to-image 요청을 검사한다. 하나라도 실패하면 exit 1로 끝난다.
# 사용법: verify.sh [노드...]
source "$(dirname "$0")/lib.sh"

# 검사용 요청. 256x256, 20 step이면 수 초 안에 끝난다.
REQUEST="{\"model\":\"${MODEL}\",\"prompt\":\"A gray steel workbench seen from above.\",\"size\":\"256x256\",\"num_inference_steps\":20,\"seed\":0}"

status=0
nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  base="http://${node}:${PORT}/v1"
  if ! curl -sf -m 5 "${base}/models" | grep -q "\"${MODEL}\""; then
    echo "FAIL ${node}: ${MODEL} is not in ${base}/models."
    status=1
    continue
  fi
  body="$(curl -sf -m 300 "${base}/images/generations" -H 'Content-Type: application/json' \
    -d "${REQUEST}")" || body=""
  if [[ "${body}" == *'"b64_json"'* ]]; then
    echo "PASS ${node}"
  else
    echo "FAIL ${node}: image generation failed."
    status=1
  fi
done
exit "${status}"
