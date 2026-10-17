#!/usr/bin/env bash
# 노드마다 /v1/models와 짧은 chat 요청을 검사한다. 하나라도 실패하면 exit 1로 끝난다.
# 사용법: verify.sh [노드...]
source "$(dirname "$0")/lib.sh"

status=0
nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  base="http://${node}:${PORT}/v1"
  if ! curl -sf -m 5 "${base}/models" | grep -q "\"${MODEL}\""; then
    echo "FAIL ${node}: ${MODEL} is not in ${base}/models."
    status=1
    continue
  fi
  body="$(curl -sf -m 120 "${base}/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word pong.\"}],\"max_tokens\":16,\"temperature\":0.7,\"top_p\":0.8}")" || body=""
  if [[ "${body}" == *'"choices"'* ]]; then
    echo "PASS ${node}"
  else
    echo "FAIL ${node}: chat completion failed."
    status=1
  fi
done
exit "${status}"
