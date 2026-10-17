# 서버 운영 스크립트 공용 함수. 각 서비스의 scripts/lib.sh가 source한다.

# 노드 이름은 저장소 루트 .env에 둔다. .env는 git에 넣지 않는다.
REPO_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"

# 키 1개의 값을 출력한다. 환경 변수가 있으면 그 값을 쓴다. 없으면 .env에서 읽는다.
# .env 전체를 source하지 않는다. 그래서 다른 값(API 키)은 셸 변수가 되지 않는다.
env_value() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then
    echo "${!key}"
  elif [[ -f "${REPO_ENV_FILE}" ]]; then
    sed -n "s/^${key}=//p" "${REPO_ENV_FILE}" | tail -n 1 | tr -d "\"'"
  fi
}

# 대상 노드에서 stdin의 bash 스크립트를 실행한다. 현재 노드이면 ssh를 쓰지 않는다.
run_on() {
  local node="$1"
  if [[ "${node}" == "$(hostname)" ]]; then
    bash -s
  else
    ssh -o BatchMode=yes "${node}" bash -s
  fi
}

# 인자가 없으면 NODES_KEY(config.env)가 가리키는 .env 값의 모든 노드를 대상으로 한다.
# 값이 없으면 실패한다. 호출하는 쪽은 결과를 변수에 먼저 받는다. 그래야 실패가 스크립트를 멈춘다.
target_nodes() {
  if [[ $# -gt 0 ]]; then
    echo "$@"
    return
  fi
  local nodes
  nodes="$(env_value "${NODES_KEY}")"
  if [[ -z "${nodes}" ]]; then
    echo "FAIL: ${NODES_KEY} is empty. Set the node names in ${REPO_ENV_FILE}." >&2
    return 1
  fi
  echo "${nodes}"
}
