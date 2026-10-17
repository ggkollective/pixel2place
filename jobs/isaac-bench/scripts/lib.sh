# 공용 함수와 설정. 개발 노드에서 실행하는 스크립트가 source한다. entrypoint.sh는 쓰지 않는다.
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${JOB_DIR}/../.." && pwd)"
# shellcheck source=../config.env
source "${JOB_DIR}/config.env"
# shellcheck source=../../../serving/scripts/common.sh
source "${REPO_ROOT}/serving/scripts/common.sh"

# .env의 필수 값을 출력한다. 값이 없으면 실패한다.
required_env_value() {
  local key="$1" value
  value="$(env_value "${key}")"
  if [[ -z "${value}" ]]; then
    echo "FAIL: ${key} is empty. Set it in ${REPO_ENV_FILE}." >&2
    return 1
  fi
  echo "${value}"
}

# Job 1개의 상태를 JSON으로 출력한다. CLI가 CLI_TIMEOUT_S 안에 답하지 않으면 실패한다.
# 로그인이 만료되면 CLI는 브라우저 로그인을 기다리며 멈춘다. 그래서 시간 제한을 둔다.
CLI_TIMEOUT_S=30
job_json() {
  timeout "${CLI_TIMEOUT_S}" nebius ai job get "$1" --format json </dev/null 2>/dev/null
}

# 버킷에 접근하는 aws 명령. 접근 키는 ~/.aws의 프로필에 있다.
s3() {
  aws --profile "${S3_PROFILE}" --endpoint-url "${S3_ENDPOINT}" s3 "$@"
}

BUCKET="$(required_env_value NEBIUS_JOB_BUCKET)"
S3_PROFILE="$(required_env_value NEBIUS_S3_PROFILE)"
# Job을 만드는 프로젝트. 비우면 nebius CLI 프로필의 기본 프로젝트를 쓴다. 프로젝트는 리전 1개에 속한다.
PROJECT_ID="$(env_value NEBIUS_JOB_PROJECT_ID)"
