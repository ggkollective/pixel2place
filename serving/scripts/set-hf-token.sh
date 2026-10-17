#!/usr/bin/env bash
# Hugging Face 토큰을 노드마다 ~/.cache/huggingface/token에 저장한다.
# 서버 컨테이너는 이 캐시 디렉터리를 마운트하므로 hf CLI와 huggingface_hub가 토큰을 자동으로 읽는다.
# 사용법: set-hf-token.sh [노드...]
#   노드를 주지 않으면 저장소 루트 .env의 서버 운영 노드에 모두 저장한다. 토큰은 화면에 표시하지 않고 붙여넣는다.
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

# 노드 목록을 가진 .env의 키. 서비스마다 1개다.
NODE_KEYS="REASONER_NODES GENERATOR_NODES RENDERER_NODES"
TOKEN_PATH=".cache/huggingface/token"
# 서버 운영에 필요한 gated 저장소. 토큰 계정이 각 저장소의 gate를 승인받아야 한다.
REQUIRED_REPOS="nvidia/Cosmos3-Nano nvidia/Cosmos-1.0-Guardrail"
# 접근 확인에 쓰는 파일. 모든 HF 저장소에 있다.
PROBE_FILE=".gitattributes"

nodes="$*"
if [[ -z "${nodes}" ]]; then
  # 세 서비스의 노드를 중복 없이 모은다.
  nodes="$(for key in ${NODE_KEYS}; do env_value "${key}"; done | tr ' ' '\n' | sort -u | xargs)"
fi
if [[ -z "${nodes}" ]]; then
  echo "FAIL: No node is set. Set ${NODE_KEYS// /, } in ${REPO_ENV_FILE}." >&2
  exit 1
fi

read -rsp "Paste the Hugging Face token and press Enter: " token
echo
if [[ -z "${token}" ]]; then
  echo "FAIL: Token is empty." >&2
  exit 1
fi

# 토큰이 프로세스 목록에 보이지 않도록 헤더를 파일 디스크립터로 넘긴다.
hf_curl() {
  curl -s -o /dev/null -w '%{http_code}' -H @<(printf 'Authorization: Bearer %s\n' "${token}") "$@"
}

if [[ "$(hf_curl https://huggingface.co/api/whoami-v2)" != "200" ]]; then
  echo "FAIL: Hugging Face does not accept the token." >&2
  exit 1
fi
echo "PASS token is valid."

status=0
for repo in ${REQUIRED_REPOS}; do
  code="$(hf_curl -I -L "https://huggingface.co/${repo}/resolve/main/${PROBE_FILE}")"
  if [[ "${code}" == "200" ]]; then
    echo "PASS access to ${repo}."
  else
    echo "WARN no access to ${repo} (HTTP ${code}). Request access on the model page."
    status=2
  fi
done

for node in ${nodes}; do
  # 토큰은 stdin으로만 보낸다. 기존 파일도 소유자 읽기·쓰기만 허용하도록 권한을 다시 정한다.
  if [[ "${node}" == "$(hostname)" ]]; then
    (umask 077 && mkdir -p "${HOME}/.cache/huggingface" && printf '%s' "${token}" >"${HOME}/${TOKEN_PATH}")
    chmod 600 "${HOME}/${TOKEN_PATH}"
  else
    printf '%s' "${token}" | ssh -o BatchMode=yes "${node}" \
      "umask 077 && mkdir -p ~/.cache/huggingface && cat > ~/${TOKEN_PATH} && chmod 600 ~/${TOKEN_PATH}"
  fi
  echo "Saved token on ${node}."
done
exit "${status}"
