#!/usr/bin/env bash
# Job이 끝날 때까지 상태 변화를 states.txt에 쓴다. 끝나면 fetch.sh로 로그와 출력을 받는다.
# 사용법: watch.sh RUN_ID
#
# 끝난 것으로 보는 조건:
# - 상태가 COMPLETED, FAILED 또는 CANCELLED다.
# - 상태가 ERROR이고 VM이 없다. ERROR 직후에는 VM이 남아 있을 수 있다.
#
# nebius CLI의 로그인은 몇 시간 뒤에 만료된다. 만료되면 CLI는 브라우저 로그인을 기다리며 멈춘다.
# 그래서 CLI가 MAX_CLI_FAILURES번 연속으로 답하지 않으면 감시를 끝낸다. Job은 계속 실행한다.
# 그때도 fetch.sh는 버킷에 있는 출력을 받는다.
source "$(dirname "$0")/lib.sh"

run_id="${1:?Usage: watch.sh RUN_ID}"
dest="${REPO_ROOT}/${EVIDENCE_DIR}/${run_id}"
job_id="$(sed -n 's/^job_id=//p' "${dest}/job.txt")"
POLL_S=20
MAX_CLI_FAILURES=3

previous=""
failures=0
status=0
while true; do
  if job="$(job_json "${job_id}")"; then
    failures=0
  else
    failures=$((failures + 1))
    if [[ "${failures}" -ge "${MAX_CLI_FAILURES}" ]]; then
      echo "FAIL: nebius ai job get did not answer ${MAX_CLI_FAILURES} times. Log in again with nebius profile create." >&2
      status=1
      break
    fi
    sleep "${POLL_S}"
    continue
  fi
  state="$(jq -r '.status.state // "UNKNOWN"' <<<"${job}")"
  instances="$(jq -r '(.status.instances // []) | length' <<<"${job}")"
  if [[ "${state}/${instances}" != "${previous}" ]]; then
    echo "$(date -u +%H:%M:%S) ${state} instances=${instances}" | tee -a "${dest}/states.txt"
    previous="${state}/${instances}"
  fi
  case "${state}" in
    COMPLETED | FAILED | CANCELLED) break ;;
    ERROR) [[ "${instances}" -eq 0 ]] && break ;;
  esac
  sleep "${POLL_S}"
done
# Job은 끝날 때 출력을 한 번 더 올린다. 그 업로드가 버킷에 보이도록 잠시 기다린다.
sleep "${POLL_S}"
"${JOB_DIR}/scripts/fetch.sh" "${run_id}"
cat "${dest}/out/stages.txt" 2>/dev/null || true
exit "${status}"
