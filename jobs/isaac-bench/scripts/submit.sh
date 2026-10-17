#!/usr/bin/env bash
# Job 입력을 올리고 Nebius Serverless Job 1개를 만든다. Job은 비용이 든다.
# 사용법: submit.sh MODE [bench-cell 인자...]
#   MODE: rehearse(입력, 환경 기록과 출력만. GPU가 필요 없다) | probe(환경 기록과 hello 캡처)
#         | smoke(probe와 pytest -m sim) | bench(probe와 bench-cell)
#   예: submit.sh bench --count 30 --seed 0 --shard 0/2
# 환경 변수 REGION, PLATFORM, PRESET, TIMEOUT은 config.env의 값을 바꾼다.
# 다른 리전에서는 그 리전의 버킷(NEBIUS_JOB_BUCKET)과 프로젝트(NEBIUS_JOB_PROJECT_ID)를 함께 준다.
# Job이 끝날 때까지 기다리고 결과를 받으려면 watch.sh RUN_ID를 실행한다.
# 출력: 실행 ID와 Job ID. 같은 값을 근거 디렉터리의 job.txt에 쓴다.
#
# 시작 스크립트와 manifest는 --inject-file로 넣는다. 입력과 출력의 주소는 환경 변수로 준다.
source "$(dirname "$0")/lib.sh"

mode="${1:?Usage: submit.sh MODE [bench-cell arguments...]}"
shift
case "${mode}" in
  rehearse | probe | smoke | bench) ;;
  *)
    echo "FAIL: MODE ${mode} is not rehearse, probe, smoke or bench." >&2
    exit 1
    ;;
esac

run_id="$(date -u +%Y%m%dT%H%M%SZ)-${mode}"
"${JOB_DIR}/scripts/package.sh" "${run_id}"

dest="${REPO_ROOT}/${EVIDENCE_DIR}/${run_id}"
# shellcheck source=/dev/null
source "${dest}/urls.env"

parent_args=()
[[ -n "${PROJECT_ID}" ]] && parent_args=(--parent-id "${PROJECT_ID}")
job_id="$(
  nebius ai job create \
    "${parent_args[@]}" \
    --name "${JOB_NAME_PREFIX}-${run_id,,}" \
    --image "${IMAGE}" \
    --container-command bash \
    --args "${INJECT_DIR}/entrypoint.sh" \
    --inject-file "${dest}/inject/entrypoint.sh:${INJECT_DIR}/entrypoint.sh" \
    --inject-file "${dest}/inject/manifest.txt:${INJECT_DIR}/manifest.txt" \
    --platform "${PLATFORM}" \
    --preset "${PRESET}" \
    --timeout "${TIMEOUT}" \
    --disk-size "${DISK_SIZE}" \
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env "P2P_MODE=${mode}" \
    --env "P2P_SOURCE_URL=${P2P_SOURCE_URL}" \
    --env "P2P_USD_URL=${P2P_USD_URL}" \
    --env "P2P_OUTPUT_URL=${P2P_OUTPUT_URL}" \
    --env "P2P_BENCH_ARGS=$*" \
    --async \
    --format 'jsonpath={.resource_id}'
)"

{
  echo "run_id=${run_id}"
  echo "job_id=${job_id}"
  echo "mode=${mode}"
  echo "region=${REGION}"
  echo "platform=${PLATFORM}"
  echo "preset=${PRESET}"
  echo "timeout=${TIMEOUT}"
  echo "image=${IMAGE}"
  echo "bench_args=$*"
} >"${dest}/job.txt"
echo "run_id=${run_id} job_id=${job_id}"
