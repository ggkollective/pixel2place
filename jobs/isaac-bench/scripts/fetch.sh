#!/usr/bin/env bash
# Job의 상태, 로그와 출력을 근거 디렉터리에 받는다. 실행 중인 Job에도 쓸 수 있다.
# 사용법: fetch.sh RUN_ID
# submit.sh에 REGION이나 NEBIUS_JOB_BUCKET을 환경 변수로 줬으면 여기에도 같은 값을 준다.
# 출력: <근거 디렉터리>/<실행 ID>/job.json, job.log, out/
# nebius CLI가 답하지 않아도 버킷의 출력은 받는다. 버킷은 ~/.aws의 접근 키를 쓴다.
source "$(dirname "$0")/lib.sh"

run_id="${1:?Usage: fetch.sh RUN_ID}"
dest="${REPO_ROOT}/${EVIDENCE_DIR}/${run_id}"
if [[ ! -f "${dest}/job.txt" ]]; then
  echo "FAIL: ${dest}/job.txt does not exist. Make the job with submit.sh." >&2
  exit 1
fi
job_id="$(sed -n 's/^job_id=//p' "${dest}/job.txt")"

state="unknown"
if job="$(job_json "${job_id}")"; then
  echo "${job}" >"${dest}/job.json"
  state="$(jq -r '.status.state // "unknown"' <<<"${job}")"
  # 로그는 콘솔과 CLI에서만 받을 수 있다. Job의 최대 시간은 168h다.
  nebius ai job logs "${job_id}" --since 168h --timestamps </dev/null >"${dest}/job.log" 2>&1 || true
else
  echo "The nebius CLI did not answer. The job state and the job log are not saved."
fi
# Job은 출력을 out.tar.gz 1개로 올린다. Job이 올리기 전에는 파일이 없다.
if s3 cp --only-show-errors "s3://${BUCKET}/${run_id}/out/out.tar.gz" "${dest}/out.tar.gz" 2>/dev/null; then
  mkdir -p "${dest}/out"
  tar -xzf "${dest}/out.tar.gz" -C "${dest}/out"
else
  echo "No output archive in the bucket yet."
fi
echo "state=${state} dir=${dest}"
