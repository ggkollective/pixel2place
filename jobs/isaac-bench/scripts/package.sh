#!/usr/bin/env bash
# Job 입력을 버킷에 올리고 서명된 주소를 만든다.
# 사용법: package.sh RUN_ID
#
# 버킷의 <실행 ID>/in/에 올리는 파일:
# 1. source.tar.gz: HEAD의 git archive. 커밋하지 않은 변경은 넣지 않는다.
# 2. so101-usd.tar.gz: 변환한 SO-101 USD 디렉터리
#
# 근거 디렉터리의 <실행 ID>/에 쓰는 파일:
# 1. inject/entrypoint.sh: 작업 트리의 파일. 커밋 전에도 고쳐서 시험할 수 있다.
# 2. inject/manifest.txt: 커밋, 작업 트리 상태와 각 파일의 SHA-256
# 3. urls.env: 입력을 받는 주소 2개와 출력을 올리는 주소 1개. 주소는 URL_EXPIRES_S 뒤에 만료된다.
#
# Job은 버킷을 마운트하지 않는다. 마운트는 root만 접근할 수 있다.
source "$(dirname "$0")/lib.sh"

run_id="${1:?Usage: package.sh RUN_ID}"
usd_file="$(env_value SO101_USD_PATH)"
usd_file="${usd_file:-${HOME}/.cache/pixel2place/so101/usd/so101_new_calib/so101_new_calib.usda}"
if [[ ! -f "${usd_file}" ]]; then
  echo "FAIL: SO-101 USD ${usd_file} does not exist. Make it with sim/import_so101.py." >&2
  exit 1
fi
# USD는 <usd>/so101_new_calib/so101_new_calib.usda다. <usd> 디렉터리를 통째로 묶는다.
usd_root="$(dirname "$(dirname "${usd_file}")")"

dest="${REPO_ROOT}/${EVIDENCE_DIR}/${run_id}"
mkdir -p "${dest}/inject"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT
git -C "${REPO_ROOT}" archive --format=tar.gz -o "${stage}/source.tar.gz" HEAD
tar -C "$(dirname "${usd_root}")" -czf "${stage}/so101-usd.tar.gz" "$(basename "${usd_root}")"
cp "${JOB_DIR}/scripts/entrypoint.sh" "${dest}/inject/entrypoint.sh"
{
  echo "run_id=${run_id}"
  echo "source_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "worktree_changes=$(git -C "${REPO_ROOT}" status --porcelain | wc -l)"
  echo "usd_dir=$(basename "${usd_root}")"
  (cd "${stage}" && sha256sum source.tar.gz so101-usd.tar.gz)
  (cd "${dest}/inject" && sha256sum entrypoint.sh)
} >"${dest}/inject/manifest.txt"

prefix="s3://${BUCKET}/${run_id}"
s3 cp --recursive --only-show-errors "${stage}" "${prefix}/in/"
# aws CLI는 GET 주소만 서명한다. PUT 주소는 boto3로 서명한다.
output_url="$(
  uv run --quiet --no-project --with boto3 python - \
    "${S3_PROFILE}" "${S3_ENDPOINT}" "${BUCKET}" "${run_id}/out/out.tar.gz" "${URL_EXPIRES_S}" <<'PY'
import sys

import boto3

profile, endpoint, bucket, key, expires = sys.argv[1:6]
client = boto3.Session(profile_name=profile).client("s3", endpoint_url=endpoint)
print(
    client.generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=int(expires)
    )
)
PY
)"
# 주소에는 &가 있다. submit.sh가 이 파일을 source한다. 그래서 값을 작은따옴표로 감싼다.
{
  echo "P2P_SOURCE_URL='$(s3 presign "${prefix}/in/source.tar.gz" --expires-in "${URL_EXPIRES_S}")'"
  echo "P2P_USD_URL='$(s3 presign "${prefix}/in/so101-usd.tar.gz" --expires-in "${URL_EXPIRES_S}")'"
  echo "P2P_OUTPUT_URL='${output_url}'"
} >"${dest}/urls.env"
chmod 600 "${dest}/urls.env"
echo "PASS inputs are in s3://<bucket>/${run_id}/in/"
