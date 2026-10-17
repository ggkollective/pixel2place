#!/usr/bin/env bash
# SO-101 URDF와 메시를 고정 커밋에서 받는다. 이미 받았으면 해시만 검사한다.
# 사용법: fetch_so101.sh
# 출력: ~/.cache/pixel2place/so101/<커밋>/Simulation/SO101/so101_new_calib.urdf
#
# 출처: github.com/TheRobotStudio/SO-ARM100 (Apache-2.0)
set -euo pipefail

REPO_URL="https://github.com/TheRobotStudio/SO-ARM100.git"
COMMIT="5f6d2b876a53a4872e405b991dd925556c9e38a4"
SUBDIR="Simulation/SO101"
URDF_FILE="so101_new_calib.urdf"
# 받은 URDF가 고정 커밋의 파일과 같은지 검사한다.
URDF_SHA256="3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
DEST="${HOME}/.cache/pixel2place/so101/${COMMIT}"

if [[ ! -f "${DEST}/${SUBDIR}/${URDF_FILE}" ]]; then
  rm -rf "${DEST}"
  git clone --quiet --filter=blob:none --no-checkout --sparse "${REPO_URL}" "${DEST}"
  git -C "${DEST}" sparse-checkout set "${SUBDIR}"
  git -C "${DEST}" -c advice.detachedHead=false checkout --quiet "${COMMIT}"
fi

actual="$(sha256sum "${DEST}/${SUBDIR}/${URDF_FILE}" | cut -d' ' -f1)"
if [[ "${actual}" != "${URDF_SHA256}" ]]; then
  echo "FAIL: ${URDF_FILE} hash is ${actual}. Expected ${URDF_SHA256}." >&2
  exit 1
fi
echo "license=$(head -1 "${DEST}/LICENSE" | xargs)"
echo "${DEST}/${SUBDIR}/${URDF_FILE}"
