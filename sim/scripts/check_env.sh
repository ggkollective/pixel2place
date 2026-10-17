#!/usr/bin/env bash
# Isaac Sim이 이 노드에서 headless로 동작하는지 확인한다. 하나라도 실패하면 exit 1로 끝난다.
# 사용법: ISAAC_SIM_PATH=<release 디렉터리> check_env.sh [출력 디렉터리]
#
# 단계:
# 1. 호환성 검사(GUI 앱)를 --no-window로 실행하고 결과 줄을 확인한다.
# 2. sim/hello_capture.py로 장면을 캡처하고 MP4를 만든다. 실행 중 GPU 메모리 증가량을 잰다.
# 3. 생성 서버가 영향을 받지 않았는지 verify.sh로 확인한다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-${HOME}/.cache/pixel2place/sim/hello}"
# 호환성 검사는 결과 줄을 쓴 뒤에도 종료하지 않는다. 그래서 이 시간 안에 결과 줄을 찾고 종료한다.
COMPAT_TIMEOUT_S=300
HELLO_TIMEOUT_S=1200
MEMORY_SAMPLE_S=2

if [[ -z "${ISAAC_SIM_PATH:-}" ]]; then
  echo "FAIL: Set ISAAC_SIM_PATH to the Isaac Sim release directory." >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

gpu_used_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
    | awk '{sum += $1} END {print sum + 0}'
}

echo "== compatibility check"
compat_log="${OUT_DIR}/compatibility_check.log"
"${ISAAC_SIM_PATH}/isaac-sim.compatibility_check.sh" --no-window >"${compat_log}" 2>&1 &
compat_pid=$!
result=""
for ((i = 0; i < COMPAT_TIMEOUT_S; i++)); do
  result="$(grep -o 'System checking result: [A-Z]*' "${compat_log}" || true)"
  [[ -n "${result}" ]] && break
  sleep 1
done
pkill -P "${compat_pid}" 2>/dev/null || true
kill "${compat_pid}" 2>/dev/null || true
wait "${compat_pid}" 2>/dev/null || true
if [[ "${result}" != *PASSED ]]; then
  echo "FAIL: compatibility check: ${result:-no result line}. Log: ${compat_log}" >&2
  exit 1
fi
echo "PASS ${result}"

echo "== headless capture"
baseline_mib="$(gpu_used_mib)"
peak_mib="${baseline_mib}"
hello_log="${OUT_DIR}/hello_capture.log"
timeout "${HELLO_TIMEOUT_S}" "${ISAAC_SIM_PATH}/python.sh" "${REPO_ROOT}/sim/hello_capture.py" \
  --out-dir "${OUT_DIR}/frames" >"${hello_log}" 2>&1 &
hello_pid=$!
while kill -0 "${hello_pid}" 2>/dev/null; do
  used="$(gpu_used_mib)"
  ((used > peak_mib)) && peak_mib="${used}"
  sleep "${MEMORY_SAMPLE_S}"
done
if ! wait "${hello_pid}" || [[ ! -f "${OUT_DIR}/frames/summary.json" ]]; then
  echo "FAIL: headless capture. Log: ${hello_log}" >&2
  exit 1
fi
ffmpeg -loglevel error -y -framerate 30 -i "${OUT_DIR}/frames/frame_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p "${OUT_DIR}/hello.mp4"
echo "PASS $(cat "${OUT_DIR}/frames/summary.json" | tr -d '\n ')"
echo "gpu_memory_increase_mib=$((peak_mib - baseline_mib)) video=${OUT_DIR}/hello.mp4"

echo "== generator server"
"${REPO_ROOT}/serving/cosmos-generator/scripts/verify.sh"
