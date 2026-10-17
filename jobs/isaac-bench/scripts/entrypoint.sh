#!/usr/bin/env bash
# Nebius Serverless Job 안에서 실행한다. 이미지는 nvcr.io/nvidia/isaac-sim이다. root 권한을 쓰지 않는다.
# 이 파일과 manifest.txt는 Job의 --inject-file로 같은 디렉터리에 들어온다.
# 설정은 환경 변수로 받는다.
#   P2P_MODE: rehearse | probe | smoke | bench
#   P2P_SOURCE_URL: source.tar.gz를 받는 주소
#   P2P_USD_URL: so101-usd.tar.gz를 받는 주소
#   P2P_OUTPUT_URL: out.tar.gz를 PUT으로 올리는 주소
#   P2P_BENCH_ARGS: bench 모드의 bench-cell 인자. 공백으로 나눈다.
#
# 버킷 마운트를 쓰지 않는다. 마운트는 root만 접근할 수 있다. Isaac 이미지는 root가 아닌 사용자로 실행한다.
# 그래서 입력과 출력은 만료 시간이 있는 서명된 주소로 주고받는다.
#
# 단계:
# 1. 입력을 받아서 컨테이너 디스크에 푼다.
# 2. 환경을 기록한다. GPU, 드라이버, Vulkan ICD와 그래픽 라이브러리다.
#    rehearse 모드는 여기서 끝난다. GPU 없이 입력과 출력의 경로만 확인한다.
# 3. VM이 NVIDIA 그래픽 라이브러리를 컨테이너에 넣었는지 확인한다. Isaac의 렌더러가 이 라이브러리를 쓴다.
# 4. 이미지의 warmup.sh와 sim/hello_capture.py를 실행한다. 이 실행이 셰이더 캐시를 만든다.
# 5. smoke와 bench 모드: uv 환경과 ffmpeg를 준비한다.
# 6. smoke 모드: pytest -m sim을 실행한다.
# 7. bench 모드: pixel2place bench-cell을 실행한다.
#
# 출력은 컨테이너 디스크에 쓴다. 1분마다(bench 모드는 5분마다)와 끝날 때 out.tar.gz 1개로 묶어서 올린다.
set -uo pipefail

ISAAC_DIR="${ISAAC_DIR:-/isaac-sim}"
INJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${P2P_WORK_DIR:-/tmp/p2p}"
SRC_DIR="${WORK_DIR}/src"
OUT_DIR="${WORK_DIR}/out"
BIN_DIR="${WORK_DIR}/bin"
# 출력을 올리는 주기(초). bench 모드는 출력이 커서 더 긴 주기를 쓴다.
SYNC_INTERVAL_S=60
BENCH_SYNC_INTERVAL_S=300
HELLO_TIMEOUT_S=600
WARMUP_TIMEOUT_S=1200
SMOKE_TIMEOUT_S=1500
# Isaac 이미지의 HEALTHCHECK는 이 디렉터리의 가장 새로운 Kit 로그에서 "AppReady"를 찾는다.
# python.sh로 실행한 Isaac은 Kit 로그를 다른 경로에 쓴다. 그래서 검사가 통과하지 않는다.
# Nebius는 검사가 통과하기 전의 Job을 STARTING으로 둔다. 생성 약 21분 뒤에 그 Job을 ERROR로 끝내고 VM을 지운다.
# 그래서 이 스크립트가 시작할 때 표식 파일을 만든다. 실행 중에도 수정 시각을 새로 쓴다.
HEALTH_LOG="${ISAAC_DIR}/.nvidia-omniverse/logs/Kit/p2p-job/1.0/kit_p2p_job.log"
HEALTH_INTERVAL_S=20
ICD_DIRS=(/etc/vulkan/icd.d /usr/share/vulkan/icd.d)

log() {
  echo "[p2p-job] $(date -u +%H:%M:%S) $*"
}

# 단계 1개의 결과와 걸린 시간을 stages.txt에 쓴다.
STAGE_STARTED=0
stage_start() {
  STAGE_STARTED="$(date +%s)"
  log "Stage $1 started."
}
stage_end() {
  local name="$1" result="$2"
  echo "${name} ${result} $(($(date +%s) - STAGE_STARTED))s" >>"${OUT_DIR}/stages.txt"
  log "Stage ${name} result: ${result}."
}

mark_ready() {
  mkdir -p "$(dirname "${HEALTH_LOG}")" && echo "AppReady" >"${HEALTH_LOG}"
}

# 출력 디렉터리를 out.tar.gz로 묶어서 올린다. 같은 주소에 다시 올리면 이전 파일을 바꾼다.
sync_out() {
  local archive="${WORK_DIR}/out.tar.gz"
  tar -czf "${archive}" -C "${OUT_DIR}" . 2>/dev/null
  # 실행 중인 단계가 파일을 쓰고 있으면 tar가 1로 끝난다. 그 묶음도 올린다.
  [[ -s "${archive}" ]] && upload "${archive}" "${P2P_OUTPUT_URL}"
}

finish() {
  local code="$1"
  [[ -n "${SYNC_PID:-}" ]] && kill "${SYNC_PID}" 2>/dev/null
  [[ -n "${HEALTH_PID:-}" ]] && kill "${HEALTH_PID}" 2>/dev/null
  [[ -n "${GPU_PID:-}" ]] && kill "${GPU_PID}" 2>/dev/null
  echo "exit_code=${code}" >>"${OUT_DIR}/stages.txt"
  if sync_out; then
    log "Outputs are in the bucket."
  else
    log "The upload to the bucket failed."
  fi
  exit "${code}"
}

# 파일 1개를 서명된 주소에 PUT으로 올린다.
upload() {
  curl -fsS --retry 3 -T "$1" "$2" >/dev/null
}

download() {
  curl -fsSL --retry 3 -o "$2" "$1"
}

record_environment() {
  {
    echo "== time"
    date -u
    echo "== user"
    id
    echo "== system"
    uname -a
    head -n 4 /etc/os-release
    nproc
    free -g
    df -h /tmp
    echo "== gpu"
    nvidia-smi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "== variables"
    env | grep -E '^(NVIDIA_|VK_|LD_LIBRARY_PATH|ACCEPT_EULA|OMNI_|__EGL)' | sort
    echo "== vulkan and egl files"
    ls -l "${ICD_DIRS[@]}" /etc/glvnd/egl_vendor.d /usr/share/glvnd/egl_vendor.d 2>&1
    echo "== graphics libraries"
    ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-(glcore|rtcore|glvkspirv|gpucomp)|libnvoptix'
    echo "== tools"
    for tool in curl wget tar git dpkg xz ffmpeg python3; do
      echo "${tool}: $(command -v "${tool}" || echo none)"
    done
    echo "== isaac directory"
    ls -ld "${ISAAC_DIR}" "${ISAAC_DIR}/python.sh" "${ISAAC_DIR}/kit/cache" "${HOME:-/nonexistent}"
    echo "== kit logs that the image health check reads"
    find "${ISAAC_DIR}/.nvidia-omniverse/logs" -name 'kit_*.log' 2>&1
    echo "== injected files"
    ls -ln "${INJECT_DIR}"
  } >"${OUT_DIR}/environment.txt" 2>&1
}

# Vulkan ICD 파일과 libGLX_nvidia가 둘 다 있으면 성공이다.
graphics_ready() {
  ldconfig -p | grep -q 'libGLX_nvidia' || return 1
  local dir
  for dir in "${ICD_DIRS[@]}"; do
    compgen -G "${dir}/nvidia*.json" >/dev/null && return 0
  done
  return 1
}

# hello 캡처를 1회 실행한다. 인자는 시도 이름이다.
# 실패하면 Kit 로그를 시도 디렉터리에 복사한다. Kit 로그는 렌더러의 Info 줄을 가진다.
hello_attempt() {
  local name="$1" dir="${OUT_DIR}/hello/$1"
  mkdir -p "${dir}"
  find "${ISAAC_DIR}/kit/logs" -name '*.log' -delete 2>/dev/null
  timeout "${HELLO_TIMEOUT_S}" "${ISAAC_DIR}/python.sh" "${SRC_DIR}/sim/hello_capture.py" \
    --out-dir "${dir}/frames" >"${dir}/hello_capture.log" 2>&1
  if [[ -f "${dir}/frames/summary.json" ]]; then
    log "Hello capture ${name} passed. $(tr -d '\n ' <"${dir}/frames/summary.json")"
    return 0
  fi
  find "${ISAAC_DIR}/kit/logs" -name '*.log' -exec cp {} "${dir}/kit.log" \; 2>/dev/null
  log "Hello capture ${name} failed. Last log lines:"
  tail -n 12 "${dir}/hello_capture.log" | cut -c1-300
  return 1
}

# 새 VM에는 셰이더 캐시가 없다. 캐시가 없으면 첫 캡처가 "renderer failed to advance"로 실패한다.
# 그래서 이미지의 warmup.sh를 먼저 실행한다. 그 뒤의 첫 캡처는 첫 프레임에서 몇 분 기다린다.
# 캡처가 실패하면 1회 다시 실행한다.
run_hello() {
  if [[ -x "${ISAAC_DIR}/warmup.sh" ]]; then
    local started
    started="$(date +%s)"
    log "Isaac warm-up started."
    (cd "${ISAAC_DIR}" && timeout "${WARMUP_TIMEOUT_S}" ./warmup.sh) >"${OUT_DIR}/hello/warmup.log" 2>&1
    log "Isaac warm-up ended. exit=$? seconds=$(($(date +%s) - started))"
  else
    log "The image has no warmup.sh."
  fi
  hello_attempt attempt1 || hello_attempt attempt2
}

prepare_python() {
  export UV_CACHE_DIR="${WORK_DIR}/uv-cache"
  export UV_PYTHON_INSTALL_DIR="${WORK_DIR}/uv-python"
  download https://astral.sh/uv/install.sh "${WORK_DIR}/uv-install.sh" || return 1
  # UV_UNMANAGED_INSTALL은 셸 프로필과 HOME을 바꾸지 않고 이 디렉터리에만 설치한다.
  UV_UNMANAGED_INSTALL="${BIN_DIR}" sh "${WORK_DIR}/uv-install.sh" >"${OUT_DIR}/uv_install.log" 2>&1 || return 1
  (cd "${SRC_DIR}" && uv sync --locked) >"${OUT_DIR}/uv_sync.log" 2>&1 || return 1
  if ! command -v ffmpeg >/dev/null; then
    # imageio-ffmpeg wheel은 libx264가 있는 정적 ffmpeg를 가진다.
    local ffmpeg_exe
    ffmpeg_exe="$(cd "${SRC_DIR}" && uv run --with imageio-ffmpeg python -c \
      'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>>"${OUT_DIR}/uv_sync.log")" || return 1
    ln -sf "${ffmpeg_exe}" "${BIN_DIR}/ffmpeg"
  fi
  ffmpeg -version | head -n 1
}

run_smoke() {
  local attempt dir
  for attempt in 1 2; do
    dir="${OUT_DIR}/smoke/attempt${attempt}"
    mkdir -p "${dir}"
    # 테스트 이름과 결과를 Job 로그에 바로 쓴다. VM이 중간에 없어져도 진행한 곳을 알 수 있다.
    (cd "${SRC_DIR}" && timeout "${SMOKE_TIMEOUT_S}" uv run pytest -m sim -v -p no:cacheprovider \
      --basetemp="${dir}/tmp" --junitxml="${dir}/junit.xml") 2>&1 | tee "${dir}/pytest.log"
    local code="${PIPESTATUS[0]}"
    # skip만 있는 실행도 exit 0이다. 그래서 skip이 없는지 함께 확인한다.
    if [[ "${code}" -eq 0 ]] && grep -q 'skipped="0"' "${dir}/junit.xml"; then
      return 0
    fi
    log "Smoke attempt ${attempt} failed. exit=${code}"
  done
  return 1
}

run_bench() {
  local -a bench_args
  read -r -a bench_args <<<"${P2P_BENCH_ARGS:-}"
  log "bench-cell started. arguments: ${bench_args[*]}"
  (cd "${SRC_DIR}" && uv run pixel2place bench-cell --out-dir "${OUT_DIR}/bench" "${bench_args[@]}") \
    2>&1 | tee "${OUT_DIR}/bench.log" | grep --line-buffered -v '^\[place_task\]'
  return "${PIPESTATUS[0]}"
}

: "${P2P_MODE:?P2P_MODE is not set.}"
: "${P2P_SOURCE_URL:?P2P_SOURCE_URL is not set.}"
: "${P2P_USD_URL:?P2P_USD_URL is not set.}"
: "${P2P_OUTPUT_URL:?P2P_OUTPUT_URL is not set.}"
mkdir -p "${SRC_DIR}" "${OUT_DIR}" "${BIN_DIR}"
export PATH="${BIN_DIR}:${PATH}"
export ISAAC_SIM_PATH="${ISAAC_DIR}"
[[ "${P2P_MODE}" == bench ]] && SYNC_INTERVAL_S="${BENCH_SYNC_INTERVAL_S}"
log "Job started. mode=${P2P_MODE}"
mark_ready
(while sleep "${HEALTH_INTERVAL_S}"; do mark_ready; done) >/dev/null 2>&1 &
HEALTH_PID=$!

stage_start inputs
download "${P2P_SOURCE_URL}" "${WORK_DIR}/source.tar.gz" \
  && download "${P2P_USD_URL}" "${WORK_DIR}/so101-usd.tar.gz" \
  && tar -xzf "${WORK_DIR}/source.tar.gz" -C "${SRC_DIR}" \
  && tar -xzf "${WORK_DIR}/so101-usd.tar.gz" -C "${WORK_DIR}" \
  && cp "${INJECT_DIR}/manifest.txt" "${OUT_DIR}/manifest.txt" \
  && (cd "${WORK_DIR}" && grep -E ' (source|so101-usd)\.tar\.gz$' "${OUT_DIR}/manifest.txt" | sha256sum -c --quiet)
if [[ $? -ne 0 ]]; then
  stage_end inputs FAIL
  finish 1
fi
usd_dir="$(sed -n 's/^usd_dir=//p' "${OUT_DIR}/manifest.txt")"
export SO101_USD_PATH="${WORK_DIR}/${usd_dir}/so101_new_calib/so101_new_calib.usda"
stage_end inputs PASS

# 이 반복은 표준 출력을 닫는다. 남은 sleep이 Job의 출력 스트림을 붙잡지 않게 한다.
(while sleep "${SYNC_INTERVAL_S}"; do sync_out; done) >/dev/null 2>&1 &
SYNC_PID=$!
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 10 \
  >"${OUT_DIR}/gpu_usage.csv" 2>/dev/null &
GPU_PID=$!

stage_start environment
record_environment
stage_end environment PASS
[[ "${P2P_MODE}" == rehearse ]] && finish 0

stage_start graphics
if graphics_ready; then
  stage_end graphics PASS
else
  # 라이브러리가 없어도 hello 캡처를 실행한다. Isaac의 오류 줄을 근거로 남긴다.
  stage_end graphics MISSING
fi

stage_start hello
mkdir -p "${OUT_DIR}/hello"
if run_hello; then
  stage_end hello PASS
  sync_out
else
  stage_end hello FAIL
  finish 1
fi
[[ "${P2P_MODE}" == probe ]] && finish 0

stage_start python
if prepare_python; then
  stage_end python PASS
else
  stage_end python FAIL
  finish 1
fi

if [[ "${P2P_MODE}" == smoke ]]; then
  stage_start smoke
  if run_smoke; then
    stage_end smoke PASS
    finish 0
  fi
  stage_end smoke FAIL
  finish 1
fi

stage_start bench
if run_bench; then
  stage_end bench PASS
  finish 0
fi
stage_end bench FAIL
finish 1
