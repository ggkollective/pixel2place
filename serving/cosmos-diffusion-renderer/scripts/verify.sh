#!/usr/bin/env bash
# 노드마다 /health, /v1/models, 작은 inverse 요청을 검사한다. 하나라도 실패하면 exit 1로 끝난다.
# 사용법: verify.sh [노드...]
source "$(dirname "$0")/lib.sh"

# 검사용 입력은 256x256 그라디언트 PNG다. host python3 표준 라이브러리로 만든다.
frame="$(mktemp --suffix=.png)"
trap 'rm -f "${frame}"' EXIT
python3 - "${frame}" <<'PY'
import struct
import sys
import zlib

size = 256
raw = b"".join(
    b"\x00" + bytes(value for x in range(size) for value in (x, y, 128)) for y in range(size)
)


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


with open(sys.argv[1], "wb") as file:
    file.write(b"\x89PNG\r\n\x1a\n")
    file.write(chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)))
    file.write(chunk(b"IDAT", zlib.compress(raw)))
    file.write(chunk(b"IEND", b""))
PY

status=0
nodes="$(target_nodes "$@")"
for node in ${nodes}; do
  base="http://${node}:${PORT}"
  if ! curl -sf -m 5 "${base}/health" | grep -q '"ready"'; then
    echo "FAIL ${node}: ${base}/health is not ready."
    status=1
    continue
  fi
  if ! curl -sf -m 5 "${base}/v1/models" | grep -q "\"${INVERSE_MODEL}\""; then
    echo "FAIL ${node}: ${INVERSE_MODEL} is not in ${base}/v1/models."
    status=1
    continue
  fi
  body="$(curl -sf -m 600 "${base}/v1/inverse" -F "frames=@${frame}" -F passes=basecolor \
    -F height=256 -F width=256 -F response_format=b64_json)" || body=""
  if [[ "${body}" == *'"b64_json"'* ]]; then
    echo "PASS ${node}"
  else
    echo "FAIL ${node}: inverse rendering failed."
    status=1
  fi
done
exit "${status}"
