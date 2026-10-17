# 공용 함수와 설정. 다른 스크립트가 source한다.
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config.env
source "${SERVING_DIR}/config.env"
# shellcheck source=../../scripts/common.sh
source "${SERVING_DIR}/../scripts/common.sh"
