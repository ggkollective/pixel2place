"""CLI 프로세스의 로깅 설정."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(*, verbose: bool = False) -> None:
    """루트 로거에 stderr 핸들러를 한 번 설정한다."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # openai/httpx의 요청 단위 로그를 줄인다.
    logging.getLogger("httpx").setLevel(logging.WARNING)
