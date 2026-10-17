"""실행 프로그램 설정을 환경 변수와 `.env` 파일에서 읽는다."""

from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated

from pydantic import Field, HttpUrl, SecretStr, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Cosmos3-Nano 서버의 생성 속도는 약 13 tokens/s다. 계산 과정을 포함한 응답이 수 분 걸린다.
DEFAULT_REASONER_TIMEOUT_S = 900.0
# timeout 뒤의 재시도는 대기 시간을 두 배로 늘린다. 그래서 1회만 재시도한다.
DEFAULT_REASONER_MAX_RETRIES = 1
# 응답 1회의 최대 생성 토큰 수. 13 tokens/s에서 약 5분이다.
DEFAULT_REASONER_MAX_TOKENS = 4096
# 영상 1개(640x640, 93 frame, 35 step)의 생성에 약 190초가 걸린다.
DEFAULT_GENERATOR_TIMEOUT_S = 900.0
DEFAULT_NEMOTRON_MODEL = "nvidia/Nemotron-3_5-Lightning"
# 작업 지시 접수와 계획 검토의 모델이다. 오프라인 평가의 모든 기준을 넘은 모델이다.
DEFAULT_JUDGMENT_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
DEFAULT_NEMOTRON_TIMEOUT_S = 120.0
# reasoning 토큰도 응답 한도에 포함된다. 잘린 응답은 접속 검증에 쓰지 않는다.
DEFAULT_NEMOTRON_MAX_TOKENS = 1024
TOKEN_FACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Isaac 실행은 기동(첫 실행은 셰이더 캐시 때문에 약 165초)과 부품마다 약 12초가 걸린다.
DEFAULT_SIM_TIMEOUT_S = 1800.0
DEFAULT_SO101_USD_PATH = (
    Path.home() / ".cache/pixel2place/so101/usd/so101_new_calib/so101_new_calib.usda"
)
DEFAULT_DEMO_HOST = IPv4Address("127.0.0.1")

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseSettings):
    """Cosmos Reasoner 서버 연결 설정.

    서버는 OpenAI 호환 API를 제공한다. base URL과 모델 ID에는 기본값이 없다.
    두 값이 없으면 생성할 때 검증 오류가 난다.
    API 키는 선택이다. 서버가 키를 검사하지 않으면 비워 둔다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    reasoner_base_url: str
    cosmos_reasoner_model: str
    reasoner_api_key: SecretStr | None = None
    reasoner_timeout_s: float = DEFAULT_REASONER_TIMEOUT_S
    reasoner_max_retries: int = DEFAULT_REASONER_MAX_RETRIES
    reasoner_max_tokens: int = Field(default=DEFAULT_REASONER_MAX_TOKENS, ge=1)


class GeneratorSettings(BaseSettings):
    """Cosmos 생성 서버 연결 설정.

    `Settings`와 분리한다. 지시문 확장 없이 생성할 때는 Reasoner 설정이 없어도 된다.
    base URL과 모델 ID에는 기본값이 없다. 빈 문자열도 거부한다.
    생성 요청은 비싸다. 그래서 재시도하지 않는다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    generator_base_url: NonBlankStr
    cosmos_generator_model: NonBlankStr
    generator_timeout_s: float = Field(default=DEFAULT_GENERATOR_TIMEOUT_S, gt=0)


class TokenFactorySettings(BaseSettings):
    """Token Factory 인증과 Nemotron 접속 검증 설정.

    Cosmos 설정과 분리한다. 공식 HTTPS 주소만 허용한다.
    인증 키를 URL, 로그와 결과 파일에 넣지 않는다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    nebius_api_key: SecretStr
    nebius_base_url: HttpUrl = HttpUrl(TOKEN_FACTORY_BASE_URL)
    nemotron_model: NonBlankStr = DEFAULT_NEMOTRON_MODEL
    nemotron_timeout_s: float = Field(default=DEFAULT_NEMOTRON_TIMEOUT_S, gt=0, allow_inf_nan=False)
    nemotron_max_tokens: int = Field(default=DEFAULT_NEMOTRON_MAX_TOKENS, ge=1, le=4096)

    @field_validator("nebius_api_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        """비어 있거나 공백이 붙은 키를 거부한다."""
        key = value.get_secret_value()
        if not key or key != key.strip():
            raise ValueError("NEBIUS_API_KEY must be nonempty and have no surrounding whitespace.")
        return value

    @field_validator("nebius_base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl) -> HttpUrl:
        """인증 정보를 보낼 공식 API 주소만 허용한다."""
        if (
            value.scheme != "https"
            or value.host != "api.tokenfactory.nebius.com"
            or value.port != 443
            or value.username is not None
            or value.password is not None
            or value.query is not None
            or value.fragment is not None
            or (value.path or "").rstrip("/") != "/v1"
        ):
            raise ValueError("NEBIUS_BASE_URL must be the official Token Factory HTTPS API URL.")
        return value

    @field_validator("nemotron_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        """다른 공급자나 Nemotron이 아닌 모델을 거부한다."""
        if not value.startswith("nvidia/") or "nemotron" not in value.lower():
            raise ValueError("NEMOTRON_MODEL must be an NVIDIA Nemotron model ID.")
        return value


class SimSettings(BaseSettings):
    """Isaac Sim 실행 설정.

    `isaac_sim_path`는 Isaac Sim release 디렉터리다. `python.sh`가 이 디렉터리에 있다.
    `isaac_sim_path`에는 기본값이 없다.
    `so101_usd_path`는 `sim/import_so101.py`가 만든 SO-101 USD다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    isaac_sim_path: Path
    so101_usd_path: Path = DEFAULT_SO101_USD_PATH
    sim_timeout_s: float = Field(default=DEFAULT_SIM_TIMEOUT_S, gt=0)


class DemoSettings(BaseSettings):
    """데모 웹 서버가 여는 주소.

    기본값은 loopback이다. 다른 장치에서 접속하려면 `DEMO_HOST`에 서버 노드의 IPv4 주소를 둔다.
    서버는 같은 값을 허용 Host로 쓴다. 그래서 모든 인터페이스 주소를 거부한다.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    demo_host: IPv4Address = DEFAULT_DEMO_HOST

    @field_validator("demo_host")
    @classmethod
    def validate_host(cls, value: IPv4Address) -> IPv4Address:
        """인터페이스 1개의 주소만 허용한다."""
        if value.is_unspecified:
            raise ValueError("DEMO_HOST must be the address of one interface.")
        return value
