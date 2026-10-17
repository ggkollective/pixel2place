"""조율 프로그램 CLI의 입력 검사와 외부 클라이언트 수명."""

from pathlib import Path

from openai import OpenAI
from pydantic import HttpUrl, TypeAdapter

from pixel2place.agent.prompts import system_prompt
from pixel2place.agent.runtime import AgentRuntime
from pixel2place.agent.runtime_contracts import (
    MAX_REQUEST_RETRIES,
    AgentBudget,
    AgentMetadata,
    AgentReport,
)
from pixel2place.agent.tools import AssemblyTools
from pixel2place.cell.config import CellConfig
from pixel2place.cell.scene import SimScene
from pixel2place.config import (
    DEFAULT_JUDGMENT_MODEL,
    NonBlankStr,
    Settings,
    SimSettings,
    TokenFactorySettings,
)
from pixel2place.reasoner.client import UNUSED_API_KEY, CosmosReasoner
from pixel2place.reasoner.nemotron import create_nemotron_client

DISPOSITION_MODEL = DEFAULT_JUDGMENT_MODEL
"""처분을 켠 실행에서 `checked`의 요청을 보내는 모델. 처분 역할의 오프라인 평가로 정했다.

조율 모델은 재작업 도구를 받은 요청에서 불량을 고르지 못했다.
"""
MAX_PLACE_RUNS = 2
"""처분을 켠 실행의 배치 한도. 첫 배치와 재작업 1회다."""


def run_agent_cell(
    prompt: str,
    *,
    out_dir: Path,
    cell: CellConfig,
    scene: SimScene,
    token_settings: TokenFactorySettings,
    cosmos_settings: Settings,
    sim_settings: SimSettings,
    review: bool = False,
    disposition: bool = False,
) -> AgentReport:
    """입력을 고정한다. 도구를 조율한다. 종료 때 두 SDK 클라이언트를 닫는다.

    `review`는 계획 검토를 켠다. 모델이 준비한 계획을 목표와 비교하고 재계획을 고를 수 있다.
    `disposition`은 처분을 켠다. 모델이 실행 뒤의 카메라 판정으로 출하, 재작업과 불량을 고른다.
    출하는 참값 판정과 카메라 판정의 통과를 요구한다. 배치 실행의 한도는 2회가 된다.
    검토 요청과 처분 요청은 판단 모델로 보낸다. 다른 요청은 `token_settings`의 조율 모델로 보낸다.
    일시 오류가 난 Nemotron 요청은 다시 보낸다. Cosmos 요청은 다시 보내지 않는다.
    """
    review_model = DEFAULT_JUDGMENT_MODEL if review else None
    disposition_model = DISPOSITION_MODEL if disposition else None
    budget = AgentBudget(
        request_retries=MAX_REQUEST_RETRIES,
        max_place_runs=MAX_PLACE_RUNS if disposition else 1,
    )
    cosmos_url = TypeAdapter(HttpUrl).validate_python(cosmos_settings.reasoner_base_url)
    cosmos_model = TypeAdapter(NonBlankStr).validate_python(cosmos_settings.cosmos_reasoner_model)
    token_settings = token_settings.model_copy(
        update={
            "nemotron_timeout_s": budget.nemotron_timeout_s,
            "nemotron_max_tokens": budget.max_tokens,
        }
    )
    key = cosmos_settings.reasoner_api_key
    with (
        create_nemotron_client(token_settings) as client,
        OpenAI(
            base_url=str(cosmos_url),
            api_key=key.get_secret_value() if key else UNUSED_API_KEY,
            timeout=budget.cosmos_timeout_s,
            max_retries=budget.sdk_retries,
        ) as cosmos,
    ):
        reasoner = CosmosReasoner(
            client=cosmos,
            model=cosmos_model,
            max_tokens=budget.cosmos_max_tokens,
        )
        tools = AssemblyTools(
            out_dir=out_dir,
            prompt=prompt,
            cell=cell,
            scene=scene,
            reasoner=reasoner,
            sim_settings=sim_settings,
            agent=AgentMetadata(
                model=token_settings.nemotron_model,
                cosmos_model=cosmos_model,
                budget=budget,
                system_prompt=system_prompt(review=review, disposition=disposition),
                review=review,
                review_model=review_model,
                disposition=disposition,
                disposition_model=disposition_model,
            ),
        )
        return AgentRuntime(
            tools=tools,
            client=client,
            model=token_settings.nemotron_model,
            budget=budget,
            prompt=prompt,
            review_model=review_model,
            disposition_model=disposition_model,
        ).run()
