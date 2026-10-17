"""localhost 또는 사설 네트워크 HTTP의 입력·응답과 작업 수명.

실행·산출물 계약은 JobManager에 맡긴다.
모델 인증값과 임의 파일은 브라우저에 전달하지 않는다.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pixel2place.demo.access import check_origin, install_access, load_access_code
from pixel2place.demo.contracts import (
    AssetName,
    IntakeRequest,
    IntakeView,
    OrderOptions,
    OrderPreview,
    ReferenceView,
    RunRequest,
    RunSummary,
    RunView,
    ShiftRequest,
    ShiftView,
)
from pixel2place.demo.intake import IntakeBusyError, IntakeMismatchError, IntakeUnavailableError
from pixel2place.demo.jobs import VIDEO_ASSETS, JobManager, RunBusyError
from pixel2place.demo.orders import order_options, preview_order
from pixel2place.demo.shifts import ShiftDesk
from pixel2place.logconfig import setup_logging
from pixel2place.order import Order, OrderNotFeasibleError, parse_order

MONITOR_INTERVAL_S = 0.5
# 가장 큰 교대 요청보다 크다. 그 요청은 부품 16개의 주문 8개이고 들여쓴 JSON으로 약 22,000바이트다.
MAX_BODY_BYTES = 65_536
logger = logging.getLogger(__name__)


async def _monitor(manager: JobManager, shifts: ShiftDesk) -> None:
    """브라우저가 없어도 종료한 워커의 상태와 남은 프로세스를 정리한다.

    열린 교대의 다음 주문도 여기서 시작한다. 한 단계의 오류는 다른 단계를 막지 않는다.
    """
    while True:
        for step in (manager.list, shifts.advance):
            try:
                await run_in_threadpool(step)
            except (OSError, ValueError, KeyError):
                logger.error("Run monitor could not update the run or shift records.")
        await asyncio.sleep(MONITOR_INTERVAL_S)


def json_body[T](parse: Callable[[bytes], T]) -> Callable[[Request], Awaitable[T]]:
    """요청 본문을 JSON 모드로 검증하는 dependency. `parse`는 계약의 JSON 검증 함수다.

    FastAPI의 기본 검증은 Python 모드다. strict 계약의 enum과 tuple을 거부한다.
    """

    async def read(request: Request) -> T:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise RequestValidationError([])
        try:
            return parse(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from None

    return read


def create_app(
    *,
    manager: JobManager,
    shifts: ShiftDesk,
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver"),
    access_code: str | None = None,
) -> FastAPI:
    """서버 한 개의 작업 수명을 소유한다. API는 경로와 작업 정책을 받지 않는다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        monitor = asyncio.create_task(_monitor(manager, shifts))
        try:
            yield
        finally:
            monitor.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await monitor
            finally:
                try:
                    await run_in_threadpool(shifts.close)
                finally:
                    await run_in_threadpool(manager.close)

    app = FastAPI(title="pixel2place demo", lifespan=lifespan, docs_url=None, redoc_url=None)
    if access_code is not None:
        install_access(app, code=access_code)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "The run input is not valid."}, status_code=422)

    @app.exception_handler(RunBusyError)
    async def busy(request: Request, error: RunBusyError) -> JSONResponse:
        return JSONResponse(
            {"detail": "An assembly run, a shift or a demo server is already active."},
            status_code=409,
        )

    @app.exception_handler(IntakeBusyError)
    async def intake_busy(request: Request, error: IntakeBusyError) -> JSONResponse:
        return JSONResponse({"detail": "An order reading is already active."}, status_code=409)

    @app.exception_handler(IntakeUnavailableError)
    async def intake_unavailable(request: Request, error: IntakeUnavailableError) -> JSONResponse:
        return JSONResponse(
            {"detail": "The order reading is not available on this server."}, status_code=503
        )

    @app.exception_handler(IntakeMismatchError)
    async def intake_mismatch(request: Request, error: IntakeMismatchError) -> JSONResponse:
        return JSONResponse(
            {"detail": "The order is not the order of an accepted reading."}, status_code=422
        )

    @app.exception_handler(OrderNotFeasibleError)
    async def not_feasible(request: Request, error: OrderNotFeasibleError) -> JSONResponse:
        return JSONResponse({"detail": "The work order is not feasible."}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse(
            {"detail": "The run, shift or asset is not available."}, status_code=404
        )

    @app.exception_handler(ValueError)
    async def invalid_record(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse({"detail": "The run records are not valid."}, status_code=409)

    @app.exception_handler(OSError)
    async def io_error(request: Request, error: OSError) -> JSONResponse:
        return JSONResponse(
            {"detail": "The assembly worker or its records are not available."}, status_code=503
        )

    @app.get("/api/runs")
    def runs() -> tuple[RunSummary, ...]:
        return manager.list()

    @app.get("/api/reference/{variant}")
    def reference(variant: Literal["before", "after"]) -> ReferenceView:
        return manager.reference(variant)

    @app.get("/api/reference/{variant}/assets/{name}", response_class=FileResponse)
    def reference_asset(
        variant: Literal["before", "after"], name: Literal["video", "result"]
    ) -> FileResponse:
        path = manager.reference_asset(variant=variant, name=name)
        return FileResponse(path, media_type="video/mp4" if name == "video" else "image/png")

    @app.get("/api/orders/options")
    def options() -> OrderOptions:
        return order_options()

    @app.post("/api/orders/preview")
    def preview(body: Annotated[Order, Depends(json_body(parse_order))]) -> OrderPreview:
        return preview_order(body)

    @app.post("/api/orders/intake")
    def intake(
        body: Annotated[IntakeRequest, Depends(json_body(IntakeRequest.model_validate_json))],
        request: Request,
    ) -> IntakeView:
        """판단 모델에 유료 요청 1회를 보낸다. 조작 요청과 같은 Origin 검사를 한다."""
        check_origin(request)
        view = manager.intakes.read(body.text)
        logger.info(
            "Read an order from text.",
            extra={"intake_id": view.intake_id, "outcome": view.outcome.value},
        )
        return view

    @app.post("/api/runs", status_code=202)
    def start_run(
        body: Annotated[RunRequest, Depends(json_body(RunRequest.model_validate_json))],
        request: Request,
    ) -> RunView:
        check_origin(request)
        return manager.start(body)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> RunView:
        return manager.get(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str, request: Request) -> RunView:
        check_origin(request)
        return manager.cancel(run_id)

    @app.get("/api/runs/{run_id}/assets/{name}", response_class=FileResponse)
    def asset(run_id: str, name: AssetName) -> FileResponse:
        path = manager.asset(run_id, name)
        return FileResponse(path, media_type="video/mp4" if name in VIDEO_ASSETS else "image/png")

    @app.get("/api/shifts")
    def recent_shifts() -> tuple[ShiftView, ...]:
        return shifts.recent()

    @app.post("/api/shifts", status_code=202)
    def start_shift(
        body: Annotated[ShiftRequest, Depends(json_body(ShiftRequest.model_validate_json))],
        request: Request,
    ) -> ShiftView:
        """주문마다 실제 CLI 프로세스를 1개씩 차례로 시작한다."""
        check_origin(request)
        return shifts.start(body)

    @app.get("/api/shifts/{shift_id}")
    def shift(shift_id: str) -> ShiftView:
        return shifts.get(shift_id)

    @app.post("/api/shifts/{shift_id}/cancel")
    def cancel_shift(shift_id: str, request: Request) -> ShiftView:
        check_origin(request)
        return shifts.cancel(shift_id)

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/ui", StaticFiles(directory=static), name="ui")

        @app.get("/", response_class=FileResponse)
        def index() -> FileResponse:
            return FileResponse(static / "index.html", media_type="text/html")

    return app


def serve(*, root: Path, workspace: Path, host: str, port: int) -> None:
    """`host` 주소에서 인증한 데모 서버를 연다. 같은 주소만 Host로 허용한다."""
    import uvicorn

    manager = JobManager(root=root, workspace=workspace)
    try:
        code = load_access_code(manager.root)
        shifts = ShiftDesk(jobs=manager)
        uvicorn.run(
            create_app(manager=manager, shifts=shifts, allowed_hosts=(host,), access_code=code),
            host=host,
            port=port,
            workers=1,
        )
    finally:
        manager.close()
