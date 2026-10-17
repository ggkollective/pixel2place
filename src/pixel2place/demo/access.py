"""공유 접속 코드로 데모의 화면과 API를 보호한다."""

import hmac
import os
import re
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

COOKIE_NAME = "pixel2place_demo"
_CODE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_LOGIN_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pixel2place · Demo access</title>
<style>
body{margin:0;background:#eef0f2;color:#1f262e;font:15px system-ui,sans-serif;
display:grid;place-items:center;min-height:100vh}
main{box-sizing:border-box;width:min(420px,90vw);background:white;border:1px solid
#d5dae0;border-top:4px solid #f2a900;border-radius:10px;padding:36px}
h1{font-size:25px;margin:0 0 24px}
p{line-height:1.6;color:#5f6b77}label{display:block;font-size:13px;margin:24px 0 10px}
input{box-sizing:border-box;width:100%;font:inherit;padding:13px;border:1px solid
#d5dae0;border-radius:7px}button{width:100%;margin-top:16px;border:0;border-radius:7px;
background:#f2a900;color:#1f262e;padding:14px;font:inherit;font-weight:700;cursor:pointer}
.error{color:#c8412f}small{color:#8a95a1}
</style></head><body><main><small>PIXEL2PLACE · Nebius x NVIDIA Global AI Hackathon Demo</small>
<h1>Open the assembly cell.</h1><p>Enter the demo access code from the service owner.</p>
<!-- message --><form method="post" action="/access">
<label for="code">Access code</label><input id="code" name="code" type="password"
autocomplete="current-password" maxlength="80" required autofocus>
<button type="submit">Open demo</button></form></main></body></html>"""


def load_access_code(root: Path) -> str:
    """소유자만 읽을 수 있는 파일에 코드를 저장한다. symlink는 거부한다."""
    descriptor = os.open(root / ".access-code", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+", encoding="ascii") as file:
        os.fchmod(file.fileno(), 0o600)
        code = file.read(128).strip()
        if code:
            if not _CODE.fullmatch(code):
                raise ValueError("The demo access code file is not valid.")
        else:
            code = secrets.token_urlsafe(32)
            file.seek(0)
            file.truncate()
            file.write(code + "\n")
            file.flush()
            os.fsync(file.fileno())
    return code


def check_origin(request: Request) -> None:
    """조작 요청의 Origin이 같은 Host인지 검사한다."""
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="The request origin is not allowed.")


def install_access(app: FastAPI, *, code: str) -> None:
    """로그인 화면 외의 화면과 모든 API에 인증 쿠키를 요구한다."""
    if not _CODE.fullmatch(code):
        raise ValueError("The demo access code is not valid.")

    def matches(value: str) -> bool:
        return hmac.compare_digest(value.encode(), code.encode())

    @app.middleware("http")
    async def require_access(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == "/access" or matches(request.cookies.get(COOKIE_NAME, "")):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "The demo access code is required."}, status_code=401)
        return RedirectResponse("/access", status_code=303)

    @app.get("/access", response_class=HTMLResponse)
    def access_form() -> HTMLResponse:
        return HTMLResponse(_LOGIN_HTML.replace("<!-- message -->", ""))

    @app.post("/access")
    async def sign_in(request: Request) -> Response:
        check_origin(request)
        body = await request.body()
        if len(body) > 512:
            raise HTTPException(status_code=400, detail="The access input is not valid.")
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="The access input is not valid.") from None
        values = parse_qs(text).get("code", [""])
        if not matches(values[0]):
            return HTMLResponse(
                _LOGIN_HTML.replace(
                    "<!-- message -->", '<p class="error">Invalid access code.</p>'
                ),
                status_code=401,
            )
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE_NAME, code, httponly=True, samesite="strict", max_age=43200)
        return response
