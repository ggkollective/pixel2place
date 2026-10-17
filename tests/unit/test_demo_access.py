"""로그인 입력 오류가 인증 상태와 서버 응답을 깨뜨리지 않는지 검사한다."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pixel2place.demo.access import COOKIE_NAME, install_access

pytestmark = pytest.mark.unit
ACCESS_CODE = "a" * 43


@pytest.fixture
def client() -> Iterator[TestClient]:
    """접속 코드가 필요한 API 한 개를 제공한다."""
    app = FastAPI()
    install_access(app, code=ACCESS_CODE)

    @app.get("/api/protected")
    def protected() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app, raise_server_exceptions=False) as session:
        yield session


@pytest.mark.parametrize("body", [b"code=\xff", b"\x80", b"code=" + b"a" * 508])
def test_invalid_access_input_does_not_authenticate(client: TestClient, body: bytes) -> None:
    """잘못된 바이트와 크기 초과는 400으로 거부한다. 다음 로그인은 동작한다."""
    response = client.post("/access", content=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "The access input is not valid."}
    assert COOKIE_NAME not in client.cookies
    assert client.get("/api/protected").status_code == 401
    assert (
        client.post("/access", data={"code": ACCESS_CODE}, follow_redirects=False).status_code
        == 303
    )
    assert client.get("/api/protected").json() == {"ok": True}


def test_only_valid_same_origin_login_grants_access(client: TestClient) -> None:
    """다른 출처와 잘못된 코드를 거부한다. 올바른 코드만 API를 연다."""
    assert client.get("/api/protected").status_code == 401
    assert client.post("/access", data={"code": "wrong"}).status_code == 401
    assert (
        client.post(
            "/access", data={"code": ACCESS_CODE}, headers={"Origin": "https://outside.example"}
        ).status_code
        == 403
    )
    assert COOKIE_NAME not in client.cookies

    response = client.post("/access", data={"code": ACCESS_CODE}, follow_redirects=False)

    assert response.status_code == 303
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert client.get("/api/protected").json() == {"ok": True}
