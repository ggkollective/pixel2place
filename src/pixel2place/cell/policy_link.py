"""Isaac 실행기와 제어기 프로세스 사이의 통신.

제어기는 Isaac `python.sh` 밖의 프로세스에서 돈다. 두 프로세스는 TCP로 메시지를 주고받는다.
제어기 프로세스가 서버다. 실행기(`sim/place_task.py --policy`)가 클라이언트다.

메시지 1개의 형식:
1. 헤더 길이. 4바이트, big-endian 부호 없는 정수다.
2. 헤더. UTF-8 JSON 객체다. `type`이 메시지 종류다. `arrays`는 배열의 이름, dtype과 모양이다.
3. 배열의 원시 바이트. `arrays`의 순서다.

요청과 응답:
- `hello`: 요청은 `protocol`과 `dof_names`다. 응답은 `hello`다.
- `reset`: 요청은 에피소드의 문맥(`context`)이다. 응답은 `ok`와 선택 값 `max_steps`다.
- `act`: 요청은 `step`과 배열 `state`, `front`, `wrist`다. 응답은 `actions`와 배열 `actions`다.
- 서버가 요청을 처리하지 못하면 `error`와 `message`로 답한다.

실행기는 pixel2place를 import하지 않는다. 그래서 실행기에 같은 형식의 코드가 따로 있다.
"""

import json
import logging
import socket
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
HEADER_LENGTH_FORMAT = ">I"
HEADER_LENGTH_BYTES = struct.calcsize(HEADER_LENGTH_FORMAT)
# 헤더는 작은 JSON이다. 이 값을 넘는 길이는 형식 오류다.
MAX_HEADER_BYTES = 1 << 20
# 이미지 2개와 관절 값이다. 이 값을 넘는 배열은 형식 오류다.
MAX_ARRAY_BYTES = 64 << 20
MESSAGE_HELLO = "hello"
MESSAGE_RESET = "reset"
MESSAGE_ACT = "act"
MESSAGE_OK = "ok"
MESSAGE_ACTIONS = "actions"
MESSAGE_ERROR = "error"
ARRAY_STATE = "state"
ARRAY_FRONT = "front"
ARRAY_WRIST = "wrist"
ARRAY_ACTIONS = "actions"
# 관절 순서. 실행기의 순서와 같아야 한다.
DOF_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LOOPBACK_HOST = "127.0.0.1"
# 서버가 연결을 기다리는 동안 중단 요청을 확인하는 간격(초)
ACCEPT_POLL_S = 0.2

Arrays = dict[str, NDArray[Any]]


class PolicyLinkError(RuntimeError):
    """메시지 형식이 틀렸거나 연결이 끊겼다."""


@dataclass(frozen=True)
class PolicyAddress:
    """제어기 서버의 주소."""

    host: str
    port: int

    @classmethod
    def parse(cls, text: str) -> "PolicyAddress":
        """`HOST:PORT`를 읽는다."""
        host, separator, port = text.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError(f"Controller address {text!r} is not HOST:PORT.")
        return cls(host=host, port=int(port))

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class Observation:
    """`act` 요청의 관측."""

    step: int
    """지금까지 진행한 프레임 수에서 1을 뺀 값. 기록한 에피소드의 프레임 번호와 같다."""
    state: NDArray[np.float32]
    """측정 관절 값. 모양은 (6,)이다."""
    front: NDArray[np.uint8]
    """외부 사선 카메라 이미지. 모양은 (높이, 너비, 3)이다."""
    wrist: NDArray[np.uint8]


class Policy(Protocol):
    """제어기 서버가 부르는 제어기."""

    def reset(self, context: Mapping[str, Any]) -> int | None:
        """에피소드를 시작한다. 프레임 수의 한도를 정하면 그 값을 반환한다."""
        ...

    def act(self, observation: Observation) -> NDArray[np.float32]:
        """다음 프레임들의 관절 목표. 모양은 (프레임, 6)이다. 단위는 rad다."""
        ...


def send_message(
    connection: socket.socket, header: Mapping[str, Any], arrays: Arrays | None = None
) -> None:
    """메시지 1개를 보낸다."""
    arrays = arrays or {}
    described = {
        **header,
        "arrays": [
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in arrays.items()
        ],
    }
    encoded = json.dumps(described).encode("utf-8")
    payload = [struct.pack(HEADER_LENGTH_FORMAT, len(encoded)), encoded]
    payload += [np.ascontiguousarray(array).tobytes() for array in arrays.values()]
    connection.sendall(b"".join(payload))


def receive_message(connection: socket.socket) -> tuple[dict[str, Any], Arrays]:
    """메시지 1개를 받는다. `(헤더, 배열)`을 반환한다.

    Raises:
        PolicyLinkError: 연결이 끊겼거나 형식이 틀렸을 때.
    """
    (length,) = struct.unpack(HEADER_LENGTH_FORMAT, _receive_exact(connection, HEADER_LENGTH_BYTES))
    if length > MAX_HEADER_BYTES:
        raise PolicyLinkError(f"Message header of {length} bytes is too long.")
    try:
        header = json.loads(_receive_exact(connection, length).decode("utf-8"))
        specs = header.pop("arrays")
        arrays: Arrays = {}
        for spec in specs:
            dtype = np.dtype(spec["dtype"])
            shape = tuple(int(size) for size in spec["shape"])
            size = dtype.itemsize * int(np.prod(shape, dtype=np.int64))
            if dtype.hasobject or size > MAX_ARRAY_BYTES:
                raise PolicyLinkError(f"Array {spec['name']!r} is not valid.")
            data = _receive_exact(connection, size)
            arrays[str(spec["name"])] = np.frombuffer(data, dtype=dtype).reshape(shape)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PolicyLinkError(f"Message is not valid: {exc}") from exc
    return header, arrays


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise PolicyLinkError("Connection closed before the message ended.")
        chunks += chunk
    return bytes(chunks)


class PolicyServer:
    """제어기 1개를 TCP로 제공한다. 연결을 1개씩 차례로 처리한다.

    `start`가 서버 스레드를 시작한다. `port`가 0이면 운영체제가 포트를 고른다.
    `address`가 실제 주소다. `close`가 서버를 끝낸다.
    """

    def __init__(self, policy: Policy, *, host: str = LOOPBACK_HOST, port: int = 0) -> None:
        self.policy = policy
        self._listener = socket.create_server((host, port))
        self._listener.settimeout(ACCEPT_POLL_S)
        bound_host, bound_port = self._listener.getsockname()[:2]
        self.address = PolicyAddress(host=bound_host, port=bound_port)
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        """`close`를 부를 때까지 연결을 받는다."""
        while not self._stopped.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                self._serve_connection(connection)

    def close(self) -> None:
        self._stopped.set()
        self._listener.close()
        if self._thread is not None:
            self._thread.join()

    def __enter__(self) -> "PolicyServer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        """연결 1개의 요청을 끝까지 처리한다. 제어기의 예외는 `error` 응답으로 보낸다."""
        while True:
            try:
                header, arrays = receive_message(connection)
            except (PolicyLinkError, OSError):
                return
            try:
                reply, reply_arrays = self._handle(header, arrays)
            except Exception as exc:
                logger.warning("Controller request failed.", extra={"error": str(exc)})
                reply, reply_arrays = {"type": MESSAGE_ERROR, "message": str(exc)}, {}
            try:
                send_message(connection, reply, reply_arrays)
            except OSError:
                return

    def _handle(self, header: Mapping[str, Any], arrays: Arrays) -> tuple[dict[str, Any], Arrays]:
        kind = header.get("type")
        if kind == MESSAGE_HELLO:
            if header.get("protocol") != PROTOCOL_VERSION:
                raise PolicyLinkError(f"Protocol {header.get('protocol')!r} is not supported.")
            if tuple(header.get("dof_names", ())) != DOF_NAMES:
                raise PolicyLinkError("Joint names do not agree.")
            return {"type": MESSAGE_HELLO, "protocol": PROTOCOL_VERSION}, {}
        if kind == MESSAGE_RESET:
            max_steps = self.policy.reset(header.get("context", {}))
            return {"type": MESSAGE_OK, "max_steps": max_steps}, {}
        if kind == MESSAGE_ACT:
            observation = Observation(
                step=int(header["step"]),
                state=arrays[ARRAY_STATE].astype(np.float32),
                front=arrays[ARRAY_FRONT],
                wrist=arrays[ARRAY_WRIST],
            )
            actions = np.asarray(self.policy.act(observation), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != len(DOF_NAMES) or not len(actions):
                raise PolicyLinkError(f"The controller returned actions of shape {actions.shape}.")
            return {"type": MESSAGE_ACTIONS}, {ARRAY_ACTIONS: actions}
        raise PolicyLinkError(f"Message type {kind!r} is not known.")


class PolicyClient:
    """제어기 서버의 클라이언트. 가짜 실행기와 테스트가 쓴다. 실제 실행기는 자기 코드를 쓴다."""

    def __init__(self, address: PolicyAddress, *, timeout_s: float) -> None:
        self._connection = socket.create_connection((address.host, address.port), timeout=timeout_s)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def hello(self) -> None:
        self._request(
            {"type": MESSAGE_HELLO, "protocol": PROTOCOL_VERSION, "dof_names": list(DOF_NAMES)}
        )

    def reset(self, context: Mapping[str, Any]) -> int | None:
        header, _ = self._request({"type": MESSAGE_RESET, "context": dict(context)})
        return header.get("max_steps")

    def act(self, observation: Observation) -> NDArray[np.float32]:
        _, arrays = self._request(
            {"type": MESSAGE_ACT, "step": observation.step},
            {
                ARRAY_STATE: observation.state,
                ARRAY_FRONT: observation.front,
                ARRAY_WRIST: observation.wrist,
            },
        )
        return arrays[ARRAY_ACTIONS].astype(np.float32)

    def _request(
        self, header: Mapping[str, Any], arrays: Arrays | None = None
    ) -> tuple[dict[str, Any], Arrays]:
        send_message(self._connection, header, arrays)
        reply, reply_arrays = receive_message(self._connection)
        if reply.get("type") == MESSAGE_ERROR:
            raise PolicyLinkError(f"Controller server returned an error: {reply.get('message')}")
        return reply, reply_arrays
