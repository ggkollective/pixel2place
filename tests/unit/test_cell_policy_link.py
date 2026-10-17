import socket
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from pixel2place.cell.policy_link import (
    ARRAY_ACTIONS,
    DOF_NAMES,
    HEADER_LENGTH_FORMAT,
    MAX_HEADER_BYTES,
    MESSAGE_ACT,
    MESSAGE_ERROR,
    MESSAGE_HELLO,
    PROTOCOL_VERSION,
    Observation,
    PolicyAddress,
    PolicyClient,
    PolicyLinkError,
    PolicyServer,
    receive_message,
    send_message,
)

TIMEOUT_S = 5.0
IMAGE = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
STATE = np.linspace(0.0, 0.5, len(DOF_NAMES), dtype=np.float32)


class StubPolicy:
    """받은 문맥과 관측을 기억하고 관절 값에 1을 더한 목표 2개를 준다."""

    def __init__(self, *, max_steps: int | None = 7) -> None:
        self.max_steps = max_steps
        self.contexts: list[Mapping[str, Any]] = []
        self.observations: list[Observation] = []
        self.actions: Any = None
        self.error: Exception | None = None

    def reset(self, context: Mapping[str, Any]) -> int | None:
        self.contexts.append(context)
        return self.max_steps

    def act(self, observation: Observation) -> Any:
        if self.error is not None:
            raise self.error
        self.observations.append(observation)
        if self.actions is not None:
            return self.actions
        return np.stack([observation.state + 1, observation.state + 2])


@pytest.fixture
def policy() -> StubPolicy:
    return StubPolicy()


@pytest.fixture
def server(policy):
    with PolicyServer(policy) as running:
        yield running


def _observation(step: int = 3) -> Observation:
    return Observation(step=step, state=STATE, front=IMAGE, wrist=IMAGE[::-1])


def _raw_request(server: PolicyServer, header: dict, arrays: dict | None = None) -> dict:
    """클라이언트 클래스를 거치지 않고 요청 1개를 보낸다. 응답의 헤더를 반환한다."""
    with socket.create_connection(
        (server.address.host, server.address.port), timeout=TIMEOUT_S
    ) as connection:
        send_message(connection, header, arrays)
        reply, _ = receive_message(connection)
    return reply


@pytest.mark.unit
class TestPolicyAddress:
    """제어기 서버 주소 테스트."""

    def test_parse(self):
        """HOST:PORT를 읽는다. 문자열로 바꾸면 같은 형식이다."""
        address = PolicyAddress.parse("127.0.0.1:5555")
        assert address == PolicyAddress(host="127.0.0.1", port=5555)
        assert str(address) == "127.0.0.1:5555"

    @pytest.mark.parametrize("text", ["127.0.0.1", ":5555", "host:port", "host:", ""])
    def test_invalid(self, text):
        """HOST:PORT가 아니면 ValueError다."""
        with pytest.raises(ValueError, match="HOST:PORT"):
            PolicyAddress.parse(text)


@pytest.mark.unit
class TestMessages:
    """메시지 형식 테스트."""

    def test_round_trip_keeps_arrays(self):
        """헤더와 배열의 dtype, 모양과 값을 그대로 받는다."""
        sender, receiver = socket.socketpair()
        with sender, receiver:
            send_message(sender, {"type": MESSAGE_ACT, "step": 2}, {"state": STATE, "front": IMAGE})
            header, arrays = receive_message(receiver)
        assert header == {"type": MESSAGE_ACT, "step": 2}
        assert list(arrays) == ["state", "front"]
        assert arrays["state"].dtype == np.float32
        np.testing.assert_array_equal(arrays["state"], STATE)
        assert arrays["front"].shape == IMAGE.shape
        np.testing.assert_array_equal(arrays["front"], IMAGE)

    def test_message_without_arrays(self):
        """배열이 없는 메시지는 헤더만 준다."""
        sender, receiver = socket.socketpair()
        with sender, receiver:
            send_message(sender, {"type": MESSAGE_HELLO})
            header, arrays = receive_message(receiver)
        assert header == {"type": MESSAGE_HELLO}
        assert arrays == {}

    def test_closed_connection(self):
        """메시지가 끝나기 전에 연결이 끊기면 PolicyLinkError다."""
        sender, receiver = socket.socketpair()
        with receiver:
            sender.sendall(struct.pack(HEADER_LENGTH_FORMAT, 100) + b"{")
            sender.close()
            with pytest.raises(PolicyLinkError, match="closed"):
                receive_message(receiver)

    def test_header_too_long(self):
        """헤더 길이가 한도를 넘으면 PolicyLinkError다."""
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.sendall(struct.pack(HEADER_LENGTH_FORMAT, MAX_HEADER_BYTES + 1))
            with pytest.raises(PolicyLinkError, match="too long"):
                receive_message(receiver)

    @pytest.mark.parametrize(
        "header",
        [
            b"not json",
            b'{"type": "act"}',
            b'{"type": "act", "arrays": [{"name": "a", "dtype": "O", "shape": [1]}]}',
            b'{"type": "act", "arrays": [{"name": "a", "dtype": "nothing", "shape": [1]}]}',
            b'{"type": "act", "arrays": [{"name": "a", "dtype": "<f4"}]}',
        ],
    )
    def test_invalid_header(self, header):
        """JSON이 아니거나 배열 설명이 틀린 헤더는 PolicyLinkError다."""
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.sendall(struct.pack(HEADER_LENGTH_FORMAT, len(header)) + header)
            with pytest.raises(PolicyLinkError, match="not valid"):
                receive_message(receiver)


@pytest.mark.unit
class TestPolicyServer:
    """제어기 서버와 클라이언트 테스트."""

    def test_hello_reset_and_act(self, server, policy):
        """클라이언트가 문맥과 관측을 보내고 관절 목표의 묶음을 받는다."""
        context = {"slot": 2, "instruction": "Place it.", "layout": "002_slot_r5c2", "repeat": 4}
        with PolicyClient(server.address, timeout_s=TIMEOUT_S) as client:
            client.hello()
            assert client.reset(context) == policy.max_steps
            actions = client.act(_observation(step=3))
        assert policy.contexts == [context]
        assert actions.dtype == np.float32
        np.testing.assert_allclose(actions, np.stack([STATE + 1, STATE + 2]))
        (observation,) = policy.observations
        assert observation.step == 3
        np.testing.assert_array_equal(observation.state, STATE)
        np.testing.assert_array_equal(observation.front, IMAGE)
        np.testing.assert_array_equal(observation.wrist, IMAGE[::-1])

    def test_reset_without_step_limit(self, tmp_path):
        """제어기가 프레임 수의 한도를 정하지 않으면 reset은 None을 준다."""
        with (
            PolicyServer(StubPolicy(max_steps=None)) as server,
            PolicyClient(server.address, timeout_s=TIMEOUT_S) as client,
        ):
            assert client.reset({}) is None

    def test_serves_connections_in_turn(self, server, policy):
        """연결이 끝나면 다음 연결을 받는다."""
        for step in range(2):
            with PolicyClient(server.address, timeout_s=TIMEOUT_S) as client:
                client.hello()
                client.act(_observation(step=step))
        assert [observation.step for observation in policy.observations] == [0, 1]

    def test_port_zero_gets_free_port(self, server):
        """포트 0으로 만든 서버는 운영체제가 고른 포트를 주소에 쓴다."""
        assert server.address.port > 0

    def test_wrong_protocol(self, server):
        """프로토콜 버전이 다르면 error로 답한다."""
        reply = _raw_request(
            server,
            {"type": MESSAGE_HELLO, "protocol": PROTOCOL_VERSION + 1, "dof_names": DOF_NAMES},
        )
        assert reply["type"] == MESSAGE_ERROR
        assert "not supported" in reply["message"]

    def test_wrong_joint_names(self, server):
        """관절 이름의 순서가 다르면 error로 답한다."""
        reply = _raw_request(
            server,
            {
                "type": MESSAGE_HELLO,
                "protocol": PROTOCOL_VERSION,
                "dof_names": list(reversed(DOF_NAMES)),
            },
        )
        assert reply["type"] == MESSAGE_ERROR
        assert "Joint names" in reply["message"]

    def test_unknown_message_type(self, server):
        """모르는 메시지 종류는 error로 답한다."""
        reply = _raw_request(server, {"type": "stop"})
        assert reply["type"] == MESSAGE_ERROR
        assert "not known" in reply["message"]

    def test_policy_error_is_error_reply(self, server, policy):
        """제어기의 예외는 클라이언트에서 PolicyLinkError가 된다. 서버는 다음 요청을 처리한다."""
        policy.error = RuntimeError("Model is not loaded.")
        with PolicyClient(server.address, timeout_s=TIMEOUT_S) as client:
            with pytest.raises(PolicyLinkError, match="Model is not loaded"):
                client.act(_observation())
            policy.error = None
            assert client.act(_observation()).shape == (2, len(DOF_NAMES))

    @pytest.mark.parametrize(
        "actions",
        [np.zeros(len(DOF_NAMES)), np.zeros((2, 5)), np.zeros((0, len(DOF_NAMES)))],
    )
    def test_invalid_actions_are_error_reply(self, server, policy, actions):
        """제어기가 준 목표의 모양이 (프레임, 6)이 아니면 error로 답한다."""
        policy.actions = actions
        with (
            PolicyClient(server.address, timeout_s=TIMEOUT_S) as client,
            pytest.raises(PolicyLinkError, match="shape"),
        ):
            client.act(_observation())

    def test_missing_observation_array_is_error_reply(self, server):
        """act 요청에 배열이 없으면 error로 답한다."""
        reply = _raw_request(server, {"type": MESSAGE_ACT, "step": 0})
        assert reply["type"] == MESSAGE_ERROR

    def test_close_stops_listening(self, policy):
        """close 뒤에는 접속할 수 없다."""
        server = PolicyServer(policy)
        server.start()
        address = server.address
        server.close()
        with pytest.raises(OSError):
            socket.create_connection((address.host, address.port), timeout=1.0)

    def test_actions_array_name(self, server):
        """act 응답의 배열 이름은 actions다."""
        with socket.create_connection(
            (server.address.host, server.address.port), timeout=TIMEOUT_S
        ) as connection:
            send_message(
                connection,
                {"type": MESSAGE_ACT, "step": 0},
                {"state": STATE, "front": IMAGE, "wrist": IMAGE},
            )
            _, arrays = receive_message(connection)
        assert list(arrays) == [ARRAY_ACTIONS]
