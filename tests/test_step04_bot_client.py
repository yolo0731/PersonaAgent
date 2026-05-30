from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from fastapi.testclient import TestClient

from bot_client.protocol.codec import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
    encode_packet,
)
from bot_client.protocol.frame_decoder import FrameDecoder


def _body(*fields: tuple[TlvType, str | int]) -> bytes:
    body = bytearray()
    for tlv_type, value in fields:
        if isinstance(value, str):
            append_string(tlv_type, value, body)
        else:
            append_uint64(tlv_type, value, body)
    return bytes(body)


def _packet(msg_type: MessageType, seq_id: int, body: bytes = b"") -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)


async def _read_packets(
    reader: asyncio.StreamReader,
    decoder: FrameDecoder,
) -> list[Packet]:
    data = await reader.read(4096)
    if not data:
        raise ConnectionError("mock server connection closed")
    return decoder.feed(data)


class MockLiteIMServer:
    def __init__(
        self,
        handler: Callable[
            [Packet, asyncio.StreamWriter, MockLiteIMServer],
            Awaitable[None],
        ],
    ) -> None:
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None
        self.received: list[Packet] = []
        self.connections = 0
        self.login_count = 0
        self.heartbeat_event = asyncio.Event()
        self.second_login_event = asyncio.Event()

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("mock server has not started")
        sockets = self._server.sockets
        if not sockets:
            raise RuntimeError("mock server has no listening socket")
        return int(sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, 0)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def wait_for_message(self, msg_type: MessageType, timeout: float = 1.0) -> Packet:
        async def _wait() -> Packet:
            while True:
                for packet in self.received:
                    if packet.header.msg_type == msg_type:
                        return packet
                await asyncio.sleep(0.005)

        return await asyncio.wait_for(_wait(), timeout)

    async def wait_for_second_login(self, timeout: float = 1.0) -> None:
        await asyncio.wait_for(self.second_login_event.wait(), timeout)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connections += 1
        decoder = FrameDecoder()
        try:
            while True:
                for packet in await _read_packets(reader, decoder):
                    self.received.append(packet)
                    if packet.header.msg_type == MessageType.LoginRequest:
                        self.login_count += 1
                        if self.login_count >= 2:
                            self.second_login_event.set()
                    if packet.header.msg_type == MessageType.HeartbeatRequest:
                        self.heartbeat_event.set()
                    await self._handler(packet, writer, self)
        except (ConnectionError, asyncio.CancelledError):
            return
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.fixture
async def mock_server(
    request: pytest.FixtureRequest,
) -> MockLiteIMServer:
    handler = request.param
    assert callable(handler)
    server = MockLiteIMServer(handler)
    await server.start()
    try:
        yield server
    finally:
        await server.close()


def test_bot_client_settings_have_safe_defaults_without_env_file() -> None:
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(_env_file=None)

    assert settings.liteim_host == "127.0.0.1"
    assert settings.liteim_port == 9000
    assert settings.bot_username == "persona_agent_bot"
    assert settings.bot_password == "change_me"
    assert settings.bot_nickname == "PersonaAgent"
    assert settings.request_timeout_seconds == 5.0
    assert settings.heartbeat_interval_seconds == 30.0
    assert settings.reconnect_initial_delay_seconds == 0.2
    assert settings.reconnect_max_delay_seconds == 5.0


async def _auth_and_heartbeat_handler(
    packet: Packet,
    writer: asyncio.StreamWriter,
    _server: MockLiteIMServer,
) -> None:
    if packet.header.msg_type == MessageType.LoginRequest:
        response = _packet(
            MessageType.LoginResponse,
            packet.header.seq_id,
            _body(
                (TlvType.UserId, 1001),
                (TlvType.Username, "agent_bot"),
                (TlvType.Nickname, "Agent Bot"),
                (TlvType.SessionId, 5001),
            ),
        )
        writer.write(encode_packet(response))
        await writer.drain()
    elif packet.header.msg_type == MessageType.HeartbeatRequest:
        writer.write(encode_packet(_packet(MessageType.HeartbeatResponse, packet.header.seq_id)))
        await writer.drain()


@pytest.mark.parametrize("mock_server", [_auth_and_heartbeat_handler], indirect=True)
async def test_bot_client_login_and_pending_response_match(
    mock_server: MockLiteIMServer,
) -> None:
    from bot_client.connection.client import BotClient
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(
        _env_file=None,
        liteim_host=mock_server.host,
        liteim_port=mock_server.port,
        bot_username="agent_bot",
        bot_password="secret",
        bot_nickname="Agent Bot",
        request_timeout_seconds=0.2,
        heartbeat_interval_seconds=60.0,
    )
    client = BotClient(settings)

    await client.connect()
    identity = await client.login()
    response = await client.request(MessageType.HeartbeatRequest)
    await client.close()

    assert identity.user_id == 1001
    assert identity.username == "agent_bot"
    assert identity.nickname == "Agent Bot"
    assert identity.session_id == 5001
    assert response.header.msg_type == MessageType.HeartbeatResponse
    assert response.header.seq_id != 0
    assert client.pending_count == 0
    assert [packet.header.msg_type for packet in mock_server.received] == [
        MessageType.LoginRequest,
        MessageType.HeartbeatRequest,
    ]


async def _login_only_handler(
    packet: Packet,
    writer: asyncio.StreamWriter,
    _server: MockLiteIMServer,
) -> None:
    if packet.header.msg_type == MessageType.LoginRequest:
        writer.write(
            encode_packet(
                _packet(
                    MessageType.LoginResponse,
                    packet.header.seq_id,
                    _body(
                        (TlvType.UserId, 1001),
                        (TlvType.Username, "agent_bot"),
                        (TlvType.Nickname, "Agent Bot"),
                        (TlvType.SessionId, 5001),
                    ),
                )
            )
        )
        await writer.drain()


@pytest.mark.parametrize("mock_server", [_login_only_handler], indirect=True)
async def test_pending_request_timeout_clears_pending(
    mock_server: MockLiteIMServer,
) -> None:
    from bot_client.connection.client import BotClient, BotClientTimeoutError
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(
        _env_file=None,
        liteim_host=mock_server.host,
        liteim_port=mock_server.port,
        bot_username="agent_bot",
        bot_password="secret",
        request_timeout_seconds=0.03,
        heartbeat_interval_seconds=60.0,
    )
    client = BotClient(settings)

    await client.connect()
    await client.login()
    with pytest.raises(BotClientTimeoutError):
        await client.request(MessageType.HeartbeatRequest)
    await client.close()

    assert client.pending_count == 0


@pytest.mark.parametrize("mock_server", [_auth_and_heartbeat_handler], indirect=True)
async def test_heartbeat_loop_sends_request_after_login(
    mock_server: MockLiteIMServer,
) -> None:
    from bot_client.connection.client import BotClient
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(
        _env_file=None,
        liteim_host=mock_server.host,
        liteim_port=mock_server.port,
        bot_username="agent_bot",
        bot_password="secret",
        request_timeout_seconds=0.2,
        heartbeat_interval_seconds=0.02,
    )
    client = BotClient(settings)

    await client.connect()
    await client.login()
    await asyncio.wait_for(mock_server.heartbeat_event.wait(), timeout=1.0)
    await client.close()

    heartbeat = await mock_server.wait_for_message(MessageType.HeartbeatRequest)
    assert heartbeat.header.seq_id > 1


async def _close_after_first_login_handler(
    packet: Packet,
    writer: asyncio.StreamWriter,
    server: MockLiteIMServer,
) -> None:
    if packet.header.msg_type != MessageType.LoginRequest:
        return

    writer.write(
        encode_packet(
            _packet(
                MessageType.LoginResponse,
                packet.header.seq_id,
                _body(
                    (TlvType.UserId, 1001),
                    (TlvType.Username, "agent_bot"),
                    (TlvType.Nickname, "Agent Bot"),
                    (TlvType.SessionId, 5000 + server.login_count),
                ),
            )
        )
    )
    await writer.drain()
    if server.login_count == 1:
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize("mock_server", [_close_after_first_login_handler], indirect=True)
async def test_supervisor_reconnects_after_disconnect_and_logs_in_again(
    mock_server: MockLiteIMServer,
) -> None:
    from bot_client.connection.supervisor import BotClientSupervisor
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(
        _env_file=None,
        liteim_host=mock_server.host,
        liteim_port=mock_server.port,
        bot_username="agent_bot",
        bot_password="secret",
        request_timeout_seconds=0.2,
        heartbeat_interval_seconds=60.0,
        reconnect_initial_delay_seconds=0.01,
        reconnect_max_delay_seconds=0.05,
    )
    supervisor = BotClientSupervisor(settings)

    await supervisor.start()
    await mock_server.wait_for_second_login()
    await supervisor.stop()

    assert mock_server.login_count >= 2
    assert supervisor.last_error is None


async def test_supervisor_connection_failure_does_not_affect_agent_service(
    unused_tcp_port: int,
) -> None:
    from agent_service.main import create_app
    from bot_client.connection.supervisor import BotClientSupervisor
    from bot_client.runtime.config import BotClientSettings

    settings = BotClientSettings(
        _env_file=None,
        liteim_host="127.0.0.1",
        liteim_port=unused_tcp_port,
        bot_username="agent_bot",
        bot_password="secret",
        reconnect_initial_delay_seconds=0.01,
        reconnect_max_delay_seconds=0.02,
    )
    supervisor = BotClientSupervisor(settings)

    await supervisor.start()
    for _ in range(30):
        if supervisor.last_error is not None:
            break
        await asyncio.sleep(0.01)
    response = TestClient(create_app()).get("/health")
    await supervisor.stop()

    assert supervisor.last_error is not None
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "personaagent"}
