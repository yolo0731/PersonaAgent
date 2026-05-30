from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep

import httpx
from fastapi.testclient import TestClient

from bot_client.connection.client import BotIdentity
from bot_client.protocol.codec import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
)
from bot_client.protocol.parsers import FriendProfile, parse_incoming_message


def _chat_payload(run_id: str = "run-1") -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": "hello agent",
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def _packet(msg_type: MessageType, body: bytes) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=99), body=body)


def _private_message_body(message_id: int = 7001, text: str = "hello agent") -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, 10011002, body)
    append_uint64(TlvType.SenderId, 1002, body)
    append_uint64(TlvType.ReceiverId, 1001, body)
    append_string(TlvType.MessageText, text, body)
    append_string(TlvType.ClientMessageId, "alice-7001", body)
    append_uint64(TlvType.TimestampMs, 1_700_000_001_000, body)
    return bytes(body)


@dataclass
class FakeReliabilityClient:
    identity: BotIdentity = field(
        default_factory=lambda: BotIdentity(
            user_id=1001,
            username="agent_bot",
            nickname="Agent Bot",
            session_id=5001,
        )
    )
    delivery_acks: list[int] = field(default_factory=list)
    read_acks: list[tuple[int, int]] = field(default_factory=list)
    sent_private_messages: list[tuple[int, str, str | None]] = field(default_factory=list)

    async def pull_offline_messages(self, limit: int = 100) -> list[object]:
        return []

    async def ack_offline_messages(self, message_ids: list[int]) -> None:
        return None

    async def send_delivery_ack(self, message_id: int) -> None:
        self.delivery_acks.append(message_id)

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None:
        self.read_acks.append((conversation_id, message_id))

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> None:
        self.sent_private_messages.append((receiver_id, text, client_message_id))


def test_chat_endpoint_returns_mock_reply_command() -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app

    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.post("/chat", json=_chat_payload("run-chat-ok"))

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["error"] is None
    assert body["command"]["run_id"] == "run-chat-ok"
    assert body["command"]["should_send"] is True
    assert body["command"]["receiver_id"] == 1002
    assert body["command"]["text"] == "mock reply: hello agent"
    assert body["command"]["client_message_id"] == "pa-run-chat-ok"


def test_chat_endpoint_returns_structured_error_envelope() -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app
    from agent_service.schemas import ChatRequest

    async def failing_handler(_request: ChatRequest) -> object:
        raise RuntimeError("boom")

    client = TestClient(create_app(Settings(_env_file=None), chat_handler=failing_handler))

    response = client.post("/chat", json=_chat_payload("run-chat-error"))

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["command"]["run_id"] == "run-chat-error"
    assert body["command"]["should_send"] is False
    assert body["error"] == {
        "code": "agent_service_error",
        "message": "boom",
        "retryable": True,
    }


def test_chat_request_from_liteim_message_maps_fields_and_run_id() -> None:
    from bot_client.agent.api import chat_request_from_message

    message = parse_incoming_message(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    request = chat_request_from_message(message)

    assert request.run_id == "liteim-message-7001"
    assert request.conversation_type == 1
    assert request.conversation_id == 10011002
    assert request.message_id == 7001
    assert request.sender_id == 1002
    assert request.receiver_id == 1001
    assert request.text == "hello agent"
    assert request.client_message_id == "alice-7001"


def test_chat_request_from_liteim_message_includes_recent_context() -> None:
    from bot_client.agent.api import chat_request_from_message

    current = parse_incoming_message(
        _packet(MessageType.PrivateMessagePush, _private_message_body(message_id=7003))
    )
    recent = [
        parse_incoming_message(
            _packet(
                MessageType.HistoryResponse,
                _private_message_body(message_id=7001, text="前一句上下文"),
            )
        ),
        parse_incoming_message(
            _packet(
                MessageType.HistoryResponse,
                _private_message_body(message_id=7002, text="后一句上下文"),
            )
        ),
    ]

    request = chat_request_from_message(current, recent_context=recent)

    assert [item.message_id for item in request.recent_context] == [7001, 7002]
    assert [item.text for item in request.recent_context] == [
        "前一句上下文",
        "后一句上下文",
    ]


async def test_agent_api_client_posts_chat_request_and_returns_command() -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app
    from agent_service.schemas import ChatRequest
    from bot_client.agent.api import AgentApiClient

    app = create_app(Settings(_env_file=None))
    transport = httpx.ASGITransport(app=app)
    api_client = AgentApiClient(
        base_url="http://agent.test",
        timeout_seconds=0.2,
        transport=transport,
    )

    command = await api_client.chat(ChatRequest.model_validate(_chat_payload("run-http-ok")))

    assert command.run_id == "run-http-ok"
    assert command.should_send is True
    assert command.receiver_id == 1002
    assert command.text == "mock reply: hello agent"
    assert command.client_message_id == "pa-run-http-ok"


async def test_chat_endpoint_runs_sync_handler_without_blocking_health() -> None:
    import asyncio

    from agent_service.config import Settings
    from agent_service.main import create_app
    from agent_service.schemas import ChatRequest, no_reply_command

    def slow_handler(request: ChatRequest) -> object:
        sleep(0.2)
        return no_reply_command(request, "slow_handler_done")

    app = create_app(Settings(_env_file=None), chat_handler=slow_handler)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        base_url="http://agent.test",
        timeout=1.0,
        transport=transport,
    ) as client:
        started_at = perf_counter()
        chat_task = asyncio.create_task(client.post("/chat", json=_chat_payload("run-slow")))
        await asyncio.sleep(0.01)
        health_response = await client.get("/health")
        health_elapsed = perf_counter() - started_at
        chat_response = await chat_task

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert health_elapsed < 0.15
    assert chat_response.status_code == 200


async def test_agent_api_client_fails_closed_on_timeout() -> None:
    from agent_service.schemas import ChatRequest
    from bot_client.agent.api import AgentApiClient

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    api_client = AgentApiClient(
        base_url="http://agent.test",
        timeout_seconds=0.01,
        transport=httpx.MockTransport(timeout_handler),
    )

    command = await api_client.chat(ChatRequest.model_validate(_chat_payload("run-timeout")))

    assert command.run_id == "run-timeout"
    assert command.should_send is False
    assert command.receiver_id == 1002
    assert command.reason == "agent_service_unavailable"


async def test_agent_service_processor_no_send_when_command_should_not_send(
    tmp_path: Path,
) -> None:
    from agent_service.schemas import AgentReplyCommand
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    class NoSendAgentClient:
        async def chat_for_message(self, message: object) -> AgentReplyCommand:
            return AgentReplyCommand(
                run_id="run-no-send",
                source_message_id=7001,
                should_send=False,
                receiver_id=1002,
                text="",
                client_message_id=None,
                reason="policy_no_reply",
            )

    state = JsonMessageState(tmp_path / "state.json")
    state.replace_friends([FriendProfile(1002, "alice", "Alice", True)])
    client = FakeReliabilityClient()
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(NoSendAgentClient()),
        require_friendship=True,
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert client.delivery_acks == [7001]
    assert client.read_acks == [(10011002, 7001)]
    assert client.sent_private_messages == []


async def test_agent_service_processor_sends_fallback_when_service_unavailable(
    tmp_path: Path,
) -> None:
    from agent_service.schemas import AgentReplyCommand
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    class UnavailableAgentClient:
        async def chat_for_message(self, message: object) -> AgentReplyCommand:
            return AgentReplyCommand(
                run_id="run-unavailable",
                source_message_id=7001,
                should_send=False,
                receiver_id=1002,
                text="",
                client_message_id=None,
                dedup_key="agent-reply:run-unavailable:7001",
                trace_summary=["agent_service_unavailable"],
                reason="agent_service_unavailable",
            )

    state = JsonMessageState(tmp_path / "state.json")
    state.replace_friends([FriendProfile(1002, "alice", "Alice", True)])
    client = FakeReliabilityClient()
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(UnavailableAgentClient()),
        require_friendship=True,
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert client.delivery_acks == [7001]
    assert client.read_acks == [(10011002, 7001)]
    assert client.sent_private_messages == [
        (1002, "我刚刚有点卡住了，你再发我一遍", "pa-unavailable-7001")
    ]


async def test_agent_service_processor_sends_command_text_and_client_message_id(
    tmp_path: Path,
) -> None:
    from agent_service.schemas import AgentReplyCommand
    from bot_client.agent.api import AgentServiceMessageProcessor
    from bot_client.messages.handler import BotMessageHandler
    from bot_client.messages.state import JsonMessageState

    class SendAgentClient:
        async def chat_for_message(self, message: object) -> AgentReplyCommand:
            return AgentReplyCommand(
                run_id="run-send",
                source_message_id=7001,
                should_send=True,
                receiver_id=1003,
                text="agent reply",
                client_message_id="pa-run-send",
                reason=None,
            )

    state = JsonMessageState(tmp_path / "state.json")
    state.replace_friends([FriendProfile(1002, "alice", "Alice", True)])
    client = FakeReliabilityClient()
    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=AgentServiceMessageProcessor(SendAgentClient()),
        require_friendship=True,
    )

    await handler.handle_packet(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )

    assert client.delivery_acks == [7001]
    assert client.read_acks == [(10011002, 7001)]
    assert client.sent_private_messages == [(1003, "agent reply", "pa-run-send")]
