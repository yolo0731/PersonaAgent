from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from agent_service.schemas import AgentReplyCommand
from bot_client.connection.client import BotIdentity
from bot_client.protocol.codec import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
)
from bot_client.protocol.parsers import FriendProfile


def _packet(msg_type: MessageType, body: bytes, seq_id: int = 0) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)


def _private_message_body(*, message_id: int = 8801, text: str = "hello agent") -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, 10011002, body)
    append_uint64(TlvType.SenderId, 1002, body)
    append_uint64(TlvType.ReceiverId, 1001, body)
    append_string(TlvType.MessageText, text, body)
    append_string(TlvType.ClientMessageId, f"demo_user-{message_id}", body)
    append_uint64(TlvType.TimestampMs, 1_700_000_002_000, body)
    return bytes(body)


@dataclass
class FakeRuntimeClient:
    identity: BotIdentity = field(
        default_factory=lambda: BotIdentity(
            user_id=1001,
            username="persona_bot",
            nickname="PersonaAgent",
            session_id=9001,
        )
    )
    push_queue: asyncio.Queue[Packet] = field(default_factory=asyncio.Queue)
    events: list[str] = field(default_factory=list)
    delivery_acks: list[int] = field(default_factory=list)
    read_acks: list[tuple[int, int]] = field(default_factory=list)
    sent_private_messages: list[tuple[int, str, str | None]] = field(default_factory=list)
    history_requests: list[tuple[int, int, int, int]] = field(default_factory=list)

    async def connect(self) -> None:
        self.events.append("connect")

    async def login(self) -> BotIdentity:
        self.events.append("login")
        return self.identity

    async def close(self) -> None:
        self.events.append("close")

    async def list_friends(self) -> list[FriendProfile]:
        self.events.append("list_friends")
        return [FriendProfile(1002, "demo_user", "demo_user", True)]

    async def list_friend_requests(self) -> list[object]:
        self.events.append("list_friend_requests")
        return []

    async def accept_friend_request(self, requester_id: int) -> object:
        raise AssertionError(f"unexpected accept {requester_id}")

    async def reject_friend_request(self, requester_id: int) -> object:
        raise AssertionError(f"unexpected reject {requester_id}")

    async def pull_offline_messages(self, limit: int = 100) -> list[object]:
        self.events.append(f"pull_offline:{limit}")
        return []

    async def ack_offline_messages(self, message_ids: list[int]) -> None:
        self.events.append(f"offline_ack:{len(message_ids)}")

    async def pull_history_messages(
        self,
        *,
        conversation_type: int,
        conversation_id: int,
        before_message_id: int = 0,
        limit: int = 8,
    ) -> list[object]:
        self.history_requests.append(
            (conversation_type, conversation_id, before_message_id, limit)
        )
        return []

    async def send_delivery_ack(self, message_id: int) -> None:
        self.delivery_acks.append(message_id)

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None:
        self.read_acks.append((conversation_id, message_id))

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> object:
        self.sent_private_messages.append((receiver_id, text, client_message_id))
        return object()


class StaticAgentClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_for_message(
        self,
        message: object,
        recent_context: list[object] | tuple[object, ...] = (),
    ) -> AgentReplyCommand:
        self.calls += 1
        return AgentReplyCommand(
            run_id="run-qt-smoke",
            source_message_id=8801,
            should_send=True,
            receiver_id=1002,
            conversation_type=1,
            conversation_id=10011002,
            text="agent reply from script",
            client_message_id="pa-run-qt-smoke",
            dedup_key="agent-reply:run-qt-smoke:8801",
            trace_summary=["finalize_reply:send_command"],
            reason="finalized_reply",
        )


async def _wait_until(predicate: object, timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


def test_cli_settings_override_bot_credentials_and_keep_qt_demo_defaults() -> None:
    from scripts.runtime.run_bot_client import build_settings, parse_args

    args = parse_args(["--username", "persona_bot", "--password", "demo_password"])
    settings = build_settings(args)

    assert settings.bot_username == "persona_bot"
    assert settings.bot_password == "demo_password"
    assert settings.liteim_host == "127.0.0.1"
    assert settings.liteim_port == 9000
    assert settings.agent_service_url == "http://127.0.0.1:8088"


def test_build_runtime_uses_agent_mode_by_default() -> None:
    from bot_client.runtime.app import AgentBotRuntime
    from scripts.runtime.run_bot_client import build_runtime, build_settings, parse_args

    args = parse_args(["--username", "persona_bot", "--password", "demo_password"])
    runtime = build_runtime(args, build_settings(args))

    assert isinstance(runtime, AgentBotRuntime)


async def test_agent_runtime_replies_to_private_push_through_agent_service(
    tmp_path: Path,
) -> None:
    from bot_client.runtime.app import AgentBotRuntime
    from bot_client.runtime.config import BotClientSettings

    client = FakeRuntimeClient()
    agent = StaticAgentClient()
    runtime = AgentBotRuntime(
        settings=BotClientSettings(
            _env_file=None,
            bot_username="persona_bot",
            bot_password="demo_password",
        ),
        client=client,
        agent_client=agent,
        state_path=tmp_path / "state.json",
    )

    await runtime.start()
    await client.push_queue.put(
        _packet(MessageType.PrivateMessagePush, _private_message_body())
    )
    await _wait_until(lambda: len(client.sent_private_messages) == 1)
    await runtime.stop()

    assert agent.calls == 1
    assert client.delivery_acks == [8801]
    assert client.read_acks == [(10011002, 8801)]
    assert client.sent_private_messages == [
        (1002, "agent reply from script", "pa-run-qt-smoke")
    ]
