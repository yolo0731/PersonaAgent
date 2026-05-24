from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from bot_client.bot_client import BotIdentity
from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
)
from bot_client.protocol_parser import FriendProfile, IncomingMessage, parse_incoming_message


def _packet(msg_type: MessageType, body: bytes, seq_id: int = 0) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)


def _message_body(
    *,
    message_id: int,
    sender_id: int = 1002,
    receiver_id: int = 1001,
    conversation_type: int = 1,
    conversation_id: int = 10011002,
    text: str = "hello echo",
) -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, conversation_type, body)
    append_uint64(TlvType.ConversationId, conversation_id, body)
    append_uint64(TlvType.SenderId, sender_id, body)
    append_uint64(TlvType.ReceiverId, receiver_id, body)
    append_string(TlvType.MessageText, text, body)
    append_uint64(TlvType.TimestampMs, 1_700_000_001_000 + message_id, body)
    return bytes(body)


def _incoming_message(
    *,
    message_id: int,
    sender_id: int = 1002,
    conversation_type: int = 1,
    conversation_id: int = 10011002,
    text: str = "hello echo",
) -> IncomingMessage:
    return parse_incoming_message(
        _packet(
            MessageType.PrivateMessagePush,
            _message_body(
                message_id=message_id,
                sender_id=sender_id,
                conversation_type=conversation_type,
                conversation_id=conversation_id,
                text=text,
            ),
        )
    )


@dataclass
class FakeEchoClient:
    friends: list[FriendProfile] = field(default_factory=list)
    offline_messages: list[IncomingMessage] = field(default_factory=list)
    identity: BotIdentity | None = field(
        default_factory=lambda: BotIdentity(
            user_id=1001,
            username="agent_bot",
            nickname="Agent Bot",
            session_id=5001,
        )
    )
    push_queue: asyncio.Queue[Packet] = field(default_factory=asyncio.Queue)
    events: list[str] = field(default_factory=list)
    delivery_acks: list[int] = field(default_factory=list)
    read_acks: list[tuple[int, int]] = field(default_factory=list)
    offline_acks: list[list[int]] = field(default_factory=list)
    sent_private_messages: list[tuple[int, str, str | None]] = field(default_factory=list)
    closed: bool = False

    async def connect(self) -> None:
        self.events.append("connect")

    async def login(self) -> BotIdentity:
        self.events.append("login")
        assert self.identity is not None
        return self.identity

    async def close(self) -> None:
        self.closed = True
        self.events.append("close")

    async def list_friends(self) -> list[FriendProfile]:
        self.events.append("list_friends")
        return self.friends

    async def list_friend_requests(self) -> list[object]:
        self.events.append("list_friend_requests")
        return []

    async def accept_friend_request(self, requester_id: int) -> object:
        raise AssertionError(f"unexpected accept {requester_id}")

    async def reject_friend_request(self, requester_id: int) -> object:
        raise AssertionError(f"unexpected reject {requester_id}")

    async def pull_offline_messages(self, limit: int = 100) -> list[IncomingMessage]:
        self.events.append(f"pull_offline:{limit}")
        return self.offline_messages

    async def ack_offline_messages(self, message_ids: list[int]) -> None:
        self.offline_acks.append(list(message_ids))
        self.events.append(f"offline_ack:{','.join(str(message_id) for message_id in message_ids)}")

    async def send_delivery_ack(self, message_id: int) -> None:
        self.delivery_acks.append(message_id)
        self.events.append(f"delivery:{message_id}")

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None:
        self.read_acks.append((conversation_id, message_id))
        self.events.append(f"read:{message_id}")

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> object:
        self.sent_private_messages.append((receiver_id, text, client_message_id))
        self.events.append(f"reply:{receiver_id}:{text}")
        return object()


async def _wait_until(predicate: object, timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


def test_echo_processor_returns_original_text_only_when_enabled() -> None:
    from bot_client.echo import EchoMessageProcessor

    message = _incoming_message(message_id=7001, text="ping")

    enabled = EchoMessageProcessor(enabled=True)
    disabled = EchoMessageProcessor(enabled=False)

    assert enabled(message).reply_text == "ping"
    assert disabled(message).reply_text is None


async def test_echo_runtime_syncs_friend_policy_before_offline_echo(
    tmp_path: Path,
) -> None:
    from bot_client.config import BotClientSettings
    from bot_client.runtime import EchoBotRuntime

    client = FakeEchoClient(
        friends=[FriendProfile(1002, "alice", "Alice", True)],
        offline_messages=[_incoming_message(message_id=7001, text="offline ping")],
    )
    settings = BotClientSettings(_env_file=None, echo_mode=True, offline_message_limit=25)
    runtime = EchoBotRuntime(
        settings=settings,
        client=client,
        state_path=tmp_path / "state.json",
    )

    await runtime.start()
    await runtime.stop()

    assert client.events[:5] == [
        "connect",
        "login",
        "list_friends",
        "list_friend_requests",
        "pull_offline:25",
    ]
    assert client.read_acks == [(10011002, 7001)]
    assert client.offline_acks == [[7001]]
    assert client.sent_private_messages == [
        (1002, "offline ping", client.sent_private_messages[0][2])
    ]
    assert client.sent_private_messages[0][2] is not None
    assert client.sent_private_messages[0][2].startswith("pa-1001-")


async def test_echo_runtime_consumes_live_private_push_and_replies(
    tmp_path: Path,
) -> None:
    from bot_client.config import BotClientSettings
    from bot_client.runtime import EchoBotRuntime

    client = FakeEchoClient(friends=[FriendProfile(1002, "alice", "Alice", True)])
    runtime = EchoBotRuntime(
        settings=BotClientSettings(_env_file=None, echo_mode=True),
        client=client,
        state_path=tmp_path / "state.json",
    )

    await runtime.start()
    await client.push_queue.put(
        _packet(
            MessageType.PrivateMessagePush,
            _message_body(message_id=7002, text="live ping"),
        )
    )
    await _wait_until(lambda: len(client.sent_private_messages) == 1)
    await runtime.stop()

    assert client.delivery_acks == [7002]
    assert client.read_acks == [(10011002, 7002)]
    assert client.sent_private_messages == [(1002, "live ping", client.sent_private_messages[0][2])]
    assert client.sent_private_messages[0][2] is not None
    assert client.sent_private_messages[0][2].startswith("pa-1001-")


async def test_echo_runtime_deduplicates_offline_messages_across_restarts(
    tmp_path: Path,
) -> None:
    from bot_client.config import BotClientSettings
    from bot_client.runtime import EchoBotRuntime

    state_path = tmp_path / "state.json"
    settings = BotClientSettings(_env_file=None, echo_mode=True)
    offline = [_incoming_message(message_id=7003, text="only once")]
    first = FakeEchoClient(
        friends=[FriendProfile(1002, "alice", "Alice", True)],
        offline_messages=offline,
    )
    second = FakeEchoClient(
        friends=[FriendProfile(1002, "alice", "Alice", True)],
        offline_messages=offline,
    )

    first_runtime = EchoBotRuntime(settings=settings, client=first, state_path=state_path)
    await first_runtime.start()
    await first_runtime.stop()

    second_runtime = EchoBotRuntime(settings=settings, client=second, state_path=state_path)
    await second_runtime.start()
    await second_runtime.stop()

    assert len(first.sent_private_messages) == 1
    assert second.sent_private_messages == []
    assert second.read_acks == []
    assert second.offline_acks == [[7003]]


async def test_echo_runtime_respects_echo_mode_disabled(tmp_path: Path) -> None:
    from bot_client.config import BotClientSettings
    from bot_client.runtime import EchoBotRuntime

    client = FakeEchoClient(friends=[FriendProfile(1002, "alice", "Alice", True)])
    runtime = EchoBotRuntime(
        settings=BotClientSettings(_env_file=None, echo_mode=False),
        client=client,
        state_path=tmp_path / "state.json",
    )

    await runtime.start()
    await client.push_queue.put(
        _packet(MessageType.PrivateMessagePush, _message_body(message_id=7004))
    )
    await _wait_until(lambda: client.read_acks == [(10011002, 7004)])
    await runtime.stop()

    assert client.delivery_acks == [7004]
    assert client.sent_private_messages == []


async def test_echo_runtime_records_group_push_without_reply(tmp_path: Path) -> None:
    from bot_client.config import BotClientSettings
    from bot_client.runtime import EchoBotRuntime

    client = FakeEchoClient(friends=[FriendProfile(1002, "alice", "Alice", True)])
    runtime = EchoBotRuntime(
        settings=BotClientSettings(_env_file=None, echo_mode=True),
        client=client,
        state_path=tmp_path / "state.json",
    )

    await runtime.start()
    await client.push_queue.put(
        _packet(
            MessageType.GroupMessagePush,
            _message_body(
                message_id=7005,
                conversation_type=2,
                conversation_id=88,
                receiver_id=88,
                text="group ping",
            ),
        )
    )
    await _wait_until(lambda: len(runtime.state.group_messages) == 1)
    await runtime.stop()

    assert runtime.state.group_messages[0].message_id == 7005
    assert runtime.state.group_messages[0].conversation_id == 88
    assert runtime.state.group_messages[0].text == "group ping"
    assert client.delivery_acks == []
    assert client.read_acks == []
    assert client.sent_private_messages == []
