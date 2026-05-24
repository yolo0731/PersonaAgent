from __future__ import annotations

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
    parse_tlv_map,
)


def _packet(msg_type: MessageType, body: bytes) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=99), body=body)


def _message_body(
    *,
    message_id: int,
    conversation_type: int = 1,
    conversation_id: int = 10011002,
    sender_id: int = 1002,
    receiver_id: int = 1001,
    text: str = "hello bot",
    timestamp_ms: int = 1_700_000_001_000,
    client_message_id: str | None = None,
) -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, conversation_type, body)
    append_uint64(TlvType.ConversationId, conversation_id, body)
    append_uint64(TlvType.SenderId, sender_id, body)
    append_uint64(TlvType.ReceiverId, receiver_id, body)
    append_string(TlvType.MessageText, text, body)
    if client_message_id is not None:
        append_string(TlvType.ClientMessageId, client_message_id, body)
    append_uint64(TlvType.TimestampMs, timestamp_ms, body)
    return bytes(body)


def _repeated_message_body() -> bytes:
    body = bytearray()
    for message_id, text, client_message_id in (
        (5001, "older offline", "alice-1"),
        (5002, "newer offline", "alice-2"),
    ):
        body.extend(
            _message_body(
                message_id=message_id,
                text=text,
                timestamp_ms=1_700_000_000_000 + message_id,
                client_message_id=client_message_id,
            )
        )
    return bytes(body)


def _receipt_body(
    *,
    kind: MessageType,
    message_id: int = 5001,
    conversation_id: int = 10011002,
    peer_id: int = 1002,
    delivery_status: int = 2,
) -> bytes:
    body = bytearray()
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, conversation_id, body)
    append_uint64(TlvType.MessageId, message_id, body)
    if kind == MessageType.ReadReceiptPush:
        append_uint64(TlvType.UserId, peer_id, body)
    else:
        append_uint64(TlvType.ReceiverId, peer_id, body)
    append_uint64(TlvType.DeliveryStatus, delivery_status, body)
    return bytes(body)


@dataclass
class FakeReliabilityClient:
    offline_messages: list[object] = field(default_factory=list)
    identity: BotIdentity = field(
        default_factory=lambda: BotIdentity(
            user_id=1001,
            username="agent_bot",
            nickname="Agent Bot",
            session_id=5001,
        )
    )
    offline_pull_limits: list[int] = field(default_factory=list)
    offline_acks: list[list[int]] = field(default_factory=list)
    delivery_acks: list[int] = field(default_factory=list)
    read_acks: list[tuple[int, int]] = field(default_factory=list)
    sent_private_messages: list[tuple[int, str, str | None]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    async def pull_offline_messages(self, limit: int = 100) -> list[object]:
        self.offline_pull_limits.append(limit)
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
    ) -> None:
        self.sent_private_messages.append((receiver_id, text, client_message_id))
        self.events.append(f"reply:{receiver_id}")


def test_protocol_parser_reads_single_and_repeated_message_packets() -> None:
    from bot_client.protocol_parser import parse_incoming_message, parse_offline_messages

    push = _packet(MessageType.PrivateMessagePush, _message_body(message_id=5001))
    message = parse_incoming_message(push)

    assert message.message_id == 5001
    assert message.conversation_type == 1
    assert message.conversation_id == 10011002
    assert message.sender_id == 1002
    assert message.receiver_id == 1001
    assert message.text == "hello bot"
    assert message.client_message_id is None

    offline = _packet(MessageType.OfflineMessagesResponse, _repeated_message_body())
    messages = parse_offline_messages(offline)

    assert [item.message_id for item in messages] == [5001, 5002]
    assert [item.text for item in messages] == ["older offline", "newer offline"]
    assert [item.client_message_id for item in messages] == ["alice-1", "alice-2"]


def test_protocol_builders_encode_liteim_reliability_requests() -> None:
    from bot_client.protocol_builders import (
        make_delivery_ack_body,
        make_offline_ack_body,
        make_offline_request_body,
        make_private_message_body,
        make_read_ack_body,
    )

    offline_request = parse_tlv_map(make_offline_request_body(limit=25))
    assert offline_request[int(TlvType.Limit)][0] == (25).to_bytes(8, "big")

    offline_ack = parse_tlv_map(make_offline_ack_body([5001, 5002]))
    assert [int.from_bytes(value, "big") for value in offline_ack[int(TlvType.MessageId)]] == [
        5001,
        5002,
    ]

    delivery_ack = parse_tlv_map(make_delivery_ack_body(5001))
    assert int.from_bytes(delivery_ack[int(TlvType.MessageId)][0], "big") == 5001

    read_ack = parse_tlv_map(make_read_ack_body(10011002, 5002))
    assert int.from_bytes(read_ack[int(TlvType.ConversationType)][0], "big") == 1
    assert int.from_bytes(read_ack[int(TlvType.ConversationId)][0], "big") == 10011002
    assert int.from_bytes(read_ack[int(TlvType.MessageId)][0], "big") == 5002

    private = parse_tlv_map(make_private_message_body(1002, "reply", "pa-1001-abc"))
    assert int.from_bytes(private[int(TlvType.ReceiverId)][0], "big") == 1002
    assert private[int(TlvType.MessageText)][0].decode() == "reply"
    assert private[int(TlvType.ClientMessageId)][0].decode() == "pa-1001-abc"


def test_json_message_state_persists_processed_messages_and_receipts(tmp_path: Path) -> None:
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import ReceiptTraceEvent

    state_path = tmp_path / "bot_state" / "state.json"
    state = JsonMessageState(state_path)

    assert not state.has_processed(5001)
    state.mark_processed(5001)
    state.record_receipt(
        ReceiptTraceEvent(
            kind="delivered",
            message_id=5001,
            conversation_id=10011002,
            peer_user_id=1002,
            delivery_status=2,
        )
    )

    reloaded = JsonMessageState(state_path)

    assert reloaded.has_processed(5001)
    assert reloaded.receipts[0].kind == "delivered"
    assert reloaded.receipts[0].message_id == 5001


async def test_message_handler_pulls_offline_after_login_and_acks_after_processing(
    tmp_path: Path,
) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import parse_offline_messages

    messages = parse_offline_messages(
        _packet(MessageType.OfflineMessagesResponse, _message_body(message_id=5001))
    )
    client = FakeReliabilityClient(offline_messages=messages)
    processed: list[int] = []

    async def processor(message: object) -> MessageProcessingResult:
        processed.append(message.message_id)  # type: ignore[attr-defined]
        client.events.append(f"process:{message.message_id}")  # type: ignore[attr-defined]
        return MessageProcessingResult()

    handler = BotMessageHandler(
        client=client,
        state=JsonMessageState(tmp_path / "state.json"),
        processor=processor,
    )

    await handler.sync_offline_after_login(limit=100)

    assert client.offline_pull_limits == [100]
    assert processed == [5001]
    assert client.offline_acks == [[5001]]
    assert client.read_acks == [(10011002, 5001)]
    assert client.events == ["process:5001", "read:5001", "offline_ack:5001"]


async def test_message_handler_sends_delivery_read_and_reply_with_client_message_id(
    tmp_path: Path,
) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState

    client = FakeReliabilityClient()
    processed: list[int] = []

    async def processor(message: object) -> MessageProcessingResult:
        processed.append(message.message_id)  # type: ignore[attr-defined]
        client.events.append(f"process:{message.message_id}")  # type: ignore[attr-defined]
        return MessageProcessingResult(reply_text="pong")

    handler = BotMessageHandler(
        client=client,
        state=JsonMessageState(tmp_path / "state.json"),
        processor=processor,
    )
    packet = _packet(MessageType.PrivateMessagePush, _message_body(message_id=5001))

    await handler.handle_packet(packet)

    assert client.delivery_acks == [5001]
    assert client.read_acks == [(10011002, 5001)]
    assert processed == [5001]
    assert client.sent_private_messages == [(1002, "pong", client.sent_private_messages[0][2])]
    assert client.sent_private_messages[0][2] is not None
    assert client.sent_private_messages[0][2].startswith("pa-1001-")
    assert client.events == ["delivery:5001", "process:5001", "read:5001", "reply:1002"]


async def test_message_handler_deduplicates_processed_messages_across_restarts(
    tmp_path: Path,
) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState

    state_path = tmp_path / "state.json"
    packet = _packet(MessageType.PrivateMessagePush, _message_body(message_id=5001))
    first_client = FakeReliabilityClient()
    second_client = FakeReliabilityClient()
    processed: list[int] = []

    async def processor(message: object) -> MessageProcessingResult:
        processed.append(message.message_id)  # type: ignore[attr-defined]
        return MessageProcessingResult()

    first = BotMessageHandler(
        client=first_client,
        state=JsonMessageState(state_path),
        processor=processor,
    )
    await first.handle_packet(packet)

    second = BotMessageHandler(
        client=second_client,
        state=JsonMessageState(state_path),
        processor=processor,
    )
    await second.handle_packet(packet)

    assert processed == [5001]
    assert first_client.delivery_acks == [5001]
    assert second_client.delivery_acks == [5001]
    assert first_client.read_acks == [(10011002, 5001)]
    assert second_client.read_acks == []


async def test_message_handler_ignores_messages_sent_by_bot_itself(tmp_path: Path) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState

    client = FakeReliabilityClient()
    processed: list[int] = []

    async def processor(message: object) -> MessageProcessingResult:
        processed.append(message.message_id)  # type: ignore[attr-defined]
        return MessageProcessingResult(reply_text="should not send")

    handler = BotMessageHandler(
        client=client,
        state=JsonMessageState(tmp_path / "state.json"),
        processor=processor,
    )
    packet = _packet(
        MessageType.PrivateMessagePush,
        _message_body(message_id=5001, sender_id=1001, receiver_id=1002),
    )

    await handler.handle_packet(packet)

    assert processed == []
    assert client.delivery_acks == []
    assert client.read_acks == []
    assert client.sent_private_messages == []


async def test_message_handler_records_delivery_and_read_receipts(tmp_path: Path) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState

    state = JsonMessageState(tmp_path / "state.json")
    handler = BotMessageHandler(
        client=FakeReliabilityClient(),
        state=state,
        processor=lambda message: MessageProcessingResult(),
    )

    await handler.handle_packet(
        _packet(
            MessageType.DeliveryReceiptPush,
            _receipt_body(kind=MessageType.DeliveryReceiptPush),
        )
    )
    await handler.handle_packet(
        _packet(
            MessageType.ReadReceiptPush,
            _receipt_body(kind=MessageType.ReadReceiptPush, delivery_status=3),
        )
    )

    assert [event.kind for event in state.receipts] == ["delivered", "read"]
    assert [event.message_id for event in state.receipts] == [5001, 5001]
