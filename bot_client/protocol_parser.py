from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    ProtocolError,
    ProtocolErrorCode,
    TlvType,
    get_string,
    get_uint64,
    parse_tlv_map,
)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    message_id: int
    conversation_type: int
    conversation_id: int
    sender_id: int
    receiver_id: int
    text: str
    timestamp_ms: int
    client_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReceiptTraceEvent:
    kind: Literal["delivered", "read"]
    message_id: int
    conversation_id: int
    peer_user_id: int
    delivery_status: int


def parse_incoming_message(packet: Packet) -> IncomingMessage:
    fields = parse_tlv_map(packet.body)
    client_values = fields.get(int(TlvType.ClientMessageId), [])
    client_message_id = client_values[0].decode("utf-8") if client_values else None
    return IncomingMessage(
        message_id=get_uint64(fields, TlvType.MessageId),
        conversation_type=get_uint64(fields, TlvType.ConversationType),
        conversation_id=get_uint64(fields, TlvType.ConversationId),
        sender_id=get_uint64(fields, TlvType.SenderId),
        receiver_id=get_uint64(fields, TlvType.ReceiverId),
        text=get_string(fields, TlvType.MessageText),
        timestamp_ms=get_uint64(fields, TlvType.TimestampMs),
        client_message_id=client_message_id,
    )


def parse_offline_messages(packet: Packet) -> list[IncomingMessage]:
    fields = parse_tlv_map(packet.body)
    message_ids = _uint64_values(fields, TlvType.MessageId)
    if not message_ids:
        return []

    conversation_types = _uint64_values(fields, TlvType.ConversationType)
    conversation_ids = _uint64_values(fields, TlvType.ConversationId)
    sender_ids = _uint64_values(fields, TlvType.SenderId)
    receiver_ids = _uint64_values(fields, TlvType.ReceiverId)
    texts = _string_values(fields, TlvType.MessageText)
    timestamp_ms_values = _uint64_values(fields, TlvType.TimestampMs)
    client_message_ids = _optional_string_values(fields, TlvType.ClientMessageId, len(message_ids))

    count = len(message_ids)
    if not all(
        len(values) == count
        for values in (
            conversation_types,
            conversation_ids,
            sender_ids,
            receiver_ids,
            texts,
            timestamp_ms_values,
            client_message_ids,
        )
    ):
        raise ProtocolError(
            ProtocolErrorCode.ParseError,
            "offline response message fields mismatch",
        )

    return [
        IncomingMessage(
            message_id=message_ids[index],
            conversation_type=conversation_types[index],
            conversation_id=conversation_ids[index],
            sender_id=sender_ids[index],
            receiver_id=receiver_ids[index],
            text=texts[index],
            timestamp_ms=timestamp_ms_values[index],
            client_message_id=client_message_ids[index],
        )
        for index in range(count)
    ]


def parse_receipt(packet: Packet) -> ReceiptTraceEvent:
    fields = parse_tlv_map(packet.body)
    if packet.header.msg_type == MessageType.ReadReceiptPush:
        kind: Literal["delivered", "read"] = "read"
        peer_user_id = get_uint64(fields, TlvType.UserId)
    else:
        kind = "delivered"
        peer_user_id = get_uint64(fields, TlvType.ReceiverId)

    return ReceiptTraceEvent(
        kind=kind,
        message_id=get_uint64(fields, TlvType.MessageId),
        conversation_id=get_uint64(fields, TlvType.ConversationId),
        peer_user_id=peer_user_id,
        delivery_status=get_uint64(fields, TlvType.DeliveryStatus),
    )


def _uint64_values(fields: dict[int, list[bytes]], tlv_type: TlvType) -> list[int]:
    values = fields.get(int(tlv_type), [])
    output: list[int] = []
    for value in values:
        if len(value) != 8:
            raise ProtocolError(ProtocolErrorCode.ParseError, "uint64 tlv field must be 8 bytes")
        output.append(int.from_bytes(value, "big"))
    return output


def _string_values(fields: dict[int, list[bytes]], tlv_type: TlvType) -> list[str]:
    try:
        return [value.decode("utf-8") for value in fields.get(int(tlv_type), [])]
    except UnicodeDecodeError as exc:
        raise ProtocolError(ProtocolErrorCode.ParseError, "string tlv field is not utf-8") from exc


def _optional_string_values(
    fields: dict[int, list[bytes]],
    tlv_type: TlvType,
    expected_count: int,
) -> list[str | None]:
    values = _string_values(fields, tlv_type)
    if not values:
        return [None] * expected_count
    if len(values) == expected_count:
        output: list[str | None] = [value for value in values]
        return output
    raise ProtocolError(
        ProtocolErrorCode.ParseError,
        "optional string tlv field count does not match messages",
    )
