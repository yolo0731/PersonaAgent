from __future__ import annotations

from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
)


def make_packet(msg_type: MessageType, seq_id: int, body: bytes = b"") -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)


def make_login_body(username: str, password: str) -> bytes:
    body = bytearray()
    append_string(TlvType.Username, username, body)
    append_string(TlvType.Password, password, body)
    return bytes(body)


def make_register_body(username: str, password: str, nickname: str) -> bytes:
    body = bytearray(make_login_body(username, password))
    if nickname:
        append_string(TlvType.Nickname, nickname, body)
    return bytes(body)
