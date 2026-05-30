# 构造 LiteIM 请求 body：登录注册、好友操作、离线 ACK、送达/已读 ACK、私聊发送。
from __future__ import annotations

from bot_client.protocol.codec import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
)

# 把 BotClient 要发送给 LiteIM 的请求参数组装成二进制 TLV body。

# 根据消息类型、序列号和请求体，构造一个 Packet 对象
def make_packet(msg_type: MessageType, seq_id: int, body: bytes = b"") -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)

# 构造登录消息 TLV body
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


def make_friend_action_body(requester_id: int) -> bytes:
    if requester_id == 0:
        raise ValueError("friend requester id must not be zero")
    body = bytearray()
    append_uint64(TlvType.TargetUserId, requester_id, body)
    return bytes(body)


def make_offline_request_body(limit: int) -> bytes:
    body = bytearray()
    append_uint64(TlvType.Limit, limit, body)
    return bytes(body)


def make_offline_ack_body(message_ids: list[int]) -> bytes:
    if not message_ids:
        raise ValueError("offline ack message ids must not be empty")
    body = bytearray()
    for message_id in message_ids:
        if message_id == 0:
            raise ValueError("offline ack message id must not be zero")
        append_uint64(TlvType.MessageId, message_id, body)
    return bytes(body)


def make_delivery_ack_body(message_id: int) -> bytes:
    if message_id == 0:
        raise ValueError("delivery ack message id must not be zero")
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    return bytes(body)


def make_read_ack_body(conversation_id: int, message_id: int) -> bytes:
    if conversation_id == 0:
        raise ValueError("read ack conversation id must not be zero")
    if message_id == 0:
        raise ValueError("read ack message id must not be zero")
    body = bytearray()
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, conversation_id, body)
    append_uint64(TlvType.MessageId, message_id, body)
    return bytes(body)


def make_history_request_body(
    *,
    conversation_type: int,
    conversation_id: int,
    before_message_id: int = 0,
    limit: int,
) -> bytes:
    if conversation_type == 0:
        raise ValueError("conversation type must not be zero")
    if conversation_id == 0:
        raise ValueError("conversation id must not be zero")
    if limit <= 0:
        raise ValueError("history limit must be positive")
    body = bytearray()
    append_uint64(TlvType.ConversationType, conversation_type, body)
    append_uint64(TlvType.ConversationId, conversation_id, body)
    if before_message_id != 0:
        append_uint64(TlvType.MessageId, before_message_id, body)
    append_uint64(TlvType.Limit, limit, body)
    return bytes(body)

# 包装客户端发送消息的 TLV body，包含接收者 ID、消息文本和客户端消息 ID（用于去重）
def make_private_message_body(
    receiver_id: int,
    text: str,
    client_message_id: str,
) -> bytes:
    if receiver_id == 0:
        raise ValueError("receiver id must not be zero")
    if not text:
        raise ValueError("message text must not be empty")
    if not client_message_id:
        raise ValueError("client message id must not be empty")
    body = bytearray()
    append_uint64(TlvType.ReceiverId, receiver_id, body)
    append_string(TlvType.MessageText, text, body)
    append_string(TlvType.ClientMessageId, client_message_id, body)
    return bytes(body)
