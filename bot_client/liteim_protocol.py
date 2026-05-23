from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeAlias

PACKET_MAGIC = 0x4C494D31
PACKET_VERSION = 1
PACKET_FLAGS_NONE = 0
PACKET_HEADER_SIZE = 20
MAX_PACKET_BODY_LENGTH = 1024 * 1024

TLV_HEADER_SIZE = 6
MAX_TLV_VALUE_LENGTH = 1024 * 1024

_PACKET_HEADER = struct.Struct("!IBBHQI")
_TLV_HEADER = struct.Struct("!HI")

BytesLike: TypeAlias = bytes | bytearray | memoryview
MessageTypeValue: TypeAlias = "MessageType | int"
TlvTypeValue: TypeAlias = "TlvType | int"
TlvMap: TypeAlias = dict[int, list[bytes]]


class ProtocolErrorCode(IntEnum):
    Ok = 0
    InvalidArgument = 1
    NotFound = 2
    AlreadyExists = 3
    IoError = 4
    ParseError = 5
    ConfigError = 6
    InternalError = 7
    ResourceExhausted = 8


class ProtocolError(Exception):
    def __init__(self, code: ProtocolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MessageType(IntEnum):
    Unknown = 0

    HeartbeatRequest = 1
    HeartbeatResponse = 2

    RegisterRequest = 100
    RegisterResponse = 101
    LoginRequest = 102
    LoginResponse = 103
    LogoutRequest = 104
    LogoutResponse = 105

    AddFriendRequest = 200
    AddFriendResponse = 201
    ListFriendsRequest = 202
    ListFriendsResponse = 203
    AcceptFriendRequest = 204
    AcceptFriendResponse = 205
    RejectFriendRequest = 206
    RejectFriendResponse = 207
    ListFriendRequestsRequest = 208
    ListFriendRequestsResponse = 209
    FriendAcceptedPush = 210

    PrivateMessageRequest = 300
    PrivateMessageResponse = 301
    PrivateMessagePush = 302

    CreateGroupRequest = 400
    CreateGroupResponse = 401
    JoinGroupRequest = 402
    JoinGroupResponse = 403
    ListGroupsRequest = 404
    ListGroupsResponse = 405
    GroupMessageRequest = 406
    GroupMessageResponse = 407
    GroupMessagePush = 408

    OfflineMessagesRequest = 500
    OfflineMessagesResponse = 501
    HistoryRequest = 502
    HistoryResponse = 503
    OfflineMessagesAckRequest = 504
    OfflineMessagesAckResponse = 505
    DeliveryAckRequest = 506
    DeliveryAckResponse = 507
    DeliveryReceiptPush = 508
    ReadAckRequest = 509
    ReadAckResponse = 510
    ReadReceiptPush = 511

    ErrorResponse = 900


class TlvType(IntEnum):
    Unknown = 0

    Username = 1
    Password = 2
    UserId = 3
    Nickname = 4
    Token = 5
    SessionId = 6

    FriendId = 20
    TargetUserId = 21
    OnlineStatus = 22
    FriendRequestStatus = 23

    GroupId = 30
    GroupName = 31

    ConversationType = 40
    ConversationId = 41
    MessageId = 42
    MessageText = 43
    SenderId = 44
    ReceiverId = 45
    TimestampMs = 46
    Offset = 47
    Limit = 48
    UnreadCount = 49
    DeliveryStatus = 50
    ClientMessageId = 51

    ErrorCode = 90
    ErrorMessage = 91


@dataclass(slots=True)
class PacketHeader:
    magic: int = PACKET_MAGIC
    version: int = PACKET_VERSION
    flags: int = PACKET_FLAGS_NONE
    msg_type: MessageTypeValue = MessageType.Unknown
    seq_id: int = 0
    body_len: int = 0


@dataclass(slots=True)
class Packet:
    header: PacketHeader = field(default_factory=PacketHeader)
    body: bytes = b""


_MESSAGE_NAMES: dict[MessageType, str] = {
    MessageType.Unknown: "UNKNOWN",
    MessageType.HeartbeatRequest: "HEARTBEAT_REQUEST",
    MessageType.HeartbeatResponse: "HEARTBEAT_RESPONSE",
    MessageType.RegisterRequest: "REGISTER_REQUEST",
    MessageType.RegisterResponse: "REGISTER_RESPONSE",
    MessageType.LoginRequest: "LOGIN_REQUEST",
    MessageType.LoginResponse: "LOGIN_RESPONSE",
    MessageType.LogoutRequest: "LOGOUT_REQUEST",
    MessageType.LogoutResponse: "LOGOUT_RESPONSE",
    MessageType.AddFriendRequest: "ADD_FRIEND_REQUEST",
    MessageType.AddFriendResponse: "ADD_FRIEND_RESPONSE",
    MessageType.ListFriendsRequest: "LIST_FRIENDS_REQUEST",
    MessageType.ListFriendsResponse: "LIST_FRIENDS_RESPONSE",
    MessageType.AcceptFriendRequest: "ACCEPT_FRIEND_REQUEST",
    MessageType.AcceptFriendResponse: "ACCEPT_FRIEND_RESPONSE",
    MessageType.RejectFriendRequest: "REJECT_FRIEND_REQUEST",
    MessageType.RejectFriendResponse: "REJECT_FRIEND_RESPONSE",
    MessageType.ListFriendRequestsRequest: "LIST_FRIEND_REQUESTS_REQUEST",
    MessageType.ListFriendRequestsResponse: "LIST_FRIEND_REQUESTS_RESPONSE",
    MessageType.FriendAcceptedPush: "FRIEND_ACCEPTED_PUSH",
    MessageType.PrivateMessageRequest: "PRIVATE_MESSAGE_REQUEST",
    MessageType.PrivateMessageResponse: "PRIVATE_MESSAGE_RESPONSE",
    MessageType.PrivateMessagePush: "PRIVATE_MESSAGE_PUSH",
    MessageType.CreateGroupRequest: "CREATE_GROUP_REQUEST",
    MessageType.CreateGroupResponse: "CREATE_GROUP_RESPONSE",
    MessageType.JoinGroupRequest: "JOIN_GROUP_REQUEST",
    MessageType.JoinGroupResponse: "JOIN_GROUP_RESPONSE",
    MessageType.ListGroupsRequest: "LIST_GROUPS_REQUEST",
    MessageType.ListGroupsResponse: "LIST_GROUPS_RESPONSE",
    MessageType.GroupMessageRequest: "GROUP_MESSAGE_REQUEST",
    MessageType.GroupMessageResponse: "GROUP_MESSAGE_RESPONSE",
    MessageType.GroupMessagePush: "GROUP_MESSAGE_PUSH",
    MessageType.OfflineMessagesRequest: "OFFLINE_MESSAGES_REQUEST",
    MessageType.OfflineMessagesResponse: "OFFLINE_MESSAGES_RESPONSE",
    MessageType.HistoryRequest: "HISTORY_REQUEST",
    MessageType.HistoryResponse: "HISTORY_RESPONSE",
    MessageType.OfflineMessagesAckRequest: "OFFLINE_MESSAGES_ACK_REQUEST",
    MessageType.OfflineMessagesAckResponse: "OFFLINE_MESSAGES_ACK_RESPONSE",
    MessageType.DeliveryAckRequest: "DELIVERY_ACK_REQUEST",
    MessageType.DeliveryAckResponse: "DELIVERY_ACK_RESPONSE",
    MessageType.DeliveryReceiptPush: "DELIVERY_RECEIPT_PUSH",
    MessageType.ReadAckRequest: "READ_ACK_REQUEST",
    MessageType.ReadAckResponse: "READ_ACK_RESPONSE",
    MessageType.ReadReceiptPush: "READ_RECEIPT_PUSH",
    MessageType.ErrorResponse: "ERROR_RESPONSE",
}

_REQUEST_TYPES = {
    MessageType.HeartbeatRequest,
    MessageType.RegisterRequest,
    MessageType.LoginRequest,
    MessageType.LogoutRequest,
    MessageType.AddFriendRequest,
    MessageType.ListFriendsRequest,
    MessageType.AcceptFriendRequest,
    MessageType.RejectFriendRequest,
    MessageType.ListFriendRequestsRequest,
    MessageType.PrivateMessageRequest,
    MessageType.CreateGroupRequest,
    MessageType.JoinGroupRequest,
    MessageType.ListGroupsRequest,
    MessageType.GroupMessageRequest,
    MessageType.OfflineMessagesRequest,
    MessageType.HistoryRequest,
    MessageType.OfflineMessagesAckRequest,
    MessageType.DeliveryAckRequest,
    MessageType.ReadAckRequest,
}

_RESPONSE_TYPES = {
    MessageType.HeartbeatResponse,
    MessageType.RegisterResponse,
    MessageType.LoginResponse,
    MessageType.LogoutResponse,
    MessageType.AddFriendResponse,
    MessageType.ListFriendsResponse,
    MessageType.AcceptFriendResponse,
    MessageType.RejectFriendResponse,
    MessageType.ListFriendRequestsResponse,
    MessageType.PrivateMessageResponse,
    MessageType.CreateGroupResponse,
    MessageType.JoinGroupResponse,
    MessageType.ListGroupsResponse,
    MessageType.GroupMessageResponse,
    MessageType.OfflineMessagesResponse,
    MessageType.HistoryResponse,
    MessageType.OfflineMessagesAckResponse,
    MessageType.DeliveryAckResponse,
    MessageType.ReadAckResponse,
    MessageType.ErrorResponse,
}

_PUSH_TYPES = {
    MessageType.PrivateMessagePush,
    MessageType.FriendAcceptedPush,
    MessageType.DeliveryReceiptPush,
    MessageType.ReadReceiptPush,
    MessageType.GroupMessagePush,
}

_TLV_NAMES: dict[TlvType, str] = {
    TlvType.Unknown: "UNKNOWN",
    TlvType.Username: "USERNAME",
    TlvType.Password: "PASSWORD",
    TlvType.UserId: "USER_ID",
    TlvType.Nickname: "NICKNAME",
    TlvType.Token: "TOKEN",
    TlvType.SessionId: "SESSION_ID",
    TlvType.FriendId: "FRIEND_ID",
    TlvType.TargetUserId: "TARGET_USER_ID",
    TlvType.OnlineStatus: "ONLINE_STATUS",
    TlvType.FriendRequestStatus: "FRIEND_REQUEST_STATUS",
    TlvType.GroupId: "GROUP_ID",
    TlvType.GroupName: "GROUP_NAME",
    TlvType.ConversationType: "CONVERSATION_TYPE",
    TlvType.ConversationId: "CONVERSATION_ID",
    TlvType.MessageId: "MESSAGE_ID",
    TlvType.MessageText: "MESSAGE_TEXT",
    TlvType.SenderId: "SENDER_ID",
    TlvType.ReceiverId: "RECEIVER_ID",
    TlvType.TimestampMs: "TIMESTAMP_MS",
    TlvType.Offset: "OFFSET",
    TlvType.Limit: "LIMIT",
    TlvType.UnreadCount: "UNREAD_COUNT",
    TlvType.DeliveryStatus: "DELIVERY_STATUS",
    TlvType.ClientMessageId: "CLIENT_MESSAGE_ID",
    TlvType.ErrorCode: "ERROR_CODE",
    TlvType.ErrorMessage: "ERROR_MESSAGE",
}


def _known_message_type(value: MessageTypeValue) -> MessageType | None:
    try:
        return MessageType(int(value))
    except ValueError:
        return None


def _known_tlv_type(value: TlvTypeValue) -> TlvType | None:
    try:
        return TlvType(int(value))
    except ValueError:
        return None


def _require_uint(value: int, max_value: int, field_name: str) -> int:
    if value < 0 or value > max_value:
        raise ProtocolError(
            ProtocolErrorCode.InvalidArgument,
            f"{field_name} is out of unsigned integer range",
        )
    return value


def _as_bytes(data: BytesLike | None, field_name: str) -> bytes:
    if data is None:
        raise ProtocolError(ProtocolErrorCode.InvalidArgument, f"{field_name} is null")
    return bytes(data)


def validate_header(header: PacketHeader) -> None:
    if header.magic != PACKET_MAGIC:
        raise ProtocolError(ProtocolErrorCode.ParseError, "invalid packet magic")
    if header.version != PACKET_VERSION:
        raise ProtocolError(ProtocolErrorCode.ParseError, "unsupported packet version")
    if header.flags != PACKET_FLAGS_NONE:
        raise ProtocolError(ProtocolErrorCode.ParseError, "unsupported packet flags")
    if header.body_len > MAX_PACKET_BODY_LENGTH:
        raise ProtocolError(ProtocolErrorCode.ParseError, "packet body is too large")
    _require_uint(int(header.msg_type), 0xFFFF, "packet message type")
    _require_uint(header.seq_id, 0xFFFFFFFFFFFFFFFF, "packet seq_id")
    _require_uint(header.body_len, 0xFFFFFFFF, "packet body_len")


def encode_packet(packet: Packet) -> bytes:
    body = _as_bytes(packet.body, "packet body")
    if len(body) > MAX_PACKET_BODY_LENGTH:
        raise ProtocolError(ProtocolErrorCode.InvalidArgument, "packet body is too large")

    header = PacketHeader(
        magic=packet.header.magic,
        version=packet.header.version,
        flags=packet.header.flags,
        msg_type=packet.header.msg_type,
        seq_id=packet.header.seq_id,
        body_len=len(body),
    )
    validate_header(header)
    return _PACKET_HEADER.pack(
        header.magic,
        header.version,
        header.flags,
        int(header.msg_type),
        header.seq_id,
        header.body_len,
    ) + body


def parse_header(data: BytesLike | None) -> PacketHeader:
    header_data = _as_bytes(data, "packet header data")
    if len(header_data) < PACKET_HEADER_SIZE:
        raise ProtocolError(ProtocolErrorCode.ParseError, "packet header is incomplete")

    magic, version, flags, raw_msg_type, seq_id, body_len = _PACKET_HEADER.unpack(
        header_data[:PACKET_HEADER_SIZE]
    )
    known_msg_type = _known_message_type(raw_msg_type)
    msg_type = known_msg_type if known_msg_type is not None else raw_msg_type
    header = PacketHeader(
        magic=magic,
        version=version,
        flags=flags,
        msg_type=msg_type,
        seq_id=seq_id,
        body_len=body_len,
    )
    validate_header(header)
    return header


def message_type_name(value: MessageTypeValue) -> str:
    known = _known_message_type(value)
    if known is None:
        return "UNKNOWN"
    return _MESSAGE_NAMES[known]


def is_request_type(value: MessageTypeValue) -> bool:
    known = _known_message_type(value)
    return known in _REQUEST_TYPES if known is not None else False


def is_response_type(value: MessageTypeValue) -> bool:
    known = _known_message_type(value)
    return known in _RESPONSE_TYPES if known is not None else False


def is_push_type(value: MessageTypeValue) -> bool:
    known = _known_message_type(value)
    return known in _PUSH_TYPES if known is not None else False


def tlv_type_name(value: TlvTypeValue) -> str:
    known = _known_tlv_type(value)
    if known is None:
        return "UNKNOWN"
    return _TLV_NAMES[known]


def _append_value(tlv_type: TlvTypeValue, value: bytes, output: bytearray | None = None) -> bytes:
    raw_type = int(tlv_type)
    if raw_type == int(TlvType.Unknown):
        raise ProtocolError(ProtocolErrorCode.InvalidArgument, "cannot encode unknown tlv type")
    _require_uint(raw_type, 0xFFFF, "tlv type")
    if len(value) > MAX_TLV_VALUE_LENGTH:
        raise ProtocolError(ProtocolErrorCode.InvalidArgument, "tlv value is too large")

    encoded = _TLV_HEADER.pack(raw_type, len(value)) + value
    if output is not None:
        output.extend(encoded)
    return encoded


def append_string(
    tlv_type: TlvTypeValue, value: str, output: bytearray | None = None
) -> bytes:
    return _append_value(tlv_type, value.encode("utf-8"), output)


def append_uint64(
    tlv_type: TlvTypeValue, value: int, output: bytearray | None = None
) -> bytes:
    _require_uint(value, 0xFFFFFFFFFFFFFFFF, "uint64 tlv value")
    return _append_value(tlv_type, struct.pack("!Q", value), output)


def parse_tlv_map(data: BytesLike | None) -> TlvMap:
    body = _as_bytes(data, "tlv body data")
    output: TlvMap = {}
    offset = 0
    while offset < len(body):
        if len(body) - offset < TLV_HEADER_SIZE:
            raise ProtocolError(ProtocolErrorCode.ParseError, "tlv header is incomplete")

        raw_type, value_len = _TLV_HEADER.unpack(body[offset : offset + TLV_HEADER_SIZE])
        offset += TLV_HEADER_SIZE

        if value_len > MAX_TLV_VALUE_LENGTH:
            raise ProtocolError(ProtocolErrorCode.ParseError, "tlv value is too large")
        if value_len > len(body) - offset:
            raise ProtocolError(ProtocolErrorCode.ParseError, "tlv length exceeds body size")

        output.setdefault(raw_type, []).append(body[offset : offset + value_len])
        offset += value_len

    return output


def _find_values(tlv_map: TlvMap, tlv_type: TlvTypeValue, field_kind: str) -> list[bytes]:
    values = tlv_map.get(int(tlv_type))
    if not values:
        raise ProtocolError(ProtocolErrorCode.NotFound, f"missing required {field_kind} field")
    return values


def _read_uint64(value: bytes) -> int:
    return int(struct.unpack("!Q", value)[0])


def get_string(tlv_map: TlvMap, tlv_type: TlvTypeValue) -> str:
    value = _find_values(tlv_map, tlv_type, "string")[0]
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError(ProtocolErrorCode.ParseError, "string tlv field is not utf-8") from exc


def get_uint64(tlv_map: TlvMap, tlv_type: TlvTypeValue) -> int:
    value = _find_values(tlv_map, tlv_type, "uint64")[0]
    if len(value) != 8:
        raise ProtocolError(ProtocolErrorCode.ParseError, "uint64 tlv field must be 8 bytes")
    return _read_uint64(value)


def get_repeated_string(tlv_map: TlvMap, tlv_type: TlvTypeValue) -> list[str]:
    values = _find_values(tlv_map, tlv_type, "repeated string")
    try:
        return [value.decode("utf-8") for value in values]
    except UnicodeDecodeError as exc:
        raise ProtocolError(ProtocolErrorCode.ParseError, "string tlv field is not utf-8") from exc


def get_repeated_uint64(tlv_map: TlvMap, tlv_type: TlvTypeValue) -> list[int]:
    result: list[int] = []
    for value in _find_values(tlv_map, tlv_type, "repeated uint64"):
        if len(value) != 8:
            raise ProtocolError(ProtocolErrorCode.ParseError, "uint64 tlv field must be 8 bytes")
        result.append(_read_uint64(value))
    return result
