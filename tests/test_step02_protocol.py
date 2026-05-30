import struct

import pytest

from bot_client.protocol.codec import (
    MAX_PACKET_BODY_LENGTH,
    PACKET_FLAGS_NONE,
    PACKET_HEADER_SIZE,
    PACKET_MAGIC,
    PACKET_VERSION,
    MessageType,
    Packet,
    PacketHeader,
    ProtocolError,
    ProtocolErrorCode,
    TlvType,
    append_string,
    append_uint64,
    encode_packet,
    get_repeated_string,
    get_repeated_uint64,
    get_string,
    get_uint64,
    is_push_type,
    is_request_type,
    is_response_type,
    message_type_name,
    parse_header,
    parse_tlv_map,
    tlv_type_name,
)
from bot_client.protocol.frame_decoder import FrameDecoder


def _packet(
    msg_type: MessageType = MessageType.PrivateMessageRequest,
    body: bytes = b"hello",
) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=42), body=body)


def test_message_type_values_names_and_categories_match_liteim_v1() -> None:
    expected_values = {
        MessageType.HeartbeatRequest: (1, "HEARTBEAT_REQUEST", "request"),
        MessageType.HeartbeatResponse: (2, "HEARTBEAT_RESPONSE", "response"),
        MessageType.RegisterRequest: (100, "REGISTER_REQUEST", "request"),
        MessageType.RegisterResponse: (101, "REGISTER_RESPONSE", "response"),
        MessageType.LoginRequest: (102, "LOGIN_REQUEST", "request"),
        MessageType.LoginResponse: (103, "LOGIN_RESPONSE", "response"),
        MessageType.LogoutRequest: (104, "LOGOUT_REQUEST", "request"),
        MessageType.LogoutResponse: (105, "LOGOUT_RESPONSE", "response"),
        MessageType.AddFriendRequest: (200, "ADD_FRIEND_REQUEST", "request"),
        MessageType.AddFriendResponse: (201, "ADD_FRIEND_RESPONSE", "response"),
        MessageType.ListFriendsRequest: (202, "LIST_FRIENDS_REQUEST", "request"),
        MessageType.ListFriendsResponse: (203, "LIST_FRIENDS_RESPONSE", "response"),
        MessageType.AcceptFriendRequest: (204, "ACCEPT_FRIEND_REQUEST", "request"),
        MessageType.AcceptFriendResponse: (205, "ACCEPT_FRIEND_RESPONSE", "response"),
        MessageType.RejectFriendRequest: (206, "REJECT_FRIEND_REQUEST", "request"),
        MessageType.RejectFriendResponse: (207, "REJECT_FRIEND_RESPONSE", "response"),
        MessageType.ListFriendRequestsRequest: (208, "LIST_FRIEND_REQUESTS_REQUEST", "request"),
        MessageType.ListFriendRequestsResponse: (209, "LIST_FRIEND_REQUESTS_RESPONSE", "response"),
        MessageType.FriendAcceptedPush: (210, "FRIEND_ACCEPTED_PUSH", "push"),
        MessageType.PrivateMessageRequest: (300, "PRIVATE_MESSAGE_REQUEST", "request"),
        MessageType.PrivateMessageResponse: (301, "PRIVATE_MESSAGE_RESPONSE", "response"),
        MessageType.PrivateMessagePush: (302, "PRIVATE_MESSAGE_PUSH", "push"),
        MessageType.CreateGroupRequest: (400, "CREATE_GROUP_REQUEST", "request"),
        MessageType.CreateGroupResponse: (401, "CREATE_GROUP_RESPONSE", "response"),
        MessageType.JoinGroupRequest: (402, "JOIN_GROUP_REQUEST", "request"),
        MessageType.JoinGroupResponse: (403, "JOIN_GROUP_RESPONSE", "response"),
        MessageType.ListGroupsRequest: (404, "LIST_GROUPS_REQUEST", "request"),
        MessageType.ListGroupsResponse: (405, "LIST_GROUPS_RESPONSE", "response"),
        MessageType.GroupMessageRequest: (406, "GROUP_MESSAGE_REQUEST", "request"),
        MessageType.GroupMessageResponse: (407, "GROUP_MESSAGE_RESPONSE", "response"),
        MessageType.GroupMessagePush: (408, "GROUP_MESSAGE_PUSH", "push"),
        MessageType.OfflineMessagesRequest: (500, "OFFLINE_MESSAGES_REQUEST", "request"),
        MessageType.OfflineMessagesResponse: (501, "OFFLINE_MESSAGES_RESPONSE", "response"),
        MessageType.HistoryRequest: (502, "HISTORY_REQUEST", "request"),
        MessageType.HistoryResponse: (503, "HISTORY_RESPONSE", "response"),
        MessageType.OfflineMessagesAckRequest: (504, "OFFLINE_MESSAGES_ACK_REQUEST", "request"),
        MessageType.OfflineMessagesAckResponse: (505, "OFFLINE_MESSAGES_ACK_RESPONSE", "response"),
        MessageType.DeliveryAckRequest: (506, "DELIVERY_ACK_REQUEST", "request"),
        MessageType.DeliveryAckResponse: (507, "DELIVERY_ACK_RESPONSE", "response"),
        MessageType.DeliveryReceiptPush: (508, "DELIVERY_RECEIPT_PUSH", "push"),
        MessageType.ReadAckRequest: (509, "READ_ACK_REQUEST", "request"),
        MessageType.ReadAckResponse: (510, "READ_ACK_RESPONSE", "response"),
        MessageType.ReadReceiptPush: (511, "READ_RECEIPT_PUSH", "push"),
        MessageType.ErrorResponse: (900, "ERROR_RESPONSE", "response"),
    }

    for msg_type, (raw_value, name, category) in expected_values.items():
        assert int(msg_type) == raw_value
        assert message_type_name(msg_type) == name
        assert is_request_type(msg_type) is (category == "request")
        assert is_response_type(msg_type) is (category == "response")
        assert is_push_type(msg_type) is (category == "push")

    assert message_type_name(MessageType.Unknown) == "UNKNOWN"
    assert message_type_name(999) == "UNKNOWN"
    assert not is_request_type(999)
    assert not is_response_type(999)
    assert not is_push_type(999)


def test_tlv_type_values_and_names_match_liteim_v1() -> None:
    expected_values = {
        TlvType.Username: (1, "USERNAME"),
        TlvType.Password: (2, "PASSWORD"),
        TlvType.UserId: (3, "USER_ID"),
        TlvType.Nickname: (4, "NICKNAME"),
        TlvType.Token: (5, "TOKEN"),
        TlvType.SessionId: (6, "SESSION_ID"),
        TlvType.FriendId: (20, "FRIEND_ID"),
        TlvType.TargetUserId: (21, "TARGET_USER_ID"),
        TlvType.OnlineStatus: (22, "ONLINE_STATUS"),
        TlvType.FriendRequestStatus: (23, "FRIEND_REQUEST_STATUS"),
        TlvType.GroupId: (30, "GROUP_ID"),
        TlvType.GroupName: (31, "GROUP_NAME"),
        TlvType.ConversationType: (40, "CONVERSATION_TYPE"),
        TlvType.ConversationId: (41, "CONVERSATION_ID"),
        TlvType.MessageId: (42, "MESSAGE_ID"),
        TlvType.MessageText: (43, "MESSAGE_TEXT"),
        TlvType.SenderId: (44, "SENDER_ID"),
        TlvType.ReceiverId: (45, "RECEIVER_ID"),
        TlvType.TimestampMs: (46, "TIMESTAMP_MS"),
        TlvType.Offset: (47, "OFFSET"),
        TlvType.Limit: (48, "LIMIT"),
        TlvType.UnreadCount: (49, "UNREAD_COUNT"),
        TlvType.DeliveryStatus: (50, "DELIVERY_STATUS"),
        TlvType.ClientMessageId: (51, "CLIENT_MESSAGE_ID"),
        TlvType.ErrorCode: (90, "ERROR_CODE"),
        TlvType.ErrorMessage: (91, "ERROR_MESSAGE"),
    }

    for tlv_type, (raw_value, name) in expected_values.items():
        assert int(tlv_type) == raw_value
        assert tlv_type_name(tlv_type) == name

    assert tlv_type_name(TlvType.Unknown) == "UNKNOWN"
    assert tlv_type_name(999) == "UNKNOWN"


def test_packet_encode_parse_header_and_body_round_trip() -> None:
    body = "你好，LiteIM 👋".encode()
    encoded = encode_packet(_packet(body=body))

    assert len(encoded) == PACKET_HEADER_SIZE + len(body)
    header = parse_header(encoded)

    assert header.magic == PACKET_MAGIC
    assert header.version == PACKET_VERSION
    assert header.flags == PACKET_FLAGS_NONE
    assert header.msg_type == MessageType.PrivateMessageRequest
    assert header.seq_id == 42
    assert header.body_len == len(body)
    assert encoded[PACKET_HEADER_SIZE:] == body


def test_packet_header_uses_liteim_network_byte_order() -> None:
    packet = Packet(
        header=PacketHeader(
            msg_type=MessageType.GroupMessageRequest,
            seq_id=0x0102030405060708,
        ),
        body=b"x",
    )

    encoded = encode_packet(packet)

    assert encoded[:PACKET_HEADER_SIZE] == bytes(
        [
            0x4C,
            0x49,
            0x4D,
            0x31,
            0x01,
            0x00,
            0x01,
            0x96,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
            0x00,
            0x00,
            0x00,
            0x01,
        ]
    )


def test_packet_validation_errors_match_liteim_error_categories() -> None:
    with pytest.raises(ProtocolError) as invalid_magic:
        parse_header(b"\x00" + encode_packet(_packet())[1:PACKET_HEADER_SIZE])
    assert invalid_magic.value.code == ProtocolErrorCode.ParseError

    with pytest.raises(ProtocolError) as incomplete:
        parse_header(b"\x00" * (PACKET_HEADER_SIZE - 1))
    assert incomplete.value.code == ProtocolErrorCode.ParseError

    oversized_header = struct.pack(
        "!IBBHQI",
        PACKET_MAGIC,
        PACKET_VERSION,
        PACKET_FLAGS_NONE,
        int(MessageType.PrivateMessageRequest),
        1,
        MAX_PACKET_BODY_LENGTH + 1,
    )
    with pytest.raises(ProtocolError) as oversized:
        parse_header(oversized_header)
    assert oversized.value.code == ProtocolErrorCode.ParseError

    with pytest.raises(ProtocolError) as encode_oversized:
        encode_packet(Packet(header=PacketHeader(), body=b"x" * (MAX_PACKET_BODY_LENGTH + 1)))
    assert encode_oversized.value.code == ProtocolErrorCode.InvalidArgument


def test_parse_header_preserves_unknown_message_type_values() -> None:
    unknown_header = struct.pack(
        "!IBBHQI",
        PACKET_MAGIC,
        PACKET_VERSION,
        PACKET_FLAGS_NONE,
        999,
        7,
        0,
    )
    explicit_unknown_header = struct.pack(
        "!IBBHQI",
        PACKET_MAGIC,
        PACKET_VERSION,
        PACKET_FLAGS_NONE,
        int(MessageType.Unknown),
        8,
        0,
    )

    assert parse_header(unknown_header).msg_type == 999
    assert parse_header(explicit_unknown_header).msg_type == MessageType.Unknown


def test_tlv_string_uint64_repeated_and_utf8_values_round_trip() -> None:
    body = bytearray()
    append_string(TlvType.Username, "first", body)
    append_string(TlvType.Username, "second", body)
    append_uint64(TlvType.UserId, 1001, body)
    append_uint64(TlvType.UserId, 1002, body)
    append_string(TlvType.MessageText, "你好 👋", body)

    tlv_map = parse_tlv_map(body)

    assert get_string(tlv_map, TlvType.Username) == "first"
    assert get_uint64(tlv_map, TlvType.UserId) == 1001
    assert get_string(tlv_map, TlvType.MessageText) == "你好 👋"
    assert get_repeated_string(tlv_map, TlvType.Username) == ["first", "second"]
    assert get_repeated_uint64(tlv_map, TlvType.UserId) == [1001, 1002]


def test_tlv_helpers_use_network_byte_order_and_reject_invalid_shapes() -> None:
    encoded = append_uint64(TlvType.MessageId, 0x0102030405060708)

    assert encoded == bytes(
        [
            0x00,
            0x2A,
            0x00,
            0x00,
            0x00,
            0x08,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
        ]
    )

    with pytest.raises(ProtocolError) as unknown_encode:
        append_string(TlvType.Unknown, "bad")
    assert unknown_encode.value.code == ProtocolErrorCode.InvalidArgument

    with pytest.raises(ProtocolError) as short_header:
        parse_tlv_map(b"\x00" * 5)
    assert short_header.value.code == ProtocolErrorCode.ParseError

    broken_length = bytearray(append_string(TlvType.Username, "a"))
    broken_length[5] = 2
    with pytest.raises(ProtocolError) as broken:
        parse_tlv_map(broken_length)
    assert broken.value.code == ProtocolErrorCode.ParseError

    bad_uint64 = parse_tlv_map(append_string(TlvType.UserId, "bad"))
    with pytest.raises(ProtocolError) as wrong_size:
        get_uint64(bad_uint64, TlvType.UserId)
    assert wrong_size.value.code == ProtocolErrorCode.ParseError


def test_parse_tlv_map_preserves_unknown_raw_tlv_ids() -> None:
    body = struct.pack("!HI", 999, 3) + b"raw"

    tlv_map = parse_tlv_map(body)

    assert tlv_map[999] == [b"raw"]
    assert tlv_type_name(999) == "UNKNOWN"


def test_frame_decoder_handles_complete_split_and_sticky_packets() -> None:
    decoder = FrameDecoder()
    first = encode_packet(_packet(MessageType.LoginRequest, b"alice"))
    second = encode_packet(
        Packet(
            header=PacketHeader(msg_type=MessageType.PrivateMessageRequest, seq_id=43),
            body=b"hi",
        )
    )

    assert decoder.feed(first[:5]) == []
    assert decoder.buffered_bytes == 5

    packets = decoder.feed(first[5:] + second)

    assert len(packets) == 2
    assert packets[0].header.msg_type == MessageType.LoginRequest
    assert packets[0].body == b"alice"
    assert packets[1].header.seq_id == 43
    assert packets[1].body == b"hi"
    assert decoder.buffered_bytes == 0


def test_frame_decoder_error_state_and_reset_match_liteim_behavior() -> None:
    decoder = FrameDecoder()
    invalid = bytearray(encode_packet(_packet()))
    invalid[0] = 0

    with pytest.raises(ProtocolError) as first_error:
        decoder.feed(invalid)
    assert first_error.value.code == ProtocolErrorCode.ParseError
    assert decoder.has_error

    valid = encode_packet(Packet(header=PacketHeader(seq_id=99), body=b"ok"))
    with pytest.raises(ProtocolError) as rejected:
        decoder.feed(valid)
    assert rejected.value.code == ProtocolErrorCode.ParseError

    decoder.reset()
    packets = decoder.feed(valid)

    assert len(packets) == 1
    assert packets[0].header.seq_id == 99
    assert packets[0].body == b"ok"
    assert not decoder.has_error


def test_frame_decoder_waits_for_declared_max_body_without_error() -> None:
    decoder = FrameDecoder()
    header_only = struct.pack(
        "!IBBHQI",
        PACKET_MAGIC,
        PACKET_VERSION,
        PACKET_FLAGS_NONE,
        int(MessageType.PrivateMessageRequest),
        7,
        MAX_PACKET_BODY_LENGTH,
    )

    assert decoder.feed(header_only) == []
    assert not decoder.has_error
    assert decoder.buffered_bytes == PACKET_HEADER_SIZE
