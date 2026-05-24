from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    PacketHeader,
    TlvType,
    append_string,
    append_uint64,
    parse_tlv_map,
)


def _packet(msg_type: MessageType, body: bytes, seq_id: int = 99) -> Packet:
    return Packet(header=PacketHeader(msg_type=msg_type, seq_id=seq_id), body=body)


def _friend_fields(
    *,
    user_id: int,
    username: str,
    nickname: str,
    online: bool = False,
    request_status: int | None = None,
) -> bytes:
    body = bytearray()
    append_uint64(TlvType.FriendId, user_id, body)
    append_string(TlvType.Username, username, body)
    append_string(TlvType.Nickname, nickname, body)
    append_uint64(TlvType.OnlineStatus, 1 if online else 0, body)
    if request_status is not None:
        append_uint64(TlvType.FriendRequestStatus, request_status, body)
    return bytes(body)


def _private_message_body(*, message_id: int, sender_id: int) -> bytes:
    body = bytearray()
    append_uint64(TlvType.MessageId, message_id, body)
    append_uint64(TlvType.ConversationType, 1, body)
    append_uint64(TlvType.ConversationId, 10011002, body)
    append_uint64(TlvType.SenderId, sender_id, body)
    append_uint64(TlvType.ReceiverId, 1001, body)
    append_string(TlvType.MessageText, "hello agent", body)
    append_uint64(TlvType.TimestampMs, 1_700_000_001_000, body)
    return bytes(body)


@dataclass
class FakeFriendPolicyClient:
    friends: list[object] = field(default_factory=list)
    requests: list[object] = field(default_factory=list)
    listed_friends: int = 0
    listed_requests: int = 0
    accepted: list[int] = field(default_factory=list)
    rejected: list[int] = field(default_factory=list)

    async def list_friends(self) -> list[object]:
        self.listed_friends += 1
        return self.friends

    async def list_friend_requests(self) -> list[object]:
        self.listed_requests += 1
        return self.requests

    async def accept_friend_request(self, requester_id: int) -> object:
        self.accepted.append(requester_id)
        for request in self.requests:
            if request.profile.user_id == requester_id:  # type: ignore[attr-defined]
                return request
        raise AssertionError(f"unexpected accept {requester_id}")

    async def reject_friend_request(self, requester_id: int) -> object:
        self.rejected.append(requester_id)
        for request in self.requests:
            if request.profile.user_id == requester_id:  # type: ignore[attr-defined]
                return request
        raise AssertionError(f"unexpected reject {requester_id}")


@dataclass
class FakeReliabilityClient:
    identity: object | None = None
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


def test_friend_protocol_parser_and_builders_match_liteim_fields() -> None:
    from bot_client.protocol_builders import make_friend_action_body
    from bot_client.protocol_parser import (
        FRIEND_REQUEST_ACCEPTED,
        FRIEND_REQUEST_PENDING,
        parse_friend_action,
        parse_friend_requests,
        parse_friends,
    )

    friends_packet = _packet(
        MessageType.ListFriendsResponse,
        _friend_fields(user_id=1002, username="alice", nickname="Alice", online=True),
    )
    friends = parse_friends(friends_packet)

    assert friends[0].user_id == 1002
    assert friends[0].username == "alice"
    assert friends[0].nickname == "Alice"
    assert friends[0].online is True

    requests_packet = _packet(
        MessageType.ListFriendRequestsResponse,
        _friend_fields(
            user_id=1003,
            username="bob",
            nickname="Bob",
            online=False,
            request_status=FRIEND_REQUEST_PENDING,
        )
        + _friend_fields(
            user_id=1004,
            username="carol",
            nickname="Carol",
            online=True,
            request_status=FRIEND_REQUEST_PENDING,
        ),
    )
    requests = parse_friend_requests(requests_packet)

    assert [request.profile.user_id for request in requests] == [1003, 1004]
    assert [request.status for request in requests] == [
        FRIEND_REQUEST_PENDING,
        FRIEND_REQUEST_PENDING,
    ]

    accepted_packet = _packet(
        MessageType.AcceptFriendResponse,
        _friend_fields(
            user_id=1003,
            username="bob",
            nickname="Bob",
            online=False,
            request_status=FRIEND_REQUEST_ACCEPTED,
        ),
    )
    accepted = parse_friend_action(accepted_packet)

    assert accepted.profile.user_id == 1003
    assert accepted.status == FRIEND_REQUEST_ACCEPTED

    action_body = parse_tlv_map(make_friend_action_body(1003))
    assert int.from_bytes(action_body[int(TlvType.TargetUserId)][0], "big") == 1003


def test_friend_policy_settings_parse_allowlists() -> None:
    from bot_client.config import BotClientSettings
    from bot_client.friend_policy import FriendAccessPolicy
    from bot_client.protocol_parser import FriendProfile

    settings = BotClientSettings(
        _env_file=None,
        allowed_user_ids="1002, 1003",
        allowed_usernames="alice,bob",
        auto_accept_friend_requests=False,
        reject_non_allowlisted_friend_requests=False,
    )
    policy = FriendAccessPolicy.from_settings(settings)

    assert policy.allowed_user_ids == {1002, 1003}
    assert policy.allowed_usernames == {"alice", "bob"}
    assert policy.auto_accept_friend_requests is False
    assert policy.reject_non_allowlisted_friend_requests is False
    assert policy.allows(FriendProfile(1002, "mallory", "Mallory", False))
    assert policy.allows(FriendProfile(9999, "alice", "Alice", False))
    assert not policy.allows(FriendProfile(9999, "mallory", "Mallory", False))


async def test_friend_policy_sync_accepts_allowlisted_and_rejects_non_allowlisted(
    tmp_path: Path,
) -> None:
    from bot_client.friend_policy import FriendAccessPolicy, FriendPolicyHandler
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import (
        FRIEND_REQUEST_PENDING,
        FriendProfile,
        FriendRequest,
    )

    alice = FriendRequest(
        profile=FriendProfile(1002, "alice", "Alice", True),
        status=FRIEND_REQUEST_PENDING,
    )
    mallory = FriendRequest(
        profile=FriendProfile(1003, "mallory", "Mallory", False),
        status=FRIEND_REQUEST_PENDING,
    )
    existing_friend = FriendProfile(1004, "diana", "Diana", False)
    client = FakeFriendPolicyClient(friends=[existing_friend], requests=[alice, mallory])
    state = JsonMessageState(tmp_path / "state.json")
    handler = FriendPolicyHandler(
        client=client,
        state=state,
        policy=FriendAccessPolicy(
            allowed_user_ids={1002},
            allowed_usernames=set(),
            auto_accept_friend_requests=True,
            reject_non_allowlisted_friend_requests=True,
        ),
    )

    await handler.sync_after_login()

    assert client.listed_friends == 1
    assert client.listed_requests == 1
    assert client.accepted == [1002]
    assert client.rejected == [1003]
    assert state.is_friend(1002)
    assert not state.is_friend(1003)
    assert state.is_friend(1004)
    assert [event.action for event in state.friend_policy_events] == [
        "friend_list_synced",
        "accepted",
        "rejected",
    ]


async def test_friend_policy_can_leave_non_allowlisted_requests_pending(
    tmp_path: Path,
) -> None:
    from bot_client.friend_policy import FriendAccessPolicy, FriendPolicyHandler
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import (
        FRIEND_REQUEST_PENDING,
        FriendProfile,
        FriendRequest,
    )

    request = FriendRequest(
        profile=FriendProfile(1003, "mallory", "Mallory", False),
        status=FRIEND_REQUEST_PENDING,
    )
    client = FakeFriendPolicyClient(requests=[request])
    state = JsonMessageState(tmp_path / "state.json")
    handler = FriendPolicyHandler(
        client=client,
        state=state,
        policy=FriendAccessPolicy(
            allowed_user_ids=set(),
            allowed_usernames=set(),
            auto_accept_friend_requests=True,
            reject_non_allowlisted_friend_requests=False,
        ),
    )

    await handler.sync_after_login()

    assert client.accepted == []
    assert client.rejected == []
    assert not state.is_friend(1003)
    assert state.friend_policy_events[-1].action == "left_pending"


async def test_friend_accepted_push_updates_friend_state(tmp_path: Path) -> None:
    from bot_client.friend_policy import FriendAccessPolicy, FriendPolicyHandler
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import FRIEND_REQUEST_ACCEPTED

    state = JsonMessageState(tmp_path / "state.json")
    handler = FriendPolicyHandler(
        client=FakeFriendPolicyClient(),
        state=state,
        policy=FriendAccessPolicy(),
    )
    packet = _packet(
        MessageType.FriendAcceptedPush,
        _friend_fields(
            user_id=1002,
            username="alice",
            nickname="Alice",
            online=True,
            request_status=FRIEND_REQUEST_ACCEPTED,
        ),
    )

    await handler.handle_packet(packet)

    assert state.is_friend(1002)
    assert state.friends[0].username == "alice"
    assert state.friend_policy_events[-1].action == "accepted_push"


async def test_message_handler_blocks_private_messages_from_non_friends(
    tmp_path: Path,
) -> None:
    from bot_client.message_handler import BotMessageHandler, MessageProcessingResult
    from bot_client.message_state import JsonMessageState
    from bot_client.protocol_parser import FriendProfile

    state = JsonMessageState(tmp_path / "state.json")
    state.replace_friends([FriendProfile(1002, "alice", "Alice", True)])
    client = FakeReliabilityClient()
    processed: list[int] = []

    async def processor(message: object) -> MessageProcessingResult:
        processed.append(message.message_id)  # type: ignore[attr-defined]
        return MessageProcessingResult(reply_text="should not send")

    handler = BotMessageHandler(
        client=client,
        state=state,
        processor=processor,
        require_friendship=True,
    )

    await handler.handle_packet(
        _packet(
            MessageType.PrivateMessagePush,
            _private_message_body(message_id=5001, sender_id=9999),
        )
    )

    assert client.delivery_acks == [5001]
    assert processed == []
    assert client.read_acks == []
    assert client.sent_private_messages == []
    assert state.friend_policy_events[-1].action == "blocked_non_friend_message"


async def test_bot_client_friend_methods_send_expected_requests() -> None:
    from bot_client.bot_client import BotClient
    from bot_client.config import BotClientSettings
    from bot_client.protocol_parser import FRIEND_REQUEST_ACCEPTED, FRIEND_REQUEST_REJECTED

    settings = BotClientSettings(_env_file=None)
    client = BotClient(settings)
    sent: list[tuple[MessageType, bytes]] = []

    async def fake_request(
        msg_type: MessageType,
        body: bytes = b"",
        timeout: float | None = None,
    ) -> Packet:
        del timeout
        sent.append((msg_type, body))
        if msg_type == MessageType.ListFriendsRequest:
            return _packet(
                MessageType.ListFriendsResponse,
                _friend_fields(user_id=1002, username="alice", nickname="Alice"),
            )
        if msg_type == MessageType.ListFriendRequestsRequest:
            return _packet(
                MessageType.ListFriendRequestsResponse,
                _friend_fields(
                    user_id=1003,
                    username="bob",
                    nickname="Bob",
                    request_status=0,
                ),
            )
        if msg_type == MessageType.AcceptFriendRequest:
            return _packet(
                MessageType.AcceptFriendResponse,
                _friend_fields(
                    user_id=1003,
                    username="bob",
                    nickname="Bob",
                    request_status=FRIEND_REQUEST_ACCEPTED,
                ),
            )
        if msg_type == MessageType.RejectFriendRequest:
            return _packet(
                MessageType.RejectFriendResponse,
                _friend_fields(
                    user_id=1004,
                    username="carol",
                    nickname="Carol",
                    request_status=FRIEND_REQUEST_REJECTED,
                ),
            )
        raise AssertionError(f"unexpected request {msg_type}")

    client.request = fake_request  # type: ignore[method-assign]

    friends = await client.list_friends()
    requests = await client.list_friend_requests()
    accepted = await client.accept_friend_request(1003)
    rejected = await client.reject_friend_request(1004)

    assert [friend.user_id for friend in friends] == [1002]
    assert [request.profile.user_id for request in requests] == [1003]
    assert accepted.status == FRIEND_REQUEST_ACCEPTED
    assert rejected.status == FRIEND_REQUEST_REJECTED
    assert [item[0] for item in sent] == [
        MessageType.ListFriendsRequest,
        MessageType.ListFriendRequestsRequest,
        MessageType.AcceptFriendRequest,
        MessageType.RejectFriendRequest,
    ]
    assert int.from_bytes(parse_tlv_map(sent[2][1])[int(TlvType.TargetUserId)][0], "big") == 1003
    assert int.from_bytes(parse_tlv_map(sent[3][1])[int(TlvType.TargetUserId)][0], "big") == 1004
