from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

from bot_client.config import BotClientSettings
from bot_client.frame_decoder import FrameDecoder
from bot_client.liteim_protocol import (
    MessageType,
    Packet,
    ProtocolError,
    TlvType,
    encode_packet,
    get_string,
    get_uint64,
    is_push_type,
    is_response_type,
    parse_tlv_map,
)
from bot_client.protocol_builders import (
    make_delivery_ack_body,
    make_friend_action_body,
    make_login_body,
    make_offline_ack_body,
    make_offline_request_body,
    make_packet,
    make_private_message_body,
    make_read_ack_body,
    make_register_body,
)
from bot_client.protocol_parser import (
    FriendProfile,
    FriendRequest,
    IncomingMessage,
    parse_friend_action,
    parse_friend_requests,
    parse_friends,
    parse_incoming_message,
    parse_offline_messages,
)


class BotClientState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOGGED_IN = "logged_in"
    CLOSING = "closing"


class BotClientError(Exception):
    """Base exception for BotClient runtime failures."""


class BotClientConnectionError(BotClientError):
    """Raised when the LiteIM TCP connection is unavailable or closes."""


class BotClientTimeoutError(BotClientError):
    """Raised when a pending LiteIM request does not receive its response."""


class BotClientProtocolError(BotClientError):
    """Raised when LiteIM returns an unexpected or malformed packet."""


@dataclass(frozen=True, slots=True)
class BotIdentity:
    user_id: int
    username: str
    nickname: str
    session_id: int


@dataclass(slots=True)
class PendingRequest:
    request_type: MessageType
    future: asyncio.Future[Packet]
    created_at: float


_EXPECTED_RESPONSES: dict[MessageType, MessageType] = {
    MessageType.HeartbeatRequest: MessageType.HeartbeatResponse,
    MessageType.RegisterRequest: MessageType.RegisterResponse,
    MessageType.LoginRequest: MessageType.LoginResponse,
    MessageType.LogoutRequest: MessageType.LogoutResponse,
    MessageType.AddFriendRequest: MessageType.AddFriendResponse,
    MessageType.ListFriendsRequest: MessageType.ListFriendsResponse,
    MessageType.AcceptFriendRequest: MessageType.AcceptFriendResponse,
    MessageType.RejectFriendRequest: MessageType.RejectFriendResponse,
    MessageType.ListFriendRequestsRequest: MessageType.ListFriendRequestsResponse,
    MessageType.PrivateMessageRequest: MessageType.PrivateMessageResponse,
    MessageType.CreateGroupRequest: MessageType.CreateGroupResponse,
    MessageType.JoinGroupRequest: MessageType.JoinGroupResponse,
    MessageType.ListGroupsRequest: MessageType.ListGroupsResponse,
    MessageType.GroupMessageRequest: MessageType.GroupMessageResponse,
    MessageType.OfflineMessagesRequest: MessageType.OfflineMessagesResponse,
    MessageType.HistoryRequest: MessageType.HistoryResponse,
    MessageType.OfflineMessagesAckRequest: MessageType.OfflineMessagesAckResponse,
    MessageType.DeliveryAckRequest: MessageType.DeliveryAckResponse,
    MessageType.ReadAckRequest: MessageType.ReadAckResponse,
}


class BotClient:
    def __init__(self, settings: BotClientSettings) -> None:
        self._settings = settings
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._decoder = FrameDecoder()
        self._seq_id = 1
        self._pending: dict[int, PendingRequest] = {}
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._state = BotClientState.DISCONNECTED
        self._identity: BotIdentity | None = None
        self._closing = False
        self._disconnected_event = asyncio.Event()
        self._disconnected_event.set()
        self.push_queue: asyncio.Queue[Packet] = asyncio.Queue()

    @property
    def state(self) -> BotClientState:
        return self._state

    @property
    def identity(self) -> BotIdentity | None:
        return self._identity

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    @property
    def is_logged_in(self) -> bool:
        return self._identity is not None and self._state == BotClientState.LOGGED_IN

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._state = BotClientState.CONNECTING
        self._closing = False
        self._decoder.reset()
        self._pending.clear()
        self._seq_id = 1
        self._identity = None
        self._disconnected_event.clear()
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self._settings.liteim_host,
                self._settings.liteim_port,
            )
        except OSError as exc:
            self._state = BotClientState.DISCONNECTED
            self._disconnected_event.set()
            raise BotClientConnectionError(str(exc)) from exc

        self._state = BotClientState.CONNECTED
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def register(self) -> BotIdentity:
        body = make_register_body(
            self._settings.bot_username,
            self._settings.bot_password,
            self._settings.bot_nickname,
        )
        packet = await self.request(MessageType.RegisterRequest, body)
        if packet.header.msg_type != MessageType.RegisterResponse:
            raise BotClientProtocolError("unexpected register response")
        fields = parse_tlv_map(packet.body)
        return BotIdentity(
            user_id=get_uint64(fields, TlvType.UserId),
            username=get_string(fields, TlvType.Username),
            nickname=get_string(fields, TlvType.Nickname),
            session_id=0,
        )

    async def login(self) -> BotIdentity:
        body = make_login_body(self._settings.bot_username, self._settings.bot_password)
        packet = await self.request(MessageType.LoginRequest, body)
        if packet.header.msg_type != MessageType.LoginResponse:
            raise BotClientProtocolError("unexpected login response")

        fields = parse_tlv_map(packet.body)
        identity = BotIdentity(
            user_id=get_uint64(fields, TlvType.UserId),
            username=get_string(fields, TlvType.Username),
            nickname=get_string(fields, TlvType.Nickname),
            session_id=get_uint64(fields, TlvType.SessionId),
        )
        self._identity = identity
        self._state = BotClientState.LOGGED_IN
        self._start_heartbeat()
        return identity

    async def logout(self) -> Packet | None:
        if not self.is_connected or self._identity is None:
            self._identity = None
            self._state = (
                BotClientState.CONNECTED if self.is_connected else BotClientState.DISCONNECTED
            )
            return None

        response = await self.request(MessageType.LogoutRequest)
        if response.header.msg_type != MessageType.LogoutResponse:
            raise BotClientProtocolError("unexpected logout response")
        self._identity = None
        self._state = BotClientState.CONNECTED
        await self._cancel_heartbeat_task()
        return response

    async def list_friends(self) -> list[FriendProfile]:
        response = await self.request(MessageType.ListFriendsRequest)
        if response.header.msg_type != MessageType.ListFriendsResponse:
            raise BotClientProtocolError("unexpected list friends response")
        return parse_friends(response)

    async def list_friend_requests(self) -> list[FriendRequest]:
        response = await self.request(MessageType.ListFriendRequestsRequest)
        if response.header.msg_type != MessageType.ListFriendRequestsResponse:
            raise BotClientProtocolError("unexpected list friend requests response")
        return parse_friend_requests(response)

    async def accept_friend_request(self, requester_id: int) -> FriendRequest:
        response = await self.request(
            MessageType.AcceptFriendRequest,
            make_friend_action_body(requester_id),
        )
        if response.header.msg_type != MessageType.AcceptFriendResponse:
            raise BotClientProtocolError("unexpected accept friend response")
        return parse_friend_action(response)

    async def reject_friend_request(self, requester_id: int) -> FriendRequest:
        response = await self.request(
            MessageType.RejectFriendRequest,
            make_friend_action_body(requester_id),
        )
        if response.header.msg_type != MessageType.RejectFriendResponse:
            raise BotClientProtocolError("unexpected reject friend response")
        return parse_friend_action(response)

    async def pull_offline_messages(self, limit: int | None = None) -> list[IncomingMessage]:
        body = make_offline_request_body(
            limit if limit is not None else self._settings.offline_message_limit
        )
        response = await self.request(MessageType.OfflineMessagesRequest, body)
        if response.header.msg_type != MessageType.OfflineMessagesResponse:
            raise BotClientProtocolError("unexpected offline messages response")
        return parse_offline_messages(response)

    async def ack_offline_messages(self, message_ids: list[int]) -> None:
        response = await self.request(
            MessageType.OfflineMessagesAckRequest,
            make_offline_ack_body(message_ids),
        )
        if response.header.msg_type != MessageType.OfflineMessagesAckResponse:
            raise BotClientProtocolError("unexpected offline ack response")

    async def send_delivery_ack(self, message_id: int) -> None:
        response = await self.request(
            MessageType.DeliveryAckRequest,
            make_delivery_ack_body(message_id),
        )
        if response.header.msg_type != MessageType.DeliveryAckResponse:
            raise BotClientProtocolError("unexpected delivery ack response")

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None:
        response = await self.request(
            MessageType.ReadAckRequest,
            make_read_ack_body(conversation_id, message_id),
        )
        if response.header.msg_type != MessageType.ReadAckResponse:
            raise BotClientProtocolError("unexpected read ack response")

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> IncomingMessage:
        response = await self.request(
            MessageType.PrivateMessageRequest,
            make_private_message_body(
                receiver_id,
                text,
                client_message_id or self._new_client_message_id(),
            ),
        )
        if response.header.msg_type != MessageType.PrivateMessageResponse:
            raise BotClientProtocolError("unexpected private message response")
        return parse_incoming_message(response)

    async def request(
        self,
        msg_type: MessageType,
        body: bytes = b"",
        timeout: float | None = None,
    ) -> Packet:
        writer = self._writer
        if writer is None or writer.is_closing():
            raise BotClientConnectionError("LiteIM connection is not available")

        expected_response = _EXPECTED_RESPONSES.get(msg_type)
        if expected_response is None:
            raise BotClientProtocolError(f"{msg_type.name} is not a supported request type")

        seq_id = self._next_seq_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Packet] = loop.create_future()
        self._pending[seq_id] = PendingRequest(msg_type, future, time.monotonic())

        packet = make_packet(msg_type, seq_id, body)
        encoded = encode_packet(packet)
        try:
            async with self._send_lock:
                writer.write(encoded)
                await writer.drain()
            response = await asyncio.wait_for(
                future,
                timeout if timeout is not None else self._settings.request_timeout_seconds,
            )
        except TimeoutError as exc:
            self._drop_pending(seq_id)
            raise BotClientTimeoutError(f"{msg_type.name} timed out") from exc
        except (OSError, ConnectionError) as exc:
            self._drop_pending(seq_id)
            raise BotClientConnectionError(str(exc)) from exc
        except asyncio.CancelledError:
            self._drop_pending(seq_id)
            raise

        if response.header.msg_type == MessageType.ErrorResponse:
            raise BotClientProtocolError("LiteIM returned ErrorResponse")
        if response.header.msg_type != expected_response:
            raise BotClientProtocolError(
                f"expected {expected_response.name}, got {response.header.msg_type!r}"
            )
        return response

    async def wait_disconnected(self) -> None:
        await self._disconnected_event.wait()

    async def close(self) -> None:
        self._closing = True
        self._state = BotClientState.CLOSING
        await self._cancel_heartbeat_task()
        await self._close_transport()
        await self._cancel_receive_task()
        self._fail_pending(BotClientConnectionError("BotClient closed"))
        self._identity = None
        self._state = BotClientState.DISCONNECTED
        self._disconnected_event.set()

    def _next_seq_id(self) -> int:
        seq_id = self._seq_id
        self._seq_id += 1
        return seq_id

    def _new_client_message_id(self) -> str:
        user_id = self._identity.user_id if self._identity is not None else 0
        return f"pa-{user_id}-{uuid.uuid4().hex}"

    def _start_heartbeat(self) -> None:
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _cancel_heartbeat_task(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(self._settings.heartbeat_interval_seconds)
                if self._identity is None or not self.is_connected:
                    continue
                await self.request(MessageType.HeartbeatRequest)
        except (BotClientError, ProtocolError, OSError):
            if not self._closing:
                await self._mark_disconnected()
        except asyncio.CancelledError:
            return

    async def _receive_loop(self) -> None:
        assert self._reader is not None
        try:
            while not self._closing:
                data = await self._reader.read(4096)
                if not data:
                    break
                for packet in self._decoder.feed(data):
                    await self._dispatch_packet(packet)
        except (ProtocolError, OSError) as exc:
            self._fail_pending(BotClientConnectionError(str(exc)))
        except asyncio.CancelledError:
            return
        finally:
            if not self._closing:
                await self._mark_disconnected()

    async def _dispatch_packet(self, packet: Packet) -> None:
        pending = self._pending.pop(packet.header.seq_id, None)
        if pending is not None:
            if not pending.future.done():
                pending.future.set_result(packet)
            return

        if is_push_type(packet.header.msg_type):
            await self.push_queue.put(packet)
            return
        if is_response_type(packet.header.msg_type):
            return
        await self.push_queue.put(packet)

    def _drop_pending(self, seq_id: int) -> None:
        pending = self._pending.pop(seq_id, None)
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    def _fail_pending(self, exc: Exception) -> None:
        pending_items = list(self._pending.items())
        self._pending.clear()
        for _, pending in pending_items:
            if not pending.future.done():
                pending.future.set_exception(exc)

    async def _mark_disconnected(self) -> None:
        self._identity = None
        self._state = BotClientState.DISCONNECTED
        await self._cancel_heartbeat_task()
        self._fail_pending(BotClientConnectionError("LiteIM connection closed"))
        await self._close_transport()
        self._disconnected_event.set()

    async def _close_transport(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is None:
            return
        writer.close()
        with contextlib.suppress(OSError, ConnectionError):
            await writer.wait_closed()

    async def _cancel_receive_task(self) -> None:
        task = self._receive_task
        self._receive_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
