from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from bot_client.bot_client import BotIdentity
from bot_client.liteim_protocol import MessageType, Packet
from bot_client.message_state import FriendPolicyTraceEvent, JsonMessageState
from bot_client.protocol_parser import (
    IncomingMessage,
    parse_incoming_message,
    parse_receipt,
)

PRIVATE_CONVERSATION_TYPE = 1


@dataclass(frozen=True, slots=True)
class MessageProcessingResult:
    reply_text: str | None = None


MessageProcessor = Callable[
    [IncomingMessage],
    MessageProcessingResult | Awaitable[MessageProcessingResult],
]


class ReliabilityClient(Protocol):
    identity: BotIdentity | None

    async def pull_offline_messages(self, limit: int = 100) -> list[IncomingMessage]: ...

    async def ack_offline_messages(self, message_ids: list[int]) -> None: ...

    async def send_delivery_ack(self, message_id: int) -> None: ...

    async def send_read_ack(self, conversation_id: int, message_id: int) -> None: ...

    async def send_private_message(
        self,
        receiver_id: int,
        text: str,
        client_message_id: str | None = None,
    ) -> object: ...


class BotMessageHandler:
    def __init__(
        self,
        client: ReliabilityClient,
        state: JsonMessageState,
        processor: MessageProcessor,
        *,
        require_friendship: bool = False,
    ) -> None:
        self._client = client
        self._state = state
        self._processor = processor
        self._require_friendship = require_friendship

    async def sync_offline_after_login(self, limit: int = 100) -> None:
        messages = await self._client.pull_offline_messages(limit)
        acked_message_ids: list[int] = []
        for message in messages:
            if self._is_self_message(message):
                continue
            handled = await self._process_message(message, send_delivery_ack=False)
            if handled or self._state.has_processed(message.message_id):
                acked_message_ids.append(message.message_id)
        if acked_message_ids:
            await self._client.ack_offline_messages(acked_message_ids)

    async def handle_packet(self, packet: Packet) -> None:
        if packet.header.msg_type == MessageType.PrivateMessagePush:
            message = parse_incoming_message(packet)
            await self._process_message(message, send_delivery_ack=True)
            return
        if packet.header.msg_type == MessageType.GroupMessagePush:
            self._state.record_group_message(parse_incoming_message(packet))
            return
        if packet.header.msg_type in {
            MessageType.DeliveryReceiptPush,
            MessageType.ReadReceiptPush,
        }:
            self._state.record_receipt(parse_receipt(packet))

    async def _process_message(
        self,
        message: IncomingMessage,
        *,
        send_delivery_ack: bool,
    ) -> bool:
        if self._is_self_message(message):
            return False
        if send_delivery_ack:
            await self._client.send_delivery_ack(message.message_id)
        if self._state.has_processed(message.message_id):
            return False
        if self._blocks_non_friend_message(message):
            self._state.record_friend_policy_event(
                FriendPolicyTraceEvent(
                    action="blocked_non_friend_message",
                    user_id=message.sender_id,
                    username="",
                    reason="not_friend",
                )
            )
            self._state.mark_processed(message.message_id)
            return False

        result = await self._call_processor(message)
        if message.conversation_type == PRIVATE_CONVERSATION_TYPE:
            await self._client.send_read_ack(message.conversation_id, message.message_id)
        if result.reply_text:
            await self._client.send_private_message(
                message.sender_id,
                result.reply_text,
                self._new_client_message_id(),
            )
        self._state.mark_processed(message.message_id)
        return True

    async def _call_processor(self, message: IncomingMessage) -> MessageProcessingResult:
        result = self._processor(message)
        if inspect.isawaitable(result):
            return await result
        return result

    def _is_self_message(self, message: IncomingMessage) -> bool:
        identity = self._client.identity
        return identity is not None and message.sender_id == identity.user_id

    def _blocks_non_friend_message(self, message: IncomingMessage) -> bool:
        return (
            self._require_friendship
            and message.conversation_type == PRIVATE_CONVERSATION_TYPE
            and not self._state.is_friend(message.sender_id)
        )

    def _new_client_message_id(self) -> str:
        identity = self._client.identity
        user_id = identity.user_id if identity is not None else 0
        return f"pa-{user_id}-{uuid.uuid4().hex}"
