# 负责离线同步、push 处理、ACK 顺序、去重、自发消息过滤和 receipt 记录
from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from bot_client.connection.client import BotIdentity
from bot_client.messages.state import (
    AgentReplyTraceEvent,
    FriendPolicyTraceEvent,
    JsonMessageState,
)
from bot_client.protocol.codec import MessageType, Packet
from bot_client.protocol.parsers import (
    IncomingMessage,
    parse_incoming_message,
    parse_receipt,
)

PRIVATE_CONVERSATION_TYPE = 1
#@dataclass自动生成构造函数等样板代码，适合纯数据对象。
#frozen=True:创建后不允许修改字段, slots=True:限制这个对象只能有声明过的字段，并减少一点内存开销
@dataclass(frozen=True, slots=True)
class MessageProcessingResult:
    reply_text: str | None = None
    client_message_id: str | None = None
    receiver_id: int | None = None
    dedup_key: str | None = None
    trace_summary: list[str] = field(default_factory=list)


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

    # 处理推送包
    async def handle_packet(self, packet: Packet) -> None:
        # 如果是推送包，根据消息类型进行处理，私聊消息需要调用处理器生成回复并发送
        if packet.header.msg_type == MessageType.PrivateMessagePush:
            message = parse_incoming_message(packet)
            await self._process_message(message, send_delivery_ack=True)
            return
        # 群消息和回执消息只需要记录状态，不需要回复
        if packet.header.msg_type == MessageType.GroupMessagePush:
            self._state.record_group_message(parse_incoming_message(packet))
            return
        if packet.header.msg_type in {
            MessageType.DeliveryReceiptPush,
            MessageType.ReadReceiptPush,
        }:
            self._state.record_receipt(parse_receipt(packet))

    # 处理 BotClient 收到的一条消息，决定 ACK、AgentService 调用和 LiteIM 回复。
    async def _process_message(
        self,
        message: IncomingMessage,
        *,
        send_delivery_ack: bool,
    ) -> bool:
        #忽略自己发的消息
        if self._is_self_message(message):
            return False
        # 如果是实时 push，先发送投递 ACK
        if send_delivery_ack:
            await self._client.send_delivery_ack(message.message_id)
        #如果这条消息处理过，就不重复处理
        if self._state.has_processed(message.message_id):
            return False
        # 如果要求好友关系，但发送者不是好友，就拦截
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

        # 调用处理器生成回复结果
        result = await self._call_processor(message)
        # 私聊消息处理后发送已读 ACK
        if message.conversation_type == PRIVATE_CONVERSATION_TYPE:
            await self._client.send_read_ack(message.conversation_id, message.message_id)
        # 如果 Agent 生成了回复，就通过 BotClient 发回 LiteIM
        if result.reply_text:
            receiver_id = result.receiver_id or message.sender_id
            client_message_id = result.client_message_id or self._new_client_message_id()
            if result.dedup_key and self._state.has_sent_agent_reply(result.dedup_key):
                self._state.mark_processed(message.message_id)
                return True
            try:
                await self._client.send_private_message(
                    receiver_id,
                    result.reply_text,
                    client_message_id,
                )
            except Exception as exc:
                if result.dedup_key:
                    self._state.record_agent_reply_event(
                        AgentReplyTraceEvent(
                            status="failed",
                            dedup_key=result.dedup_key,
                            source_message_id=message.message_id,
                            receiver_id=receiver_id,
                            client_message_id=client_message_id,
                            reason=str(exc),
                            trace_summary=result.trace_summary,
                        )
                    )
                self._state.mark_processed(message.message_id)
                return True
            if result.dedup_key:
                self._state.record_agent_reply_event(
                    AgentReplyTraceEvent(
                        status="sent",
                        dedup_key=result.dedup_key,
                        source_message_id=message.message_id,
                        receiver_id=receiver_id,
                        client_message_id=client_message_id,
                        reason="sent",
                        trace_summary=result.trace_summary,
                    )
                )
        self._state.mark_processed(message.message_id)
        return True

    # 调用处理器生成回复结果，如果处理器是异步的就 await，处理器内部可能会访问状态来决定回复内容
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
