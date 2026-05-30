from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from agent_service.schemas import (
    AgentReplyCommand,
    ChatContextMessage,
    ChatRequest,
    ChatResponse,
    no_reply_command,
)
from bot_client.messages.handler import MessageProcessingResult
from bot_client.protocol.parsers import IncomingMessage
from bot_client.runtime.config import BotClientSettings


def chat_request_from_message(
    message: IncomingMessage,
    *,
    run_id: str | None = None,
    recent_context: Sequence[IncomingMessage] = (),
) -> ChatRequest:
    return ChatRequest(
        run_id=run_id or f"liteim-message-{message.message_id}",
        conversation_type=message.conversation_type,
        conversation_id=message.conversation_id,
        message_id=message.message_id,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        text=message.text,
        timestamp_ms=message.timestamp_ms,
        client_message_id=message.client_message_id,
        recent_context=[_context_message_from_liteim(item) for item in recent_context],
    )


class AgentCommandClient(Protocol):
    async def chat_for_message(
        self,
        message: IncomingMessage,
        recent_context: Sequence[IncomingMessage] = (),
    ) -> AgentReplyCommand: ...


HistoryLoader = Callable[..., Awaitable[list[IncomingMessage]]]

_SERVICE_UNAVAILABLE_REASON = "agent_service_unavailable"
_SERVICE_UNAVAILABLE_FALLBACK_TEXT = "我刚刚有点卡住了，你再发我一遍"


@dataclass(frozen=True, slots=True)
class AgentApiClient:
    base_url: str
    timeout_seconds: float = 5.0
    transport: httpx.AsyncBaseTransport | None = None

    @classmethod
    def from_settings(cls, settings: BotClientSettings) -> AgentApiClient:
        return cls(
            base_url=settings.agent_service_url,
            timeout_seconds=settings.agent_request_timeout_seconds,
        )

    async def chat(self, request: ChatRequest) -> AgentReplyCommand:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post("/chat", json=request.model_dump())
                response.raise_for_status()
            envelope = ChatResponse.model_validate(response.json())
            if not envelope.ok:
                return envelope.command
            return envelope.command
        except Exception:
            return _unavailable_command(request)

    async def chat_for_message(
        self,
        message: IncomingMessage,
        recent_context: Sequence[IncomingMessage] = (),
    ) -> AgentReplyCommand:
        return await self.chat(
            chat_request_from_message(message, recent_context=recent_context)
        )


@dataclass(frozen=True, slots=True)
class AgentServiceMessageProcessor:
    client: AgentCommandClient
    history_loader: HistoryLoader | None = None
    recent_context_limit: int = 0

    # 把 LiteIM 消息转成 AgentService HTTP 请求，再把回复命令转成处理结果。
    # 期间会尝试加载最近的消息上下文以供 AgentService 参考。
    async def __call__(self, message: IncomingMessage) -> MessageProcessingResult:
        recent_context = await self._load_recent_context(message)
        command = await self._call_client(message, recent_context)
        if not command.should_send:
            if command.reason == _SERVICE_UNAVAILABLE_REASON:
                return MessageProcessingResult(
                    reply_text=_SERVICE_UNAVAILABLE_FALLBACK_TEXT,
                    client_message_id=f"pa-unavailable-{message.message_id}",
                    receiver_id=message.sender_id,
                    dedup_key=command.dedup_key,
                    trace_summary=command.trace_summary,
                )
            return MessageProcessingResult()
        return MessageProcessingResult(
            reply_text=command.text,
            client_message_id=command.client_message_id,
            receiver_id=command.receiver_id,
            dedup_key=command.dedup_key,
            trace_summary=command.trace_summary,
        )

    async def _call_client(
        self,
        message: IncomingMessage,
        recent_context: Sequence[IncomingMessage],
    ) -> AgentReplyCommand:
        method = self.client.chat_for_message
        try:
            if len(inspect.signature(method).parameters) <= 1:
                return await method(message)
        except (TypeError, ValueError):
            pass
        return await method(message, recent_context)

    async def _load_recent_context(self, message: IncomingMessage) -> list[IncomingMessage]:
        if self.history_loader is None or self.recent_context_limit <= 0:
            return []
        try:
            return await self.history_loader(
                conversation_type=message.conversation_type,
                conversation_id=message.conversation_id,
                before_message_id=message.message_id,
                limit=self.recent_context_limit,
            )
        except Exception:
            return []


def _unavailable_command(request: ChatRequest) -> AgentReplyCommand:
    return no_reply_command(request, _SERVICE_UNAVAILABLE_REASON)


def _context_message_from_liteim(message: IncomingMessage) -> ChatContextMessage:
    return ChatContextMessage(
        message_id=message.message_id,
        conversation_type=message.conversation_type,
        conversation_id=message.conversation_id,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        text=message.text,
        timestamp_ms=message.timestamp_ms,
        client_message_id=message.client_message_id,
    )
