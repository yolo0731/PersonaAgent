from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from agent_service.schemas import AgentReplyCommand, ChatRequest, ChatResponse
from bot_client.config import BotClientSettings
from bot_client.message_handler import MessageProcessingResult
from bot_client.protocol_parser import IncomingMessage


def chat_request_from_message(
    message: IncomingMessage,
    *,
    run_id: str | None = None,
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
    )


class AgentCommandClient(Protocol):
    async def chat_for_message(self, message: IncomingMessage) -> AgentReplyCommand: ...


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

    async def chat_for_message(self, message: IncomingMessage) -> AgentReplyCommand:
        return await self.chat(chat_request_from_message(message))


@dataclass(frozen=True, slots=True)
class AgentServiceMessageProcessor:
    client: AgentCommandClient

    async def __call__(self, message: IncomingMessage) -> MessageProcessingResult:
        command = await self.client.chat_for_message(message)
        if not command.should_send:
            return MessageProcessingResult()
        return MessageProcessingResult(
            reply_text=command.text,
            client_message_id=command.client_message_id,
            receiver_id=command.receiver_id,
        )


def _unavailable_command(request: ChatRequest) -> AgentReplyCommand:
    return AgentReplyCommand(
        run_id=request.run_id,
        source_message_id=request.message_id,
        should_send=False,
        receiver_id=request.sender_id,
        text="",
        client_message_id=None,
        reason="agent_service_unavailable",
    )
