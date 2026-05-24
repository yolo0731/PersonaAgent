from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    run_id: str = Field(min_length=1)
    conversation_type: int = Field(ge=1)
    conversation_id: int = Field(ge=1)
    message_id: int = Field(ge=1)
    sender_id: int = Field(ge=1)
    receiver_id: int = Field(ge=1)
    text: str
    timestamp_ms: int = Field(ge=0)
    client_message_id: str | None = None


class AgentReplyCommand(BaseModel):
    run_id: str = Field(min_length=1)
    source_message_id: int = Field(ge=1)
    should_send: bool
    receiver_id: int = Field(ge=1)
    conversation_type: int | None = Field(default=None, ge=1)
    conversation_id: int | None = Field(default=None, ge=1)
    text: str = ""
    client_message_id: str | None = None
    dedup_key: str | None = Field(default=None, min_length=1)
    trace_summary: list[str] = Field(default_factory=list)
    reason: str | None = None


class ErrorEnvelope(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = True


class ChatResponse(BaseModel):
    ok: bool
    command: AgentReplyCommand
    error: ErrorEnvelope | None = None


def mock_reply_command(request: ChatRequest) -> AgentReplyCommand:
    return send_reply_command(
        request,
        text=f"mock reply: {request.text}",
        reason="mock_chat",
    )


def send_reply_command(
    request: ChatRequest,
    *,
    text: str,
    reason: str,
    trace_summary: Sequence[str] = (),
) -> AgentReplyCommand:
    return AgentReplyCommand(
        run_id=request.run_id,
        source_message_id=request.message_id,
        should_send=True,
        receiver_id=request.sender_id,
        conversation_type=request.conversation_type,
        conversation_id=request.conversation_id,
        text=text,
        client_message_id=_reply_client_message_id(request),
        dedup_key=reply_dedup_key(request),
        trace_summary=list(trace_summary),
        reason=reason,
    )


def no_reply_command(
    request: ChatRequest,
    reason: str,
    *,
    trace_summary: Sequence[str] = (),
) -> AgentReplyCommand:
    return AgentReplyCommand(
        run_id=request.run_id,
        source_message_id=request.message_id,
        should_send=False,
        receiver_id=request.sender_id,
        conversation_type=request.conversation_type,
        conversation_id=request.conversation_id,
        text="",
        client_message_id=None,
        dedup_key=reply_dedup_key(request),
        trace_summary=list(trace_summary),
        reason=reason,
    )


def reply_dedup_key(request: ChatRequest) -> str:
    return f"agent-reply:{request.run_id}:{request.message_id}"


def _reply_client_message_id(request: ChatRequest) -> str:
    return f"pa-{request.run_id}"
