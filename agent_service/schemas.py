from __future__ import annotations

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
    text: str = ""
    client_message_id: str | None = None
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
    return AgentReplyCommand(
        run_id=request.run_id,
        source_message_id=request.message_id,
        should_send=True,
        receiver_id=request.sender_id,
        text=f"mock reply: {request.text}",
        client_message_id=f"pa-{request.run_id}",
        reason="mock_chat",
    )


def no_reply_command(request: ChatRequest, reason: str) -> AgentReplyCommand:
    return AgentReplyCommand(
        run_id=request.run_id,
        source_message_id=request.message_id,
        should_send=False,
        receiver_id=request.sender_id,
        text="",
        client_message_id=None,
        reason=reason,
    )
