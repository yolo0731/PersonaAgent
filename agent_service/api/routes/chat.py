from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from agent_service.config import Settings
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    no_reply_command,
)

ChatHandler = Callable[[ChatRequest], AgentReplyCommand | Awaitable[AgentReplyCommand]]


def register_chat_routes(
    app: FastAPI,
    *,
    settings: Settings,
    chat_handler: ChatHandler,
) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.service_name}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            command = await call_chat_handler(chat_handler, request)
            return ChatResponse(ok=True, command=command)
        except Exception as exc:
            return ChatResponse(
                ok=False,
                command=no_reply_command(request, "agent_service_error"),
                error=ErrorEnvelope(
                    code="agent_service_error",
                    message=str(exc),
                    retryable=True,
                ),
            )


async def call_chat_handler(
    handler: ChatHandler,
    request: ChatRequest,
) -> AgentReplyCommand:
    call_method = type(handler).__call__ if callable(handler) else None
    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(call_method):
        result = handler(request)
    else:
        result = await run_in_threadpool(handler, request)
    if inspect.isawaitable(result):
        return await result
    return result
