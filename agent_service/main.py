"""FastAPI entrypoint for AgentService."""

import inspect
from collections.abc import Awaitable, Callable

from fastapi import FastAPI

from agent_service.config import Settings
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    no_reply_command,
)
from agent_service.workflow import run_agent_chat

ChatHandler = Callable[[ChatRequest], AgentReplyCommand | Awaitable[AgentReplyCommand]]


async def _default_chat_handler(request: ChatRequest) -> AgentReplyCommand:
    return run_agent_chat(request)


async def _call_chat_handler(
    handler: ChatHandler,
    request: ChatRequest,
) -> AgentReplyCommand:
    result = handler(request)
    if inspect.isawaitable(result):
        return await result
    return result


def create_app(
    settings: Settings | None = None,
    chat_handler: ChatHandler | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")
    handler = chat_handler or _default_chat_handler

    # Settings 挂在 app.state 上，后续 /chat、LLM、trace 都从这里读取运行配置。
    app.state.settings = app_settings
    app.state.chat_handler = handler

    @app.get("/health")
    def health() -> dict[str, str]:
        # /health 只证明服务进程可用，不触发 DeepSeek、RAG 或 LiteIM 依赖。
        return {"status": "ok", "service": app_settings.service_name}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            command = await _call_chat_handler(handler, request)
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

    return app


app = create_app()
