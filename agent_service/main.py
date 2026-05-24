"""FastAPI entrypoint for AgentService."""

import inspect
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException

from agent_service.config import Settings
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.review import (
    ApproveReviewRequest,
    EditReviewRequest,
    HumanReviewNotFoundError,
    HumanReviewRecord,
    HumanReviewStore,
)
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    no_reply_command,
)
from agent_service.workflow import resume_agent_review, run_agent_chat

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
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")
    store = review_store or HumanReviewStore(app_settings.agent_state_db_path)
    handler = chat_handler or (
        lambda request: run_agent_chat(
            request,
            review_store=store,
            knowledge_retriever=knowledge_retriever,
            rag_top_k=app_settings.rag_top_k,
        )
    )

    # Settings 挂在 app.state 上，后续 /chat、LLM、trace 都从这里读取运行配置。
    app.state.settings = app_settings
    app.state.chat_handler = handler
    app.state.review_store = store
    app.state.knowledge_retriever = knowledge_retriever

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

    @app.post("/human-review/{thread_id}/edit", response_model=HumanReviewRecord)
    def edit_review(thread_id: str, request: EditReviewRequest) -> HumanReviewRecord:
        try:
            return store.edit(thread_id, request.edited_text)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc

    @app.post("/human-review/{thread_id}/approve", response_model=HumanReviewRecord)
    def approve_review(
        thread_id: str,
        request: ApproveReviewRequest | None = None,
    ) -> HumanReviewRecord:
        try:
            return store.approve(
                thread_id,
                request.edited_text if request is not None else None,
            )
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc

    @app.post("/human-review/{thread_id}/reject", response_model=HumanReviewRecord)
    def reject_review(thread_id: str) -> HumanReviewRecord:
        try:
            return store.reject(thread_id)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc

    @app.post("/human-review/{thread_id}/resume", response_model=ChatResponse)
    def resume_review(thread_id: str) -> ChatResponse:
        try:
            command = resume_agent_review(thread_id, store)
        except HumanReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail="review thread not found") from exc
        return ChatResponse(ok=True, command=command)

    return app


app = create_app()
