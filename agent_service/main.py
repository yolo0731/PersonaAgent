"""FastAPI entrypoint for AgentService."""

from __future__ import annotations

from fastapi import FastAPI

from agent_service.api.routes.chat import ChatHandler, register_chat_routes
from agent_service.api.routes.review import register_review_routes
from agent_service.config import Settings
from agent_service.container import AgentServiceContainer, create_container
from agent_service.dialogue_policy import DialoguePolicy
from agent_service.llm import LLMClient
from agent_service.memory.memory_store import MemoryStore
from agent_service.persona import PersonaEngine
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.review import HumanReviewStore
from agent_service.schemas import AgentReplyCommand, ChatRequest
from agent_service.style.learning import StyleLearningStore
from agent_service.style.pair_store import StylePairStore
from agent_service.style.style_store import StyleStore
from agent_service.tools import ToolRegistry


def create_app(
    settings: Settings | None = None,
    chat_handler: ChatHandler | None = None,
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    style_pair_store: StylePairStore | None = None,
    style_learning_store: StyleLearningStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    dialogue_policy: DialoguePolicy | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="PersonaAgent AgentService")

    store = review_store or HumanReviewStore(app_settings.agent_state_db_path)
    container: AgentServiceContainer | None = None

    def get_container() -> AgentServiceContainer:
        nonlocal container
        if container is None:
            container = create_container(
                app_settings,
                review_store=store,
                knowledge_retriever=knowledge_retriever,
                memory_store=memory_store,
                style_store=style_store,
                style_pair_store=style_pair_store,
                style_learning_store=style_learning_store,
                tool_registry=tool_registry,
                persona_engine=persona_engine,
                llm_client=llm_client,
                dialogue_policy=dialogue_policy,
            )
        return container

    def default_chat_handler(request: ChatRequest) -> AgentReplyCommand:
        return get_container().chat(request)

    handler = chat_handler or default_chat_handler

    app.state.settings = app_settings
    app.state.chat_handler = handler
    app.state.review_store = store
    app.state.get_container = get_container

    register_chat_routes(app, settings=app_settings, chat_handler=handler)
    register_review_routes(app, settings=app_settings, store=store)
    return app


app = create_app()
