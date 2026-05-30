from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from agent_service.config import Settings
from agent_service.dialogue_policy import DialoguePolicy, DialoguePolicyMode
from agent_service.governance.data_manifest import ProcessedStyleSample
from agent_service.llm import LLMClient, MockLLMClient, OpenAILLMClient
from agent_service.memory.memory_store import MemoryStore
from agent_service.persona import PersonaEngine
from agent_service.rag.chunker import RecursiveTextChunker
from agent_service.rag.document_loader import DocumentLoader
from agent_service.rag.embeddings import (
    EmbeddingClient,
    GeminiEmbeddingClient,
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
)
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.rag.vector_store import ChromaVectorStore
from agent_service.review import HumanReviewStore
from agent_service.schemas import AgentReplyCommand, ChatRequest
from agent_service.style.learning import StyleLearningStore
from agent_service.style.pair_store import StylePairStore
from agent_service.style.style_store import StyleStore
from agent_service.tools import ToolRegistry
from agent_service.tools.builtin import build_default_tool_registry
from agent_service.workflow import run_agent_chat


@dataclass(frozen=True)
class AgentServiceContainer:
    settings: Settings
    review_store: HumanReviewStore
    memory_store: MemoryStore
    style_store: StyleStore
    style_pair_store: StylePairStore | None
    style_learning_store: StyleLearningStore | None
    knowledge_retriever: KnowledgeRetriever
    tool_registry: ToolRegistry
    persona_engine: PersonaEngine
    llm_client: LLMClient
    dialogue_policy: DialoguePolicy

    def chat(self, request: ChatRequest) -> AgentReplyCommand:
        return run_agent_chat(
            request,
            dialogue_policy=self.dialogue_policy,
            review_store=self.review_store,
            knowledge_retriever=self.knowledge_retriever,
            memory_store=self.memory_store,
            style_store=self.style_store,
            style_pair_store=self.style_pair_store,
            tool_registry=self.tool_registry,
            persona_engine=self.persona_engine,
            llm_client=self.llm_client,
            rag_top_k=self.settings.rag_top_k,
            memory_top_k=self.settings.memory_top_k,
            style_top_k=self.settings.style_top_k,
            style_pair_top_k=self.settings.style_pair_top_k,
            style_persona_id=self.settings.style_persona_id,
            style_on_smalltalk=self.settings.style_on_smalltalk,
            style_on_private_chat=self.settings.style_on_private_chat,
            auto_memory_on_chat=self.settings.auto_memory_on_chat,
            auto_memory_user_name=self.settings.auto_memory_user_name,
            auto_memory_persona_name=self.settings.auto_memory_persona_name,
            style_learning_store=self.style_learning_store,
        )


def create_container(
    settings: Settings,
    *,
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
) -> AgentServiceContainer:
    embedding_client = build_embedding_client(settings)
    store = review_store or HumanReviewStore(settings.agent_state_db_path)
    memories = memory_store or build_memory_store(settings, embedding_client)
    styles = style_store or build_style_store(settings, embedding_client)
    style_pairs = style_pair_store or build_style_pair_store(settings)
    knowledge = knowledge_retriever or build_knowledge_retriever(settings, embedding_client)
    tools = tool_registry or build_default_tool_registry()
    persona = persona_engine or PersonaEngine.from_file(
        settings.persona_config_path,
        profile_path=settings.style_profile_path,
    )
    style_learner = style_learning_store or build_style_learning_store(
        settings,
        style_store=styles,
        persona=persona,
    )
    llm = llm_client or build_llm_client(settings)
    policy = dialogue_policy or build_dialogue_policy(settings, llm)
    return AgentServiceContainer(
        settings=settings,
        review_store=store,
        memory_store=memories,
        style_store=styles,
        style_pair_store=style_pairs,
        style_learning_store=style_learner,
        knowledge_retriever=knowledge,
        tool_registry=tools,
        persona_engine=persona,
        llm_client=llm,
        dialogue_policy=policy,
    )


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    provider = settings.embedding_provider.strip().lower()
    if provider == "mock":
        return MockEmbeddingClient()
    if provider == "gemini":
        return GeminiEmbeddingClient(
            api_key=settings.gemini_api_key or settings.embedding_api_key,
            model=settings.embedding_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.embedding_request_timeout_seconds,
        )
    if provider in {"openai", "openai-compatible"}:
        return OpenAIEmbeddingClient(
            api_key=settings.embedding_api_key or settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or settings.openai_base_url,
            timeout_seconds=settings.embedding_request_timeout_seconds,
        )
    raise RuntimeError(f"unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def embedding_collection_name(base_name: str, settings: Settings) -> str:
    raw_name = f"{base_name}_{settings.embedding_provider}_{settings.embedding_model}"
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_")
    if len(normalized) <= 63:
        return normalized
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:54].rstrip('_-')}_{digest}"


def build_memory_store(settings: Settings, embedding_client: EmbeddingClient) -> MemoryStore:
    return MemoryStore(
        sqlite_path=settings.memory_db_path,
        chroma_path=settings.chroma_path,
        embedding_client=embedding_client,
        collection_name=embedding_collection_name("memory", settings),
        top_k=settings.memory_top_k,
    )


def build_knowledge_retriever(
    settings: Settings,
    embedding_client: EmbeddingClient,
) -> KnowledgeRetriever:
    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(
            settings.chroma_path,
            collection_name=embedding_collection_name("knowledge", settings),
        ),
        embedding_client=embedding_client,
        chunker=RecursiveTextChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        ),
        top_k=settings.rag_top_k,
    )
    docs_path = Path(settings.knowledge_docs_path)
    if docs_path.exists():
        documents = DocumentLoader().load_directory(docs_path)
        if documents:
            retriever.index_documents(documents)
    return retriever


def build_style_store(settings: Settings, embedding_client: EmbeddingClient) -> StyleStore:
    store = StyleStore(
        chroma_path=settings.chroma_path,
        embedding_client=embedding_client,
        top_k=settings.style_top_k,
        collection_name=embedding_collection_name("style", settings),
    )
    samples_path = Path(settings.style_samples_path)
    if samples_path.exists():
        samples = load_processed_style_samples(samples_path)
        if samples:
            store.index_samples(samples)
    return store


def load_processed_style_samples(path: Path) -> list[ProcessedStyleSample]:
    samples: list[ProcessedStyleSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            samples.append(ProcessedStyleSample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return samples


def build_style_pair_store(settings: Settings) -> StylePairStore | None:
    path = Path(settings.style_pairs_path)
    if not path.exists():
        return None
    return StylePairStore.from_jsonl(path)


def build_style_learning_store(
    settings: Settings,
    *,
    style_store: StyleStore,
    persona: PersonaEngine,
) -> StyleLearningStore | None:
    if not settings.style_reinforcement_enabled:
        return None
    return StyleLearningStore(
        samples_path=settings.style_reinforcement_samples_path,
        style_store=style_store,
        persona_id=settings.style_persona_id or persona.config.persona_id,
        consent_id=settings.style_reinforcement_consent_id,
        subject_user_id=settings.style_reinforcement_subject_user_id,
    )


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient(model=settings.llm_model)
    return OpenAILLMClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.llm_request_timeout_seconds,
    )


def build_dialogue_policy(settings: Settings, llm_client: LLMClient) -> DialoguePolicy:
    mode: DialoguePolicyMode = "llm" if settings.dialogue_policy_mode == "llm" else "rule"
    return DialoguePolicy(
        mode=mode,
        llm_client=llm_client,
        max_retries=settings.dialogue_policy_max_retries,
        timeout_seconds=settings.dialogue_policy_timeout_seconds,
    )
