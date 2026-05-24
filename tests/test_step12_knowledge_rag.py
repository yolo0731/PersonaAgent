from __future__ import annotations

from pathlib import Path


def _chat_payload(text: str = "PersonaAgent 项目怎么做 RAG？") -> dict[str, object]:
    return {
        "run_id": "run-rag",
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def test_document_loader_and_chunker_create_required_metadata(tmp_path: Path) -> None:
    from agent_service.rag.chunker import RecursiveTextChunker
    from agent_service.rag.document_loader import DocumentLoader

    docs_dir = tmp_path / "knowledge_docs"
    docs_dir.mkdir()
    (docs_dir / "liteim.md").write_text(
        "# LiteIM Reactor\n\nLiteIM uses epoll and one-loop-per-thread Reactor.",
        encoding="utf-8",
    )

    documents = DocumentLoader().load_directory(docs_dir)
    chunks = RecursiveTextChunker(chunk_size=48, chunk_overlap=8).split_documents(documents)

    assert len(documents) == 1
    assert documents[0].doc_id == "liteim"
    assert documents[0].title == "LiteIM Reactor"
    assert chunks
    for chunk in chunks:
        assert chunk.doc_id == "liteim"
        assert chunk.source.endswith("liteim.md")
        assert chunk.title == "LiteIM Reactor"
        assert chunk.chunk_id.startswith("liteim:")
        assert chunk.active is True
        assert chunk.metadata == {
            "doc_id": chunk.doc_id,
            "source": chunk.source,
            "title": chunk.title,
            "chunk_id": chunk.chunk_id,
            "active": True,
        }


def test_chroma_knowledge_retriever_persists_and_returns_top_k(tmp_path: Path) -> None:
    from agent_service.rag.documents import KnowledgeDocument
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.rag.knowledge_retriever import KnowledgeRetriever
    from agent_service.rag.vector_store import ChromaVectorStore

    persist_path = tmp_path / "chroma"
    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(persist_path, collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        top_k=1,
    )

    retriever.index_documents(
        [
            KnowledgeDocument(
                doc_id="liteim-reactor",
                source="docs/liteim.md",
                title="LiteIM Reactor",
                text="LiteIM uses epoll, nonblocking socket, and Reactor threads.",
            ),
            KnowledgeDocument(
                doc_id="personaagent",
                source="docs/personaagent.md",
                title="PersonaAgent",
                text="PersonaAgent uses FastAPI, LangGraph, and Knowledge RAG.",
            ),
        ]
    )

    reopened = KnowledgeRetriever(
        vector_store=ChromaVectorStore(persist_path, collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        top_k=1,
    )
    result = reopened.retrieve("epoll Reactor")

    assert result.trace.query == "epoll Reactor"
    assert result.trace.top_k == 1
    assert result.trace.result_count == 1
    assert result.trace.chunk_ids == ["liteim-reactor:0000"]
    assert len(result.results) == 1
    assert result.results[0].doc_id == "liteim-reactor"
    assert "epoll" in result.results[0].text


def test_metadata_filter_excludes_inactive_documents(tmp_path: Path) -> None:
    from agent_service.rag.documents import KnowledgeDocument
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.rag.knowledge_retriever import KnowledgeRetriever
    from agent_service.rag.vector_store import ChromaVectorStore

    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(tmp_path / "chroma", collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        top_k=3,
    )
    retriever.index_documents(
        [
            KnowledgeDocument(
                doc_id="inactive",
                source="docs/inactive.md",
                title="Inactive",
                text="special retrieval marker that should be filtered",
                active=False,
            ),
            KnowledgeDocument(
                doc_id="active",
                source="docs/active.md",
                title="Active",
                text="general active knowledge",
                active=True,
            ),
        ]
    )

    result = retriever.retrieve("special retrieval marker", active_only=True)

    assert result.results
    assert {item.doc_id for item in result.results} == {"active"}
    assert all(item.metadata["active"] is True for item in result.results)


def test_empty_knowledge_collection_returns_empty_result(tmp_path: Path) -> None:
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.rag.knowledge_retriever import KnowledgeRetriever
    from agent_service.rag.vector_store import ChromaVectorStore

    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(tmp_path / "chroma", collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        top_k=5,
    )

    result = retriever.retrieve("anything")

    assert result.results == []
    assert result.trace.result_count == 0


def test_workflow_writes_knowledge_results_into_context_and_trace(tmp_path: Path) -> None:
    from agent_service.rag.documents import KnowledgeDocument
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.rag.knowledge_retriever import KnowledgeRetriever
    from agent_service.rag.vector_store import ChromaVectorStore
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    retriever = KnowledgeRetriever(
        vector_store=ChromaVectorStore(tmp_path / "chroma", collection_name="knowledge"),
        embedding_client=MockEmbeddingClient(),
        top_k=1,
    )
    retriever.index_documents(
        [
            KnowledgeDocument(
                doc_id="rag-route",
                source="docs/rag.md",
                title="Knowledge RAG",
                text="PersonaAgent Step 12 retrieves Knowledge RAG chunks before reply generation.",
            )
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload()),
        knowledge_retriever=retriever,
        rag_top_k=1,
    )

    assert state["decision"].need_knowledge is True
    assert state["retrieved_context"] == [
        "PersonaAgent Step 12 retrieves Knowledge RAG chunks before reply generation."
    ]
    assert len(state["retrieval_trace"]) == 1
    assert state["retrieval_trace"][0].chunk_ids == ["rag-route:0000"]
    retrieve_events = [event for event in state["trace"] if event.node == "retrieve_context"]
    assert retrieve_events
    assert retrieve_events[0].action == "knowledge_top_k=1"
