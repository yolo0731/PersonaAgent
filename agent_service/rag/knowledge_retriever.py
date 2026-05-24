from __future__ import annotations

from collections.abc import Sequence

from agent_service.rag.chunker import RecursiveTextChunker
from agent_service.rag.documents import (
    KnowledgeDocument,
    KnowledgeRetrieval,
    RetrievalTrace,
)
from agent_service.rag.embeddings import EmbeddingClient
from agent_service.rag.vector_store import ChromaVectorStore


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore,
        embedding_client: EmbeddingClient,
        chunker: RecursiveTextChunker | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._chunker = chunker or RecursiveTextChunker()
        self._top_k = top_k

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> int:
        chunks = self._chunker.split_documents(list(documents))
        embeddings = self._embedding_client.embed_texts([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, embeddings)
        return len(chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        active_only: bool = True,
    ) -> KnowledgeRetrieval:
        effective_top_k = top_k or self._top_k
        query_embedding = self._embedding_client.embed_query(query)
        results = self._vector_store.query(
            query_embedding,
            top_k=effective_top_k,
            active_only=active_only,
        )
        return KnowledgeRetrieval(
            results=results,
            trace=RetrievalTrace(
                query=query,
                top_k=effective_top_k,
                result_count=len(results),
                collection=self._vector_store.collection_name,
                chunk_ids=[result.chunk_id for result in results],
            ),
        )
