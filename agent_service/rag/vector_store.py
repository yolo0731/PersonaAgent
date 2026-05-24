from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_service.rag.documents import KnowledgeChunk, MetadataValue, RetrievedKnowledgeChunk


class ChromaVectorStore:
    def __init__(self, persist_path: str | Path, *, collection_name: str = "knowledge") -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        chromadb: Any = importlib.import_module("chromadb")
        self._client: Any = chromadb.PersistentClient(path=str(self.persist_path))
        self._collection: Any = self._client.get_or_create_collection(name=collection_name)

    def upsert(self, chunks: Sequence[KnowledgeChunk], embeddings: Sequence[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=list(embeddings),
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int,
        active_only: bool = True,
    ) -> list[RetrievedKnowledgeChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if int(self._collection.count()) == 0:
            return []

        where = {"active": True} if active_only else None
        raw_result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _parse_query_result(raw_result)


def _parse_query_result(raw_result: object) -> list[RetrievedKnowledgeChunk]:
    if not isinstance(raw_result, Mapping):
        return []
    ids = _first_list(raw_result.get("ids"))
    documents = _first_list(raw_result.get("documents"))
    metadatas = _first_list(raw_result.get("metadatas"))
    distances = _first_list(raw_result.get("distances"))

    results: list[RetrievedKnowledgeChunk] = []
    for index, chunk_id_value in enumerate(ids):
        metadata = _metadata_at(metadatas, index)
        document = str(documents[index]) if index < len(documents) else ""
        distance = _float_at(distances, index)
        chunk_id = str(metadata.get("chunk_id", chunk_id_value))
        results.append(
            RetrievedKnowledgeChunk(
                chunk_id=chunk_id,
                doc_id=str(metadata.get("doc_id", "")),
                source=str(metadata.get("source", "")),
                title=str(metadata.get("title", "")),
                text=document,
                active=bool(metadata.get("active", True)),
                score=1.0 / (1.0 + max(distance, 0.0)),
                metadata=metadata,
            )
        )
    return results


def _first_list(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    if not value:
        return []
    first = value[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return []
    return list(first)


def _metadata_at(values: list[object], index: int) -> dict[str, MetadataValue]:
    if index >= len(values):
        return {}
    value = values[index]
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, MetadataValue] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)):
            metadata[str(key)] = item
    return metadata


def _float_at(values: list[object], index: int) -> float:
    if index >= len(values):
        return 0.0
    value = values[index]
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
