from __future__ import annotations

import importlib
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_service.rag.documents import MetadataValue, RetrievalTrace
from agent_service.rag.embeddings import EmbeddingClient


class MemoryRecord(BaseModel):
    memory_id: str = Field(min_length=1)
    user_id: int = Field(ge=1)
    content: str = Field(min_length=1)
    source_message_id: int = Field(ge=1)
    active: bool = True
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: str

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "source_message_id": self.source_message_id,
            "active": self.active,
            "importance": self.importance,
            "created_at": self.created_at,
        }


class RetrievedMemory(BaseModel):
    memory_id: str = Field(min_length=1)
    user_id: int = Field(ge=1)
    content: str = Field(min_length=1)
    source_message_id: int = Field(ge=1)
    active: bool
    importance: float
    created_at: str
    score: float
    metadata: dict[str, MetadataValue]


class MemoryRetrieval(BaseModel):
    results: list[RetrievedMemory]
    trace: RetrievalTrace


class MemoryNotFoundError(KeyError):
    """Raised when a user-scoped memory record is missing."""


class MemoryStore:
    def __init__(
        self,
        *,
        sqlite_path: str | Path,
        chroma_path: str | Path,
        embedding_client: EmbeddingClient,
        collection_name: str = "memory",
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._sqlite_path = Path(sqlite_path)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_client = embedding_client
        self._collection_name = collection_name
        self._top_k = top_k
        self._ensure_schema()

        chromadb: Any = importlib.import_module("chromadb")
        chroma_root = Path(chroma_path)
        chroma_root.mkdir(parents=True, exist_ok=True)
        self._client: Any = chromadb.PersistentClient(path=str(chroma_root))
        self._collection: Any = self._client.get_or_create_collection(name=collection_name)

    def save_memory(
        self,
        *,
        user_id: int,
        content: str,
        source_message_id: int,
        importance: float = 0.5,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            memory_id=memory_id or f"mem-{user_id}-{source_message_id}",
            user_id=user_id,
            content=content,
            source_message_id=source_message_id,
            active=True,
            importance=importance,
            created_at=_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, content, source_message_id,
                    active, importance, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    content = excluded.content,
                    source_message_id = excluded.source_message_id,
                    active = excluded.active,
                    importance = excluded.importance,
                    created_at = excluded.created_at
                """,
                (
                    record.memory_id,
                    record.user_id,
                    record.content,
                    record.source_message_id,
                    int(record.active),
                    record.importance,
                    record.created_at,
                ),
            )
        self._upsert_vector(record)
        return record

    def deactivate_memory(self, memory_id: str, *, user_id: int) -> MemoryRecord:
        self._require_memory(memory_id, user_id=user_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET active = 0
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            )
        updated = self._require_memory(memory_id, user_id=user_id)
        self._upsert_vector(updated)
        return updated

    def list_memories(self, *, user_id: int, active_only: bool = True) -> list[MemoryRecord]:
        where = "WHERE user_id = ?"
        params: list[object] = [user_id]
        if active_only:
            where += " AND active = 1"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, user_id, content, source_message_id,
                       active, importance, created_at
                FROM memories
                {where}
                ORDER BY created_at ASC, memory_id ASC
                """,
                params,
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def retrieve_memory(
        self,
        *,
        user_id: int,
        query: str,
        top_k: int | None = None,
        active_only: bool = True,
    ) -> MemoryRetrieval:
        effective_top_k = top_k or self._top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive")
        if int(self._collection.count()) == 0:
            return self._empty_retrieval(query, effective_top_k)

        where: dict[str, object]
        if active_only:
            where = {"$and": [{"user_id": user_id}, {"active": True}]}
        else:
            where = {"user_id": user_id}

        raw_result = self._collection.query(
            query_embeddings=[self._embedding_client.embed_query(query)],
            n_results=effective_top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results = _parse_memory_query_result(raw_result)
        return MemoryRetrieval(
            results=results,
            trace=RetrievalTrace(
                query=query,
                top_k=effective_top_k,
                result_count=len(results),
                collection=self._collection_name,
                chunk_ids=[result.memory_id for result in results],
            ),
        )

    def _empty_retrieval(self, query: str, top_k: int) -> MemoryRetrieval:
        return MemoryRetrieval(
            results=[],
            trace=RetrievalTrace(
                query=query,
                top_k=top_k,
                result_count=0,
                collection=self._collection_name,
                chunk_ids=[],
            ),
        )

    def _upsert_vector(self, record: MemoryRecord) -> None:
        self._collection.upsert(
            ids=[record.memory_id],
            documents=[record.content],
            embeddings=[self._embedding_client.embed_query(record.content)],
            metadatas=[record.metadata],
        )

    def _require_memory(self, memory_id: str, *, user_id: int) -> MemoryRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT memory_id, user_id, content, source_message_id,
                       active, importance, created_at
                FROM memories
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(memory_id)
        return _record_from_row(row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_active
                ON memories(user_id, active)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn


def _parse_memory_query_result(raw_result: object) -> list[RetrievedMemory]:
    if not isinstance(raw_result, Mapping):
        return []
    ids = _first_list(raw_result.get("ids"))
    documents = _first_list(raw_result.get("documents"))
    metadatas = _first_list(raw_result.get("metadatas"))
    distances = _first_list(raw_result.get("distances"))

    results: list[RetrievedMemory] = []
    for index, memory_id_value in enumerate(ids):
        metadata = _metadata_at(metadatas, index)
        memory_id = str(metadata.get("memory_id", memory_id_value))
        content = str(documents[index]) if index < len(documents) else ""
        distance = _float_at(distances, index)
        results.append(
            RetrievedMemory(
                memory_id=memory_id,
                user_id=int(metadata.get("user_id", 0)),
                content=content,
                source_message_id=int(metadata.get("source_message_id", 0)),
                active=bool(metadata.get("active", True)),
                importance=float(metadata.get("importance", 0.5)),
                created_at=str(metadata.get("created_at", "")),
                score=1.0 / (1.0 + max(distance, 0.0)),
                metadata=metadata,
            )
        )
    return results


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        user_id=int(row["user_id"]),
        content=str(row["content"]),
        source_message_id=int(row["source_message_id"]),
        active=bool(row["active"]),
        importance=float(row["importance"]),
        created_at=str(row["created_at"]),
    )


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


def _now() -> str:
    return datetime.now(UTC).isoformat()
