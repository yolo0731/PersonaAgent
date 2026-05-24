from __future__ import annotations

from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool


class KnowledgeDocument(BaseModel):
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    active: bool = True


class KnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    active: bool = True

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "active": self.active,
        }


class RetrievedKnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    active: bool
    score: float
    metadata: dict[str, MetadataValue]


class RetrievalTrace(BaseModel):
    query: str
    top_k: int = Field(ge=1)
    result_count: int = Field(ge=0)
    collection: str
    chunk_ids: list[str] = Field(default_factory=list)


class KnowledgeRetrieval(BaseModel):
    results: list[RetrievedKnowledgeChunk]
    trace: RetrievalTrace
