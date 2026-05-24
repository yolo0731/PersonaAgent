from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_service.governance.data_manifest import ProcessedStyleSample
from agent_service.rag.documents import MetadataValue, RetrievalTrace
from agent_service.rag.embeddings import EmbeddingClient
from agent_service.style.features import StyleFeatureExtractor, StyleFeatures


class RetrievedStyleSample(BaseModel):
    sample_id: str = Field(min_length=1)
    consent_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    active: bool
    score: float
    metadata: dict[str, MetadataValue]


class StyleSummary(BaseModel):
    persona_id: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    text: str
    examples: list[str] = Field(default_factory=list)


class StyleRetrieval(BaseModel):
    results: list[RetrievedStyleSample]
    trace: RetrievalTrace
    features: StyleFeatures
    summary: StyleSummary
    fallback_reason: str | None = None


class StyleStore:
    def __init__(
        self,
        *,
        chroma_path: str | Path,
        embedding_client: EmbeddingClient,
        top_k: int = 8,
        min_samples: int = 2,
        collection_name: str = "style",
        feature_extractor: StyleFeatureExtractor | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self._persist_path = Path(chroma_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        chromadb: Any = importlib.import_module("chromadb")
        self._client: Any = chromadb.PersistentClient(path=str(self._persist_path))
        self._collection: Any = self._client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name
        self._embedding_client = embedding_client
        self._top_k = top_k
        self._min_samples = min_samples
        self._feature_extractor = feature_extractor or StyleFeatureExtractor()

    def index_samples(self, samples: Sequence[ProcessedStyleSample]) -> int:
        if not samples:
            return 0
        embeddings = self._embedding_client.embed_texts([sample.text for sample in samples])
        self._collection.upsert(
            ids=[sample.sample_id for sample in samples],
            documents=[sample.text for sample in samples],
            embeddings=embeddings,
            metadatas=[_metadata_for_sample(sample) for sample in samples],
        )
        return len(samples)

    def retrieve_style(
        self,
        *,
        persona_id: str,
        query: str,
        top_k: int | None = None,
        consent_ids: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> StyleRetrieval:
        effective_top_k = top_k or self._top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = self._query_candidates(
            persona_id=persona_id,
            query=query,
            top_k=effective_top_k,
        )
        allowed_consents = set(consent_ids) if consent_ids is not None else None
        filtered = [
            candidate
            for candidate in candidates
            if candidate.persona_id == persona_id
            and (not active_only or candidate.active)
            and (allowed_consents is None or candidate.consent_id in allowed_consents)
        ][:effective_top_k]

        feature_samples = [_sample_from_result(result) for result in filtered]
        features = self._feature_extractor.extract(feature_samples)
        if len(filtered) < self._min_samples:
            return StyleRetrieval(
                results=[],
                trace=RetrievalTrace(
                    query=query,
                    top_k=effective_top_k,
                    result_count=0,
                    collection=self.collection_name,
                    chunk_ids=[],
                ),
                features=features,
                summary=StyleSummary(
                    persona_id=persona_id,
                    sample_count=len(filtered),
                    text="insufficient authorized style samples",
                    examples=[],
                ),
                fallback_reason="insufficient_authorized_style_samples",
            )

        summary = _make_style_summary(persona_id=persona_id, results=filtered, features=features)
        return StyleRetrieval(
            results=filtered,
            trace=RetrievalTrace(
                query=query,
                top_k=effective_top_k,
                result_count=len(filtered),
                collection=self.collection_name,
                chunk_ids=[result.sample_id for result in filtered],
            ),
            features=features,
            summary=summary,
        )

    def _query_candidates(
        self,
        *,
        persona_id: str,
        query: str,
        top_k: int,
    ) -> list[RetrievedStyleSample]:
        if int(self._collection.count()) == 0:
            return []
        query_embedding = self._embedding_client.embed_query(query)
        count = int(self._collection.count())
        n_results = min(count, max(top_k * 10, self._min_samples, top_k))
        raw_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"persona_id": persona_id},
            include=["documents", "metadatas", "distances"],
        )
        return _parse_query_result(raw_result)


def _metadata_for_sample(sample: ProcessedStyleSample) -> dict[str, MetadataValue]:
    return {
        "sample_id": sample.sample_id,
        "record_id": sample.record_id,
        "consent_id": sample.consent_id,
        "persona_id": sample.persona_id,
        "speaker_user_id": sample.speaker_user_id,
        "source": sample.source,
        "active": sample.active,
        "revoked": sample.revoked,
        "timestamp_ms": sample.timestamp_ms,
    }


def _parse_query_result(raw_result: object) -> list[RetrievedStyleSample]:
    if not isinstance(raw_result, Mapping):
        return []
    ids = _first_list(raw_result.get("ids"))
    documents = _first_list(raw_result.get("documents"))
    metadatas = _first_list(raw_result.get("metadatas"))
    distances = _first_list(raw_result.get("distances"))

    results: list[RetrievedStyleSample] = []
    for index, sample_id_value in enumerate(ids):
        metadata = _metadata_at(metadatas, index)
        sample_id = str(metadata.get("sample_id", sample_id_value))
        consent_id = str(metadata.get("consent_id", ""))
        persona_id = str(metadata.get("persona_id", ""))
        document = str(documents[index]) if index < len(documents) else ""
        if not sample_id or not consent_id or not persona_id or not document:
            continue
        distance = _float_at(distances, index)
        results.append(
            RetrievedStyleSample(
                sample_id=sample_id,
                consent_id=consent_id,
                persona_id=persona_id,
                text=document,
                active=bool(metadata.get("active", True)),
                score=1.0 / (1.0 + max(distance, 0.0)),
                metadata=metadata,
            )
        )
    return results


def _sample_from_result(result: RetrievedStyleSample) -> ProcessedStyleSample:
    speaker_user_id = result.metadata.get("speaker_user_id", 0)
    timestamp_ms = result.metadata.get("timestamp_ms", 0)
    return ProcessedStyleSample(
        sample_id=result.sample_id,
        record_id=str(result.metadata.get("record_id", result.sample_id)),
        consent_id=result.consent_id,
        persona_id=result.persona_id,
        speaker_user_id=int(speaker_user_id) if isinstance(speaker_user_id, int) else 1,
        source=str(result.metadata.get("source", "style_collection")),
        text=result.text,
        allowed_usage=["style_simulation"],
        forbidden_usage=[],
        active=result.active,
        revoked=bool(result.metadata.get("revoked", False)),
        pii_redactions={"email": 0, "phone": 0, "id_card": 0},
        timestamp_ms=int(timestamp_ms) if isinstance(timestamp_ms, int) else 0,
    )


def _make_style_summary(
    *,
    persona_id: str,
    results: Sequence[RetrievedStyleSample],
    features: StyleFeatures,
) -> StyleSummary:
    tone = _top_items_text(features.tone_particle_counts)
    catchphrases = _top_items_text(features.catchphrase_counts)
    punctuation = _top_items_text(features.punctuation_counts)
    text = (
        f"persona={persona_id}; samples={features.sample_count}; "
        f"avg_sentence_length={features.average_sentence_length}; "
        f"emoji_per_sample={features.emoji_per_sample}; "
        f"punctuation={punctuation}; tone_particles={tone}; catchphrases={catchphrases}"
    )
    return StyleSummary(
        persona_id=persona_id,
        sample_count=features.sample_count,
        text=text,
        examples=[result.text for result in results],
    )


def _top_items_text(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(f"{key}:{value}" for key, value in ordered[:5])


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
