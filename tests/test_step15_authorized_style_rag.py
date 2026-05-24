from __future__ import annotations

from pathlib import Path

from agent_service.governance.data_manifest import ProcessedStyleSample


def _chat_payload(text: str = "请按我的风格回复一下") -> dict[str, object]:
    return {
        "run_id": "run-style",
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def _sample(
    sample_id: str,
    *,
    persona_id: str = "1002",
    consent_id: str = "consent-1002-style",
    text: str = "好呀！收到啦～",
    active: bool = True,
) -> ProcessedStyleSample:
    return ProcessedStyleSample(
        sample_id=sample_id,
        record_id=f"raw-{sample_id}",
        consent_id=consent_id,
        persona_id=persona_id,
        speaker_user_id=int(persona_id),
        source="fixture/style.jsonl",
        text=text,
        allowed_usage=["style_simulation"],
        forbidden_usage=[],
        active=active,
        revoked=not active,
        pii_redactions={"email": 0, "phone": 0, "id_card": 0},
        timestamp_ms=1_700_000_001_000,
    )


def _style_store(tmp_path: Path, *, top_k: int = 5, min_samples: int = 2):
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.style.style_store import StyleStore

    return StyleStore(
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=top_k,
        min_samples=min_samples,
    )


def test_style_retrieval_is_persona_scoped_and_excludes_inactive(tmp_path: Path) -> None:
    store = _style_store(tmp_path)
    store.index_samples(
        [
            _sample("style-1002-active", text="好呀，我会温和直接地回复。"),
            _sample("style-1002-active-2", text="收到啦，我会先说重点。"),
            _sample("style-1002-inactive", text="这条 inactive 风格不能被检索。", active=False),
            _sample("style-2002-active", persona_id="2002", text="另一位用户的辛辣详细风格。"),
        ]
    )

    result = store.retrieve_style(persona_id="1002", query="温和直接回复", top_k=5)

    assert result.fallback_reason is None
    assert {item.persona_id for item in result.results} == {"1002"}
    assert {item.sample_id for item in result.results} == {
        "style-1002-active",
        "style-1002-active-2",
    }
    assert all(item.active for item in result.results)
    assert result.trace.collection == "style"
    assert set(result.trace.chunk_ids) == {"style-1002-active", "style-1002-active-2"}


def test_style_retrieval_filters_by_consent_id(tmp_path: Path) -> None:
    store = _style_store(tmp_path, min_samples=1)
    store.index_samples(
        [
            _sample("style-consent-a", consent_id="consent-a", text="好呀，收到啦。"),
            _sample("style-consent-b", consent_id="consent-b", text="没问题，马上安排。"),
        ]
    )

    result = store.retrieve_style(
        persona_id="1002",
        query="收到 安排",
        top_k=5,
        consent_ids=["consent-a"],
    )

    assert [item.consent_id for item in result.results] == ["consent-a"]
    assert result.trace.chunk_ids == ["style-consent-a"]


def test_style_feature_extractor_computes_deterministic_stats() -> None:
    from agent_service.style.features import StyleFeatureExtractor

    features = StyleFeatureExtractor().extract(
        [
            _sample("style-1", text="好呀！🙂"),
            _sample("style-2", text="收到啦！🙂🙂"),
        ]
    )

    assert features.sample_count == 2
    assert features.average_sentence_length == 2.5
    assert features.punctuation_counts == {"！": 2}
    assert features.emoji_count == 3
    assert features.emoji_per_sample == 1.5
    assert features.tone_particle_counts == {"呀": 1, "啦": 1}
    assert features.catchphrase_counts == {"好呀": 1, "收到": 1}
    assert features.reply_length_distribution.min_chars == 4
    assert features.reply_length_distribution.max_chars == 6
    assert features.reply_length_distribution.average_chars == 5.0
    assert features.reply_length_distribution.short_count == 2


def test_style_retrieval_returns_fallback_when_authorized_samples_are_insufficient(
    tmp_path: Path,
) -> None:
    store = _style_store(tmp_path, min_samples=2)
    store.index_samples([_sample("style-only-one", text="好呀，收到啦。")])

    result = store.retrieve_style(persona_id="1002", query="请按我的风格回复", top_k=5)

    assert result.results == []
    assert result.summary.examples == []
    assert result.summary.sample_count == 1
    assert result.fallback_reason == "insufficient_authorized_style_samples"
    assert result.trace.result_count == 0


def test_workflow_injects_style_summary_examples_and_trace(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _style_store(tmp_path, top_k=2)
    store.index_samples(
        [
            _sample("style-a", text="好呀！我会直接说重点。"),
            _sample("style-b", text="收到啦！我的邮箱是 [REDACTED_EMAIL]。"),
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload()),
        style_store=store,
        style_top_k=2,
    )

    assert state["decision"].need_style is True
    assert len(state["retrieval_trace"]) == 1
    assert state["retrieval_trace"][0].collection == "style"
    assert set(state["retrieval_trace"][0].chunk_ids) == {"style-a", "style-b"}
    assert state["retrieved_context"][0].startswith("style_summary:")
    assert all("style_example:" in item for item in state["retrieved_context"][1:])
    assert any("[REDACTED_EMAIL]" in item for item in state["retrieved_context"])
    assert all("alice@example.com" not in item for item in state["retrieved_context"])
    assert [event.action for event in state["trace"] if event.node == "retrieve_context"] == [
        "style_top_k=2"
    ]
