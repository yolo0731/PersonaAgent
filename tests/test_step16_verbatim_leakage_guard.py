from __future__ import annotations

from pathlib import Path

from agent_service.governance.data_manifest import ProcessedStyleSample


def _source_text() -> str:
    return "好呀！我会直接说重点。"


def _style_sample(sample_id: str = "style-a", text: str | None = None) -> ProcessedStyleSample:
    return ProcessedStyleSample(
        sample_id=sample_id,
        record_id=f"raw-{sample_id}",
        consent_id="consent-1002-style",
        persona_id="1002",
        speaker_user_id=1002,
        source="fixture/style.jsonl",
        text=text or _source_text(),
        allowed_usage=["style_simulation"],
        forbidden_usage=[],
        active=True,
        revoked=False,
        pii_redactions={"email": 0, "phone": 0, "id_card": 0},
        timestamp_ms=1_700_000_001_000,
    )


def _chat_payload(text: str) -> dict[str, object]:
    return {
        "run_id": "run-leakage",
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def _style_store(tmp_path: Path):
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.style.style_store import StyleStore

    return StyleStore(
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=2,
        min_samples=1,
    )


def test_direct_style_sample_copy_is_blocked() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    guard = VerbatimLeakageGuard()
    assessment = guard.assess(
        _source_text(),
        [LeakageSource(source_id="style-a", text=_source_text())],
    )

    assert assessment.action == "block"
    assert assessment.reason == "direct_verbatim_copy"
    assert assessment.safe_text == ""
    assert assessment.metrics.verbatim_leakage_rate == 1.0
    assert assessment.metrics.max_ngram_overlap == 1.0
    assert assessment.metrics.source_ids == ["style-a"]


def test_high_overlap_reply_is_rewritten_without_copying_sample_text() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    source = "好呀，我会直接说重点，然后给你一个简短答案"
    draft = "好呀，我会直接说重点，然后给你一个简短说明"

    assessment = VerbatimLeakageGuard().assess(
        draft,
        [LeakageSource(source_id="style-a", text=source)],
    )

    assert assessment.action == "rewrite"
    assert assessment.reason == "high_verbatim_overlap"
    assert assessment.safe_text == "我会保持相近的简洁语气，但不会复述授权样本原文。"
    assert "好呀，我会直接说重点" not in assessment.safe_text
    assert assessment.metrics.max_ngram_overlap > 0.45


def test_pii_leak_is_blocked_even_without_style_overlap() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    assessment = VerbatimLeakageGuard().assess(
        "可以联系 alice@example.com，我稍后处理。",
        [LeakageSource(source_id="style-a", text=_source_text())],
    )

    assert assessment.action == "block"
    assert assessment.reason == "pii_leak_detected"
    assert assessment.metrics.pii_leak_count == 1


def test_style_source_id_leak_is_blocked() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    assessment = VerbatimLeakageGuard().assess(
        "这句话参考了 [style-source:style-a] 的写法。",
        [LeakageSource(source_id="style-a", text=_source_text())],
    )

    assert assessment.action == "block"
    assert assessment.reason == "style_source_id_leak"
    assert assessment.metrics.source_ids == ["style-a"]


def test_style_similar_but_non_verbatim_reply_passes() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    assessment = VerbatimLeakageGuard().assess(
        "可以，我先给结论，再补充两个要点。",
        [LeakageSource(source_id="style-a", text=_source_text())],
    )

    assert assessment.action == "pass"
    assert assessment.safe_text == "可以，我先给结论，再补充两个要点。"
    assert assessment.metrics.verbatim_leakage_rate == 0.0
    assert assessment.metrics.pii_leak_count == 0


def test_short_common_style_phrases_do_not_trigger_verbatim_rewrite() -> None:
    from agent_service.safety.verbatim_guard import LeakageSource, VerbatimLeakageGuard

    assessment = VerbatimLeakageGuard().assess(
        "记得，1月1日。",
        [LeakageSource(source_id="style-short", text="记得")],
    )

    assert assessment.action == "pass"
    assert assessment.safe_text == "记得，1月1日。"
    assert assessment.metrics.source_ids == []


def test_workflow_rewrites_draft_that_repeats_retrieved_style_sample(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _style_store(tmp_path)
    store.index_samples([_style_sample(), _style_sample("style-b", text="收到啦，我会先说重点。")])

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload(f"请按我的风格回复：{_source_text()}")),
        style_store=store,
        style_top_k=2,
    )

    assert state["decision"].need_style is True
    assert state["safety_result"].blocked is False
    assert state["safety_result"].reason == "direct_verbatim_copy"
    assert state["final_command"].should_send is True
    assert state["final_command"].reason == "finalized_reply"
    assert state["final_command"].text == "我会保持相近的简洁语气，但不会复述授权样本原文。"
    assert any(event.action == "rewritten:direct_verbatim_copy" for event in state["trace"])
