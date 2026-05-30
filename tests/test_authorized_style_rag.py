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
    source: str = "fixture/style.jsonl",
    allowed_usage: list[str] | None = None,
) -> ProcessedStyleSample:
    return ProcessedStyleSample(
        sample_id=sample_id,
        record_id=f"raw-{sample_id}",
        consent_id=consent_id,
        persona_id=persona_id,
        speaker_user_id=int(persona_id) if persona_id.isdigit() else 1002,
        source=source,
        text=text,
        allowed_usage=allowed_usage or ["style_simulation"],
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


def test_style_text_filter_rejects_ocr_noise_but_keeps_chatty_short_replies() -> None:
    from agent_service.style.filters import is_learnable_style_text

    rejected = [
        "",
        "加-",
        "￥439 *",
        "2025年5月18日 14:57",
        "26/4/23",
        "期四",
        "惠",
        "res += maxR - height[right]; right--; // 右",
        "height[st.top()])",
        "public:",
        "[图片]",
        "mock reply: 你好",
    ]
    accepted = [
        "嗯",
        "示例回复二",
        "示例回复一",
        "示例上下文",
        "这个确实有点烦，但是能搞",
    ]

    assert [is_learnable_style_text(text) for text in rejected] == [False] * len(rejected)
    assert [is_learnable_style_text(text) for text in accepted] == [True] * len(accepted)


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


def test_workflow_can_apply_configured_style_persona_to_smalltalk(
    tmp_path: Path,
) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _style_store(tmp_path, top_k=2)
    store.index_samples(
        [
            _sample("style-demo_persona-a", persona_id="demo_persona", text="当前用户。"),
            _sample("style-demo_persona-b", persona_id="demo_persona", text="示例上下文。"),
            _sample("style-1002", persona_id="1002", text="这条不是目标 persona。"),
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("你在干嘛")),
        style_store=store,
        style_top_k=2,
        style_persona_id="demo_persona",
        style_on_smalltalk=True,
    )

    assert state["decision"].need_style is False
    assert state["retrieval_trace"][0].collection == "style"
    assert set(state["retrieval_trace"][0].chunk_ids) == {
        "style-demo_persona-a",
        "style-demo_persona-b",
    }
    assert "persona=demo_persona" in state["retrieved_context"][0]


def test_create_app_indexes_configured_authorized_style_samples(tmp_path: Path) -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app

    samples_path = tmp_path / "style_samples.local.jsonl"
    samples = [
        _sample(
            "style-demo_persona-a",
            persona_id="demo_persona",
            text="先别慌嘛，慢慢来。",
        ),
        _sample(
            "style-demo_persona-b",
            persona_id="demo_persona",
            text="这个确实有点烦，但是能搞。",
        ),
    ]
    samples_path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples),
        encoding="utf-8",
    )

    app = create_app(
        Settings(
            _env_file=None,
            embedding_provider="mock",
            chroma_path=str(tmp_path / "chroma"),
            agent_state_db_path=str(tmp_path / "agent_state.sqlite3"),
            memory_db_path=str(tmp_path / "memory.sqlite3"),
            knowledge_docs_path=str(tmp_path / "knowledge_docs"),
            style_samples_path=str(samples_path),
        )
    )

    retrieval = app.state.get_container().style_store.retrieve_style(
        persona_id="demo_persona",
        query="项目有点烦",
        top_k=2,
    )

    assert retrieval.fallback_reason is None
    assert set(retrieval.trace.chunk_ids) == {"style-demo_persona-a", "style-demo_persona-b"}


def test_workflow_can_apply_configured_style_persona_to_knowledge_chat(
    tmp_path: Path,
) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _style_store(tmp_path, top_k=2)
    store.index_samples(
        [
            _sample(
                "style-demo_persona-a",
                persona_id="demo_persona",
                text="先别慌嘛，慢慢来。",
            ),
            _sample(
                "style-demo_persona-b",
                persona_id="demo_persona",
                text="这个确实有点烦，但是能搞。",
            ),
            _sample("style-1002", persona_id="1002", text="这条不是目标 persona。"),
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("我在做项目，马上要面试了")),
        style_store=store,
        style_top_k=2,
        style_persona_id="demo_persona",
        style_on_private_chat=True,
    )

    assert state["decision"].need_knowledge is True
    assert [trace.collection for trace in state["retrieval_trace"]] == [
        "style",
    ]
    assert set(state["retrieval_trace"][0].chunk_ids) == {
        "style-demo_persona-a",
        "style-demo_persona-b",
    }
    assert any(
        event.node == "retrieve_context" and "style_top_k=2" in event.action
        for event in state["trace"]
    )


def test_workflow_keeps_style_context_when_private_memory_query_uses_memory_rag(
    tmp_path: Path,
) -> None:
    from agent_service.memory.memory_store import MemoryStore
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    memory_store = MemoryStore(
        sqlite_path=tmp_path / "memory.sqlite3",
        chroma_path=tmp_path / "memory_chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=3,
    )
    memory = memory_store.save_memory(
        user_id=1002,
        content="演示用户生日是1月1日",
        source_message_id=6999,
    )
    style_store = _style_store(tmp_path, top_k=2)
    style_store.index_samples(
        [
            _sample("style-demo_persona-a", persona_id="demo_persona", text="记得呀。"),
            _sample("style-demo_persona-b", persona_id="demo_persona", text="这个我不能记错。"),
        ]
    )

    payload = _chat_payload("你记得我生日吗")
    payload["message_id"] = 7002

    state = run_agent_workflow(
        ChatRequest.model_validate(payload),
        memory_store=memory_store,
        style_store=style_store,
        memory_top_k=3,
        style_top_k=2,
        style_persona_id="demo_persona",
        style_on_private_chat=True,
    )

    assert state["decision"].need_memory is True
    assert [trace.collection for trace in state["retrieval_trace"]] == ["memory", "style"]
    assert state["retrieval_trace"][0].chunk_ids == [memory.memory_id]
    assert set(state["retrieval_trace"][1].chunk_ids) == {
        "style-demo_persona-a",
        "style-demo_persona-b",
    }
    assert f"memory: {memory.content}" in state["retrieved_context"]
    assert any(item.startswith("style_summary:") for item in state["retrieved_context"])
    assert any(
        event.node == "retrieve_context"
        and "memory_top_k=1" in event.action
        and "style_top_k=2" in event.action
        for event in state["trace"]
    )


def test_workflow_retrieves_authorized_style_pairs_for_private_chat(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.style.pair_store import StylePairStore
    from agent_service.workflow import run_agent_workflow

    pairs_path = tmp_path / "style_pairs.local.jsonl"
    pairs_path.write_text(
        "\n".join(
            [
                (
                    '{"pair_id":"pair-a","persona_id":"demo_persona","self_speaker":"当前用户",'
                    '"target_speaker":"目标样本","self_text":"我马上要面试了",'
                    '"target_reply":"别慌，你先把最稳的讲熟","timestamp_ms":1}'
                ),
                (
                    '{"pair_id":"pair-noise","persona_id":"demo_persona","self_speaker":"当前用户",'
                    '"target_speaker":"目标样本","self_text":"这个多少钱",'
                    '"target_reply":"￥439 *","timestamp_ms":2}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pair_store = StylePairStore.from_jsonl(pairs_path)
    style_store = _style_store(tmp_path, top_k=2, min_samples=1)
    style_store.index_samples(
        [_sample("style-demo_persona-a", persona_id="demo_persona", text="先别慌嘛")]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("我马上要面试了，有点紧张")),
        style_store=style_store,
        style_pair_store=pair_store,
        style_top_k=1,
        style_pair_top_k=2,
        style_persona_id="demo_persona",
        style_on_private_chat=True,
    )

    assert any(item.startswith("style_pair:pair-a:") for item in state["retrieved_context"])
    assert all("￥439" not in item for item in state["retrieved_context"])
    assert any(
        event.node == "retrieve_context" and "style_pair_top_k=1" in event.action
        for event in state["trace"]
    )


def test_style_pair_store_prefers_strong_phrase_overlap(tmp_path: Path) -> None:
    from agent_service.style.pair_store import StylePairStore

    pairs_path = tmp_path / "style_pairs.local.jsonl"
    pairs_path.write_text(
        "\n".join(
            [
                (
                    '{"pair_id":"sleep","persona_id":"demo_persona","self_speaker":"当前用户",'
                    '"target_speaker":"目标样本","self_text":"我马上睡觉",'
                    '"target_reply":"11点不上床","timestamp_ms":3}'
                ),
                (
                    '{"pair_id":"interview","persona_id":"demo_persona","self_speaker":"当前用户",'
                    '"target_speaker":"目标样本","self_text":"准备面试了",'
                    '"target_reply":"别慌，你先把最稳的讲熟","timestamp_ms":1}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = StylePairStore.from_jsonl(pairs_path).retrieve_pairs(
        persona_id="demo_persona",
        query="我马上要面试了，有点紧张",
        top_k=2,
    )

    assert [item.pair.pair_id for item in result.results] == ["interview", "sleep"]


def test_style_store_does_not_index_agent_runtime_replies_as_authorized_style(
    tmp_path: Path,
) -> None:
    style_store = _style_store(tmp_path, min_samples=1)

    indexed = style_store.index_samples(
        [
            _sample(
                "runtime-style-a",
                persona_id="demo_persona",
                text="记住啦，1月1日是你的生日。",
                source="agent_runtime_reply:26879",
                consent_id="consent-demo-persona-runtime-style",
            )
        ]
    )
    retrieval = style_store.retrieve_style(persona_id="demo_persona", query="生日", top_k=1)

    assert indexed == 0
    assert retrieval.results == []
    assert retrieval.fallback_reason == "insufficient_authorized_style_samples"


def test_style_learning_store_appends_feedback_without_indexing_it(
    tmp_path: Path,
) -> None:
    from agent_service.style.learning import StyleLearningStore

    feedback_path = tmp_path / "runtime_style_feedback.local.jsonl"
    style_store = _style_store(tmp_path, min_samples=1)
    learner = StyleLearningStore(
        samples_path=feedback_path,
        style_store=style_store,
        persona_id="demo_persona",
        consent_id="consent-demo-persona-runtime-style",
        subject_user_id=10001,
    )

    sample = learner.learn_reply(
        text="好呀，先休息一下",
        source_message_id=8801,
        timestamp_ms=1_700_000_001_000,
    )
    duplicate = learner.learn_reply(
        text="好呀，先休息一下",
        source_message_id=8801,
        timestamp_ms=1_700_000_001_000,
    )
    retrieval = style_store.retrieve_style(persona_id="demo_persona", query="注意休息", top_k=1)

    assert duplicate is None
    assert sample is not None
    assert sample.sample_id == "runtime-style-consent-demo-persona-runtime-style-8801"
    assert sample.text == "好呀，先休息一下"
    assert sample.allowed_usage == ["style_feedback"]
    assert feedback_path.read_text(encoding="utf-8").count("\n") == 1
    assert retrieval.results == []
    assert retrieval.fallback_reason == "insufficient_authorized_style_samples"


def test_style_learning_store_rejects_safety_rewrite_feedback(tmp_path: Path) -> None:
    from agent_service.safety.verbatim_guard import SAFE_REWRITE_TEXT
    from agent_service.style.learning import StyleLearningStore

    feedback_path = tmp_path / "runtime_style_feedback.local.jsonl"
    learner = StyleLearningStore(
        samples_path=feedback_path,
        style_store=_style_store(tmp_path, min_samples=1),
        persona_id="demo_persona",
        consent_id="consent-demo-persona-runtime-style",
        subject_user_id=10001,
    )

    sample = learner.learn_reply(
        text=SAFE_REWRITE_TEXT,
        source_message_id=8802,
        timestamp_ms=1_700_000_001_000,
    )

    assert sample is None
    assert not feedback_path.exists()


def test_workflow_reinforces_style_after_safe_smalltalk_reply(
    tmp_path: Path,
) -> None:
    from agent_service.llm import MockLLMClient
    from agent_service.schemas import ChatRequest
    from agent_service.style.learning import StyleLearningStore
    from agent_service.workflow import run_agent_workflow

    samples_path = tmp_path / "runtime_style_feedback.local.jsonl"
    style_store = _style_store(tmp_path, top_k=2, min_samples=1)
    style_store.index_samples(
        [_sample("style-seed", persona_id="demo_persona", text="好困，先睡觉")]
    )
    learner = StyleLearningStore(
        samples_path=samples_path,
        style_store=style_store,
        persona_id="demo_persona",
        consent_id="consent-demo-persona-runtime-style",
        subject_user_id=10001,
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("你在干嘛")),
        style_store=style_store,
        style_top_k=2,
        style_persona_id="demo_persona",
        style_on_smalltalk=True,
        llm_client=MockLLMClient({"reply_text": "刚刷完题，准备躺会儿"}),
        style_learning_store=learner,
    )

    assert state["final_command"].should_send is True
    assert samples_path.exists()
    assert "刚刷完题，准备躺会儿" in samples_path.read_text(encoding="utf-8")
    assert any(
        event.node == "finalize_reply" and event.action == "style_feedback_saved=1"
        for event in state["trace"]
    )
