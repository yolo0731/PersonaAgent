from __future__ import annotations

from pathlib import Path


def _chat_payload(
    text: str,
    *,
    run_id: str = "run-memory",
    sender_id: int = 1002,
    message_id: int = 7001,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": message_id,
        "sender_id": sender_id,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": f"alice-{message_id}",
    }


def _memory_store(tmp_path: Path):
    from agent_service.memory.memory_store import MemoryStore
    from agent_service.rag.embeddings import MockEmbeddingClient

    return MemoryStore(
        sqlite_path=tmp_path / "memory.sqlite3",
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=3,
    )


def test_memory_store_saves_lists_and_deactivates_records(tmp_path: Path) -> None:
    from agent_service.memory.memory_store import MemoryStore

    store: MemoryStore = _memory_store(tmp_path)

    record = store.save_memory(
        user_id=1002,
        content="我喜欢清淡直接的回答",
        source_message_id=7001,
        importance=0.8,
    )

    assert record.memory_id == "mem-1002-7001"
    assert record.user_id == 1002
    assert record.content == "我喜欢清淡直接的回答"
    assert record.source_message_id == 7001
    assert record.active is True
    assert record.importance == 0.8
    assert record.created_at
    assert store.list_memories(user_id=1002) == [record]

    deactivated = store.deactivate_memory(record.memory_id, user_id=1002)

    assert deactivated.active is False
    assert store.list_memories(user_id=1002) == []
    assert store.list_memories(user_id=1002, active_only=False)[0].active is False


def test_memory_retrieval_is_user_scoped_and_excludes_inactive(tmp_path: Path) -> None:
    store = _memory_store(tmp_path)
    alice = store.save_memory(
        user_id=1002,
        content="我喜欢清淡直接的回答",
        source_message_id=7001,
    )
    store.save_memory(
        user_id=2002,
        content="我喜欢辛辣详细的回答",
        source_message_id=8001,
    )

    alice_result = store.retrieve_memory(user_id=1002, query="清淡回答", top_k=3)
    bob_result = store.retrieve_memory(user_id=2002, query="清淡回答", top_k=3)

    assert [item.memory_id for item in alice_result.results] == [alice.memory_id]
    assert {item.user_id for item in alice_result.results} == {1002}
    assert {item.user_id for item in bob_result.results} == {2002}
    assert alice_result.trace.chunk_ids == [alice.memory_id]

    store.deactivate_memory(alice.memory_id, user_id=1002)
    inactive_result = store.retrieve_memory(user_id=1002, query="清淡回答", top_k=3)

    assert inactive_result.results == []
    assert inactive_result.trace.result_count == 0


def test_remember_instruction_saves_memory_in_workflow(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _memory_store(tmp_path)

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("/remember 我喜欢清淡回答")),
        memory_store=store,
        memory_top_k=3,
    )

    memories = store.list_memories(user_id=1002)
    assert len(memories) == 1
    assert memories[0].content == "我喜欢清淡回答"
    assert state["retrieved_context"] == ["memory_saved: 我喜欢清淡回答"]
    assert state["retrieval_trace"][0].chunk_ids == [memories[0].memory_id]
    assert [event.action for event in state["trace"] if event.node == "retrieve_context"] == [
        "memory_saved"
    ]


def test_suffix_remember_instruction_saves_memory_in_workflow(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _memory_store(tmp_path)

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("对的，1月1日是我的生日，记住")),
        memory_store=store,
        memory_top_k=3,
    )

    memories = store.list_memories(user_id=1002)
    assert len(memories) == 1
    assert memories[0].content == "对的，1月1日是我的生日"
    assert state["retrieved_context"] == ["memory_saved: 对的，1月1日是我的生日"]


def test_forget_instruction_deactivates_memory_in_workflow(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _memory_store(tmp_path)
    record = store.save_memory(
        user_id=1002,
        content="我喜欢清淡回答",
        source_message_id=7000,
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload(f"/forget {record.memory_id}")),
        memory_store=store,
        memory_top_k=3,
    )

    assert store.list_memories(user_id=1002) == []
    assert store.list_memories(user_id=1002, active_only=False)[0].active is False
    assert state["retrieved_context"] == [f"memory_deactivated: {record.memory_id}"]
    assert [event.action for event in state["trace"] if event.node == "retrieve_context"] == [
        "memory_deactivated"
    ]


def test_memory_query_injects_user_memory_into_workflow_context(tmp_path: Path) -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _memory_store(tmp_path)
    record = store.save_memory(
        user_id=1002,
        content="我喜欢清淡回答",
        source_message_id=7001,
    )
    store.save_memory(
        user_id=2002,
        content="我喜欢辛辣回答",
        source_message_id=8001,
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("我之前让你记住了什么？", message_id=7002)),
        memory_store=store,
        memory_top_k=3,
    )

    assert state["retrieved_context"] == ["memory: 我喜欢清淡回答"]
    assert state["retrieval_trace"][0].chunk_ids == [record.memory_id]
    assert [event.action for event in state["trace"] if event.node == "retrieve_context"] == [
        "memory_top_k=1"
    ]


def test_smalltalk_reads_prior_memory_and_saves_new_chat_turn(tmp_path: Path) -> None:
    from agent_service.llm import MockLLMClient
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    store = _memory_store(tmp_path)
    prior = store.save_memory(
        user_id=1002,
        content="演示用户最近在准备项目演示",
        source_message_id=6999,
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("今天有点紧张", message_id=7002)),
        memory_store=store,
        memory_top_k=3,
        llm_client=MockLLMClient({"reply_text": "没事，先把能讲清楚的讲清楚"}),
        auto_memory_on_chat=True,
        auto_memory_user_name="演示用户",
        auto_memory_persona_name="示例伙伴",
    )

    memories = store.list_memories(user_id=1002)
    contents = {memory.content for memory in memories}

    assert f"memory: {prior.content}" in state["retrieved_context"]
    assert "演示用户说：今天有点紧张" in contents
    assert "示例伙伴回复：没事，先把能讲清楚的讲清楚" in contents
    assert any(
        event.node == "finalize_reply" and event.action == "auto_memory_saved=2"
        for event in state["trace"]
    )
