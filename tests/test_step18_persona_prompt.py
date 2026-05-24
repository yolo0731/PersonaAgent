from __future__ import annotations

from pathlib import Path

import pytest


def _chat_payload(
    text: str = "请回答项目问题",
    *,
    run_id: str = "run-persona",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def test_default_persona_yaml_loads_versioned_identity_notice() -> None:
    from agent_service.persona.engine import PersonaEngine

    engine = PersonaEngine.from_default()

    assert engine.config.prompt_version == "persona-v1"
    assert "AI Agent" in engine.config.identity_notice
    assert engine.config.style_instruction
    assert engine.config.safety_boundaries


def test_persona_yaml_requires_identity_notice(tmp_path: Path) -> None:
    from agent_service.persona.engine import PersonaConfigError, PersonaEngine

    config_path = tmp_path / "persona.yaml"
    config_path.write_text(
        """
persona_id: persona-agent
display_name: PersonaAgent
prompt_version: persona-v1
style_instruction: Keep answers concise.
safety_boundaries:
  - Do not impersonate a real person.
prompt_templates:
  system: "{identity_notice}"
  developer: "{context_block}"
  user: "{user_text}"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PersonaConfigError, match="identity_notice"):
        PersonaEngine.from_file(config_path)


def test_persona_prompt_contains_context_sections_and_metadata() -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.rag.documents import RetrievalTrace
    from agent_service.schemas import ChatRequest

    prompt = PersonaEngine.from_default().build_prompt(
        request=ChatRequest.model_validate(_chat_payload("LiteIM 项目是什么？")),
        retrieved_context=[
            "knowledge: LiteIM uses epoll and one-loop-per-thread Reactor.",
            "memory: 用户喜欢先给结论",
            "style_summary: samples=2; avg_sentence_length=12.0",
            "style_example:style-a: 好的，我先说重点。",
        ],
        retrieval_trace=[
            RetrievalTrace(
                query="LiteIM 项目是什么？",
                top_k=3,
                result_count=1,
                collection="knowledge",
                chunk_ids=["knowledge-doc-1"],
            ),
            RetrievalTrace(
                query="LiteIM 项目是什么？",
                top_k=3,
                result_count=1,
                collection="memory",
                chunk_ids=["mem-1002-7001"],
            ),
            RetrievalTrace(
                query="LiteIM 项目是什么？",
                top_k=3,
                result_count=1,
                collection="style",
                chunk_ids=["style-a"],
            ),
        ],
    )

    joined = "\n".join(message.content for message in prompt.messages)

    assert "PersonaAgent is an AI Agent" in joined
    assert "Knowledge context" in joined
    assert "LiteIM uses epoll" in joined
    assert "Memory context" in joined
    assert "用户喜欢先给结论" in joined
    assert "Style guidance" in joined
    assert "avg_sentence_length=12.0" in joined
    assert "Do not claim to be a real person" in joined
    assert prompt.metadata.prompt_version == "persona-v1"
    assert prompt.metadata.used_context_ids == ["knowledge-doc-1", "mem-1002-7001", "style-a"]


def test_persona_prompt_uses_style_fallback_when_samples_are_empty() -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.schemas import ChatRequest

    prompt = PersonaEngine.from_default().build_prompt(
        request=ChatRequest.model_validate(_chat_payload("请正常回复")),
        retrieved_context=["style_fallback: insufficient_authorized_style_samples"],
        retrieval_trace=[],
    )

    joined = "\n".join(message.content for message in prompt.messages)

    assert "No authorized style samples are available" in joined
    assert "Keep a clear, concise, and non-impersonating assistant style." in joined


def test_workflow_records_prompt_metadata_and_version_trace() -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("hello persona", run_id="run-persona-workflow")),
        persona_engine=PersonaEngine.from_default(),
    )

    assert state["prompt_metadata"] is not None
    assert state["prompt_metadata"].prompt_version == "persona-v1"
    assert state["prompt_messages"]
    assert any("prompt_version=persona-v1" in event.action for event in state["trace"])
