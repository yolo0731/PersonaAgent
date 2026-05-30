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
            "style_pair:pair-a: 演示用户: 我马上要面试了 -> 示例伙伴: 别慌，你先把最稳的讲熟",
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
    assert "Authorized style examples" in joined
    assert "好的，我先说重点" in joined
    assert "Relevant dialogue pairs" in joined
    assert "我马上要面试了" in joined
    assert "别慌，你先把最稳的讲熟" in joined
    assert "Do not claim to be a real person" in joined
    assert prompt.metadata.prompt_version == "persona-v1"
    assert prompt.metadata.used_context_ids == ["knowledge-doc-1", "mem-1002-7001", "style-a"]


def test_persona_prompt_filters_noisy_style_examples_before_prompting() -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.schemas import ChatRequest

    prompt = PersonaEngine.from_default().build_prompt(
        request=ChatRequest.model_validate(_chat_payload("我马上要面试了")),
        retrieved_context=[
            "style_summary: samples=2",
            "style_example:style-clean: 压力怕是有点大",
            "style_example:style-noisy: 但我见侍考研具头还好上父和浙不都有一堆点击",
        ],
        retrieval_trace=[],
    )

    joined = "\n".join(message.content for message in prompt.messages)

    assert "压力怕是有点大" in joined
    assert "但我见侍考研具头" not in joined


def test_persona_prompt_injects_local_demo_style_profile(tmp_path: Path) -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.schemas import ChatRequest

    config_path = tmp_path / "demo_persona_config.yaml"
    profile_path = tmp_path / "demo_persona_style_profile.local.md"
    config_path.write_text(
        """
persona_id: demo_persona
display_name: 示例伙伴
prompt_version: demo_persona-v1
identity_notice: >-
  示例伙伴 is an authorized PersonaAgent AI style companion for 演示用户.
style_instruction: >-
  Use the authorized 目标样本 style while keeping replies natural and concise.
safety_boundaries:
  - Do not expose private chat records.
  - Do not make real-world commitments on behalf of the real person.
prompt_templates:
  system: "{identity_notice}\\n\\nPersona name: {display_name}\\n\\n{safety_boundaries}"
  developer: "{style_instruction}\\n\\n{persona_profile}\\n\\n{context_block}"
  user: "{user_text}"
""".strip(),
        encoding="utf-8",
    )
    profile_path.write_text(
        "\n".join(
            [
                "# 示例伙伴本地风格资料",
                "- 用户本人：演示用户，正在准备技术面试，偏好简短自然的回复。",
                "- 风格对象：示例伙伴，授权演示对象，表达理性、简洁、温和。",
                "- 关系：情侣。",
                "- 回复方式：短句、自然、亲密但不客服化。",
            ]
        ),
        encoding="utf-8",
    )

    prompt = PersonaEngine.from_file(config_path, profile_path=profile_path).build_prompt(
        request=ChatRequest.model_validate(_chat_payload("你在干嘛")),
        retrieved_context=["style_summary: persona=demo_persona; samples=8"],
        retrieval_trace=[],
    )

    joined = "\n".join(message.content for message in prompt.messages)

    assert "Persona name: 示例伙伴" in joined
    assert "用户本人：演示用户" in joined
    assert "风格对象：示例伙伴" in joined
    assert "短句、自然、亲密但不客服化" in joined
    assert prompt.metadata.persona_id == "demo_persona"


def test_default_persona_template_uses_configured_local_style_profile(tmp_path: Path) -> None:
    from agent_service.persona.engine import PersonaEngine
    from agent_service.schemas import ChatRequest

    profile_path = tmp_path / "style_profile.local.md"
    profile_path.write_text(
        "Local profile: short, natural, warm, uses fewer formal assistant phrases.",
        encoding="utf-8",
    )

    prompt = PersonaEngine.from_file(
        Path("agent_service/persona/persona.yaml"),
        profile_path=profile_path,
    ).build_prompt(
        request=ChatRequest.model_validate(_chat_payload("你在干嘛")),
        retrieved_context=["style_summary: persona=demo_persona; samples=8"],
        retrieval_trace=[],
    )

    joined = "\n".join(message.content for message in prompt.messages)

    assert "Local profile: short, natural, warm" in joined


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
