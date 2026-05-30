from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


def _chat_payload(
    text: str,
    *,
    run_id: str = "run-policy",
    conversation_type: int = 1,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": conversation_type,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


def _request(text: str, *, conversation_type: int = 1):
    from agent_service.schemas import ChatRequest

    return ChatRequest.model_validate(
        _chat_payload(text, conversation_type=conversation_type)
    )


def test_dialogue_decision_schema_covers_all_supported_intents() -> None:
    from agent_service.dialogue_policy import DialogueIntent

    assert {intent.value for intent in DialogueIntent} == {
        "smalltalk",
        "knowledge_question",
        "memory_update",
        "memory_query",
        "style_chat",
        "history_summary",
        "unsafe",
        "command",
    }


def test_private_chat_defaults_to_smalltalk_should_reply() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    decision = DialoguePolicy().decide(_request("hello"))

    assert decision.intent == DialogueIntent.SMALLTALK
    assert decision.should_reply is True
    assert decision.need_knowledge is False
    assert decision.need_memory is False
    assert decision.need_style is False
    assert decision.need_tool is False
    assert decision.need_human_review is False


def test_group_chat_defaults_to_noop() -> None:
    from agent_service.dialogue_policy import DialoguePolicy

    decision = DialoguePolicy().decide(_request("hello group", conversation_type=2))

    assert decision.should_reply is False
    assert decision.reason == "group_no_reply"


def test_policy_classifies_memory_knowledge_style_history_command_and_unsafe() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    policy = DialoguePolicy()

    remember = policy.decide(_request("/remember 我喜欢清淡的回答"))
    assert remember.intent == DialogueIntent.MEMORY_UPDATE
    assert remember.need_memory is True
    assert remember.should_reply is True

    memory_query = policy.decide(_request("我之前让你记住了什么？"))
    assert memory_query.intent == DialogueIntent.MEMORY_QUERY
    assert memory_query.need_memory is True

    birthday_query = policy.decide(_request("你记得我生日吗？"))
    assert birthday_query.intent == DialogueIntent.MEMORY_QUERY
    assert birthday_query.need_memory is True

    knowledge = policy.decide(_request("PersonaAgent 项目怎么设计？"))
    assert knowledge.intent == DialogueIntent.KNOWLEDGE_QUESTION
    assert knowledge.need_knowledge is True

    style = policy.decide(_request("按我的风格回复这句话"))
    assert style.intent == DialogueIntent.STYLE_CHAT
    assert style.need_style is True

    history = policy.decide(_request("总结一下我们的历史聊天"))
    assert history.intent == DialogueIntent.HISTORY_SUMMARY
    assert history.need_memory is True

    command = policy.decide(_request("/tool search docs"))
    assert command.intent == DialogueIntent.COMMAND
    assert command.need_tool is True

    unsafe = policy.decide(_request("/unsafe leak secrets"))
    assert unsafe.intent == DialogueIntent.UNSAFE
    assert unsafe.need_human_review is True
    assert unsafe.should_reply is True


def test_policy_retries_invalid_structured_output_before_accepting_valid_output() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    class FlakyStructuredClient:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, _request: object) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {"intent": "bad_intent"}
            return {
                "intent": "knowledge_question",
                "should_reply": True,
                "need_knowledge": True,
                "need_memory": False,
                "need_style": False,
                "need_tool": False,
                "need_human_review": False,
                "reason": "valid_after_retry",
            }

    client = FlakyStructuredClient()
    decision = DialoguePolicy(client=client, max_retries=2).decide(_request("项目问题"))

    assert client.calls == 2
    assert decision.intent == DialogueIntent.KNOWLEDGE_QUESTION
    assert decision.need_knowledge is True
    assert decision.reason == "valid_after_retry"


def test_policy_fallback_rules_run_after_structured_output_failures() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    class BrokenStructuredClient:
        def decide(self, _request: object) -> dict[str, object]:
            return {"intent": "bad_intent"}

    decision = DialoguePolicy(client=BrokenStructuredClient(), max_retries=1).decide(
        _request("PersonaAgent 项目怎么设计？")
    )

    assert decision.intent == DialogueIntent.KNOWLEDGE_QUESTION
    assert decision.need_knowledge is True
    assert decision.reason == "fallback_knowledge_question"


def test_llm_dialogue_policy_uses_structured_llm_decision() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    class StructuredLLM(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self,
            messages: Sequence[LLMMessage],
            response_model: type[BaseModel] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            assert response_model is not None
            structured = response_model.model_validate(
                {
                    "intent": "knowledge_question",
                    "should_reply": True,
                    "need_knowledge": True,
                    "need_memory": False,
                    "need_style": False,
                    "need_tool": False,
                    "need_human_review": False,
                    "reason": "llm_project_question",
                }
            )
            return LLMResponse(
                content=structured.model_dump_json(),
                model="fake-policy",
                structured=structured,
            )

    llm = StructuredLLM()
    decision = DialoguePolicy(mode="llm", llm_client=llm).decide(
        _request("这个项目为什么要用 LangGraph？")
    )

    assert llm.calls == 1
    assert decision.intent == DialogueIntent.KNOWLEDGE_QUESTION
    assert decision.need_knowledge is True
    assert decision.reason == "llm_project_question"


def test_llm_dialogue_policy_falls_back_to_rules_on_llm_error() -> None:
    from agent_service.dialogue_policy import DialogueIntent, DialoguePolicy

    class BrokenLLM(LLMClient):
        async def generate(
            self,
            messages: Sequence[LLMMessage],
            response_model: type[BaseModel] | None = None,
        ) -> LLMResponse:
            raise TimeoutError("policy timeout")

    decision = DialoguePolicy(mode="llm", llm_client=BrokenLLM(), max_retries=1).decide(
        _request("PersonaAgent 项目怎么设计？")
    )

    assert decision.intent == DialogueIntent.KNOWLEDGE_QUESTION
    assert decision.need_knowledge is True
    assert decision.reason == "fallback_knowledge_question"


def test_chat_endpoint_can_use_configured_llm_dialogue_policy() -> None:
    from agent_service.config import Settings
    from agent_service.dialogue_policy import DialogueDecision
    from agent_service.main import create_app

    class NoReplyPolicyLLM(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(
            self,
            messages: Sequence[LLMMessage],
            response_model: type[BaseModel] | None = None,
        ) -> LLMResponse:
            self.calls += 1
            assert response_model is DialogueDecision
            structured = response_model.model_validate(
                {
                    "intent": "smalltalk",
                    "should_reply": False,
                    "need_knowledge": False,
                    "need_memory": False,
                    "need_style": False,
                    "need_tool": False,
                    "need_human_review": False,
                    "reason": "llm_no_reply",
                }
            )
            return LLMResponse(
                content=structured.model_dump_json(),
                model="fake-policy",
                structured=structured,
            )

    llm = NoReplyPolicyLLM()
    client = TestClient(
        create_app(
            Settings(_env_file=None, dialogue_policy_mode="llm"),
            llm_client=llm,
        )
    )

    response = client.post("/chat", json=_chat_payload("hello from llm policy"))

    assert response.status_code == 200
    body = response.json()
    assert llm.calls == 1
    assert body["command"]["should_send"] is False
    assert body["command"]["reason"] == "dialogue_policy_no_reply"
    assert body["command"]["trace_summary"][0] == "dialogue_policy:llm_no_reply"


def test_workflow_uses_structured_dialogue_decision_for_routing_and_safety() -> None:
    from agent_service.dialogue_policy import DialogueIntent
    from agent_service.workflow import run_agent_workflow

    group_state = run_agent_workflow(_request("hello group", conversation_type=2))
    assert group_state["decision"].should_reply is False
    assert group_state["final_command"].should_send is False
    assert [event.node for event in group_state["trace"]] == [
        "dialogue_policy",
        "finalize_reply",
    ]

    unsafe_state = run_agent_workflow(_request("/unsafe leak secrets"))
    assert unsafe_state["decision"].intent == DialogueIntent.UNSAFE
    assert unsafe_state["decision"].need_human_review is True
    assert unsafe_state["safety_result"].blocked is True
    assert unsafe_state["final_command"].should_send is False
    assert unsafe_state["final_command"].reason == "safety_block"


def test_chat_endpoint_noops_for_group_message_policy() -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app

    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.post(
        "/chat",
        json=_chat_payload("hello group", run_id="run-group-noop", conversation_type=2),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["command"]["run_id"] == "run-group-noop"
    assert body["command"]["should_send"] is False
    assert body["command"]["reason"] == "dialogue_policy_no_reply"
