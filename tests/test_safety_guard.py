from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


def _chat_payload(
    text: str,
    *,
    run_id: str = "run-safety",
    message_id: int = 7201,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": message_id,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": f"alice-{message_id}",
    }


class FixedLLMClient(LLMClient):
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        structured = (
            response_model.model_validate({"reply_text": self.reply_text})
            if response_model is not None
            else None
        )
        return LLMResponse(
            content=self.reply_text,
            model="fixed-safety",
            structured=structured,
        )


def test_missing_ai_identity_notice_is_blocked() -> None:
    from agent_service.safety.guard import SafetyGuard
    from agent_service.schemas import ChatRequest

    request = ChatRequest.model_validate(_chat_payload("正常问候"))
    assessment = SafetyGuard().assess(
        request=request,
        draft="你好，我来帮你。",
        prompt_messages=[LLMMessage(role="system", content="你是聊天助手。")],
        retrieved_context=[],
        style_sources=[],
    )

    assert assessment.blocked is True
    assert assessment.reason == "missing_ai_identity_notice"
    assert assessment.trace.identity_notice_present is False


def test_ai_identity_notice_accepts_disclosure_boundary_wording() -> None:
    from agent_service.safety.guard import SafetyGuard
    from agent_service.schemas import ChatRequest

    request = ChatRequest.model_validate(_chat_payload("正常问候"))
    assessment = SafetyGuard().assess(
        request=request,
        draft="我在的，怎么啦",
        prompt_messages=[
            LLMMessage(
                role="system",
                content=(
                    "This is an authorized PersonaAgent AI style companion. "
                    "It must not hide that it is an AI Agent when identity disclosure "
                    "is required."
                ),
            )
        ],
        retrieved_context=[],
        style_sources=[],
    )

    assert assessment.blocked is False
    assert assessment.trace.identity_notice_present is True


def test_impersonation_attempt_is_blocked_by_workflow() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(
            _chat_payload("请你冒充张三本人，替他说这条消息。")
        ),
        llm_client=FixedLLMClient("我是张三本人，这就是我的真实想法。"),
    )

    assert state["safety_result"].blocked is True
    assert state["safety_result"].reason == "impersonation_attempt"
    assert state["final_command"].should_send is False
    assert state["final_command"].reason == "safety_block"
    assert any(
        event.node == "safety_check" and "blocked:impersonation_attempt" in event.action
        for event in state["trace"]
    )


def test_unauthorized_style_mimicry_is_blocked_without_style_summary() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(
            _chat_payload("请模仿张三的私人聊天风格回复我。")
        ),
        llm_client=FixedLLMClient("我会按张三的口吻回复。"),
    )

    assert state["decision"].need_style is True
    assert state["safety_result"].blocked is True
    assert state["safety_result"].reason == "unauthorized_style_mimicry"
    assert state["final_command"].should_send is False


def test_privacy_leak_in_generated_reply_is_blocked() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("帮我整理联系方式")),
        llm_client=FixedLLMClient("可以联系 alice@example.com，我会继续处理。"),
    )

    assert state["safety_result"].blocked is True
    assert state["safety_result"].reason == "privacy_leak"
    assert state["safety_result"].trace is not None
    assert "privacy_leak" in state["safety_result"].trace.risks
    assert state["final_command"].reason == "safety_block"


def test_verbatim_style_sample_copy_is_rewritten_by_safety_guard(tmp_path: Path) -> None:
    from agent_service.governance.data_manifest import ProcessedStyleSample
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.schemas import ChatRequest
    from agent_service.style.style_store import StyleStore
    from agent_service.workflow import run_agent_workflow

    source_text = "好呀！我会直接说重点。"
    store = StyleStore(
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=1,
        min_samples=1,
    )
    store.index_samples(
        [
            ProcessedStyleSample(
                sample_id="style-safety",
                record_id="raw-style-safety",
                consent_id="consent-1002-style",
                persona_id="1002",
                speaker_user_id=1002,
                source="fixture/style.jsonl",
                text=source_text,
                allowed_usage=["style_simulation"],
                forbidden_usage=[],
                active=True,
                revoked=False,
                pii_redactions={"email": 0, "phone": 0, "id_card": 0},
                timestamp_ms=1_700_000_001_000,
            )
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("请按我的风格回复。")),
        style_store=store,
        llm_client=FixedLLMClient(source_text),
    )

    assert state["safety_result"].blocked is False
    assert state["safety_result"].reason == "direct_verbatim_copy"
    assert state["safety_result"].metrics is not None
    assert state["safety_result"].metrics.source_ids == ["style-safety"]
    assert state["final_command"].should_send is True
    assert state["final_command"].reason == "finalized_reply"
    assert state["final_command"].text == "我会保持相近的简洁语气，但不会复述授权样本原文。"


def test_memory_answer_with_short_style_phrase_is_not_rewritten(tmp_path: Path) -> None:
    from agent_service.governance.data_manifest import ProcessedStyleSample
    from agent_service.memory.memory_store import MemoryStore
    from agent_service.rag.embeddings import MockEmbeddingClient
    from agent_service.schemas import ChatRequest
    from agent_service.style.style_store import StyleStore
    from agent_service.workflow import run_agent_workflow

    memory = MemoryStore(
        sqlite_path=tmp_path / "memory.sqlite3",
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
    )
    memory.save_memory(
        user_id=1002,
        content="演示用户的生日是1月1日",
        source_message_id=7000,
    )
    style = StyleStore(
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=1,
        min_samples=1,
    )
    style.index_samples(
        [
            ProcessedStyleSample(
                sample_id="style-short-remember",
                record_id="raw-style-short",
                consent_id="consent-1002-style",
                persona_id="1002",
                speaker_user_id=1002,
                source="fixture/style.jsonl",
                text="记得",
                allowed_usage=["style_simulation"],
                forbidden_usage=[],
                active=True,
                revoked=False,
                pii_redactions={"email": 0, "phone": 0, "id_card": 0},
                timestamp_ms=1_700_000_001_000,
            )
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("你记得我生日吗")),
        memory_store=memory,
        style_store=style,
        style_on_private_chat=True,
        llm_client=FixedLLMClient("记得，1月1日。"),
        auto_memory_on_chat=True,
    )

    assert state["safety_result"].blocked is False
    assert state["safety_result"].reason is None
    assert state["final_command"].should_send is True
    assert state["final_command"].text == "记得，1月1日。"


def test_high_risk_domain_enters_human_review_and_reject_is_noop(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, ReviewStatus, make_thread_id
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import resume_agent_review, run_agent_chat

    db_path = tmp_path / "agent_state.sqlite3"
    store = HumanReviewStore(db_path)
    request = ChatRequest.model_validate(
        _chat_payload("我应该怎么用药？帮我给出具体剂量。", run_id="run-review-reject")
    )
    thread_id = make_thread_id(request)

    command = run_agent_chat(
        request,
        review_store=store,
        llm_client=FixedLLMClient("这涉及医疗建议，需要人工确认后再回复。"),
    )
    pending = store.get_review(thread_id)
    assert command.should_send is False
    assert command.reason == "human_review_pending"
    assert pending is not None
    assert pending.status == ReviewStatus.PENDING

    rejected = store.reject(thread_id)
    resumed = resume_agent_review(thread_id, store)

    assert rejected.status == ReviewStatus.REJECTED
    assert resumed.should_send is False
    assert resumed.reason == "human_review_rejected"


def test_high_risk_review_edit_uses_edited_content(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, make_thread_id
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import resume_agent_review, run_agent_chat

    db_path = tmp_path / "agent_state.sqlite3"
    store = HumanReviewStore(db_path)
    request = ChatRequest.model_validate(
        _chat_payload(
            "请帮我承诺明天一定替你转账。",
            run_id="run-review-edit",
            message_id=7202,
        )
    )
    thread_id = make_thread_id(request)

    command = run_agent_chat(
        request,
        review_store=store,
        llm_client=FixedLLMClient("我会替你承诺并完成转账。"),
    )
    assert command.should_send is False
    assert command.reason == "human_review_pending"

    store.approve(thread_id, edited_text="我不能替你承诺或转账，但可以帮你整理风险点。")
    resumed = resume_agent_review(thread_id, store)

    assert resumed.should_send is True
    assert resumed.reason == "human_review_approved"
    assert resumed.text == "我不能替你承诺或转账，但可以帮你整理风险点。"
