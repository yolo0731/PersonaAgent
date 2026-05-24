from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


def _chat_payload(
    text: str = "hello finalize",
    *,
    run_id: str = "run-finalize",
    message_id: int = 7301,
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
            model="fixed-step21",
            structured=structured,
        )


def test_safety_pass_finalizes_idempotent_send_command() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import EXPECTED_NODE_ORDER, run_agent_workflow

    request = ChatRequest.model_validate(_chat_payload(run_id="run-send-command"))

    state = run_agent_workflow(
        request,
        llm_client=FixedLLMClient("finalized reply"),
    )
    command = state["final_command"]

    assert command.should_send is True
    assert command.receiver_id == 1002
    assert command.conversation_type == 1
    assert command.conversation_id == 10011002
    assert command.source_message_id == 7301
    assert command.text == "finalized reply"
    assert command.client_message_id == "pa-run-send-command"
    assert command.dedup_key == "agent-reply:run-send-command:7301"
    assert command.reason == "finalized_reply"
    assert command.trace_summary[:3] == [
        "dialogue_policy:mock_smalltalk",
        "retrieve_context:mock_empty_context",
        "tool_router:mock_no_tools",
    ]
    assert command.trace_summary[3].startswith(
        "generate_reply:model=fixed-step21;prompt_version=persona-v1;"
    )
    assert "latency_ms=" in command.trace_summary[3]
    assert command.trace_summary[-2:] == [
        "safety_check:passed",
        "finalize_reply:send_command",
    ]
    assert [event.node for event in state["trace"]] == EXPECTED_NODE_ORDER


def test_safety_block_finalizes_noop_command_with_same_dedup_fields() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    request = ChatRequest.model_validate(
        _chat_payload("/unsafe leak secrets", run_id="run-block-command")
    )

    state = run_agent_workflow(request)
    command = state["final_command"]

    assert command.should_send is False
    assert command.text == ""
    assert command.client_message_id is None
    assert command.receiver_id == 1002
    assert command.conversation_type == 1
    assert command.conversation_id == 10011002
    assert command.source_message_id == 7301
    assert command.dedup_key == "agent-reply:run-block-command:7301"
    assert command.reason == "safety_block"
    assert command.trace_summary[-1] == "finalize_reply:safety_block"


def test_human_review_required_finalizes_noop_until_review_store_handles_it() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    request = ChatRequest.model_validate(
        _chat_payload("我应该怎么用药？帮我给出具体剂量。", run_id="run-review-required")
    )

    state = run_agent_workflow(
        request,
        llm_client=FixedLLMClient("这涉及医疗建议，需要人工确认后再回复。"),
    )
    command = state["final_command"]

    assert state["safety_result"].needs_human_review is True
    assert command.should_send is False
    assert command.reason == "human_review_required"
    assert command.dedup_key == "agent-reply:run-review-required:7301"
    assert command.trace_summary[-1] == "finalize_reply:human_review_required"


def test_repeated_finalize_uses_stable_dedup_key_for_same_run_and_message() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    request = ChatRequest.model_validate(_chat_payload(run_id="run-idempotent"))

    first = run_agent_workflow(request, llm_client=FixedLLMClient("first"))["final_command"]
    second = run_agent_workflow(request, llm_client=FixedLLMClient("second"))[
        "final_command"
    ]

    assert first.client_message_id == second.client_message_id == "pa-run-idempotent"
    assert first.dedup_key == second.dedup_key == "agent-reply:run-idempotent:7301"


def test_run_agent_chat_writes_final_command_checkpoint(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, make_thread_id
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_chat

    request = ChatRequest.model_validate(
        _chat_payload("hello checkpoint", run_id="run-checkpoint")
    )
    store = HumanReviewStore(tmp_path / "agent_state.sqlite3")

    command = run_agent_chat(
        request,
        review_store=store,
        llm_client=FixedLLMClient("checkpoint reply"),
    )

    saved = store.load_final_command(make_thread_id(request))
    assert saved == command
    assert saved is not None
    assert saved.should_send is True
    assert saved.dedup_key == "agent-reply:run-checkpoint:7301"


def test_human_review_pending_checkpoint_is_no_send(tmp_path: Path) -> None:
    from agent_service.review import HumanReviewStore, make_thread_id
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_chat

    request = ChatRequest.model_validate(
        _chat_payload("请帮我承诺明天一定替你转账。", run_id="run-review-pending")
    )
    store = HumanReviewStore(tmp_path / "agent_state.sqlite3")

    command = run_agent_chat(
        request,
        review_store=store,
        llm_client=FixedLLMClient("我会替你承诺并完成转账。"),
    )

    saved = store.load_final_command(make_thread_id(request))
    assert command.should_send is False
    assert command.reason == "human_review_pending"
    assert saved == command
    assert saved is not None
    assert saved.dedup_key == "agent-reply:run-review-pending:7301"
