from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel

from agent_service.dialogue_policy import DialogueDecision, DialogueIntent
from agent_service.generation import GenerationTrace, ReplyDraft
from agent_service.llm.base import LLMMessage
from agent_service.persona import PromptMetadata
from agent_service.rag.documents import RetrievalTrace
from agent_service.safety.guard import SafetyTrace
from agent_service.safety.verbatim_guard import LeakageMetrics
from agent_service.schemas import AgentReplyCommand, ChatRequest, no_reply_command

EXPECTED_NODE_ORDER = [
    "dialogue_policy",
    "retrieve_context",
    "tool_router",
    "generate_reply",
    "safety_check",
    "finalize_reply",
]


class SafetyResult(BaseModel):
    blocked: bool
    reason: str | None = None
    needs_human_review: bool = False
    metrics: LeakageMetrics | None = None
    trace: SafetyTrace | None = None


class TraceEvent(BaseModel):
    node: str
    action: str


@dataclass(frozen=True)
class ParsedToolCommand:
    name: str
    payload: dict[str, object]


class AgentState(TypedDict):
    request: ChatRequest
    run_id: str
    decision: DialogueDecision
    retrieved_context: list[str]
    retrieval_trace: list[RetrievalTrace]
    tool_calls: list[str]
    tool_results: list[str]
    prompt_messages: list[LLMMessage]
    prompt_metadata: PromptMetadata | None
    reply_draft: ReplyDraft | None
    generation_trace: GenerationTrace | None
    draft: str
    safety_result: SafetyResult
    final_command: AgentReplyCommand
    trace: list[TraceEvent]


GraphRoute = Literal["retrieve_context", "finalize_reply"]


def make_initial_agent_state(request: ChatRequest) -> AgentState:
    return AgentState(
        request=request,
        run_id=request.run_id,
        decision=DialogueDecision(
            intent=DialogueIntent.SMALLTALK,
            should_reply=True,
            reason="not_evaluated",
        ),
        retrieved_context=[],
        retrieval_trace=[],
        tool_calls=[],
        tool_results=[],
        prompt_messages=[],
        prompt_metadata=None,
        reply_draft=None,
        generation_trace=None,
        draft="",
        safety_result=SafetyResult(blocked=False),
        final_command=no_reply_command(request, "not_finalized"),
        trace=[],
    )
