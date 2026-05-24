from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent_service.dialogue_policy import DialogueDecision, DialogueIntent, DialoguePolicy
from agent_service.generation import GenerationTrace, LLMReplyGenerator, ReplyDraft
from agent_service.llm import LLMClient
from agent_service.llm.base import LLMMessage
from agent_service.memory.memory_store import MemoryNotFoundError, MemoryStore
from agent_service.memory.memory_tools import parse_forget_memory_id, parse_remember_content
from agent_service.persona import PersonaEngine, PromptMetadata
from agent_service.rag.documents import RetrievalTrace
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.review import (
    HumanReviewStore,
    ReviewStatus,
    make_thread_id,
    resume_human_review,
)
from agent_service.safety.guard import SafetyGuard, SafetyTrace
from agent_service.safety.verbatim_guard import LeakageMetrics, LeakageSource
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    no_reply_command,
    send_reply_command,
)
from agent_service.style.style_store import StyleStore
from agent_service.tools import (
    ToolErrorEnvelope,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntimeContext,
    ToolTrace,
)

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


def build_agent_graph(
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    effective_persona_engine = persona_engine or PersonaEngine.from_default()
    reply_generator = LLMReplyGenerator(
        llm_client=llm_client,
        max_retries=generation_max_retries,
    )
    graph.add_node("dialogue_policy", _dialogue_policy)
    graph.add_node(
        "retrieve_context",
        lambda state: _retrieve_context(
            state,
            knowledge_retriever=knowledge_retriever,
            memory_store=memory_store,
            style_store=style_store,
            rag_top_k=rag_top_k,
            memory_top_k=memory_top_k,
            style_top_k=style_top_k,
        ),
    )
    graph.add_node(
        "tool_router",
        lambda state: _tool_router(
            state,
            tool_registry=tool_registry,
            memory_store=memory_store,
        ),
    )
    graph.add_node(
        "generate_reply",
        lambda state: _generate_reply(
            state,
            persona_engine=effective_persona_engine,
            reply_generator=reply_generator,
        ),
    )
    graph.add_node("safety_check", _safety_check)
    graph.add_node("finalize_reply", _finalize_reply)

    graph.add_edge(START, "dialogue_policy")
    graph.add_conditional_edges(
        "dialogue_policy",
        _route_after_dialogue_policy,
        {
            "retrieve_context": "retrieve_context",
            "finalize_reply": "finalize_reply",
        },
    )
    graph.add_edge("retrieve_context", "tool_router")
    graph.add_edge("tool_router", "generate_reply")
    graph.add_edge("generate_reply", "safety_check")
    graph.add_edge("safety_check", "finalize_reply")
    graph.add_edge("finalize_reply", END)
    return graph.compile()


def run_agent_workflow(
    request: ChatRequest,
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
) -> AgentState:
    final_state = build_agent_graph(
        knowledge_retriever=knowledge_retriever,
        memory_store=memory_store,
        style_store=style_store,
        tool_registry=tool_registry,
        persona_engine=persona_engine,
        llm_client=llm_client,
        generation_max_retries=generation_max_retries,
        rag_top_k=rag_top_k,
        memory_top_k=memory_top_k,
        style_top_k=style_top_k,
    ).invoke(make_initial_agent_state(request))
    return cast(AgentState, final_state)


def run_agent_chat(
    request: ChatRequest,
    *,
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
) -> AgentReplyCommand:
    state = run_agent_workflow(
        request,
        knowledge_retriever=knowledge_retriever,
        memory_store=memory_store,
        style_store=style_store,
        tool_registry=tool_registry,
        persona_engine=persona_engine,
        llm_client=llm_client,
        generation_max_retries=generation_max_retries,
        rag_top_k=rag_top_k,
        memory_top_k=memory_top_k,
        style_top_k=style_top_k,
    )
    safety = state["safety_result"]
    thread_id = make_thread_id(request) if review_store is not None else None
    if (
        review_store is not None
        and not safety.blocked
        and (safety.needs_human_review or state["decision"].need_human_review)
    ):
        assert thread_id is not None
        existing = review_store.get_review(thread_id)
        if existing is not None and existing.status == ReviewStatus.COMPLETED:
            command = no_reply_command(request, "human_review_already_resumed")
            review_store.save_checkpoint(thread_id, state, command)
            return command
        review_store.create_pending(thread_id, state)
        command = no_reply_command(
            request,
            "human_review_pending",
            trace_summary=state["final_command"].trace_summary,
        )
        review_store.save_checkpoint(thread_id, state, command)
        return command
    command = state["final_command"]
    if review_store is not None:
        assert thread_id is not None
        review_store.save_checkpoint(thread_id, state, command)
    return command


def resume_agent_review(thread_id: str, review_store: HumanReviewStore) -> AgentReplyCommand:
    return resume_human_review(thread_id, review_store)


def _dialogue_policy(state: AgentState) -> dict[str, object]:
    decision = DialoguePolicy().decide(state["request"])
    return {
        "decision": decision,
        "trace": _append_trace(state, "dialogue_policy", decision.reason),
    }


def _route_after_dialogue_policy(state: AgentState) -> GraphRoute:
    if state["decision"].should_reply:
        return "retrieve_context"
    return "finalize_reply"


def _retrieve_context(
    state: AgentState,
    *,
    knowledge_retriever: KnowledgeRetriever | None,
    memory_store: MemoryStore | None,
    style_store: StyleStore | None,
    rag_top_k: int,
    memory_top_k: int,
    style_top_k: int,
) -> dict[str, object]:
    if state["decision"].need_memory and memory_store is not None:
        return _retrieve_memory_context(state, memory_store=memory_store, memory_top_k=memory_top_k)

    if state["decision"].need_style and style_store is not None:
        return _retrieve_style_context(state, style_store=style_store, style_top_k=style_top_k)

    if state["decision"].need_knowledge and knowledge_retriever is not None:
        retrieval = knowledge_retriever.retrieve(state["request"].text, top_k=rag_top_k)
        return {
            "retrieved_context": [result.text for result in retrieval.results],
            "retrieval_trace": [retrieval.trace],
            "trace": _append_trace(
                state,
                "retrieve_context",
                f"knowledge_top_k={retrieval.trace.result_count}",
            ),
        }
    return {
        "retrieved_context": [],
        "retrieval_trace": [],
        "trace": _append_trace(state, "retrieve_context", "mock_empty_context"),
    }


def _retrieve_style_context(
    state: AgentState,
    *,
    style_store: StyleStore,
    style_top_k: int,
) -> dict[str, object]:
    request = state["request"]
    retrieval = style_store.retrieve_style(
        persona_id=str(request.sender_id),
        query=request.text,
        top_k=style_top_k,
    )
    if retrieval.fallback_reason is not None:
        return {
            "retrieved_context": [f"style_fallback: {retrieval.fallback_reason}"],
            "retrieval_trace": [retrieval.trace],
            "trace": _append_trace(
                state,
                "retrieve_context",
                f"style_fallback={retrieval.fallback_reason}",
            ),
        }
    return {
        "retrieved_context": [
            f"style_summary: {retrieval.summary.text}",
            *[f"style_example:{sample.sample_id}: {sample.text}" for sample in retrieval.results],
        ],
        "retrieval_trace": [retrieval.trace],
        "trace": _append_trace(
            state,
            "retrieve_context",
            f"style_top_k={retrieval.trace.result_count}",
        ),
    }


def _retrieve_memory_context(
    state: AgentState,
    *,
    memory_store: MemoryStore,
    memory_top_k: int,
) -> dict[str, object]:
    request = state["request"]
    remember_content = parse_remember_content(request.text)
    if remember_content is not None:
        record = memory_store.save_memory(
            user_id=request.sender_id,
            content=remember_content,
            source_message_id=request.message_id,
        )
        trace = RetrievalTrace(
            query=request.text,
            top_k=memory_top_k,
            result_count=1,
            collection="memory",
            chunk_ids=[record.memory_id],
        )
        return {
            "retrieved_context": [f"memory_saved: {record.content}"],
            "retrieval_trace": [trace],
            "trace": _append_trace(state, "retrieve_context", "memory_saved"),
        }

    forget_memory_id = parse_forget_memory_id(request.text)
    if forget_memory_id is not None:
        try:
            record = memory_store.deactivate_memory(
                forget_memory_id,
                user_id=request.sender_id,
            )
            context = f"memory_deactivated: {record.memory_id}"
            chunk_ids = [record.memory_id]
            result_count = 1
            action = "memory_deactivated"
        except MemoryNotFoundError:
            context = f"memory_not_found: {forget_memory_id}"
            chunk_ids = []
            result_count = 0
            action = "memory_not_found"
        trace = RetrievalTrace(
            query=request.text,
            top_k=memory_top_k,
            result_count=result_count,
            collection="memory",
            chunk_ids=chunk_ids,
        )
        return {
            "retrieved_context": [context],
            "retrieval_trace": [trace],
            "trace": _append_trace(state, "retrieve_context", action),
        }

    retrieval = memory_store.retrieve_memory(
        user_id=request.sender_id,
        query=request.text,
        top_k=memory_top_k,
    )
    return {
        "retrieved_context": [f"memory: {result.content}" for result in retrieval.results],
        "retrieval_trace": [retrieval.trace],
        "trace": _append_trace(
            state,
            "retrieve_context",
            f"memory_top_k={retrieval.trace.result_count}",
        ),
    }


def _tool_router(
    state: AgentState,
    *,
    tool_registry: ToolRegistry | None,
    memory_store: MemoryStore | None,
) -> dict[str, object]:
    if not state["decision"].need_tool:
        return {
            "tool_calls": [],
            "tool_results": [],
            "trace": _append_trace(state, "tool_router", "mock_no_tools"),
        }

    if tool_registry is None:
        result = _tool_error_result(
            "tool_registry",
            "tool_registry_not_configured",
            "tool registry is not configured",
        )
        return {
            "tool_calls": [],
            "tool_results": [result.model_dump_json()],
            "trace": _append_trace(state, "tool_router", "tool:tool_registry:error"),
        }

    parsed = _parse_tool_command(state["request"].text)
    if isinstance(parsed, ToolExecutionResult):
        return {
            "tool_calls": [],
            "tool_results": [parsed.model_dump_json()],
            "trace": _append_trace(state, "tool_router", f"tool:{parsed.tool_name}:error"),
        }

    result = tool_registry.execute(
        parsed.name,
        parsed.payload,
        ToolRuntimeContext(
            request=state["request"],
            memory_store=memory_store,
            recent_context=tuple(state["retrieved_context"]),
        ),
    )
    return {
        "tool_calls": [parsed.name],
        "tool_results": [result.model_dump_json()],
        "trace": _append_trace(state, "tool_router", f"tool:{parsed.name}:{result.trace.status}"),
    }


def _generate_reply(
    state: AgentState,
    *,
    persona_engine: PersonaEngine,
    reply_generator: LLMReplyGenerator,
) -> dict[str, object]:
    prompt = persona_engine.build_prompt(
        request=state["request"],
        retrieved_context=state["retrieved_context"],
        retrieval_trace=state["retrieval_trace"],
        tool_results=state["tool_results"],
    )
    generation = reply_generator.generate(request=state["request"], prompt=prompt)
    return {
        "prompt_messages": prompt.messages,
        "prompt_metadata": prompt.metadata,
        "reply_draft": generation.draft,
        "generation_trace": generation.trace,
        "draft": generation.draft.reply_text,
        "trace": _append_trace(
            state,
            "generate_reply",
            (
                f"model={generation.trace.model};"
                f"prompt_version={prompt.metadata.prompt_version};"
                f"prompt_tokens={generation.trace.prompt_tokens};"
                f"completion_tokens={generation.trace.completion_tokens};"
                f"latency_ms={generation.trace.latency_ms:.3f};"
                f"fallback={generation.trace.fallback_used}"
            ),
        ),
    }


def _safety_check(state: AgentState) -> dict[str, object]:
    style_sources = _style_sources_from_context(state["retrieved_context"])
    assessment = SafetyGuard().assess(
        request=state["request"],
        draft=state["draft"],
        prompt_messages=state["prompt_messages"],
        retrieved_context=state["retrieved_context"],
        style_sources=style_sources,
        unsafe_decision=state["decision"].intent == DialogueIntent.UNSAFE,
    )
    result = SafetyResult(
        blocked=assessment.blocked,
        reason=assessment.reason,
        needs_human_review=assessment.needs_human_review,
        metrics=assessment.metrics,
        trace=assessment.trace,
    )
    if assessment.blocked:
        action = f"blocked:{assessment.reason}"
    elif assessment.needs_human_review:
        action = f"review:{assessment.reason}"
    elif assessment.safe_text is not None and assessment.safe_text != state["draft"]:
        action = f"rewritten:{assessment.reason}"
    else:
        action = "passed"
    updates: dict[str, object] = {
        "safety_result": result,
        "trace": _append_trace(state, "safety_check", action),
    }
    if assessment.safe_text is not None:
        updates["draft"] = assessment.safe_text
    return updates


def _finalize_reply(state: AgentState) -> dict[str, object]:
    request = state["request"]
    if not state["decision"].should_reply:
        action = "dialogue_policy_no_reply"
        command = no_reply_command(
            request,
            "dialogue_policy_no_reply",
            trace_summary=_trace_summary(state, action),
        )
    elif state["safety_result"].blocked:
        action = "safety_block"
        command = no_reply_command(
            request,
            "safety_block",
            trace_summary=_trace_summary(state, action),
        )
    elif state["safety_result"].needs_human_review:
        action = "human_review_required"
        command = no_reply_command(
            request,
            "human_review_required",
            trace_summary=_trace_summary(state, action),
        )
    else:
        action = "send_command"
        command = send_reply_command(
            request,
            text=state["draft"],
            reason="finalized_reply",
            trace_summary=_trace_summary(state, action),
        )
    return {
        "final_command": command,
        "trace": _append_trace(state, "finalize_reply", action),
    }


def _append_trace(state: AgentState, node: str, action: str) -> list[TraceEvent]:
    return [*state["trace"], TraceEvent(node=node, action=action)]


def _trace_summary(state: AgentState, final_action: str) -> list[str]:
    events = [*state["trace"], TraceEvent(node="finalize_reply", action=final_action)]
    return [f"{event.node}:{event.action}" for event in events]


def _parse_tool_command(text: str) -> ParsedToolCommand | ToolExecutionResult:
    stripped = text.strip()
    if stripped != "/tool" and not stripped.startswith("/tool "):
        return _tool_error_result(
            "tool_command",
            "tool_command_parse_error",
            "tool commands must use: /tool <tool_name> <json_payload>",
        )

    remainder = stripped[len("/tool") :].strip()
    if not remainder:
        return _tool_error_result(
            "tool_command",
            "tool_command_parse_error",
            "missing tool name",
        )

    parts = remainder.split(maxsplit=1)
    tool_name = parts[0]
    payload_text = parts[1] if len(parts) == 2 else "{}"
    try:
        raw_payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return _tool_error_result(
            tool_name,
            "tool_command_parse_error",
            "tool payload must be valid JSON",
        )
    if not isinstance(raw_payload, dict):
        return _tool_error_result(
            tool_name,
            "tool_command_parse_error",
            "tool payload must be a JSON object",
        )
    return ParsedToolCommand(name=tool_name, payload=cast(dict[str, object], raw_payload))


def _tool_error_result(tool_name: str, code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name,
        ok=False,
        error=ToolErrorEnvelope(code=code, message=message, retryable=False),
        trace=ToolTrace(tool_name=tool_name, status="error", duration_ms=0.0),
    )


def _style_sources_from_context(context: list[str]) -> list[LeakageSource]:
    sources: list[LeakageSource] = []
    prefix = "style_example:"
    for item in context:
        if not item.startswith(prefix):
            continue
        payload = item[len(prefix) :]
        source_id, separator, text = payload.partition(":")
        if separator and source_id.strip() and text.strip():
            sources.append(LeakageSource(source_id=source_id.strip(), text=text.strip()))
    return sources
