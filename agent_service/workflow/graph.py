from __future__ import annotations

import json
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_service.dialogue_policy import (
    PRIVATE_CONVERSATION_TYPE,
    DialogueIntent,
    DialoguePolicy,
)
from agent_service.generation import LLMReplyGenerator
from agent_service.llm import LLMClient
from agent_service.memory.memory_store import MemoryNotFoundError, MemoryStore
from agent_service.memory.memory_tools import parse_forget_memory_id, parse_remember_content
from agent_service.persona import PersonaEngine
from agent_service.rag.documents import RetrievalTrace
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.review import (
    HumanReviewStore,
    ReviewStatus,
    make_thread_id,
    resume_human_review,
)
from agent_service.safety.guard import SafetyGuard
from agent_service.safety.verbatim_guard import LeakageSource
from agent_service.schemas import (
    AgentReplyCommand,
    ChatRequest,
    no_reply_command,
    send_reply_command,
)
from agent_service.style.learning import StyleLearningStore
from agent_service.style.pair_store import StylePairStore
from agent_service.style.style_store import StyleStore
from agent_service.tools import (
    ToolErrorEnvelope,
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntimeContext,
    ToolTrace,
)
from agent_service.workflow.state import (
    AgentState,
    GraphRoute,
    ParsedToolCommand,
    SafetyResult,
    TraceEvent,
    make_initial_agent_state,
)


def build_agent_graph(
    *,
    dialogue_policy: DialoguePolicy | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    style_pair_store: StylePairStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
    style_pair_top_k: int = 3,
    style_persona_id: str | None = None,
    style_on_smalltalk: bool = False,
    style_on_private_chat: bool = False,
    auto_memory_on_chat: bool = False,
    auto_memory_user_name: str = "用户",
    auto_memory_persona_name: str = "PersonaAgent",
    style_learning_store: StyleLearningStore | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    effective_persona_engine = persona_engine or PersonaEngine.from_default()
    effective_dialogue_policy = dialogue_policy or DialoguePolicy()
    reply_generator = LLMReplyGenerator(
        llm_client=llm_client,
        max_retries=generation_max_retries,
    )
    graph.add_node(
        "dialogue_policy",
        lambda state: _dialogue_policy(state, dialogue_policy=effective_dialogue_policy),
    )
    graph.add_node(
        "retrieve_context",
        lambda state: _retrieve_context(
            state,
            knowledge_retriever=knowledge_retriever,
            memory_store=memory_store,
            style_store=style_store,
            style_pair_store=style_pair_store,
            rag_top_k=rag_top_k,
            memory_top_k=memory_top_k,
            style_top_k=style_top_k,
            style_pair_top_k=style_pair_top_k,
            style_persona_id=style_persona_id,
            style_on_smalltalk=style_on_smalltalk,
            style_on_private_chat=style_on_private_chat,
            auto_memory_on_chat=auto_memory_on_chat,
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
    graph.add_node(
        "finalize_reply",
        lambda state: _finalize_reply(
            state,
            memory_store=memory_store,
            auto_memory_on_chat=auto_memory_on_chat,
            auto_memory_user_name=auto_memory_user_name,
            auto_memory_persona_name=auto_memory_persona_name,
            style_learning_store=style_learning_store,
        ),
    )

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
    dialogue_policy: DialoguePolicy | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    style_pair_store: StylePairStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
    style_pair_top_k: int = 3,
    style_persona_id: str | None = None,
    style_on_smalltalk: bool = False,
    style_on_private_chat: bool = False,
    auto_memory_on_chat: bool = False,
    auto_memory_user_name: str = "用户",
    auto_memory_persona_name: str = "PersonaAgent",
    style_learning_store: StyleLearningStore | None = None,
) -> AgentState:
    final_state = build_agent_graph(
        dialogue_policy=dialogue_policy,
        knowledge_retriever=knowledge_retriever,
        memory_store=memory_store,
        style_store=style_store,
        style_pair_store=style_pair_store,
        tool_registry=tool_registry,
        persona_engine=persona_engine,
        llm_client=llm_client,
        generation_max_retries=generation_max_retries,
        rag_top_k=rag_top_k,
        memory_top_k=memory_top_k,
        style_top_k=style_top_k,
        style_pair_top_k=style_pair_top_k,
        style_persona_id=style_persona_id,
        style_on_smalltalk=style_on_smalltalk,
        style_on_private_chat=style_on_private_chat,
        auto_memory_on_chat=auto_memory_on_chat,
        auto_memory_user_name=auto_memory_user_name,
        auto_memory_persona_name=auto_memory_persona_name,
        style_learning_store=style_learning_store,
    ).invoke(make_initial_agent_state(request))
    return cast(AgentState, final_state)


def run_agent_chat(
    request: ChatRequest,
    *,
    dialogue_policy: DialoguePolicy | None = None,
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    memory_store: MemoryStore | None = None,
    style_store: StyleStore | None = None,
    style_pair_store: StylePairStore | None = None,
    tool_registry: ToolRegistry | None = None,
    persona_engine: PersonaEngine | None = None,
    llm_client: LLMClient | None = None,
    generation_max_retries: int = 2,
    rag_top_k: int = 5,
    memory_top_k: int = 5,
    style_top_k: int = 8,
    style_pair_top_k: int = 3,
    style_persona_id: str | None = None,
    style_on_smalltalk: bool = False,
    style_on_private_chat: bool = False,
    auto_memory_on_chat: bool = False,
    auto_memory_user_name: str = "用户",
    auto_memory_persona_name: str = "PersonaAgent",
    style_learning_store: StyleLearningStore | None = None,
) -> AgentReplyCommand:
    state = run_agent_workflow(
        request,
        dialogue_policy=dialogue_policy,
        knowledge_retriever=knowledge_retriever,
        memory_store=memory_store,
        style_store=style_store,
        style_pair_store=style_pair_store,
        tool_registry=tool_registry,
        persona_engine=persona_engine,
        llm_client=llm_client,
        generation_max_retries=generation_max_retries,
        rag_top_k=rag_top_k,
        memory_top_k=memory_top_k,
        style_top_k=style_top_k,
        style_pair_top_k=style_pair_top_k,
        style_persona_id=style_persona_id,
        style_on_smalltalk=style_on_smalltalk,
        style_on_private_chat=style_on_private_chat,
        auto_memory_on_chat=auto_memory_on_chat,
        auto_memory_user_name=auto_memory_user_name,
        auto_memory_persona_name=auto_memory_persona_name,
        style_learning_store=style_learning_store,
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


def _dialogue_policy(
    state: AgentState,
    *,
    dialogue_policy: DialoguePolicy,
) -> dict[str, object]:
    decision = dialogue_policy.decide(state["request"])
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
    style_pair_store: StylePairStore | None,
    rag_top_k: int,
    memory_top_k: int,
    style_top_k: int,
    style_pair_top_k: int,
    style_persona_id: str | None,
    style_on_smalltalk: bool,
    style_on_private_chat: bool,
    auto_memory_on_chat: bool,
) -> dict[str, object]:
    contexts: list[str] = []
    traces: list[RetrievalTrace] = []
    actions: list[str] = []

    if state["decision"].need_memory and memory_store is not None:
        memory_result = _retrieve_memory_context(
            state,
            memory_store=memory_store,
            memory_top_k=memory_top_k,
        )
        if state["decision"].intent == DialogueIntent.MEMORY_UPDATE:
            return memory_result
        _extend_retrieval_update(state, contexts, traces, actions, memory_result)

    if state["decision"].need_knowledge and knowledge_retriever is not None:
        retrieval = knowledge_retriever.retrieve(state["request"].text, top_k=rag_top_k)
        contexts.extend(result.text for result in retrieval.results)
        traces.append(retrieval.trace)
        actions.append(f"knowledge_top_k={retrieval.trace.result_count}")

    if _should_retrieve_memory_for_chat(state, memory_store, auto_memory_on_chat):
        assert memory_store is not None
        memory_retrieval = memory_store.retrieve_memory(
            user_id=state["request"].sender_id,
            query=state["request"].text,
            top_k=memory_top_k,
        )
        contexts.extend(f"memory: {result.content}" for result in memory_retrieval.results)
        traces.append(memory_retrieval.trace)
        actions.append(f"memory_top_k={memory_retrieval.trace.result_count}")

    if state["decision"].need_style and style_store is not None:
        style_result = _retrieve_style_context(
            state,
            style_store=style_store,
            style_pair_store=style_pair_store,
            style_top_k=style_top_k,
            style_pair_top_k=style_pair_top_k,
            style_persona_id=style_persona_id,
        )
        return _merge_retrieval_update(state, contexts, traces, actions, style_result)

    if _should_use_style_for_private_chat(
        state,
        style_store=style_store,
        style_on_smalltalk=style_on_smalltalk,
        style_on_private_chat=style_on_private_chat,
    ):
        assert style_store is not None
        style_result = _retrieve_style_context(
            state,
            style_store=style_store,
            style_pair_store=style_pair_store,
            style_top_k=style_top_k,
            style_pair_top_k=style_pair_top_k,
            style_persona_id=style_persona_id,
        )
        return _merge_retrieval_update(state, contexts, traces, actions, style_result)

    if contexts or traces:
        return {
            "retrieved_context": contexts,
            "retrieval_trace": traces,
            "trace": _append_trace(state, "retrieve_context", ";".join(actions)),
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
    style_pair_store: StylePairStore | None,
    style_top_k: int,
    style_pair_top_k: int,
    style_persona_id: str | None,
) -> dict[str, object]:
    request = state["request"]
    persona_id = style_persona_id or str(request.sender_id)
    retrieval = style_store.retrieve_style(
        persona_id=persona_id,
        query=request.text,
        top_k=style_top_k,
    )
    pair_context, pair_traces, pair_actions = _retrieve_style_pair_context(
        request_text=request.text,
        persona_id=persona_id,
        style_pair_store=style_pair_store,
        style_pair_top_k=style_pair_top_k,
    )
    if retrieval.fallback_reason is not None:
        return {
            "retrieved_context": [
                f"style_fallback: {retrieval.fallback_reason}",
                *pair_context,
            ],
            "retrieval_trace": [retrieval.trace, *pair_traces],
            "trace": _append_trace(
                state,
                "retrieve_context",
                ";".join(
                    [
                        f"style_fallback={retrieval.fallback_reason}",
                        *pair_actions,
                    ]
                ),
            ),
        }
    return {
        "retrieved_context": [
            f"style_summary: {retrieval.summary.text}",
            *[f"style_example:{sample.sample_id}: {sample.text}" for sample in retrieval.results],
            *pair_context,
        ],
        "retrieval_trace": [retrieval.trace, *pair_traces],
        "trace": _append_trace(
            state,
            "retrieve_context",
            ";".join(
                [
                    f"style_top_k={retrieval.trace.result_count}",
                    *pair_actions,
                ]
            ),
        ),
    }


def _retrieve_style_pair_context(
    *,
    request_text: str,
    persona_id: str,
    style_pair_store: StylePairStore | None,
    style_pair_top_k: int,
) -> tuple[list[str], list[RetrievalTrace], list[str]]:
    if style_pair_store is None:
        return [], [], []
    pair_retrieval = style_pair_store.retrieve_pairs(
        persona_id=persona_id,
        query=request_text,
        top_k=style_pair_top_k,
    )
    contexts = [
        (
            f"style_pair:{result.pair.pair_id}: "
            f"{result.pair.self_speaker}: {result.pair.self_text} -> "
            f"{result.pair.target_speaker}: {result.pair.target_reply}"
        )
        for result in pair_retrieval.results
    ]
    return (
        contexts,
        [pair_retrieval.trace],
        [f"style_pair_top_k={pair_retrieval.trace.result_count}"],
    )


def _merge_retrieval_update(
    state: AgentState,
    contexts: list[str],
    traces: list[RetrievalTrace],
    actions: list[str],
    update: dict[str, object],
) -> dict[str, object]:
    _extend_retrieval_update(state, contexts, traces, actions, update)
    return {
        "retrieved_context": contexts,
        "retrieval_trace": traces,
        "trace": _append_trace(state, "retrieve_context", ";".join(actions)),
    }


def _extend_retrieval_update(
    state: AgentState,
    contexts: list[str],
    traces: list[RetrievalTrace],
    actions: list[str],
    update: dict[str, object],
) -> None:
    update_context = update.get("retrieved_context", [])
    if isinstance(update_context, list):
        contexts.extend(str(item) for item in update_context)
    update_trace = update.get("retrieval_trace", [])
    if isinstance(update_trace, list):
        traces.extend(
            trace for trace in update_trace if isinstance(trace, RetrievalTrace)
        )
    update_events = update.get("trace", [])
    if isinstance(update_events, list):
        for event in update_events[len(state["trace"]) :]:
            if isinstance(event, TraceEvent) and event.node == "retrieve_context":
                actions.append(event.action)


def _should_retrieve_memory_for_chat(
    state: AgentState,
    memory_store: MemoryStore | None,
    auto_memory_on_chat: bool,
) -> bool:
    if not auto_memory_on_chat or memory_store is None:
        return False
    decision = state["decision"]
    return (
        state["request"].conversation_type == PRIVATE_CONVERSATION_TYPE
        and decision.should_reply
        and decision.intent
        in {
            DialogueIntent.SMALLTALK,
            DialogueIntent.STYLE_CHAT,
            DialogueIntent.KNOWLEDGE_QUESTION,
        }
    )


def _should_use_style_for_private_chat(
    state: AgentState,
    *,
    style_store: StyleStore | None,
    style_on_smalltalk: bool,
    style_on_private_chat: bool,
) -> bool:
    if style_store is None:
        return False
    decision = state["decision"]
    if style_on_smalltalk and decision.intent == DialogueIntent.SMALLTALK:
        return (
            state["request"].conversation_type == PRIVATE_CONVERSATION_TYPE
            and decision.should_reply
            and not decision.need_knowledge
            and not decision.need_memory
            and not decision.need_tool
            and not decision.need_human_review
        )
    if not style_on_private_chat:
        return False
    return (
        state["request"].conversation_type == PRIVATE_CONVERSATION_TYPE
        and decision.should_reply
        and not decision.need_tool
        and not decision.need_human_review
        and decision.intent
        in {
            DialogueIntent.SMALLTALK,
            DialogueIntent.KNOWLEDGE_QUESTION,
            DialogueIntent.MEMORY_QUERY,
            DialogueIntent.HISTORY_SUMMARY,
        }
    )


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


def _finalize_reply(
    state: AgentState,
    *,
    memory_store: MemoryStore | None,
    auto_memory_on_chat: bool,
    auto_memory_user_name: str,
    auto_memory_persona_name: str,
    style_learning_store: StyleLearningStore | None,
) -> dict[str, object]:
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
    trace = _append_trace(state, "finalize_reply", action)
    if command.should_send:
        saved_count = _save_auto_chat_memory(
            state,
            memory_store=memory_store,
            enabled=auto_memory_on_chat,
            user_name=auto_memory_user_name,
            persona_name=auto_memory_persona_name,
        )
        if saved_count:
            trace = [
                *trace,
                TraceEvent(
                    node="finalize_reply",
                    action=f"auto_memory_saved={saved_count}",
                ),
            ]
        reinforced = _reinforce_style_from_reply(
            state,
            style_learning_store=style_learning_store,
        )
        if reinforced:
            trace = [*trace, TraceEvent(node="finalize_reply", action="style_feedback_saved=1")]
    return {
        "final_command": command,
        "trace": trace,
    }


def _save_auto_chat_memory(
    state: AgentState,
    *,
    memory_store: MemoryStore | None,
    enabled: bool,
    user_name: str,
    persona_name: str,
) -> int:
    if not enabled or memory_store is None or not _is_auto_memory_candidate(state):
        return 0
    request = state["request"]
    draft = state["draft"].strip()
    saved_count = 0
    if request.text.strip():
        memory_store.save_memory(
            user_id=request.sender_id,
            content=f"{user_name}说：{request.text.strip()}",
            source_message_id=request.message_id,
            memory_id=f"chat-user-{request.sender_id}-{request.message_id}",
        )
        saved_count += 1
    if draft:
        memory_store.save_memory(
            user_id=request.sender_id,
            content=f"{persona_name}回复：{draft}",
            source_message_id=request.message_id,
            memory_id=f"chat-agent-{request.sender_id}-{request.message_id}",
        )
        saved_count += 1
    return saved_count


def _reinforce_style_from_reply(
    state: AgentState,
    *,
    style_learning_store: StyleLearningStore | None,
) -> bool:
    if style_learning_store is None or not _is_auto_memory_candidate(state):
        return False
    generation_trace = state["generation_trace"]
    if generation_trace is not None and generation_trace.fallback_used:
        return False
    sample = style_learning_store.learn_reply(
        text=state["draft"],
        source_message_id=state["request"].message_id,
        timestamp_ms=state["request"].timestamp_ms,
    )
    return sample is not None


def _is_auto_memory_candidate(state: AgentState) -> bool:
    return (
        state["request"].conversation_type == PRIVATE_CONVERSATION_TYPE
        and state["decision"].intent
        in {
            DialogueIntent.SMALLTALK,
            DialogueIntent.STYLE_CHAT,
            DialogueIntent.KNOWLEDGE_QUESTION,
            DialogueIntent.MEMORY_QUERY,
            DialogueIntent.HISTORY_SUMMARY,
        }
    )


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
    for item in context:
        if item.startswith("style_example:"):
            payload = item[len("style_example:") :]
            source_id, separator, text = payload.partition(":")
            if separator and source_id.strip() and text.strip():
                sources.append(LeakageSource(source_id=source_id.strip(), text=text.strip()))
        elif item.startswith("style_pair:"):
            payload = item[len("style_pair:") :]
            source_id, separator, text = payload.partition(":")
            if not separator or not source_id.strip() or not text.strip():
                continue
            _context_text, arrow, target_text = text.partition("->")
            leakage_text = target_text if arrow else text
            if leakage_text.strip():
                sources.append(
                    LeakageSource(source_id=source_id.strip(), text=leakage_text.strip())
                )
    return sources
