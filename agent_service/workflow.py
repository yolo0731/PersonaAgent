from __future__ import annotations

from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent_service.dialogue_policy import DialogueDecision, DialogueIntent, DialoguePolicy
from agent_service.rag.documents import RetrievalTrace
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.review import (
    HumanReviewStore,
    ReviewStatus,
    make_thread_id,
    resume_human_review,
)
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


class TraceEvent(BaseModel):
    node: str
    action: str


class AgentState(TypedDict):
    request: ChatRequest
    run_id: str
    decision: DialogueDecision
    retrieved_context: list[str]
    retrieval_trace: list[RetrievalTrace]
    tool_calls: list[str]
    tool_results: list[str]
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
        draft="",
        safety_result=SafetyResult(blocked=False),
        final_command=no_reply_command(request, "not_finalized"),
        trace=[],
    )


def build_agent_graph(
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    rag_top_k: int = 5,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("dialogue_policy", _dialogue_policy)
    graph.add_node(
        "retrieve_context",
        lambda state: _retrieve_context(
            state,
            knowledge_retriever=knowledge_retriever,
            rag_top_k=rag_top_k,
        ),
    )
    graph.add_node("tool_router", _tool_router)
    graph.add_node("generate_reply", _generate_reply)
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
    rag_top_k: int = 5,
) -> AgentState:
    final_state = build_agent_graph(
        knowledge_retriever=knowledge_retriever,
        rag_top_k=rag_top_k,
    ).invoke(make_initial_agent_state(request))
    return cast(AgentState, final_state)


def run_agent_chat(
    request: ChatRequest,
    *,
    review_store: HumanReviewStore | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    rag_top_k: int = 5,
) -> AgentReplyCommand:
    state = run_agent_workflow(
        request,
        knowledge_retriever=knowledge_retriever,
        rag_top_k=rag_top_k,
    )
    if review_store is not None and state["decision"].need_human_review:
        thread_id = make_thread_id(request)
        existing = review_store.get_review(thread_id)
        if existing is not None and existing.status == ReviewStatus.COMPLETED:
            return no_reply_command(request, "human_review_already_resumed")
        review_store.create_pending(thread_id, state)
        return no_reply_command(request, "human_review_pending")
    return state["final_command"]


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
    rag_top_k: int,
) -> dict[str, object]:
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


def _tool_router(state: AgentState) -> dict[str, object]:
    return {
        "tool_calls": [],
        "tool_results": [],
        "trace": _append_trace(state, "tool_router", "mock_no_tools"),
    }


def _generate_reply(state: AgentState) -> dict[str, object]:
    draft = f"mock reply: {state['request'].text}"
    return {
        "draft": draft,
        "trace": _append_trace(state, "generate_reply", "mock_draft"),
    }


def _safety_check(state: AgentState) -> dict[str, object]:
    blocked = state["decision"].intent == DialogueIntent.UNSAFE
    result = SafetyResult(
        blocked=blocked,
        reason="mock_safety_block" if blocked else None,
    )
    return {
        "safety_result": result,
        "trace": _append_trace(state, "safety_check", "blocked" if blocked else "passed"),
    }


def _finalize_reply(state: AgentState) -> dict[str, object]:
    request = state["request"]
    if not state["decision"].should_reply:
        command = no_reply_command(request, "dialogue_policy_no_reply")
        action = "dialogue_policy_no_reply"
    elif state["safety_result"].blocked:
        command = no_reply_command(request, "safety_block")
        action = "safety_block"
    else:
        command = AgentReplyCommand(
            run_id=state["run_id"],
            source_message_id=request.message_id,
            should_send=True,
            receiver_id=request.sender_id,
            text=state["draft"],
            client_message_id=f"pa-{state['run_id']}",
            reason="graph_mock",
        )
        action = "send_mock_reply"
    return {
        "final_command": command,
        "trace": _append_trace(state, "finalize_reply", action),
    }


def _append_trace(state: AgentState, node: str, action: str) -> list[TraceEvent]:
    return [*state["trace"], TraceEvent(node=node, action=action)]
