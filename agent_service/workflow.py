from __future__ import annotations

from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent_service.schemas import AgentReplyCommand, ChatRequest, no_reply_command

EXPECTED_NODE_ORDER = [
    "dialogue_policy",
    "retrieve_context",
    "tool_router",
    "generate_reply",
    "safety_check",
    "finalize_reply",
]


class DialogueDecision(BaseModel):
    should_reply: bool
    reason: str


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
        decision=DialogueDecision(should_reply=True, reason="not_evaluated"),
        retrieved_context=[],
        tool_calls=[],
        tool_results=[],
        draft="",
        safety_result=SafetyResult(blocked=False),
        final_command=no_reply_command(request, "not_finalized"),
        trace=[],
    )


def build_agent_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("dialogue_policy", _dialogue_policy)
    graph.add_node("retrieve_context", _retrieve_context)
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


def run_agent_workflow(request: ChatRequest) -> AgentState:
    final_state = build_agent_graph().invoke(make_initial_agent_state(request))
    return cast(AgentState, final_state)


def run_agent_chat(request: ChatRequest) -> AgentReplyCommand:
    return run_agent_workflow(request)["final_command"]


def _dialogue_policy(state: AgentState) -> dict[str, object]:
    text = state["request"].text.strip()
    should_reply = text != "/no-reply"
    reason = "mock_should_reply" if should_reply else "mock_no_reply"
    return {
        "decision": DialogueDecision(should_reply=should_reply, reason=reason),
        "trace": _append_trace(state, "dialogue_policy", reason),
    }


def _route_after_dialogue_policy(state: AgentState) -> GraphRoute:
    if state["decision"].should_reply:
        return "retrieve_context"
    return "finalize_reply"


def _retrieve_context(state: AgentState) -> dict[str, object]:
    return {
        "retrieved_context": [],
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
    blocked = state["request"].text.strip().lower().startswith("/unsafe")
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
