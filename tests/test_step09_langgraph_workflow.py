from __future__ import annotations

from fastapi.testclient import TestClient


def _chat_payload(run_id: str = "run-graph-ok", text: str = "hello graph") -> dict[str, object]:
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


def test_agent_graph_runs_all_six_nodes_to_final_command() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import EXPECTED_NODE_ORDER, run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("run-graph-ok", "hello graph"))
    )

    assert state["run_id"] == "run-graph-ok"
    assert state["decision"].should_reply is True
    assert state["retrieved_context"] == []
    assert state["tool_calls"] == []
    assert state["tool_results"] == []
    assert state["draft"] == "mock reply: hello graph"
    assert state["safety_result"].blocked is False
    assert state["final_command"].should_send is True
    assert state["final_command"].receiver_id == 1002
    assert state["final_command"].text == "mock reply: hello graph"
    assert state["final_command"].client_message_id == "pa-run-graph-ok"
    assert [event.node for event in state["trace"]] == EXPECTED_NODE_ORDER


def test_agent_graph_short_circuits_no_reply_decision_to_noop_command() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("run-no-reply", "/no-reply"))
    )

    assert state["decision"].should_reply is False
    assert state["retrieved_context"] == []
    assert state["tool_calls"] == []
    assert state["tool_results"] == []
    assert state["draft"] == ""
    assert state["final_command"].should_send is False
    assert state["final_command"].reason == "dialogue_policy_no_reply"
    assert [event.node for event in state["trace"]] == [
        "dialogue_policy",
        "finalize_reply",
    ]


def test_agent_graph_safety_block_prevents_send_after_full_path() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import EXPECTED_NODE_ORDER, run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("run-safety", "/unsafe leak secrets"))
    )

    assert state["decision"].should_reply is True
    assert state["safety_result"].blocked is True
    assert state["safety_result"].reason == "mock_safety_block"
    assert state["final_command"].should_send is False
    assert state["final_command"].reason == "safety_block"
    assert [event.node for event in state["trace"]] == EXPECTED_NODE_ORDER


def test_chat_endpoint_uses_default_langgraph_handler_for_no_reply() -> None:
    from agent_service.config import Settings
    from agent_service.main import create_app

    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.post("/chat", json=_chat_payload("run-http-no-reply", "/no-reply"))

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["command"]["run_id"] == "run-http-no-reply"
    assert body["command"]["should_send"] is False
    assert body["command"]["reason"] == "dialogue_policy_no_reply"
