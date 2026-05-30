from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field


def _chat_payload(text: str, *, run_id: str = "run-tools") -> dict[str, object]:
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


def _memory_store(tmp_path: Path):
    from agent_service.memory.memory_store import MemoryStore
    from agent_service.rag.embeddings import MockEmbeddingClient

    return MemoryStore(
        sqlite_path=tmp_path / "memory.sqlite3",
        chroma_path=tmp_path / "chroma",
        embedding_client=MockEmbeddingClient(),
        top_k=3,
    )


class EchoInput(BaseModel):
    text: str = Field(min_length=1)


class EchoOutput(BaseModel):
    text: str


class EmptyInput(BaseModel):
    pass


class OkOutput(BaseModel):
    ok: bool


def test_tool_registry_validates_input_schema_and_returns_error_envelope() -> None:
    from agent_service.tools import ToolRegistry, ToolRuntimeContext, ToolSpec

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            input_model=EchoInput,
            output_model=EchoOutput,
            handler=lambda tool_input, _context: EchoOutput(text=tool_input.text),
        )
    )

    result = registry.execute("echo", {"text": ""}, ToolRuntimeContext())

    assert result.ok is False
    assert result.output == {}
    assert result.error is not None
    assert result.error.code == "tool_schema_validation_error"
    assert result.trace.tool_name == "echo"
    assert result.trace.status == "error"


def test_tool_timeout_is_recorded_in_trace() -> None:
    from agent_service.tools import ToolRegistry, ToolRuntimeContext, ToolSpec

    def slow_tool(_tool_input: EmptyInput, _context: ToolRuntimeContext) -> OkOutput:
        time.sleep(0.2)
        return OkOutput(ok=True)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="slow_tool",
            input_model=EmptyInput,
            output_model=OkOutput,
            handler=slow_tool,
            timeout_seconds=0.01,
        )
    )

    result = registry.execute("slow_tool", {}, ToolRuntimeContext())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "tool_timeout"
    assert result.trace.status == "timeout"
    assert result.trace.timed_out is True
    assert result.trace.duration_ms >= 0.0


def test_builtin_tool_registry_exposes_required_safe_tools() -> None:
    from agent_service.tools.builtin import build_default_tool_registry

    registry = build_default_tool_registry()

    assert {
        "save_memory",
        "deactivate_memory",
        "get_user_profile",
        "summarize_recent_context",
        "search_recent_context",
        "liteim_context_tool",
    }.issubset(set(registry.list_tool_names()))


def test_save_memory_tool_is_idempotent(tmp_path: Path) -> None:
    from agent_service.tools import ToolRuntimeContext
    from agent_service.tools.builtin import build_default_tool_registry

    store = _memory_store(tmp_path)
    registry = build_default_tool_registry()
    context = ToolRuntimeContext(memory_store=store)
    payload = {
        "user_id": 1002,
        "content": "我喜欢简短直接的回答",
        "source_message_id": 9001,
        "importance": 0.8,
        "idempotency_key": "message-9001",
    }

    first = registry.execute("save_memory", payload, context)
    second = registry.execute("save_memory", payload, context)

    assert first.ok is True
    assert second.ok is True
    assert first.output["memory_id"] == "mem-1002-9001"
    assert second.output["memory_id"] == first.output["memory_id"]
    assert second.trace.idempotency_key == "message-9001"
    assert second.trace.idempotent_replay is True
    assert len(store.list_memories(user_id=1002)) == 1


def test_tool_failure_returns_error_and_does_not_raise() -> None:
    from agent_service.tools import ToolRegistry, ToolRuntimeContext, ToolSpec

    def failing_tool(_tool_input: EmptyInput, _context: ToolRuntimeContext) -> OkOutput:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="failing_tool",
            input_model=EmptyInput,
            output_model=OkOutput,
            handler=failing_tool,
        )
    )

    result = registry.execute("failing_tool", {}, ToolRuntimeContext())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "tool_execution_error"
    assert "boom" in result.error.message
    assert result.trace.status == "error"


def test_workflow_tool_router_writes_tool_result_into_state() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.tools.builtin import build_default_tool_registry
    from agent_service.workflow import run_agent_workflow

    state = run_agent_workflow(
        ChatRequest.model_validate(
            _chat_payload('/tool get_user_profile {"user_id": 1002}', run_id="run-profile-tool")
        ),
        tool_registry=build_default_tool_registry(),
    )

    result = json.loads(state["tool_results"][0])

    assert state["decision"].need_tool is True
    assert state["tool_calls"] == ["get_user_profile"]
    assert result["ok"] is True
    assert result["output"]["user_id"] == 1002
    assert result["output"]["profile_source"] == "chat_request"
    assert any(event.action == "tool:get_user_profile:ok" for event in state["trace"])


def test_workflow_tool_failure_does_not_crash_graph() -> None:
    from agent_service.schemas import ChatRequest
    from agent_service.tools import ToolRegistry, ToolRuntimeContext, ToolSpec
    from agent_service.workflow import run_agent_workflow

    def failing_tool(_tool_input: EmptyInput, _context: ToolRuntimeContext) -> OkOutput:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="failing_tool",
            input_model=EmptyInput,
            output_model=OkOutput,
            handler=failing_tool,
        )
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("/tool failing_tool {}")),
        tool_registry=registry,
    )
    result = json.loads(state["tool_results"][0])

    assert state["tool_calls"] == ["failing_tool"]
    assert result["ok"] is False
    assert result["error"]["code"] == "tool_execution_error"
    assert state["final_command"].should_send is True
    assert any(event.action == "tool:failing_tool:error" for event in state["trace"])
