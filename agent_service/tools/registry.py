from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agent_service.memory.memory_store import MemoryStore
from agent_service.schemas import ChatRequest

ToolHandler = Callable[[BaseModel, "ToolRuntimeContext"], BaseModel | dict[str, object]]


class ToolErrorEnvelope(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = True


class ToolTrace(BaseModel):
    tool_name: str = Field(min_length=1)
    status: Literal["ok", "error", "timeout"]
    duration_ms: float = Field(ge=0.0)
    timed_out: bool = False
    idempotency_key: str | None = None
    idempotent_replay: bool = False


class ToolExecutionResult(BaseModel):
    tool_name: str = Field(min_length=1)
    ok: bool
    output: dict[str, object] = Field(default_factory=dict)
    error: ToolErrorEnvelope | None = None
    trace: ToolTrace


@dataclass(frozen=True)
class ToolRuntimeContext:
    request: ChatRequest | None = None
    memory_store: MemoryStore | None = None
    recent_context: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    timeout_seconds: float = 2.0
    side_effect: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._idempotent_results: dict[tuple[str, str], ToolExecutionResult] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def list_tool_names(self) -> list[str]:
        return sorted(self._tools)

    def execute(
        self,
        name: str,
        payload: object,
        context: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        started_at = perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return _error_result(
                name,
                "tool_not_found",
                f"tool not found: {name}",
                started_at=started_at,
                retryable=False,
            )

        try:
            tool_input = spec.input_model.model_validate(payload)
        except ValidationError as exc:
            return _error_result(
                spec.name,
                "tool_schema_validation_error",
                str(exc),
                started_at=started_at,
                retryable=False,
            )

        idempotency_key = _idempotency_key(tool_input) if spec.side_effect else None
        if spec.side_effect and idempotency_key is None:
            return _error_result(
                spec.name,
                "tool_idempotency_key_required",
                f"side-effect tool requires idempotency_key: {spec.name}",
                started_at=started_at,
                retryable=False,
            )

        if idempotency_key is not None:
            cached = self._idempotent_results.get((spec.name, idempotency_key))
            if cached is not None:
                trace = cached.trace.model_copy(
                    update={
                        "duration_ms": _duration_ms(started_at),
                        "idempotency_key": idempotency_key,
                        "idempotent_replay": True,
                    }
                )
                return cached.model_copy(update={"trace": trace})

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(spec.handler, tool_input, context)
        try:
            raw_output = future.result(timeout=spec.timeout_seconds)
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return _error_result(
                spec.name,
                "tool_timeout",
                f"tool timed out after {spec.timeout_seconds:.3f}s",
                started_at=started_at,
                status="timeout",
                timed_out=True,
                idempotency_key=idempotency_key,
                retryable=True,
            )
        except Exception as exc:
            executor.shutdown(wait=True)
            return _error_result(
                spec.name,
                "tool_execution_error",
                str(exc),
                started_at=started_at,
                idempotency_key=idempotency_key,
                retryable=True,
            )
        executor.shutdown(wait=True)

        try:
            output_model = spec.output_model.model_validate(raw_output)
        except ValidationError as exc:
            return _error_result(
                spec.name,
                "tool_output_validation_error",
                str(exc),
                started_at=started_at,
                idempotency_key=idempotency_key,
                retryable=True,
            )

        result = ToolExecutionResult(
            tool_name=spec.name,
            ok=True,
            output=output_model.model_dump(mode="json"),
            trace=ToolTrace(
                tool_name=spec.name,
                status="ok",
                duration_ms=_duration_ms(started_at),
                idempotency_key=idempotency_key,
            ),
        )
        if idempotency_key is not None:
            self._idempotent_results[(spec.name, idempotency_key)] = result
        return result


def _idempotency_key(tool_input: BaseModel) -> str | None:
    value = getattr(tool_input, "idempotency_key", None)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _error_result(
    tool_name: str,
    code: str,
    message: str,
    *,
    started_at: float,
    status: Literal["error", "timeout"] = "error",
    timed_out: bool = False,
    idempotency_key: str | None = None,
    retryable: bool,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name or "unknown_tool",
        ok=False,
        error=ToolErrorEnvelope(code=code, message=message or code, retryable=retryable),
        trace=ToolTrace(
            tool_name=tool_name or "unknown_tool",
            status=status,
            duration_ms=_duration_ms(started_at),
            timed_out=timed_out,
            idempotency_key=idempotency_key,
        ),
    )


def _duration_ms(started_at: float) -> float:
    return max((perf_counter() - started_at) * 1000.0, 0.0)
