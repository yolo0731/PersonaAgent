from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from agent_service.config import Settings
from agent_service.dialogue_policy import DialoguePolicy
from agent_service.eval.cases import (
    EvalDatasets,
    IntegrationEvalCase,
    RealEvalCase,
    RealEvalCaseResult,
    load_eval_datasets,
)
from agent_service.eval.metrics import EvalReport, evaluate_datasets
from agent_service.eval.report import write_json_report, write_markdown_report
from agent_service.llm import LLMClient, OpenAILLMClient
from agent_service.memory.memory_store import MemoryStore
from agent_service.persona import PersonaEngine
from agent_service.rag.knowledge_retriever import KnowledgeRetriever
from agent_service.schemas import AgentReplyCommand, ChatRequest
from agent_service.style.learning import StyleLearningStore
from agent_service.style.pair_store import StylePairStore
from agent_service.style.style_store import StyleStore
from agent_service.tools import ToolRegistry
from agent_service.workflow import AgentState, run_agent_workflow


class EvalMode(StrEnum):
    MOCK = "mock"
    REAL = "real"


class EvalOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: EvalMode = EvalMode.MOCK
    prompt_token_cost_per_1k: float = Field(default=0.0, ge=0.0)
    completion_token_cost_per_1k: float = Field(default=0.0, ge=0.0)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str | None = None
    real_eval_confirm: bool = False
    max_cases: int = Field(default=200, ge=1)
    concurrency: int = Field(default=1, ge=1)
    resume: bool = False
    fail_fast: bool = False
    sample_seed: int | None = None
    settings: Settings | None = None
    llm_client: LLMClient | None = None


class EvalRunOutput(BaseModel):
    json_path: Path
    markdown_path: Path
    results_path: Path


def run_eval_suite(
    *,
    datasets_dir: str | Path,
    output_dir: str | Path,
    options: EvalOptions | None = None,
) -> EvalRunOutput:
    effective_options = options or EvalOptions()
    if effective_options.mode == EvalMode.REAL:
        _require_real_eval_key(effective_options)
        _require_real_eval_confirmation(effective_options)

    datasets = load_eval_datasets(datasets_dir)
    root = Path(output_dir)
    results_path = root / f"{effective_options.mode.value}_eval_results.jsonl"
    if effective_options.mode == EvalMode.REAL:
        real_results = _run_real_eval_cases(
            datasets.real_cases,
            output_path=results_path,
            options=effective_options,
        )
        report = _evaluate_real_results(
            real_results,
            options=effective_options,
        )
    else:
        real_results = []
        datasets = datasets.model_copy(update={"real_cases": []})
        report = evaluate_datasets(
            datasets,
            mode="mock",
            prompt_token_cost_per_1k=effective_options.prompt_token_cost_per_1k,
            completion_token_cost_per_1k=effective_options.completion_token_cost_per_1k,
        ).model_copy(update={"real_case_results": real_results})
    json_path = root / f"{effective_options.mode.value}_eval_report.json"
    markdown_path = root / f"{effective_options.mode.value}_eval_report.md"
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path, json_name=json_path.name)
    return EvalRunOutput(
        json_path=json_path,
        markdown_path=markdown_path,
        results_path=results_path,
    )


def _require_real_eval_key(options: EvalOptions) -> None:
    if options.openai_api_key or os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError("OPENAI_API_KEY is required for real eval mode")


def _require_real_eval_confirmation(options: EvalOptions) -> None:
    if options.real_eval_confirm or os.getenv("REAL_EVAL_CONFIRM") == "1":
        return
    raise RuntimeError("REAL_EVAL_CONFIRM=1 is required for real eval mode")


def _run_real_eval_cases(
    cases: list[RealEvalCase],
    *,
    output_path: Path,
    options: EvalOptions,
) -> list[RealEvalCaseResult]:
    selected_cases = _select_real_cases(cases, options=options)
    existing = _load_existing_real_results(output_path) if options.resume else {}
    pending = [case for case in selected_cases if case.case_id not in existing]
    new_results: list[RealEvalCaseResult] = []
    if pending:
        llm_client = options.llm_client or _build_real_eval_llm_client(options)
        runtime = _build_real_eval_runtime(options, llm_client=llm_client)
        if options.concurrency == 1 or len(pending) == 1 or options.fail_fast:
            for case in pending:
                result = _run_one_real_case(
                    case,
                    runtime=runtime,
                    options=options,
                )
                new_results.append(result)
                if options.fail_fast and not result.passed:
                    break
        else:
            with ThreadPoolExecutor(max_workers=options.concurrency) as executor:
                new_results = list(
                    executor.map(
                        lambda case: _run_one_real_case(
                            case,
                            runtime=runtime,
                            options=options,
                        ),
                        pending,
                    )
                )
    merged = [
        existing[case.case_id]
        for case in selected_cases
        if case.case_id in existing
    ]
    merged.extend(new_results)
    _write_real_results(output_path, merged)
    return merged


def _select_real_cases(
    cases: list[RealEvalCase],
    *,
    options: EvalOptions,
) -> list[RealEvalCase]:
    selected = list(cases)
    if options.sample_seed is not None:
        random.Random(options.sample_seed).shuffle(selected)
    return selected[: options.max_cases]


def _run_one_real_case(
    case: RealEvalCase,
    *,
    runtime: RealEvalRuntime,
    options: EvalOptions,
) -> RealEvalCaseResult:
    request = _request_from_real_case(case)
    started_at = perf_counter()
    state = run_agent_workflow(
        request,
        dialogue_policy=runtime.dialogue_policy,
        knowledge_retriever=runtime.knowledge_retriever,
        memory_store=runtime.memory_store,
        style_store=runtime.style_store,
        style_pair_store=runtime.style_pair_store,
        tool_registry=runtime.tool_registry,
        persona_engine=runtime.persona_engine,
        llm_client=runtime.llm_client,
        rag_top_k=runtime.settings.rag_top_k,
        memory_top_k=runtime.settings.memory_top_k,
        style_top_k=runtime.settings.style_top_k,
        style_pair_top_k=runtime.settings.style_pair_top_k,
        style_persona_id=runtime.settings.style_persona_id,
        style_on_smalltalk=runtime.settings.style_on_smalltalk,
        style_on_private_chat=runtime.settings.style_on_private_chat,
        auto_memory_on_chat=runtime.settings.auto_memory_on_chat,
        auto_memory_user_name=runtime.settings.auto_memory_user_name,
        auto_memory_persona_name=runtime.settings.auto_memory_persona_name,
        style_learning_store=runtime.style_learning_store,
    )
    latency_ms = max((perf_counter() - started_at) * 1000.0, 0.0)
    command = state["final_command"]
    retrieval_ids = _retrieval_ids(state)
    safety_reason = state["safety_result"].reason
    review_reason = safety_reason if state["safety_result"].needs_human_review else None
    prompt_tokens = _prompt_tokens(state)
    completion_tokens = _completion_tokens(state)
    failures = _real_case_failures(
        case,
        command=command,
        retrieval_ids=retrieval_ids,
        safety_reason=safety_reason,
        review_reason=review_reason,
    )
    return RealEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        variant=case.variant,
        user_message=case.user_message,
        expected_behavior=case.expected_behavior,
        actual_reply=command.text,
        final_command=command,
        retrieval_ids=retrieval_ids,
        safety_reason=safety_reason,
        review_reason=review_reason,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost=_token_cost(
            prompt_tokens,
            completion_tokens,
            options=options,
        ),
        passed=not failures,
        failure_reasons=failures,
        trace_summary=command.trace_summary,
    )


def _request_from_real_case(case: RealEvalCase) -> ChatRequest:
    suffix = sum(ord(char) for char in case.case_id)
    message_id = 900_000 + suffix
    return ChatRequest(
        run_id=f"eval-{case.case_id}",
        conversation_type=1,
        conversation_id=800_001,
        message_id=message_id,
        sender_id=1002,
        receiver_id=1001,
        text=case.user_message,
        timestamp_ms=1_700_000_001_000 + suffix,
        client_message_id=f"eval-client-{case.case_id}",
    )


def _real_case_failures(
    case: RealEvalCase,
    *,
    command: AgentReplyCommand,
    retrieval_ids: list[str],
    safety_reason: str | None,
    review_reason: str | None,
) -> list[str]:
    failures: list[str] = []
    if command.should_send != case.expected_should_send:
        failures.append(
            f"should_send expected {case.expected_should_send} got {command.should_send}"
        )
    if case.expected_reply_contains:
        lowered_reply = command.text.casefold()
        for marker in case.expected_reply_contains:
            if marker.casefold() not in lowered_reply:
                failures.append(f"reply missing marker: {marker}")
    if case.expected_retrieval_ids:
        missing = [item for item in case.expected_retrieval_ids if item not in retrieval_ids]
        if missing:
            failures.append(f"missing retrieval ids: {', '.join(missing)}")
    if case.expected_human_review and review_reason is None:
        failures.append("expected human review but no review reason was recorded")
    if not case.expected_human_review and review_reason is not None:
        failures.append(f"unexpected human review: {review_reason}")
    if case.expected_safety_reason is not None and case.expected_safety_reason != safety_reason:
        failures.append(
            f"safety reason expected {case.expected_safety_reason} got {safety_reason}"
        )
    return failures


def _retrieval_ids(state: AgentState) -> list[str]:
    ids: list[str] = []
    for trace in state["retrieval_trace"]:
        ids.extend(trace.chunk_ids)
    return ids


def _prompt_tokens(state: AgentState) -> int:
    trace = state["generation_trace"]
    return trace.prompt_tokens if trace is not None else 0


def _completion_tokens(state: AgentState) -> int:
    trace = state["generation_trace"]
    return trace.completion_tokens if trace is not None else 0


def _token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    options: EvalOptions,
) -> float:
    return round(
        (prompt_tokens / 1000.0) * options.prompt_token_cost_per_1k
        + (completion_tokens / 1000.0) * options.completion_token_cost_per_1k,
        8,
    )


def _integration_cases_from_real_results(
    results: list[RealEvalCaseResult],
) -> list[IntegrationEvalCase]:
    return [
        IntegrationEvalCase(
            case_id=result.case_id,
            variant=result.variant,
            should_send=result.final_command.should_send,
            sent=result.final_command.should_send,
            failed=not result.passed,
            duplicate_send_count=0,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            request_text=result.user_message,
        )
        for result in results
    ]


def _build_real_eval_llm_client(options: EvalOptions) -> LLMClient:
    settings = options.settings or Settings()
    return OpenAILLMClient(
        api_key=options.openai_api_key or settings.openai_api_key,
        model=options.llm_model or settings.llm_model,
        base_url=options.openai_base_url or settings.openai_base_url,
        timeout_seconds=settings.llm_request_timeout_seconds,
    )


@dataclass(frozen=True)
class RealEvalRuntime:
    settings: Settings
    llm_client: LLMClient
    dialogue_policy: DialoguePolicy
    knowledge_retriever: KnowledgeRetriever
    memory_store: MemoryStore
    style_store: StyleStore
    style_pair_store: StylePairStore | None
    tool_registry: ToolRegistry
    persona_engine: PersonaEngine
    style_learning_store: StyleLearningStore | None


def _build_real_eval_runtime(options: EvalOptions, *, llm_client: LLMClient) -> RealEvalRuntime:
    from agent_service.main import create_app

    settings = options.settings or Settings()
    app = create_app(settings=settings, llm_client=llm_client)
    container = app.state.get_container()
    return RealEvalRuntime(
        settings=settings,
        llm_client=llm_client,
        dialogue_policy=container.dialogue_policy,
        knowledge_retriever=container.knowledge_retriever,
        memory_store=container.memory_store,
        style_store=container.style_store,
        style_pair_store=container.style_pair_store,
        tool_registry=container.tool_registry,
        persona_engine=container.persona_engine,
        style_learning_store=container.style_learning_store,
    )


def _evaluate_real_results(
    results: list[RealEvalCaseResult],
    *,
    options: EvalOptions,
) -> EvalReport:
    report = evaluate_datasets(
        EvalDatasets(integration_cases=_integration_cases_from_real_results(results)),
        mode=EvalMode.REAL.value,
        prompt_token_cost_per_1k=options.prompt_token_cost_per_1k,
        completion_token_cost_per_1k=options.completion_token_cost_per_1k,
    )
    return report.model_copy(
        update={
            "sample_size": report.sample_size.model_copy(
                update={
                    "integration": 0,
                    "real": len(results),
                    "total": len(results),
                }
            ),
            "real_case_results": results,
        }
    )


def _load_existing_real_results(path: Path) -> dict[str, RealEvalCaseResult]:
    if not path.exists():
        return {}
    rows: dict[str, RealEvalCaseResult] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        result = RealEvalCaseResult.model_validate(json.loads(line))
        rows[result.case_id] = result
    return rows


def _write_real_results(path: Path, results: list[RealEvalCaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            result.model_dump_json() + "\n"
            for result in results
        ),
        encoding="utf-8",
    )
