from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent_service.eval.cases import (
    EvalDatasets,
    EvalVariant,
    IntegrationEvalCase,
    RetrievalEvalCase,
    SafetyEvalCase,
    StyleEvalCase,
)
from agent_service.safety.verbatim_guard import VerbatimLeakageGuard

AB_VARIANTS: tuple[EvalVariant, ...] = (
    "no_rag",
    "knowledge",
    "knowledge_memory",
    "knowledge_memory_style",
)


class EvalSampleSize(BaseModel):
    rag: int = Field(ge=0)
    memory: int = Field(ge=0)
    style: int = Field(ge=0)
    safety: int = Field(ge=0)
    integration: int = Field(ge=0)
    total: int = Field(ge=0)


class EvalMetrics(BaseModel):
    retrieval_hit_at_5: float = Field(ge=0.0, le=1.0)
    memory_hit_at_5: float = Field(ge=0.0, le=1.0)
    style_similarity: float = Field(ge=0.0, le=1.0)
    verbatim_leakage_rate: float = Field(ge=0.0, le=1.0)
    safety_violation_rate: float = Field(ge=0.0, le=1.0)
    human_review_trigger_rate: float = Field(ge=0.0, le=1.0)
    average_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    token_cost_per_reply: float = Field(ge=0.0)
    liteim_integration_success_rate: float = Field(ge=0.0, le=1.0)


class EvalVariantReport(BaseModel):
    sample_size: int = Field(ge=0)
    average_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    token_cost_per_reply: float = Field(ge=0.0)
    liteim_integration_success_rate: float = Field(ge=0.0, le=1.0)


class EvalReport(BaseModel):
    mode: Literal["mock", "real"]
    generated_at: str
    sample_size: EvalSampleSize
    metrics: EvalMetrics
    ab_variants: dict[str, EvalVariantReport]


def evaluate_datasets(
    datasets: EvalDatasets,
    *,
    mode: Literal["mock", "real"] = "mock",
    prompt_token_cost_per_1k: float = 0.0,
    completion_token_cost_per_1k: float = 0.0,
) -> EvalReport:
    return EvalReport(
        mode=mode,
        generated_at=datetime.now(UTC).isoformat(),
        sample_size=_sample_size(datasets),
        metrics=EvalMetrics(
            retrieval_hit_at_5=_hit_at_5(datasets.rag_cases),
            memory_hit_at_5=_hit_at_5(datasets.memory_cases),
            style_similarity=_style_similarity(datasets.style_cases),
            verbatim_leakage_rate=_verbatim_leakage_rate(datasets.safety_cases),
            safety_violation_rate=_safety_violation_rate(datasets.safety_cases),
            human_review_trigger_rate=_human_review_trigger_rate(datasets.safety_cases),
            average_latency_ms=_average_latency(datasets.integration_cases),
            p95_latency_ms=_p95_latency(datasets.integration_cases),
            token_cost_per_reply=_token_cost_per_reply(
                datasets.integration_cases,
                prompt_token_cost_per_1k=prompt_token_cost_per_1k,
                completion_token_cost_per_1k=completion_token_cost_per_1k,
            ),
            liteim_integration_success_rate=_integration_success_rate(
                datasets.integration_cases
            ),
        ),
        ab_variants={
            variant: _variant_report(
                [case for case in datasets.integration_cases if case.variant == variant],
                prompt_token_cost_per_1k=prompt_token_cost_per_1k,
                completion_token_cost_per_1k=completion_token_cost_per_1k,
            )
            for variant in AB_VARIANTS
        },
    )


def _sample_size(datasets: EvalDatasets) -> EvalSampleSize:
    rag = len(datasets.rag_cases)
    memory = len(datasets.memory_cases)
    style = len(datasets.style_cases)
    safety = len(datasets.safety_cases)
    integration = len(datasets.integration_cases)
    return EvalSampleSize(
        rag=rag,
        memory=memory,
        style=style,
        safety=safety,
        integration=integration,
        total=rag + memory + style + safety + integration,
    )


def _hit_at_5(cases: Sequence[RetrievalEvalCase]) -> float:
    if not cases:
        return 0.0
    hits = sum(
        1
        for case in cases
        if set(case.expected_ids).intersection(case.retrieved_ids[:5])
    )
    return _rate(hits, len(cases))


def _style_similarity(cases: Sequence[StyleEvalCase]) -> float:
    if not cases:
        return 0.0
    total = 0.0
    for case in cases:
        reply = case.reply_text.casefold()
        matched = sum(1 for marker in case.expected_markers if marker.casefold() in reply)
        total += _rate(matched, len(case.expected_markers))
    return _rounded(total / len(cases))


def _verbatim_leakage_rate(cases: Sequence[SafetyEvalCase]) -> float:
    if not cases:
        return 0.0
    guard = VerbatimLeakageGuard()
    total = 0.0
    for case in cases:
        if not case.style_sources:
            continue
        assessment = guard.assess(case.reply_text, case.style_sources)
        total += assessment.metrics.verbatim_leakage_rate
    return _rounded(total / len(cases))


def _safety_violation_rate(cases: Sequence[SafetyEvalCase]) -> float:
    if not cases:
        return 0.0
    violations = sum(1 for case in cases if case.expected_blocked and not case.blocked)
    return _rate(violations, len(cases))


def _human_review_trigger_rate(cases: Sequence[SafetyEvalCase]) -> float:
    if not cases:
        return 0.0
    triggered = sum(1 for case in cases if case.needs_human_review)
    return _rate(triggered, len(cases))


def _average_latency(cases: Sequence[IntegrationEvalCase]) -> float:
    if not cases:
        return 0.0
    return _rounded(sum(case.latency_ms for case in cases) / len(cases))


def _p95_latency(cases: Sequence[IntegrationEvalCase]) -> float:
    if not cases:
        return 0.0
    ordered = sorted(case.latency_ms for case in cases)
    index = max(0, _ceil(0.95 * len(ordered)) - 1)
    return _rounded(ordered[index])


def _token_cost_per_reply(
    cases: Sequence[IntegrationEvalCase],
    *,
    prompt_token_cost_per_1k: float,
    completion_token_cost_per_1k: float,
) -> float:
    if not cases:
        return 0.0
    total_cost = sum(
        (case.prompt_tokens / 1000.0) * prompt_token_cost_per_1k
        + (case.completion_tokens / 1000.0) * completion_token_cost_per_1k
        for case in cases
    )
    return _rounded(total_cost / len(cases), digits=8)


def _integration_success_rate(cases: Sequence[IntegrationEvalCase]) -> float:
    if not cases:
        return 0.0
    successes = sum(1 for case in cases if _integration_success(case))
    return _rate(successes, len(cases))


def _integration_success(case: IntegrationEvalCase) -> bool:
    return (
        case.sent == case.should_send
        and not case.failed
        and case.duplicate_send_count == 0
    )


def _variant_report(
    cases: Sequence[IntegrationEvalCase],
    *,
    prompt_token_cost_per_1k: float,
    completion_token_cost_per_1k: float,
) -> EvalVariantReport:
    return EvalVariantReport(
        sample_size=len(cases),
        average_latency_ms=_average_latency(cases),
        p95_latency_ms=_p95_latency(cases),
        token_cost_per_reply=_token_cost_per_reply(
            cases,
            prompt_token_cost_per_1k=prompt_token_cost_per_1k,
            completion_token_cost_per_1k=completion_token_cost_per_1k,
        ),
        liteim_integration_success_rate=_integration_success_rate(cases),
    )


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _rounded(count / total)


def _rounded(value: float, *, digits: int = 6) -> float:
    return round(value, digits)


def _ceil(value: float) -> int:
    integer = int(value)
    if value == integer:
        return integer
    return integer + 1
