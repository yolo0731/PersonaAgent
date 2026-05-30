from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from agent_service.safety.verbatim_guard import LeakageSource
from agent_service.schemas import AgentReplyCommand

EvalVariant = Literal["no_rag", "knowledge", "knowledge_memory", "knowledge_memory_style"]
RealEvalCategory = Literal["knowledge", "memory", "style", "safety", "tool", "integration"]


class RetrievalEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_ids: list[str] = Field(min_length=1)
    retrieved_ids: list[str] = Field(default_factory=list)


class StyleEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_markers: list[str] = Field(min_length=1)
    reply_text: str


class SafetyEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    request_text: str = Field(min_length=1)
    reply_text: str
    expected_blocked: bool
    blocked: bool
    needs_human_review: bool = False
    style_sources: list[LeakageSource] = Field(default_factory=list)


class IntegrationEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    variant: EvalVariant
    should_send: bool
    sent: bool
    failed: bool = False
    duplicate_send_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    request_text: str = ""


class RealEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    category: RealEvalCategory
    variant: EvalVariant = "knowledge_memory_style"
    user_message: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    expected_reply_contains: list[str] = Field(default_factory=list)
    expected_retrieval_ids: list[str] = Field(default_factory=list)
    expected_should_send: bool = True
    expected_human_review: bool = False
    expected_safety_reason: str | None = None


class RealEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    category: RealEvalCategory
    variant: EvalVariant
    user_message: str
    expected_behavior: str
    actual_reply: str
    final_command: AgentReplyCommand
    retrieval_ids: list[str] = Field(default_factory=list)
    safety_reason: str | None = None
    review_reason: str | None = None
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    estimated_cost: float = Field(ge=0.0)
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    trace_summary: list[str] = Field(default_factory=list)


class EvalDatasets(BaseModel):
    rag_cases: list[RetrievalEvalCase] = Field(default_factory=list)
    memory_cases: list[RetrievalEvalCase] = Field(default_factory=list)
    style_cases: list[StyleEvalCase] = Field(default_factory=list)
    safety_cases: list[SafetyEvalCase] = Field(default_factory=list)
    integration_cases: list[IntegrationEvalCase] = Field(default_factory=list)
    real_cases: list[RealEvalCase] = Field(default_factory=list)


CaseModel = TypeVar("CaseModel", bound=BaseModel)


def load_eval_datasets(datasets_dir: str | Path) -> EvalDatasets:
    root = Path(datasets_dir)
    return EvalDatasets(
        rag_cases=_load_jsonl(root / "rag_cases.jsonl", RetrievalEvalCase),
        memory_cases=_load_jsonl(root / "memory_cases.jsonl", RetrievalEvalCase),
        style_cases=_load_jsonl(root / "style_cases.jsonl", StyleEvalCase),
        safety_cases=_load_jsonl(root / "safety_cases.jsonl", SafetyEvalCase),
        integration_cases=_load_jsonl(root / "integration_cases.jsonl", IntegrationEvalCase),
        real_cases=_load_jsonl_if_exists(root / "real_cases.jsonl", RealEvalCase),
    )


def _load_jsonl(path: Path, model: type[CaseModel]) -> list[CaseModel]:
    cases: list[CaseModel] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        cases.append(model.model_validate(payload))
    return cases


def _load_jsonl_if_exists(path: Path, model: type[CaseModel]) -> list[CaseModel]:
    if not path.exists():
        return []
    return _load_jsonl(path, model)
