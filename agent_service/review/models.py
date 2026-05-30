from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from agent_service.schemas import AgentReplyCommand, ChatRequest


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    EXPIRED = "expired"


class HumanReviewRecord(BaseModel):
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: ReviewStatus
    request: ChatRequest
    edited_text: str | None = None
    risk_reason: str | None = None
    expires_at: str | None = None
    created_at: str
    updated_at: str


class EditReviewRequest(BaseModel):
    edited_text: str = Field(min_length=1)
    operator: str = Field(default="local-admin", min_length=1)


class ApproveReviewRequest(BaseModel):
    edited_text: str | None = Field(default=None, min_length=1)
    operator: str = Field(default="local-admin", min_length=1)


class HumanReviewList(BaseModel):
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    items: list[HumanReviewRecord]


class HumanReviewAuditEntry(BaseModel):
    audit_id: int = Field(ge=1)
    thread_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    action: str = Field(min_length=1)
    before_status: ReviewStatus | None = None
    after_status: ReviewStatus | None = None
    edited_text: str | None = None
    final_command: AgentReplyCommand | None = None
    created_at: str


class HumanReviewDetail(BaseModel):
    record: HumanReviewRecord
    checkpoint_status: str
    final_command: AgentReplyCommand | None = None
    agent_draft: str = ""
    retrieved_context: list[str] = Field(default_factory=list)
    tool_results: list[str] = Field(default_factory=list)
    trace_summary: list[str] = Field(default_factory=list)
    audit_log: list[HumanReviewAuditEntry] = Field(default_factory=list)


class HumanReviewNotFoundError(KeyError):
    """Raised when a review thread is missing from the local store."""


class HumanReviewInvalidTransitionError(ValueError):
    """Raised when a review mutation would not change the current state."""
