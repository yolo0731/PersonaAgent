from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from agent_service.governance.pii_redactor import PiiRedactor
from agent_service.llm.base import LLMMessage
from agent_service.safety.verbatim_guard import (
    LeakageMetrics,
    LeakageSource,
    VerbatimLeakageGuard,
)
from agent_service.schemas import ChatRequest


class SafetyTrace(BaseModel):
    policy_version: str = "safety-v1"
    identity_notice_present: bool
    risks: list[str] = Field(default_factory=list)
    high_risk_categories: list[str] = Field(default_factory=list)
    style_source_ids: list[str] = Field(default_factory=list)


class SafetyAssessment(BaseModel):
    blocked: bool
    needs_human_review: bool = False
    reason: str | None = None
    safe_text: str | None = None
    metrics: LeakageMetrics | None = None
    trace: SafetyTrace


class SafetyGuard:
    def __init__(
        self,
        *,
        leakage_guard: VerbatimLeakageGuard | None = None,
        redactor: PiiRedactor | None = None,
    ) -> None:
        self._leakage_guard = leakage_guard or VerbatimLeakageGuard()
        self._redactor = redactor or PiiRedactor()

    def assess(
        self,
        *,
        request: ChatRequest,
        draft: str,
        prompt_messages: Sequence[LLMMessage],
        retrieved_context: Sequence[str],
        style_sources: Sequence[LeakageSource],
        unsafe_decision: bool = False,
    ) -> SafetyAssessment:
        identity_notice_present = _has_ai_identity_notice(prompt_messages)
        high_risk_categories = _high_risk_categories(request.text, draft)

        if not identity_notice_present:
            return _blocked(
                "missing_ai_identity_notice",
                identity_notice_present=identity_notice_present,
                risks=["missing_ai_identity_notice"],
                high_risk_categories=high_risk_categories,
                style_source_ids=[],
            )

        if unsafe_decision:
            return _blocked(
                "unsafe_request",
                identity_notice_present=identity_notice_present,
                risks=["unsafe_request"],
                high_risk_categories=high_risk_categories,
                style_source_ids=[],
            )

        if _has_impersonation_attempt(request.text, draft):
            return _blocked(
                "impersonation_attempt",
                identity_notice_present=identity_notice_present,
                risks=["impersonation_attempt"],
                high_risk_categories=high_risk_categories,
                style_source_ids=[],
            )

        if _has_unauthorized_style_mimicry(request.text, retrieved_context):
            return _blocked(
                "unauthorized_style_mimicry",
                identity_notice_present=identity_notice_present,
                risks=["unauthorized_style_mimicry"],
                high_risk_categories=high_risk_categories,
                style_source_ids=[],
            )

        privacy_leak_count = sum(self._redactor.redact(draft).replacements.values())
        if privacy_leak_count > 0:
            return _blocked(
                "privacy_leak",
                identity_notice_present=identity_notice_present,
                risks=["privacy_leak"],
                high_risk_categories=high_risk_categories,
                style_source_ids=[],
            )

        if style_sources:
            leakage = self._leakage_guard.assess(draft, style_sources)
            if leakage.action == "block":
                return _blocked(
                    leakage.reason,
                    identity_notice_present=identity_notice_present,
                    risks=[leakage.reason],
                    high_risk_categories=high_risk_categories,
                    style_source_ids=leakage.metrics.source_ids,
                    metrics=leakage.metrics,
                )
            if leakage.action == "rewrite":
                trace = SafetyTrace(
                    identity_notice_present=identity_notice_present,
                    risks=[leakage.reason],
                    high_risk_categories=high_risk_categories,
                    style_source_ids=leakage.metrics.source_ids,
                )
                return SafetyAssessment(
                    blocked=False,
                    needs_human_review=bool(high_risk_categories),
                    reason=_review_reason(high_risk_categories) or leakage.reason,
                    safe_text=leakage.safe_text,
                    metrics=leakage.metrics,
                    trace=trace,
                )

        if high_risk_categories:
            return SafetyAssessment(
                blocked=False,
                needs_human_review=True,
                reason=_review_reason(high_risk_categories),
                safe_text=draft,
                trace=SafetyTrace(
                    identity_notice_present=identity_notice_present,
                    risks=["high_risk_domain"],
                    high_risk_categories=high_risk_categories,
                    style_source_ids=[],
                ),
            )

        return SafetyAssessment(
            blocked=False,
            safe_text=draft,
            trace=SafetyTrace(
                identity_notice_present=identity_notice_present,
                risks=[],
                high_risk_categories=[],
                style_source_ids=[],
            ),
        )


def _blocked(
    reason: str,
    *,
    identity_notice_present: bool,
    risks: list[str],
    high_risk_categories: list[str],
    style_source_ids: list[str],
    metrics: LeakageMetrics | None = None,
) -> SafetyAssessment:
    return SafetyAssessment(
        blocked=True,
        reason=reason,
        metrics=metrics,
        trace=SafetyTrace(
            identity_notice_present=identity_notice_present,
            risks=risks,
            high_risk_categories=high_risk_categories,
            style_source_ids=style_source_ids,
        ),
    )


def _has_ai_identity_notice(prompt_messages: Sequence[LLMMessage]) -> bool:
    joined = "\n".join(message.content for message in prompt_messages).casefold()
    if "ai" not in joined or ("agent" not in joined and "assistant" not in joined):
        return False
    return "not a real person" in joined or "不能冒充" in joined or "不是真人" in joined


def _has_impersonation_attempt(request_text: str, draft: str) -> bool:
    text = f"{request_text}\n{draft}".casefold()
    return any(
        marker in text
        for marker in (
            "冒充",
            "伪装成",
            "假装你是",
            "本人身份",
            "替他说",
            "pretend you are",
            "as the real",
            "claim to be",
        )
    ) or re.search(r"我是.{0,12}本人", text) is not None


def _has_unauthorized_style_mimicry(
    request_text: str,
    retrieved_context: Sequence[str],
) -> bool:
    if not _requests_specific_style_mimicry(request_text):
        return False
    return not any(item.startswith("style_summary:") for item in retrieved_context)


def _requests_specific_style_mimicry(request_text: str) -> bool:
    lowered = request_text.casefold()
    if "imitate" in lowered or "mimic" in lowered:
        return True
    if "模仿" in request_text or "口吻" in request_text:
        return True
    return "风格" in request_text and "我的风格" not in request_text and "像我" not in request_text


def _high_risk_categories(request_text: str, draft: str) -> list[str]:
    text = f"{request_text}\n{draft}".casefold()
    categories: list[str] = []
    checks = {
        "money": ("转账", "汇款", "股票", "收益", "贷款", "投资", "money", "payment", "bank"),
        "legal": ("法律", "起诉", "合同", "律师", "legal", "lawsuit", "contract"),
        "medical": ("用药", "剂量", "医生", "医疗", "症状", "medical", "medicine", "dosage"),
        "account": ("账号", "密码", "验证码", "登录", "account", "password", "otp"),
        "real_world_commitment": (
            "承诺",
            "保证",
            "签署",
            "预约",
            "下单",
            "替你",
            "替我",
            "commit",
            "promise",
            "sign",
            "book",
        ),
    }
    for category, markers in checks.items():
        if any(marker in text for marker in markers):
            categories.append(category)
    return categories


def _review_reason(categories: Sequence[str]) -> str | None:
    if not categories:
        return None
    return f"high_risk_domain:{categories[0]}"
