from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from agent_service.governance.pii_redactor import PiiRedactor

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")
SAFE_REWRITE_TEXT = "我会保持相近的简洁语气，但不会复述授权样本原文。"

LeakageAction = Literal["pass", "rewrite", "block"]


class LeakageSource(BaseModel):
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class LeakageMetrics(BaseModel):
    verbatim_leakage_rate: float = Field(ge=0.0, le=1.0)
    max_ngram_overlap: float = Field(ge=0.0, le=1.0)
    max_lcs_ratio: float = Field(ge=0.0, le=1.0)
    pii_leak_count: int = Field(ge=0)
    source_ids: list[str] = Field(default_factory=list)


class LeakageAssessment(BaseModel):
    action: LeakageAction
    reason: str
    safe_text: str
    metrics: LeakageMetrics


class VerbatimLeakageGuard:
    def __init__(
        self,
        *,
        ngram_size: int = 4,
        min_source_tokens: int = 6,
        rewrite_ngram_overlap_threshold: float = 0.45,
        rewrite_lcs_ratio_threshold: float = 0.60,
        block_lcs_ratio_threshold: float = 0.90,
        redactor: PiiRedactor | None = None,
    ) -> None:
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if min_source_tokens <= 0:
            raise ValueError("min_source_tokens must be positive")
        self._ngram_size = ngram_size
        self._min_source_tokens = min_source_tokens
        self._rewrite_ngram_overlap_threshold = rewrite_ngram_overlap_threshold
        self._rewrite_lcs_ratio_threshold = rewrite_lcs_ratio_threshold
        self._block_lcs_ratio_threshold = block_lcs_ratio_threshold
        self._redactor = redactor or PiiRedactor()

    def assess(
        self,
        candidate_text: str,
        sources: Sequence[LeakageSource],
    ) -> LeakageAssessment:
        metrics = self._metrics(candidate_text, sources)
        if metrics.pii_leak_count > 0:
            return LeakageAssessment(
                action="block",
                reason="pii_leak_detected",
                safe_text="",
                metrics=metrics,
            )
        if _leaks_source_id(candidate_text, sources):
            return LeakageAssessment(
                action="block",
                reason="style_source_id_leak",
                safe_text="",
                metrics=metrics,
            )
        if metrics.max_lcs_ratio >= self._block_lcs_ratio_threshold:
            return LeakageAssessment(
                action="block",
                reason="direct_verbatim_copy",
                safe_text="",
                metrics=metrics,
            )
        if (
            metrics.max_ngram_overlap >= self._rewrite_ngram_overlap_threshold
            or metrics.max_lcs_ratio >= self._rewrite_lcs_ratio_threshold
        ):
            return LeakageAssessment(
                action="rewrite",
                reason="high_verbatim_overlap",
                safe_text=SAFE_REWRITE_TEXT,
                metrics=metrics,
            )
        return LeakageAssessment(
            action="pass",
            reason="passed",
            safe_text=candidate_text,
            metrics=metrics,
        )

    def _metrics(self, candidate_text: str, sources: Sequence[LeakageSource]) -> LeakageMetrics:
        redaction = self._redactor.redact(candidate_text)
        candidate_tokens = _tokens(candidate_text)
        candidate_norm = "".join(candidate_tokens)
        candidate_ngrams = _ngrams(candidate_tokens, self._ngram_size)
        max_ngram_overlap = 0.0
        max_lcs_ratio = 0.0
        leaking_source_ids: list[str] = []

        for source in sources:
            source_tokens = _tokens(source.text)
            if len(source_tokens) < self._min_source_tokens:
                continue
            source_norm = "".join(source_tokens)
            source_ngrams = _ngrams(source_tokens, self._ngram_size)
            ngram_overlap = _overlap_ratio(candidate_ngrams, source_ngrams)
            lcs_ratio = _lcs_ratio(candidate_norm, source_norm)
            max_ngram_overlap = max(max_ngram_overlap, ngram_overlap)
            max_lcs_ratio = max(max_lcs_ratio, lcs_ratio)
            if (
                lcs_ratio >= self._rewrite_lcs_ratio_threshold
                or ngram_overlap >= self._rewrite_ngram_overlap_threshold
                or source.source_id.lower() in candidate_text.lower()
            ):
                leaking_source_ids.append(source.source_id)

        unique_source_ids = list(dict.fromkeys(leaking_source_ids))
        return LeakageMetrics(
            verbatim_leakage_rate=_leakage_rate(len(unique_source_ids), len(sources)),
            max_ngram_overlap=round(max_ngram_overlap, 4),
            max_lcs_ratio=round(max_lcs_ratio, 4),
            pii_leak_count=sum(redaction.replacements.values()),
            source_ids=unique_source_ids,
        )


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _ngrams(tokens: Sequence[str], ngram_size: int) -> set[tuple[str, ...]]:
    if len(tokens) < ngram_size:
        return set()
    return {
        tuple(tokens[index : index + ngram_size])
        for index in range(0, len(tokens) - ngram_size + 1)
    }


def _overlap_ratio(
    candidate_ngrams: set[tuple[str, ...]],
    source_ngrams: set[tuple[str, ...]],
) -> float:
    if not candidate_ngrams or not source_ngrams:
        return 0.0
    return len(candidate_ngrams & source_ngrams) / len(source_ngrams)


def _lcs_ratio(candidate_norm: str, source_norm: str) -> float:
    if not candidate_norm or not source_norm:
        return 0.0
    lcs_length = _longest_common_substring_length(candidate_norm, source_norm)
    return lcs_length / len(source_norm)


def _longest_common_substring_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    best = 0
    for left_index in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for right_index in range(1, len(right) + 1):
            if left[left_index - 1] == right[right_index - 1]:
                current[right_index] = previous[right_index - 1] + 1
                best = max(best, current[right_index])
        previous = current
    return best


def _leaks_source_id(candidate_text: str, sources: Sequence[LeakageSource]) -> bool:
    lowered = candidate_text.lower()
    return any(source.source_id.lower() in lowered for source in sources)


def _leakage_rate(leaking_source_count: int, total_source_count: int) -> float:
    if total_source_count == 0:
        return 0.0
    return round(leaking_source_count / total_source_count, 4)
