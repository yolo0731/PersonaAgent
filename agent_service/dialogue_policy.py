from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from agent_service.schemas import ChatRequest

PRIVATE_CONVERSATION_TYPE = 1


class DialogueIntent(StrEnum):
    SMALLTALK = "smalltalk"
    KNOWLEDGE_QUESTION = "knowledge_question"
    MEMORY_UPDATE = "memory_update"
    MEMORY_QUERY = "memory_query"
    STYLE_CHAT = "style_chat"
    HISTORY_SUMMARY = "history_summary"
    UNSAFE = "unsafe"
    COMMAND = "command"


class DialogueDecision(BaseModel):
    intent: DialogueIntent
    should_reply: bool
    need_knowledge: bool = False
    need_memory: bool = False
    need_style: bool = False
    need_tool: bool = False
    need_human_review: bool = False
    reason: str = Field(min_length=1)


class StructuredDecisionClient(Protocol):
    def decide(self, request: ChatRequest) -> dict[str, object]: ...


class MockStructuredDialogueClient:
    def decide(self, request: ChatRequest) -> dict[str, object]:
        return _rule_decision(request, reason_prefix="mock").model_dump(mode="json")


class DialoguePolicy:
    def __init__(
        self,
        *,
        client: StructuredDecisionClient | None = None,
        max_retries: int = 2,
    ) -> None:
        self._client = client or MockStructuredDialogueClient()
        self._max_retries = max(1, max_retries)

    def decide(self, request: ChatRequest) -> DialogueDecision:
        for _ in range(self._max_retries):
            try:
                return DialogueDecision.model_validate(self._client.decide(request))
            except (ValidationError, ValueError, TypeError):
                continue
        return _rule_decision(request, reason_prefix="fallback")


def _rule_decision(request: ChatRequest, *, reason_prefix: str) -> DialogueDecision:
    text = request.text.strip()
    lowered = text.lower()

    if request.conversation_type != PRIVATE_CONVERSATION_TYPE:
        return DialogueDecision(
            intent=DialogueIntent.SMALLTALK,
            should_reply=False,
            reason="group_no_reply",
        )

    if _is_unsafe(lowered):
        return DialogueDecision(
            intent=DialogueIntent.UNSAFE,
            should_reply=True,
            need_human_review=True,
            reason=f"{reason_prefix}_unsafe",
        )

    if lowered == "/no-reply":
        return DialogueDecision(
            intent=DialogueIntent.COMMAND,
            should_reply=False,
            need_tool=False,
            reason="command_no_reply",
        )

    if lowered.startswith("/remember") or text.startswith("记住"):
        return DialogueDecision(
            intent=DialogueIntent.MEMORY_UPDATE,
            should_reply=True,
            need_memory=True,
            reason=f"{reason_prefix}_memory_update",
        )

    if _is_history_summary(text, lowered):
        return DialogueDecision(
            intent=DialogueIntent.HISTORY_SUMMARY,
            should_reply=True,
            need_memory=True,
            reason=f"{reason_prefix}_history_summary",
        )

    if _is_memory_query(text, lowered):
        return DialogueDecision(
            intent=DialogueIntent.MEMORY_QUERY,
            should_reply=True,
            need_memory=True,
            reason=f"{reason_prefix}_memory_query",
        )

    if _is_style_chat(text, lowered):
        return DialogueDecision(
            intent=DialogueIntent.STYLE_CHAT,
            should_reply=True,
            need_style=True,
            reason=f"{reason_prefix}_style_chat",
        )

    if _is_knowledge_question(text, lowered):
        return DialogueDecision(
            intent=DialogueIntent.KNOWLEDGE_QUESTION,
            should_reply=True,
            need_knowledge=True,
            reason=f"{reason_prefix}_knowledge_question",
        )

    if text.startswith("/"):
        return DialogueDecision(
            intent=DialogueIntent.COMMAND,
            should_reply=True,
            need_tool=True,
            reason=f"{reason_prefix}_command",
        )

    return DialogueDecision(
        intent=DialogueIntent.SMALLTALK,
        should_reply=True,
        reason=f"{reason_prefix}_smalltalk",
    )


def _is_unsafe(lowered: str) -> bool:
    return (
        lowered.startswith("/unsafe")
        or "leak secrets" in lowered
        or "bypass authorization" in lowered
    )


def _is_history_summary(text: str, lowered: str) -> bool:
    return "history summary" in lowered or ("历史" in text and "总结" in text)


def _is_memory_query(text: str, lowered: str) -> bool:
    return (
        "memory query" in lowered
        or "what did i ask you to remember" in lowered
        or "之前" in text
        or "记住了什么" in text
    )


def _is_style_chat(text: str, lowered: str) -> bool:
    return "style" in lowered or "风格" in text or "像我" in text


def _is_knowledge_question(text: str, lowered: str) -> bool:
    return (
        "project" in lowered
        or "personaagent" in lowered
        or "liteim" in lowered
        or "项目" in text
    )
