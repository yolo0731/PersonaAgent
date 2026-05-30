from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from agent_service.llm import LLMClient, LLMResponse
from agent_service.llm.base import LLMMessage
from agent_service.memory.memory_tools import parse_remember_content
from agent_service.schemas import ChatRequest

PRIVATE_CONVERSATION_TYPE = 1
DialoguePolicyMode = Literal["rule", "llm"]


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


class LLMStructuredDialogueClient:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        timeout_seconds: float | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._timeout_seconds = timeout_seconds

    def decide(self, request: ChatRequest) -> dict[str, object]:
        response = _run_async(self._generate(request))
        if not isinstance(response.structured, DialogueDecision):
            return DialogueDecision.model_validate_json(response.content).model_dump(mode="json")
        return response.structured.model_dump(mode="json")

    async def _generate(self, request: ChatRequest) -> LLMResponse:
        coroutine = self._llm_client.generate(
            _decision_prompt(request),
            response_model=DialogueDecision,
        )
        if self._timeout_seconds is None:
            return await coroutine
        return await asyncio.wait_for(coroutine, timeout=self._timeout_seconds)


class MockStructuredDialogueClient:
    def decide(self, request: ChatRequest) -> dict[str, object]:
        return _rule_decision(request, reason_prefix="mock").model_dump(mode="json")


class DialoguePolicy:
    def __init__(
        self,
        *,
        client: StructuredDecisionClient | None = None,
        mode: DialoguePolicyMode = "rule",
        llm_client: LLMClient | None = None,
        max_retries: int = 2,
        timeout_seconds: float | None = None,
    ) -> None:
        if client is not None:
            self._client: StructuredDecisionClient | None = client
        elif mode == "llm" and llm_client is not None:
            self._client = LLMStructuredDialogueClient(
                llm_client=llm_client,
                timeout_seconds=timeout_seconds,
            )
        elif mode == "rule":
            self._client = MockStructuredDialogueClient()
        else:
            self._client = None
        self._max_retries = max(1, max_retries)

    def decide(self, request: ChatRequest) -> DialogueDecision:
        if self._client is None:
            return _rule_decision(request, reason_prefix="fallback")
        for _ in range(self._max_retries):
            try:
                return DialogueDecision.model_validate(self._client.decide(request))
            except Exception:
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

    if lowered.startswith("/remember") or parse_remember_content(text) is not None:
        return DialogueDecision(
            intent=DialogueIntent.MEMORY_UPDATE,
            should_reply=True,
            need_memory=True,
            reason=f"{reason_prefix}_memory_update",
        )

    if lowered.startswith("/forget "):
        return DialogueDecision(
            intent=DialogueIntent.MEMORY_UPDATE,
            should_reply=True,
            need_memory=True,
            reason=f"{reason_prefix}_memory_forget",
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


def _decision_prompt(request: ChatRequest) -> Sequence[LLMMessage]:
    return [
        LLMMessage(
            role="developer",
            content=(
                "Classify the incoming PersonaAgent chat request. Return only a JSON object "
                "that matches DialogueDecision. Do not generate a reply. The decision controls "
                "whether the workflow retrieves knowledge, memory, authorized style, tools, or "
                "human review."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"conversation_type={request.conversation_type}\n"
                f"conversation_id={request.conversation_id}\n"
                f"sender_id={request.sender_id}\n"
                f"message_id={request.message_id}\n"
                f"text={request.text}"
            ),
        ),
    ]


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
        or ("记得" in text and "吗" in text)
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


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()
