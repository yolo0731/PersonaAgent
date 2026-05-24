from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from agent_service.llm import LLMClient, MockLLMClient
from agent_service.persona import PersonaPrompt
from agent_service.schemas import ChatRequest


class ReplyDraft(BaseModel):
    reply_text: str = Field(min_length=1)
    reason: str = Field(default="llm_generated", min_length=1)
    used_knowledge_ids: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_style_sample_ids: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class GenerationTrace(BaseModel):
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)
    fallback_used: bool = False
    error_message: str | None = None
    used_knowledge_ids: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_style_sample_ids: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    draft: ReplyDraft
    trace: GenerationTrace


class LLMReplyGenerator:
    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_retries: int = 2,
    ) -> None:
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        self._llm_client = llm_client or MockLLMClient()
        self._max_retries = max_retries

    def generate(self, *, request: ChatRequest, prompt: PersonaPrompt) -> GenerationResult:
        started_at = perf_counter()
        context_ids = _context_ids(prompt)
        prompt_tokens = 0
        completion_tokens = 0
        attempts = 0
        model = "unknown"
        error_message: str | None = None

        for _ in range(self._max_retries):
            attempts += 1
            try:
                response = _run_async(
                    self._llm_client.generate(prompt.messages, response_model=ReplyDraft)
                )
                model = response.model
                prompt_tokens += response.prompt_tokens
                completion_tokens += response.completion_tokens
                if not isinstance(response.structured, ReplyDraft):
                    raise ValueError("LLM response did not contain a ReplyDraft")
                draft = _with_context_ids(response.structured, context_ids)
                return GenerationResult(
                    draft=draft,
                    trace=GenerationTrace(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=_duration_ms(started_at),
                        attempts=attempts,
                        used_knowledge_ids=draft.used_knowledge_ids,
                        used_memory_ids=draft.used_memory_ids,
                        used_style_sample_ids=draft.used_style_sample_ids,
                    ),
                )
            except (ValidationError, ValueError, RuntimeError, TypeError) as exc:
                error_message = str(exc)

        fallback = ReplyDraft(
            reply_text=f"mock reply: {request.text}",
            reason="llm_fallback",
            used_knowledge_ids=context_ids.knowledge,
            used_memory_ids=context_ids.memory,
            used_style_sample_ids=context_ids.style,
            fallback_used=True,
        )
        return GenerationResult(
            draft=fallback,
            trace=GenerationTrace(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=_duration_ms(started_at),
                attempts=attempts,
                fallback_used=True,
                error_message=error_message,
                used_knowledge_ids=fallback.used_knowledge_ids,
                used_memory_ids=fallback.used_memory_ids,
                used_style_sample_ids=fallback.used_style_sample_ids,
            ),
        )


class _ContextIds(BaseModel):
    knowledge: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)


def _context_ids(prompt: PersonaPrompt) -> _ContextIds:
    return _ContextIds(
        knowledge=prompt.metadata.used_knowledge_ids,
        memory=prompt.metadata.used_memory_ids,
        style=prompt.metadata.used_style_sample_ids,
    )


def _with_context_ids(draft: ReplyDraft, context_ids: _ContextIds) -> ReplyDraft:
    return draft.model_copy(
        update={
            "used_knowledge_ids": draft.used_knowledge_ids or context_ids.knowledge,
            "used_memory_ids": draft.used_memory_ids or context_ids.memory,
            "used_style_sample_ids": draft.used_style_sample_ids or context_ids.style,
        }
    )


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def _duration_ms(started_at: float) -> float:
    return max((perf_counter() - started_at) * 1000.0, 0.0)
