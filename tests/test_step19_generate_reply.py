from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


def _chat_payload(text: str = "hello llm", *, run_id: str = "run-generate") -> dict[str, object]:
    return {
        "run_id": run_id,
        "conversation_type": 1,
        "conversation_id": 10011002,
        "message_id": 7001,
        "sender_id": 1002,
        "receiver_id": 1001,
        "text": text,
        "timestamp_ms": 1_700_000_001_000,
        "client_message_id": "alice-7001",
    }


class SequencedLLMClient(LLMClient):
    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StyleAwareLLMClient(LLMClient):
    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        from agent_service.generation import ReplyDraft

        joined = "\n".join(message.content for message in messages)
        text = "可以，我先给结论。" if "tone_particles=呀:2" in joined else "普通回复"
        return LLMResponse(
            content=text,
            model="style-aware-mock",
            structured=ReplyDraft(reply_text=text, reason="style_sensitive"),
            prompt_tokens=21,
            completion_tokens=5,
        )


def _prompt_with_context():
    from agent_service.persona import PersonaEngine
    from agent_service.rag.documents import RetrievalTrace
    from agent_service.schemas import ChatRequest

    request = ChatRequest.model_validate(_chat_payload("LiteIM 项目是什么？"))
    prompt = PersonaEngine.from_default().build_prompt(
        request=request,
        retrieved_context=[
            "knowledge: LiteIM uses epoll.",
            "memory: 用户喜欢先给结论",
            "style_summary: tone_particles=呀:2; avg_sentence_length=10",
        ],
        retrieval_trace=[
            RetrievalTrace(
                query=request.text,
                top_k=3,
                result_count=1,
                collection="knowledge",
                chunk_ids=["knowledge-doc-1"],
            ),
            RetrievalTrace(
                query=request.text,
                top_k=3,
                result_count=1,
                collection="memory",
                chunk_ids=["mem-1002-7001"],
            ),
            RetrievalTrace(
                query=request.text,
                top_k=3,
                result_count=1,
                collection="style",
                chunk_ids=["style-a"],
            ),
        ],
    )
    return request, prompt


def _llm_response(
    *,
    text: str,
    model: str = "mock-step19",
    structured: BaseModel | None,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> LLMResponse:
    return LLMResponse(
        content=text,
        model=model,
        structured=structured,
        raw={"response": text},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_reply_generator_returns_structured_draft_and_trace_metadata() -> None:
    from agent_service.generation import LLMReplyGenerator, ReplyDraft

    request, prompt = _prompt_with_context()
    client = SequencedLLMClient(
        [
            _llm_response(
                text="结构化回复",
                structured=ReplyDraft(reply_text="结构化回复", reason="ok"),
            )
        ]
    )

    result = LLMReplyGenerator(llm_client=client).generate(request=request, prompt=prompt)

    assert result.draft.reply_text == "结构化回复"
    assert result.draft.used_knowledge_ids == ["knowledge-doc-1"]
    assert result.draft.used_memory_ids == ["mem-1002-7001"]
    assert result.draft.used_style_sample_ids == ["style-a"]
    assert result.trace.model == "mock-step19"
    assert result.trace.prompt_tokens == 11
    assert result.trace.completion_tokens == 7
    assert result.trace.latency_ms >= 0.0
    assert result.trace.attempts == 1
    assert result.trace.fallback_used is False


def test_reply_generator_retries_non_structured_output() -> None:
    from agent_service.generation import LLMReplyGenerator, ReplyDraft

    request, prompt = _prompt_with_context()
    client = SequencedLLMClient(
        [
            _llm_response(text="not structured", structured=None),
            _llm_response(
                text="第二次成功",
                structured=ReplyDraft(reply_text="第二次成功", reason="retry_ok"),
            ),
        ]
    )

    result = LLMReplyGenerator(llm_client=client, max_retries=2).generate(
        request=request,
        prompt=prompt,
    )

    assert result.draft.reply_text == "第二次成功"
    assert result.trace.attempts == 2
    assert len(client.calls) == 2
    assert result.trace.fallback_used is False


def test_reply_generator_falls_back_after_repeated_failures() -> None:
    from agent_service.generation import LLMReplyGenerator

    request, prompt = _prompt_with_context()
    client = SequencedLLMClient([RuntimeError("bad output"), RuntimeError("still bad")])

    result = LLMReplyGenerator(llm_client=client, max_retries=2).generate(
        request=request,
        prompt=prompt,
    )

    assert result.draft.reply_text == "mock reply: LiteIM 项目是什么？"
    assert result.draft.fallback_used is True
    assert result.trace.fallback_used is True
    assert result.trace.error_message == "still bad"
    assert result.trace.attempts == 2


def test_style_context_can_affect_mock_reply() -> None:
    from agent_service.generation import LLMReplyGenerator

    request, prompt = _prompt_with_context()

    result = LLMReplyGenerator(llm_client=StyleAwareLLMClient()).generate(
        request=request,
        prompt=prompt,
    )

    assert result.draft.reply_text == "可以，我先给结论。"
    assert result.trace.model == "style-aware-mock"


def test_workflow_generate_reply_uses_llm_client_and_records_trace() -> None:
    from agent_service.generation import ReplyDraft
    from agent_service.schemas import ChatRequest
    from agent_service.workflow import run_agent_workflow

    client = SequencedLLMClient(
        [
            _llm_response(
                text="workflow llm answer",
                model="workflow-mock",
                structured=ReplyDraft(reply_text="workflow llm answer", reason="workflow_ok"),
                prompt_tokens=13,
                completion_tokens=6,
            )
        ]
    )

    state = run_agent_workflow(
        ChatRequest.model_validate(_chat_payload("hello workflow", run_id="run-workflow-llm")),
        llm_client=client,
    )

    assert state["draft"] == "workflow llm answer"
    assert state["reply_draft"] is not None
    assert state["reply_draft"].reason == "workflow_ok"
    assert state["generation_trace"] is not None
    assert state["generation_trace"].model == "workflow-mock"
    assert state["generation_trace"].prompt_tokens == 13
    assert any("model=workflow-mock" in event.action for event in state["trace"])
