from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


class MockLLMClient(LLMClient):
    def __init__(
        self,
        fixed_response: Mapping[str, Any] | str | None = None,
        model: str = "mock",
    ) -> None:
        self._fixed_response = fixed_response
        self.model = model

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        if response_model is not None:
            payload = self._structured_payload(messages)
            structured = response_model.model_validate(payload)
            content = self._content_from_model(structured)
            return LLMResponse(
                content=content,
                model=self.model,
                structured=structured,
                raw=dict(payload),
            )

        content = self._text_payload(messages)
        return LLMResponse(content=content, model=self.model, raw={"content": content})

    def _structured_payload(self, messages: Sequence[LLMMessage]) -> Mapping[str, Any]:
        if isinstance(self._fixed_response, Mapping):
            return self._fixed_response
        if isinstance(self._fixed_response, str):
            return {"reply_text": self._fixed_response}
        last_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return {"reply_text": last_user_message or "mock reply"}

    def _text_payload(self, messages: Sequence[LLMMessage]) -> str:
        if isinstance(self._fixed_response, str):
            return self._fixed_response
        if isinstance(self._fixed_response, Mapping):
            reply_text = self._fixed_response.get("reply_text")
            if reply_text is not None:
                return str(reply_text)
            content = self._fixed_response.get("content")
            if content is not None:
                return str(content)
        last_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return last_user_message or "mock response"

    def _content_from_model(self, structured: BaseModel) -> str:
        for field_name in ("reply_text", "content", "text"):
            value = getattr(structured, field_name, None)
            if value is not None:
                return str(value)
        return structured.model_dump_json()
