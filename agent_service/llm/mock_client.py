from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


# MockLLMClient 不访问真实网络，用固定结果保证 workflow 单元测试稳定。
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
            # response_model 存在时返回可校验对象，后续 LangGraph 节点可以稳定测试。
            payload = self._structured_payload(messages)
            structured = response_model.model_validate(payload)
            content = self._content_from_model(structured)
            return LLMResponse(
                content=content,
                model=self.model,
                structured=structured,
                raw=dict(payload),
            )

        # 未指定 response_model 时返回普通文本，模拟最简单的聊天回复。
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
        return {"reply_text": f"mock reply: {_clean_user_message(last_user_message)}"}

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


def _clean_user_message(content: str) -> str:
    marker = "User message:"
    if marker in content:
        return content.split(marker, maxsplit=1)[1].strip()
    return content or "mock reply"
