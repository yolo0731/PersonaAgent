from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


# OpenAILLMClient 使用 OpenAI SDK 兼容接口；DeepSeek、本地代理等兼容端点共用这条路径。
class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI-compatible LLM client.")

        # 延迟导入 openai，避免 mock 单测在未配置真实 SDK 时被迫初始化。
        openai_module = import_module("openai")
        async_openai = openai_module.AsyncOpenAI
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client_kwargs["timeout"] = self.timeout_seconds
        client = async_openai(**client_kwargs)

        completion = await client.chat.completions.create(
            model=self.model,
            messages=_openai_compatible_messages(messages),
        )
        choice = completion.choices[0]
        content = cast(str | None, choice.message.content) or ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        structured: BaseModel | None = None
        if response_model is not None:
            structured = _parse_or_wrap_structured_response(response_model, content)

        raw: dict[str, Any] = {
            "id": getattr(completion, "id", None),
            "model": getattr(completion, "model", self.model),
        }

        # SDK 原始响应收敛成 LLMResponse，后续节点只依赖统一结构。
        return LLMResponse(
            content=content,
            model=self.model,
            structured=structured,
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _openai_compatible_messages(messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for message in messages:
        role = "system" if message.role == "developer" else message.role
        output.append({"role": role, "content": message.content})
    return output


def _parse_or_wrap_structured_response(
    response_model: type[BaseModel],
    content: str,
) -> BaseModel:
    try:
        return response_model.model_validate_json(content)
    except ValidationError:
        cleaned = content.strip()
        if not cleaned:
            raise
        if "reply_text" in response_model.model_fields:
            return response_model.model_validate({"reply_text": cleaned})
        raise
