from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast

from pydantic import BaseModel

from agent_service.llm.base import LLMClient, LLMMessage, LLMResponse


# OpenAILLMClient 使用 OpenAI SDK 兼容接口；当前真实运行默认指向 DeepSeek base_url。
class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str | None, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

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
        client_kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = async_openai(**client_kwargs)

        completion = await client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in messages],
        )
        choice = completion.choices[0]
        content = cast(str | None, choice.message.content) or ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        structured: BaseModel | None = None
        if response_model is not None:
            structured = response_model.model_validate_json(content)

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
