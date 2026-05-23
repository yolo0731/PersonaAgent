from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field


# LLMMessage 是所有模型客户端共同使用的输入消息结构。
class LLMMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str


# LLMResponse 统一封装文本、结构化结果和 token 使用量，避免上层依赖具体 SDK 响应。
class LLMResponse(BaseModel):
    content: str
    model: str
    structured: BaseModel | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Generate an LLM response, optionally parsed into a Pydantic schema."""
