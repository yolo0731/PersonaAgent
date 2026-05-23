from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str


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
