from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent_service.llm.base import LLMMessage
from agent_service.rag.documents import RetrievalTrace
from agent_service.schemas import ChatRequest


class PersonaConfigError(ValueError):
    """Raised when persona configuration cannot be loaded or validated."""


class PromptTemplates(BaseModel):
    system: str = Field(min_length=1)
    developer: str = Field(min_length=1)
    user: str = Field(min_length=1)


class PersonaConfig(BaseModel):
    persona_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    identity_notice: str = Field(min_length=1)
    style_instruction: str = Field(min_length=1)
    safety_boundaries: list[str] = Field(min_length=1)
    prompt_templates: PromptTemplates

    @field_validator("identity_notice")
    @classmethod
    def _identity_notice_must_disclose_agent(cls, value: str) -> str:
        lowered = value.casefold()
        if "ai" not in lowered and "agent" not in lowered:
            raise ValueError("identity_notice must disclose that PersonaAgent is an AI Agent")
        return value


class PromptMetadata(BaseModel):
    prompt_version: str
    persona_id: str
    used_context_ids: list[str] = Field(default_factory=list)
    used_knowledge_ids: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_style_sample_ids: list[str] = Field(default_factory=list)
    style_fallback_used: bool = False


class PersonaPrompt(BaseModel):
    messages: list[LLMMessage]
    metadata: PromptMetadata


class PersonaEngine:
    def __init__(self, config: PersonaConfig) -> None:
        self.config = config

    @classmethod
    def from_default(cls) -> PersonaEngine:
        return cls.from_file(Path(__file__).with_name("persona.yaml"))

    @classmethod
    def from_file(cls, path: str | Path) -> PersonaEngine:
        config_path = Path(path)
        try:
            raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw_config, dict):
                raise PersonaConfigError("persona.yaml must contain a mapping")
            return cls(PersonaConfig.model_validate(raw_config))
        except ValidationError as exc:
            raise PersonaConfigError(str(exc)) from exc
        except yaml.YAMLError as exc:
            raise PersonaConfigError(f"invalid persona.yaml: {exc}") from exc
        except OSError as exc:
            raise PersonaConfigError(f"cannot read persona.yaml: {exc}") from exc

    def build_prompt(
        self,
        *,
        request: ChatRequest,
        retrieved_context: Sequence[str],
        retrieval_trace: Sequence[RetrievalTrace],
        tool_results: Sequence[str] = (),
    ) -> PersonaPrompt:
        context = _PromptContext.from_retrieved_context(retrieved_context, tool_results)
        used_context = _used_context_ids(retrieval_trace)
        context_block = context.to_prompt_block(self.config.style_instruction)
        values = {
            "display_name": self.config.display_name,
            "identity_notice": self.config.identity_notice,
            "prompt_version": self.config.prompt_version,
            "style_instruction": self.config.style_instruction,
            "safety_boundaries": _bullet_list(self.config.safety_boundaries),
            "context_block": context_block,
            "user_text": request.text,
        }
        return PersonaPrompt(
            messages=[
                LLMMessage(
                    role="system",
                    content=self.config.prompt_templates.system.format(**values),
                ),
                LLMMessage(
                    role="developer",
                    content=self.config.prompt_templates.developer.format(**values),
                ),
                LLMMessage(
                    role="user",
                    content=self.config.prompt_templates.user.format(**values),
                ),
            ],
            metadata=PromptMetadata(
                prompt_version=self.config.prompt_version,
                persona_id=self.config.persona_id,
                used_context_ids=used_context.all_ids,
                used_knowledge_ids=used_context.knowledge_ids,
                used_memory_ids=used_context.memory_ids,
                used_style_sample_ids=used_context.style_sample_ids,
                style_fallback_used=context.style_fallback_used,
            ),
        )


class _PromptContext(BaseModel):
    knowledge: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)
    style_summaries: list[str] = Field(default_factory=list)
    style_fallback_used: bool = False
    tools: list[str] = Field(default_factory=list)

    @classmethod
    def from_retrieved_context(
        cls,
        retrieved_context: Sequence[str],
        tool_results: Sequence[str],
    ) -> _PromptContext:
        context = cls(tools=list(tool_results))
        for item in retrieved_context:
            if item.startswith("knowledge:"):
                context.knowledge.append(_strip_prefix(item, "knowledge:"))
            elif item.startswith("memory:"):
                context.memory.append(_strip_prefix(item, "memory:"))
            elif item.startswith("memory_saved:") or item.startswith("memory_deactivated:"):
                context.memory.append(item)
            elif item.startswith("style_summary:"):
                context.style_summaries.append(_strip_prefix(item, "style_summary:"))
            elif item.startswith("style_fallback:"):
                context.style_fallback_used = True
            elif item.startswith("style_example:"):
                continue
            elif item.strip():
                context.knowledge.append(item.strip())
        if not context.style_summaries:
            context.style_fallback_used = True
        return context

    def to_prompt_block(self, style_instruction: str) -> str:
        sections = [
            _section("Knowledge context", self.knowledge),
            _section("Memory context", self.memory),
            _section("Style guidance", self._style_lines(style_instruction)),
        ]
        if self.tools:
            sections.append(_section("Tool results", self.tools))
        return "\n\n".join(sections)

    def _style_lines(self, style_instruction: str) -> list[str]:
        if self.style_fallback_used:
            return [
                "No authorized style samples are available.",
                style_instruction,
            ]
        return [*self.style_summaries, style_instruction]


class _UsedContextIds(BaseModel):
    all_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    style_sample_ids: list[str] = Field(default_factory=list)


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :].strip()


def _section(title: str, lines: Sequence[str]) -> str:
    if not lines:
        return f"{title}:\n- none"
    return f"{title}:\n{_bullet_list(lines)}"


def _bullet_list(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines if line.strip())


def _used_context_ids(retrieval_trace: Sequence[RetrievalTrace]) -> _UsedContextIds:
    all_ids: list[str] = []
    knowledge_ids: list[str] = []
    memory_ids: list[str] = []
    style_sample_ids: list[str] = []
    seen: set[str] = set()
    for trace in retrieval_trace:
        for context_id in trace.chunk_ids:
            if context_id not in seen:
                seen.add(context_id)
                all_ids.append(context_id)
            if trace.collection == "knowledge" and context_id not in knowledge_ids:
                knowledge_ids.append(context_id)
            elif trace.collection == "memory" and context_id not in memory_ids:
                memory_ids.append(context_id)
            elif trace.collection == "style" and context_id not in style_sample_ids:
                style_sample_ids.append(context_id)
    return _UsedContextIds(
        all_ids=all_ids,
        knowledge_ids=knowledge_ids,
        memory_ids=memory_ids,
        style_sample_ids=style_sample_ids,
    )
