from __future__ import annotations

from pydantic import BaseModel, Field

from agent_service.memory.memory_store import MemoryNotFoundError, MemoryStore
from agent_service.tools.registry import ToolRegistry, ToolRuntimeContext, ToolSpec


class SaveMemoryInput(BaseModel):
    user_id: int = Field(ge=1)
    content: str = Field(min_length=1)
    source_message_id: int = Field(ge=1)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    idempotency_key: str = Field(min_length=1)


class SaveMemoryOutput(BaseModel):
    memory_id: str
    user_id: int
    content: str
    source_message_id: int
    active: bool
    importance: float
    idempotency_key: str


class DeactivateMemoryInput(BaseModel):
    memory_id: str = Field(min_length=1)
    user_id: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1)


class DeactivateMemoryOutput(BaseModel):
    memory_id: str
    user_id: int
    active: bool
    idempotency_key: str


class GetUserProfileInput(BaseModel):
    user_id: int = Field(ge=1)


class GetUserProfileOutput(BaseModel):
    user_id: int
    display_name: str
    profile_source: str
    is_current_sender: bool


class SummarizeRecentContextInput(BaseModel):
    max_items: int = Field(default=5, ge=1, le=20)


class SummarizeRecentContextOutput(BaseModel):
    summary: str
    item_count: int


class SearchRecentContextInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchRecentContextOutput(BaseModel):
    matches: list[str]
    result_count: int


class LiteImContextToolInput(BaseModel):
    include_transport_boundary: bool = True


class LiteImContextToolOutput(BaseModel):
    boundary: str
    can_send_message: bool
    can_access_liteim_db: bool
    can_hold_liteim_tcp: bool


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="save_memory",
            input_model=SaveMemoryInput,
            output_model=SaveMemoryOutput,
            handler=_save_memory,
            side_effect=True,
        )
    )
    registry.register(
        ToolSpec(
            name="deactivate_memory",
            input_model=DeactivateMemoryInput,
            output_model=DeactivateMemoryOutput,
            handler=_deactivate_memory,
            side_effect=True,
        )
    )
    registry.register(
        ToolSpec(
            name="get_user_profile",
            input_model=GetUserProfileInput,
            output_model=GetUserProfileOutput,
            handler=_get_user_profile,
        )
    )
    registry.register(
        ToolSpec(
            name="summarize_recent_context",
            input_model=SummarizeRecentContextInput,
            output_model=SummarizeRecentContextOutput,
            handler=_summarize_recent_context,
        )
    )
    registry.register(
        ToolSpec(
            name="search_recent_context",
            input_model=SearchRecentContextInput,
            output_model=SearchRecentContextOutput,
            handler=_search_recent_context,
        )
    )
    registry.register(
        ToolSpec(
            name="liteim_context_tool",
            input_model=LiteImContextToolInput,
            output_model=LiteImContextToolOutput,
            handler=_liteim_context_tool,
        )
    )
    return registry


def _save_memory(tool_input: BaseModel, context: ToolRuntimeContext) -> SaveMemoryOutput:
    data = SaveMemoryInput.model_validate(tool_input)
    store = _require_memory_store(context)
    record = store.save_memory(
        user_id=data.user_id,
        content=data.content,
        source_message_id=data.source_message_id,
        importance=data.importance,
    )
    return SaveMemoryOutput(
        memory_id=record.memory_id,
        user_id=record.user_id,
        content=record.content,
        source_message_id=record.source_message_id,
        active=record.active,
        importance=record.importance,
        idempotency_key=data.idempotency_key,
    )


def _deactivate_memory(
    tool_input: BaseModel,
    context: ToolRuntimeContext,
) -> DeactivateMemoryOutput:
    data = DeactivateMemoryInput.model_validate(tool_input)
    store = _require_memory_store(context)
    try:
        record = store.deactivate_memory(data.memory_id, user_id=data.user_id)
    except MemoryNotFoundError as exc:
        raise RuntimeError(f"memory not found: {data.memory_id}") from exc
    return DeactivateMemoryOutput(
        memory_id=record.memory_id,
        user_id=record.user_id,
        active=record.active,
        idempotency_key=data.idempotency_key,
    )


def _get_user_profile(tool_input: BaseModel, context: ToolRuntimeContext) -> GetUserProfileOutput:
    data = GetUserProfileInput.model_validate(tool_input)
    is_current_sender = context.request is not None and context.request.sender_id == data.user_id
    return GetUserProfileOutput(
        user_id=data.user_id,
        display_name=f"user-{data.user_id}",
        profile_source="chat_request" if is_current_sender else "tool_input",
        is_current_sender=is_current_sender,
    )


def _summarize_recent_context(
    tool_input: BaseModel,
    context: ToolRuntimeContext,
) -> SummarizeRecentContextOutput:
    data = SummarizeRecentContextInput.model_validate(tool_input)
    items = list(context.recent_context[: data.max_items])
    return SummarizeRecentContextOutput(
        summary=" | ".join(items),
        item_count=len(items),
    )


def _search_recent_context(
    tool_input: BaseModel,
    context: ToolRuntimeContext,
) -> SearchRecentContextOutput:
    data = SearchRecentContextInput.model_validate(tool_input)
    lowered_query = data.query.casefold()
    matches = [
        item
        for item in context.recent_context
        if lowered_query in item.casefold()
    ][: data.top_k]
    return SearchRecentContextOutput(matches=matches, result_count=len(matches))


def _liteim_context_tool(
    tool_input: BaseModel,
    _context: ToolRuntimeContext,
) -> LiteImContextToolOutput:
    data = LiteImContextToolInput.model_validate(tool_input)
    boundary = "AgentService tools may read AgentService state only."
    if data.include_transport_boundary:
        boundary += " BotClient owns LiteIM TCP and is the only layer that sends LiteIM packets."
    return LiteImContextToolOutput(
        boundary=boundary,
        can_send_message=False,
        can_access_liteim_db=False,
        can_hold_liteim_tcp=False,
    )


def _require_memory_store(context: ToolRuntimeContext) -> MemoryStore:
    if context.memory_store is None:
        raise RuntimeError("memory_store is not configured")
    return context.memory_store
